from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from backend.core.data_structures import (
    CHROMOSOME_LENGTHS, GenomicArray, GenomicInterval, SparseCOO, VariantHashMap, VariantKey,
)
from backend.services.alphagenome_service import (
    ASSAYS, AlphaGenomeProvider, Column, MockProvider, summarize_tracks,
)


def test_array_alignment_and_delta():
    interval = GenomicInterval("chr1", 4, 8)
    ref = GenomicArray(interval, 1, [1, 2, 3, 4])
    alt = GenomicArray(interval, 1, [3, 2, 1, 4])
    assert ref.values.dtype == np.float32
    assert ref.values.flags.c_contiguous
    np.testing.assert_array_equal(alt.delta_from(ref).values, [2, 0, -2, 0])
    with pytest.raises(ValueError, match="aligned"):
        alt.delta_from(GenomicArray(GenomicInterval("chr1", 5, 9), 1, [1] * 4))
    for values in ([[1], [2], [3], [4]], [1, 2], [1, 2, np.nan, 4]):
        with pytest.raises(ValueError):
            GenomicArray(interval, 1, values)


@pytest.mark.parametrize("size", [10, 100_000])
def test_hash_lookup_has_constant_comparison_count(size):
    comparisons = 0

    @dataclass(frozen=True)
    class Key:
        index: int

        def __hash__(self):
            return self.index

        def __eq__(self, other):
            nonlocal comparisons
            comparisons += 1
            return self.index == other.index

    index = VariantHashMap()
    for number in range(size):
        index.add(Key(number), number)
    assert index[Key(size - 1)] == size - 1
    assert comparisons == 1
    with pytest.raises(ValueError, match="Duplicate"):
        index.add(Key(0), 0)


def test_sparse_coo_keeps_relation_shape_and_deduplicates():
    coo = SparseCOO(2, 3)
    assert coo.to_array().shape == (2, 0)
    coo.add(1, 2)
    coo.add(1, 2)
    np.testing.assert_array_equal(coo.to_array(), [[1], [2]])
    assert coo.to_array().dtype == np.int64
    with pytest.raises(ValueError):
        coo.add(0, 3)


def test_canonical_variant_and_interval_boundaries():
    first = VariantKey("x", 1, "a", "C")
    assert str(first) == "chrX:1:A>C"
    assert GenomicInterval.around(first, "1mb").start == 0
    last = VariantKey("CHRX", CHROMOSOME_LENGTHS["chrX"], "T", "C")
    assert GenomicInterval.around(last, "100kb").end == last.position
    for args in (("chrMT", 1, "T", "A"), ("1", 0, "A", "T"), ("1", 1, "AA", "T"),
                 ("1", 1, "A", "A"), ("1", 1.2, "A", "T")):
        with pytest.raises(ValueError):
            VariantKey(*args)


def test_display_bins_preserve_alignment_and_summary_uses_native_delta():
    interval = GenomicInterval("chr1", 4, 9)
    ref = np.array([[1], [2], [3], [4], [5]])
    alt = ref + np.array([[4], [-4], [0], [3], [0]])
    columns = (Column("Track", "."),)
    track = summarize_tracks("ATAC", interval, 1, columns, ref, alt, 2)
    np.testing.assert_array_equal(track.edges, [4, 6, 9])
    np.testing.assert_array_equal(track.delta.ravel(), [0, 1])
    assert track.mean_abs_delta == 2.2
    assert track.max_abs_delta == 4
    with pytest.raises(ValueError):
        summarize_tracks("ATAC", interval, 1, columns, ref, alt[:-1], 2)


def test_mock_is_repeatable_and_splice_sites_are_tissue_independent():
    variant = VariantKey("22", 36201698, "A", "C")
    interval = GenomicInterval.around(variant, "1mb")
    provider = MockProvider()
    first = provider.predict(variant, interval, "lung", 128)
    second = provider.predict(variant, interval, "lung", 128)
    other_tissue = provider.predict(variant, interval, "liver", 128)
    assert tuple(first) == ASSAYS
    for assay, track in first.items():
        np.testing.assert_array_equal(track.delta, second[assay].delta)
        np.testing.assert_allclose(track.delta, track.alternate - track.reference, atol=1e-7)
        assert track.edges[0] == interval.start
        assert track.edges[-1] == interval.end
        assert track.delta.shape == (128, len(track.columns))
    np.testing.assert_array_equal(first["SPLICE_SITES"].delta, other_tissue["SPLICE_SITES"].delta)
    assert not np.array_equal(first["RNA_SEQ"].delta, other_tissue["RNA_SEQ"].delta)


def test_sdk_adapter_with_real_sdk_types(monkeypatch):
    pytest.importorskip("alphagenome")
    import pandas as pd
    from alphagenome.data import track_data
    from alphagenome.models import dna_client, dna_output

    class Client:
        def predict_variant(self, *, interval, variant, requested_outputs, ontology_terms):
            assert variant.position == 36201698
            assert ontology_terms == ["UBERON:0002048"]
            assert dna_client.OutputType.SPLICE_SITES in requested_outputs
            metadata = pd.DataFrame({"name": ["Tissue signal"], "strand": ["+"]})
            values = np.ones((interval.width // 128, 1), dtype=np.float32)
            ref = track_data.TrackData(values, metadata, resolution=128, interval=interval)
            alt = track_data.TrackData(values * 2, metadata, resolution=128, interval=interval)
            return SimpleNamespace(reference=dna_output.Output(cage=ref), alternate=dna_output.Output(cage=alt))

    monkeypatch.setattr(dna_client, "DnaClient", lambda **_: Client())
    variant = VariantKey("22", 36201698, "A", "C")
    result = AlphaGenomeProvider("fixture").predict(variant, GenomicInterval.around(variant, "100kb"), "lung", 64)
    assert list(result) == ["CAGE"]
    np.testing.assert_array_equal(result["CAGE"].delta, np.ones((64, 1)))


def test_live_reference_check_uses_indexed_fasta(tmp_path, monkeypatch):
    pytest.importorskip("alphagenome")
    from alphagenome.models import dna_client
    from pyfaidx import Fasta

    from backend.services.alphagenome_service import ProviderError

    class Client:
        def predict_variant(self, **kwargs):
            empty = SimpleNamespace(rna_seq=None, atac=None, cage=None, splice_sites=None)
            return SimpleNamespace(reference=empty, alternate=empty)

    monkeypatch.setattr(dna_client, "DnaClient", lambda **_: Client())
    fasta = tmp_path / "example.fa"
    fasta.write_text(">chr22\nACTG\n")
    variant = VariantKey("22", 2, "C", "T")
    interval = GenomicInterval.around(variant, "100kb")
    provider = AlphaGenomeProvider("fixture", str(fasta))
    assert provider.validates_reference
    with pytest.raises(ProviderError, match="index could not be read"):
        provider.predict(variant, interval, "lung", 64)
    with Fasta(str(fasta)):
        pass
    assert provider.predict(variant, interval, "lung", 64) == {}
    with pytest.raises(ValueError, match="REF differs"):
        provider.predict(VariantKey("22", 2, "A", "T"), interval, "lung", 64)
