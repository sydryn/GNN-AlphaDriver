from dataclasses import dataclass
from hashlib import blake2b
from typing import Literal, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from backend.core.data_structures import GenomicArray, GenomicInterval, VariantKey

Assay = Literal["RNA_SEQ", "ATAC", "CAGE", "SPLICE_SITES"]
LIVE_SIGNAL_VERSION = "alphagenome:0.11.0:ALL_FOLDS"
ASSAYS: tuple[Assay, ...] = ("RNA_SEQ", "ATAC", "CAGE", "SPLICE_SITES")
TISSUES = {
    "lung": "UBERON:0002048",
    "liver": "UBERON:0001114",
    "brain": "UBERON:0000955",
    "colon": "UBERON:0001157",
}
UNITS = {
    "RNA_SEQ": "Normalized read signal",
    "ATAC": "Fold enrichment",
    "CAGE": "Fold enrichment",
    "SPLICE_SITES": "Splice-site probability",
}


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    strand: str


@dataclass(frozen=True, slots=True)
class TrackMatrix:
    assay: Assay
    columns: tuple[Column, ...]
    resolution: int
    edges: NDArray[np.int64]
    reference: NDArray[np.float64]
    alternate: NDArray[np.float64]
    delta: NDArray[np.float64]
    mean_abs_delta: float
    max_abs_delta: float


class ProviderError(RuntimeError):
    """An upstream service failed to return a usable prediction."""


class ProviderBusy(ProviderError):
    """The upstream service is temporarily rate-limited or unavailable."""


class TrackProvider(Protocol):
    name: Literal["mock", "alphagenome"]
    validates_reference: bool

    def predict(
        self, variant: VariantKey, interval: GenomicInterval, tissue: str, points: int
    ) -> dict[Assay, TrackMatrix]: ...


def summarize_tracks(
    assay: Assay, interval: GenomicInterval, resolution: int,
    columns: tuple[Column, ...], reference: NDArray, alternate: NDArray, points: int,
) -> TrackMatrix:
    if points < 1 or resolution < 1 or not columns:
        raise ValueError("Tracks need positive resolution, columns and point budget.")
    bins, remainder = divmod(interval.width, resolution)
    if remainder or bins < 1 or reference.shape != (bins, len(columns)) or alternate.shape != reference.shape:
        raise ValueError("REF and ALT matrices must cover the same bins and columns.")
    edges = np.arange(min(bins, points) + 1, dtype=np.int64) * bins // min(bins, points)
    counts = np.diff(edges)
    shape = (counts.size, len(columns))
    ref_view, alt_view, delta_view = (np.empty(shape, dtype=np.float64) for _ in range(3))
    total, peak = 0.0, 0.0
    for i in range(len(columns)):
        ref = GenomicArray(interval, resolution, reference[:, i])
        alt = GenomicArray(interval, resolution, alternate[:, i])
        delta = alt.delta_from(ref).values
        ref_view[:, i] = np.add.reduceat(ref.values, edges[:-1], dtype=np.float64) / counts
        alt_view[:, i] = np.add.reduceat(alt.values, edges[:-1], dtype=np.float64) / counts
        delta_view[:, i] = alt_view[:, i] - ref_view[:, i]
        magnitude = np.abs(delta)
        total += float(magnitude.sum(dtype=np.float64))
        peak = max(peak, float(magnitude.max()))
    return TrackMatrix(assay, columns, resolution, interval.start + edges * resolution,
                       ref_view, alt_view, delta_view, total / (bins * len(columns)), peak)


class MockProvider:
    name: Literal["mock"] = "mock"
    validates_reference = False

    def predict(
        self, variant: VariantKey, interval: GenomicInterval, tissue: str, points: int
    ) -> dict[Assay, TrackMatrix]:
        signals = {}
        for index, assay in enumerate(ASSAYS):
            seed_key = str(variant) if assay == "SPLICE_SITES" else f"{variant}/{tissue}"
            seed = int.from_bytes(blake2b(seed_key.encode(), digest_size=8).digest())
            rng = np.random.default_rng(seed + index)
            resolution = 128
            centers = interval.start + (np.arange(interval.width // resolution) + 0.5) * resolution
            x = (centers - (variant.position - 1)) / interval.width
            count = 2 if assay == "SPLICE_SITES" else 1
            references, alternates = [], []
            for _ in range(count):
                ref = 0.03 + 0.04 * np.sin(120 * x + rng.uniform(-1, 1)) ** 2
                for offset, width, height in zip(
                    rng.uniform(-0.5, 0.5, 15), rng.uniform(0.002, 0.025, 15),
                    rng.uniform(0.05, 0.42, 15), strict=True,
                ):
                    ref += height * np.exp(-((x - offset) / width) ** 2)
                ref += 0.14 * np.exp(-(x / 0.022) ** 2)
                alt = np.maximum(0, ref + (0.28 - 0.13 * (index % 2)) * np.exp(-(x / 0.024) ** 2)
                                 - 0.10 * np.exp(-((x - 0.05) / 0.018) ** 2))
                cap = 1 if assay == "SPLICE_SITES" else None
                references.append(np.clip(ref, 0, cap))
                alternates.append(np.clip(alt, 0, cap))
            columns = (Column("Synthetic donor", "+"), Column("Synthetic acceptor", "-")) if count == 2 else (
                Column(f"Synthetic {tissue} signal", "."),)
            signals[assay] = summarize_tracks(
                assay, interval, resolution, columns,
                np.column_stack(references), np.column_stack(alternates), points,
            )
        return signals


class AlphaGenomeProvider:
    name: Literal["alphagenome"] = "alphagenome"
    MAX_TRACKS_PER_ASSAY = 64

    def __init__(self, api_key: str, reference_fasta: str | None = None) -> None:
        import grpc
        from alphagenome.models import dna_client

        class CallDetails(NamedTuple):
            method: str
            timeout: float
            metadata: tuple
            credentials: object
            wait_for_ready: bool | None
            compression: object

        class PredictionDeadline(grpc.StreamStreamClientInterceptor):
            def intercept_stream_stream(self, continuation, details, requests):
                return continuation(CallDetails(
                    details.method, details.timeout if details.timeout is not None else 30.0,
                    details.metadata, details.credentials, details.wait_for_ready, details.compression,
                ), requests)

        channel = grpc.secure_channel(
            "dns:///gdmscience.googleapis.com:443", grpc.ssl_channel_credentials(),
            options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
        )
        self._client = dna_client.DnaClient(
            channel=grpc.intercept_channel(channel, PredictionDeadline()),
            model_version=dna_client.ModelVersion.ALL_FOLDS,
            metadata=[("x-goog-api-key", api_key)],
        )
        self._reference_fasta = reference_fasta
        self.validates_reference = bool(reference_fasta)

    def predict(
        self, variant: VariantKey, interval: GenomicInterval, tissue: str, points: int
    ) -> dict[Assay, TrackMatrix]:
        import grpc
        from alphagenome.data import genome
        from alphagenome.models import dna_client

        if self._reference_fasta:
            from pyfaidx import Fasta, FastaIndexingError

            try:
                with Fasta(self._reference_fasta, rebuild=False, build_index=False) as fasta:
                    ref = str(fasta[variant.chromosome][variant.position - 1:variant.position]).upper()
            except (OSError, FastaIndexingError, KeyError, ValueError) as error:
                raise ProviderError("The configured GRCh38 FASTA or its index could not be read.") from error
            if len(ref) != 1:
                raise ProviderError("The configured GRCh38 FASTA does not cover the requested coordinate.")
            if ref != variant.reference:
                raise ValueError("REF differs from the configured GRCh38 reference.")

        sdk_interval = genome.Interval(interval.chromosome, interval.start, interval.end)
        sdk_variant = genome.Variant(variant.chromosome, variant.position, variant.reference, variant.alternate)
        try:
            result = self._client.predict_variant(
                interval=sdk_interval, variant=sdk_variant,
                requested_outputs=[dna_client.OutputType[assay] for assay in ASSAYS],
                ontology_terms=[TISSUES[tissue]],
            )
        except grpc.RpcError as error:
            if error.code() in {grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.UNAVAILABLE,
                                grpc.StatusCode.DEADLINE_EXCEEDED}:
                raise ProviderBusy("AlphaGenome is busy. Try again shortly.") from error
            raise ProviderError("AlphaGenome could not complete the prediction.") from error
        except ValueError as error:
            raise ProviderError("AlphaGenome returned incompatible track data.") from error
        try:
            return self._summarize(result, interval, points)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise ProviderError("AlphaGenome returned incompatible track data.") from error

    def _summarize(self, result, interval: GenomicInterval, points: int) -> dict[Assay, TrackMatrix]:
        signals = {}
        for assay in ASSAYS:
            ref = getattr(result.reference, assay.lower())
            alt = getattr(result.alternate, assay.lower())
            if ref is None and alt is None:
                continue
            if ref is None or alt is None or ref.resolution != alt.resolution:
                raise ValueError("REF and ALT output types or resolutions differ.")
            if ref.values.shape[1] == 0 and alt.values.shape[1] == 0:
                continue
            if ref.values.shape[1] > self.MAX_TRACKS_PER_ASSAY:
                raise ValueError("Assay exceeds the configured track budget.")
            if not ref.metadata[["name", "strand"]].equals(alt.metadata[["name", "strand"]]):
                raise ValueError("REF and ALT track identities differ.")
            for track in (ref, alt):
                if (track.interval.chromosome, track.interval.start, track.interval.end) != (
                    interval.chromosome, interval.start, interval.end,
                ):
                    raise ValueError("Track coordinates differ from the requested interval.")
            columns = tuple(Column(str(name), str(strand)) for name, strand in
                            ref.metadata[["name", "strand"]].itertuples(index=False, name=None))
            signals[assay] = summarize_tracks(assay, interval, ref.resolution, columns,
                                              ref.values, alt.values, points)
        return signals
