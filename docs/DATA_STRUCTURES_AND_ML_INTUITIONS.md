# Data structures and ML intuitions

## A locus starts as aligned arrays

A submitted SNV position is 1-based. In a zero-based interval `[s, e)`, the variant base is at offset `pos - 1 - s`. Near chromosome ends, the interval shifts inward to preserve the SDK's required width. Each signal column at resolution `r` has `(e - s) / r` samples. Element `i` describes bases in `[s + ir, s + (i + 1)r)`. Think of the array as a ruler divided into equal-sized bins, with each assay taking a different kind of measurement along that ruler.

For matched track columns, the signed difference is

```text
delta[i, t] = ALT[i, t] - REF[i, t]
```

Both alleles need identical coordinates, resolution, and column identity. An indel would require aligning alternate coordinates to the reference before comparing bins. This project restricts input to SNVs and can check the REF base against an indexed GRCh38 FASTA. The SDK permits counterfactual REF alleles, so the FASTA check is required when serving a research checkpoint. [AlphaGenome variant scoring and alignment](https://www.alphagenomedocs.com/variant_scoring.html)

Array access by offset is O(1). Reading every value costs O(LT) for L native bins and T tracks. A million-base float32 column uses about 4 MiB per allele. The backend processes deltas one column at a time. For charts it averages contiguous native bins, using the same edges for REF and ALT. Subtraction commutes with this average. Mean absolute delta is computed before aggregation because opposite-signed changes may cancel in a display bin. Maximum absolute delta is also computed before aggregation so a sharp peak survives in the feature summary.

The four modalities have distinct meanings. RNA-seq measures predicted transcript-associated signal. ATAC-seq represents predicted chromatin accessibility. CAGE measures predicted activity at transcription starts. Splice-site tracks represent predicted donor or acceptor positions without a tissue assignment. A larger raw delta in one modality does not imply a larger functional effect than a smaller raw delta in another modality. [AlphaGenome model output metadata](https://www.alphagenomedocs.com/exploring_model_metadata.html)

## From identifiers to sparse relations

The canonical key `chr22:36201698:A>C` represents the allele change independently of the window. Hash maps associate these keys, gene IDs, and cell IDs with rows of their respective feature matrices. Lookup is expected O(1) under normal hash distribution. The test counts key comparisons across two map sizes, avoiding timing assertions that depend on the host.

A candidate variant may link to only a few genes. A dense adjacency would allocate space for every possible pair. COO keeps one source index and one destination index per observed edge. For three edges, a relation may have this form.

```text
[[0, 1, 1],
 [2, 2, 3]]
```

Here source row 0 links to destination row 2, and source row 1 links to destination rows 2 and 3. Sparse relation storage is O(E), and row indices plus node features use O(V) space for fixed feature widths. The builder deduplicates repeated pairs and keeps empty relation tensors at shape `[2, 0]`. PyG can batch independent `HeteroData` graphs by offsetting row indices separately for each node type.

The implemented graph has variant, gene, and cell nodes. Variant–gene edges encode curated links in research mode. Cell–gene edges encode expression context. Variant–cell edges identify the observed biosample. Gene–gene edges encode interactions. The bipartite relations have separately typed reverse edges so genes and cells can send messages back to variants. An interaction path can carry contextual features across two hops. The direction of an edge determines information flow and does not by itself establish biological causality.

## Heterogeneous attention

The model projects each node type into a shared hidden space. For an incoming edge, the destination supplies a query, while the source supplies a key and a value. A relation transform and learned relation prior affect the scaled query–key affinity. A softmax over incoming messages assigns weights before their values are summed. This lets a gene interaction and an expression edge contribute in different ways even if their endpoint representations are similar.

The two HGT layers update typed node embeddings. A residual path and layer normalization help keep the representation stable. A node with no incoming edges keeps its projected features. A linear head maps each variant embedding to one logit `z`, and the API reports `sigmoid(z)`. This bounded score can rank examples after training. Its agreement with observed frequencies requires held-out calibration. [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332)

Each variant has three features per assay, in the fixed order RNA-seq, ATAC, CAGE, and splice sites.

1. Mean absolute native-resolution delta
2. Maximum absolute native-resolution delta
3. Presence flag, equal to 1 when the provider returns the assay

An unavailable assay contributes three zeros. The presence flag distinguishes that case from a measured track with zero change. The UI ranks assays by the magnitude of `full_logit - masked_logit`, where masking clears that assay's three features. Positive values mean the full input produced a higher logit than the masked input. This diagnostic can reflect both signal and missingness sensitivity. It does not estimate a causal effect or split the final score into additive components.

## Focal loss under class imbalance

For binary target `y`, let `p_t` be the model probability assigned to the observed label, and let `alpha_t` equal `alpha` for positives and `1 - alpha` for negatives. The loss is

```text
loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

With `gamma = 0`, the result is weighted binary cross-entropy. As `gamma` increases, already well-classified examples contribute less. For instance, at `gamma = 2` an example with `p_t = 0.99` gets a focusing multiplier of `0.0001`, while an example with `p_t = 0.5` gets `0.25`. The implementation computes cross-entropy directly from logits and uses `-expm1(-BCE)` for the focusing base, preserving precision for easy examples. [Focal Loss](https://arxiv.org/abs/1708.02002)

The brief's 0.1% driver prevalence is a stress-test assumption. At that prevalence an always-passenger classifier reaches 99.9% accuracy while finding zero drivers. Tune class weight and focusing strength on held-out data. Report average precision, precision and recall at the review budget, and confidence intervals clustered by patient or locus. Compare with a prevalence baseline and a model that sees the same assay summaries without graph edges.

## Research artifacts

The included model supports supervised optimization with `WeightedFocalLoss`. Build each training graph using the same features and locus subgraph policy as inference. Keep labels in the same order as the variant records, and split patients and loci before constructing training, validation, and test graphs.

```python
import torch
from backend.gnn.focal_loss import WeightedFocalLoss
from backend.gnn.model import DriverHGT

model = DriverHGT()
loss_fn = WeightedFocalLoss(alpha=0.75, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
for graph, labels in training_graphs:
    optimizer.zero_grad()
    loss = loss_fn(model(graph), labels)
    loss.backward()
    optimizer.step()
```

Fit and select weights using a labeled cohort with a documented driver definition. Record how ambiguous labels and likely passengers were chosen. Reserve a separate validation set for calibration, assess reliability on a held-out cohort, and document how cohort prevalence differs from the intended review population. The service exposes the uncalibrated score while this evaluation is absent.

The network JSON contains a version, two numeric features per gene, four per cell, expression pairs ordered as `[cell, gene]`, gene interaction pairs, and up to 32 linked genes per canonical variant key. At inference the service keeps the linked genes, requested cell or tissue, and edges among those nodes. The schema does not assign biological meanings to gene and cell features. The research dataset owner defines and versions those features and their order.

```json
{
  "version": "example-network-v1",
  "genes": { "gene-1": [0.7, 0.2] },
  "cells": { "lung": [1, 0, 0, 0] },
  "expression": [["lung", "gene-1"]],
  "interactions": [],
  "variant_links": { "chr22:36201698:A>C": ["gene-1"] }
}
```

The checkpoint is a local `torch.save` dictionary loaded with `weights_only=True`. Its `state_dict` must match the default `DriverHGT` architecture. Set `feature_schema` to `mean_abs,max_abs,present:v1`, `network_version` to the network JSON version, and `signal_source` to `alphagenome:0.11.0:ALL_FOLDS`. Include a unique `model_version` and the training `window` (`100kb` or `1mb`). A mismatched window or missing locus and biosample is rejected during prediction. The tracks endpoint remains available independently of research coverage.
