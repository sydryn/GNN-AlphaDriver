from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch_geometric.data import HeteroData

from backend.core.data_structures import SparseCOO, VariantHashMap, VariantKey
from backend.services.alphagenome_service import ASSAYS, Assay, TrackMatrix

FEATURES_PER_ASSAY = 3
FEATURE_SCHEMA = "mean_abs,max_abs,present:v1"
FEATURE_WIDTHS = {"variant": len(ASSAYS) * FEATURES_PER_ASSAY, "gene": 2, "cell": 4}
REGULATES = ("variant", "linked_to", "gene")
REGULATED_BY = ("gene", "linked_from", "variant")
EXPRESSES = ("cell", "expresses", "gene")
EXPRESSED_IN = ("gene", "expressed_in", "cell")
OBSERVED_IN = ("variant", "observed_in", "cell")
OBSERVES = ("cell", "observes", "variant")
INTERACTS = ("gene", "interacts", "gene")
RELATIONS = (REGULATES, REGULATED_BY, EXPRESSES, EXPRESSED_IN, OBSERVED_IN, OBSERVES, INTERACTS)


@dataclass(frozen=True, slots=True)
class VariantRecord:
    key: VariantKey
    features: Sequence[float]
    genes: tuple[str, ...] = ()
    cells: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BiologicalNetwork:
    genes: Mapping[str, Sequence[float]]
    cells: Mapping[str, Sequence[float]]
    expression: tuple[tuple[str, str], ...] = ()
    interactions: tuple[tuple[str, str], ...] = ()


def assay_features(tracks: Mapping[Assay, TrackMatrix]) -> tuple[float, ...]:
    values = []
    for assay in ASSAYS:
        track = tracks.get(assay)
        values.extend((track.mean_abs_delta, track.max_abs_delta, 1.0) if track else (0.0, 0.0, 0.0))
    return tuple(values)


def build_graph(variants: Sequence[VariantRecord], network: BiologicalNetwork) -> HeteroData:
    if not variants:
        raise ValueError("The graph needs at least one variant.")

    graph = HeteroData()
    node_indices = {}
    node_features = {
        "variant": [(record.key, record.features) for record in variants],
        "gene": list(network.genes.items()),
        "cell": list(network.cells.items()),
    }
    for node_type, records in node_features.items():
        index = VariantHashMap()
        width = FEATURE_WIDTHS[node_type]
        for row, (key, features) in enumerate(records):
            index.add(key, row)
            if len(features) != width:
                raise ValueError(f"{node_type} features must have width {width}.")
        values = torch.tensor([features for _, features in records], dtype=torch.float32).reshape(-1, width)
        if not torch.isfinite(values).all():
            raise ValueError(f"{node_type} features must be finite.")
        graph[node_type].x = values
        node_indices[node_type] = index

    builders = {
        relation: SparseCOO(len(node_features[relation[0]]), len(node_features[relation[2]]))
        for relation in RELATIONS
    }

    def connect(relation, source, target) -> None:
        try:
            src = node_indices[relation[0]][source]
            dst = node_indices[relation[2]][target]
        except KeyError as error:
            raise ValueError(f"Edge in {relation[1]} references unknown node {error.args[0]!r}.") from error
        builders[relation].add(src, dst)

    for variant in variants:
        for gene in variant.genes:
            connect(REGULATES, variant.key, gene)
            connect(REGULATED_BY, gene, variant.key)
        for cell in variant.cells:
            connect(OBSERVED_IN, variant.key, cell)
            connect(OBSERVES, cell, variant.key)

    for cell, gene in network.expression:
        connect(EXPRESSES, cell, gene)
        connect(EXPRESSED_IN, gene, cell)

    for first, second in network.interactions:
        connect(INTERACTS, first, second)
        connect(INTERACTS, second, first)

    for relation, builder in builders.items():
        graph[relation].edge_index = torch.from_numpy(builder.to_array())
    return graph
