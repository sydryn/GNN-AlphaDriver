from collections import defaultdict
from pathlib import Path
from typing import Annotated

import torch
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from backend.core.data_structures import VariantKey
from backend.gnn.graph_builder import (
    FEATURE_SCHEMA, FEATURES_PER_ASSAY, BiologicalNetwork, VariantRecord, assay_features, build_graph,
)
from backend.gnn.model import DriverHGT
from backend.services.alphagenome_service import ASSAYS, TISSUES

MAX_GENES_PER_LOCUS = 32


class UnsupportedContext(ValueError):
    """The research model does not cover the requested locus or window."""


class NetworkData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Annotated[str, Field(min_length=1, max_length=100)]
    genes: dict[str, tuple[FiniteFloat, FiniteFloat]]
    cells: dict[str, tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]]
    expression: list[tuple[str, str]] = Field(default_factory=list)
    interactions: list[tuple[str, str]] = Field(default_factory=list)
    variant_links: dict[str, Annotated[list[str], Field(min_length=1, max_length=MAX_GENES_PER_LOCUS)]]

    @model_validator(mode="after")
    def validate_edges(self):
        for cell, gene in self.expression:
            if cell not in self.cells or gene not in self.genes:
                raise ValueError("An expression edge refers to an unknown cell or gene.")
        for source, target in self.interactions:
            if source not in self.genes or target not in self.genes:
                raise ValueError("An interaction edge refers to an unknown gene.")
        for key, genes in self.variant_links.items():
            if not self.genes.keys() >= set(genes):
                raise ValueError(f"Variant {key} links to an unknown gene.")
        return self


class ResearchNetwork:
    def __init__(self, data: NetworkData) -> None:
        self.data = data
        self.expression = set(data.expression)
        self.interactions = defaultdict(set)
        for first, second in data.interactions:
            self.interactions[first].add(second)
            self.interactions[second].add(first)

    def context(self, variant: VariantKey, tissue: str) -> BiologicalNetwork:
        genes = self.data.variant_links.get(str(variant))
        if not genes or tissue not in self.data.cells:
            raise UnsupportedContext("The research network does not cover this variant and biosample.")
        selected = set(genes)
        return BiologicalNetwork(
            genes={gene: self.data.genes[gene] for gene in genes},
            cells={tissue: self.data.cells[tissue]},
            expression=tuple((tissue, gene) for gene in genes if (tissue, gene) in self.expression),
            interactions=tuple(
                (gene, partner) for gene in genes
                for partner in sorted(selected.intersection(self.interactions.get(gene, ())))
                if gene < partner
            ),
        )


class InferenceEngine:
    def __init__(self, checkpoint: str | None = None, network: str | None = None) -> None:
        if bool(checkpoint) != bool(network):
            raise ValueError("Configure the research checkpoint and network together.")
        self.research_network = None
        self.model_version = "hgt-demo-v1"
        self.network_version = "synthetic-v1"
        self.window = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.model = DriverHGT()
        if checkpoint and network:
            data = NetworkData.model_validate_json(Path(network).read_bytes())
            artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(artifact, dict) or (
                artifact.get("feature_schema"), artifact.get("network_version"), artifact.get("signal_source"),
            ) != (FEATURE_SCHEMA, data.version, "alphagenome:0.11.0:ALL_FOLDS"):
                raise ValueError("The checkpoint feature, signal or network version does not match.")
            self.window = artifact.get("window")
            version = artifact.get("model_version")
            if self.window not in {"100kb", "1mb"} or not isinstance(version, str) or not version.strip():
                raise ValueError("The checkpoint must identify its window and model version.")
            self.model.load_state_dict(artifact["state_dict"], strict=True)
            if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
                raise ValueError("Checkpoint parameters must be finite.")
            self.model_version = version
            self.network_version = data.version
            self.research_network = ResearchNetwork(data)
        self.model.eval()

    @property
    def mode(self):
        return "research" if self.research_network else "demo"

    def context(self, variant: VariantKey, tissue: str, window: str) -> BiologicalNetwork:
        if self.research_network:
            if self.window != window:
                raise UnsupportedContext(f"The research checkpoint requires the {self.window} window.")
            return self.research_network.context(variant, tissue)
        return BiologicalNetwork(
            genes={"fixture_gene_1": (0.7, 0.2), "fixture_gene_2": (0.2, 0.8)},
            cells={tissue: tuple(float(name == tissue) for name in TISSUES)},
            expression=((tissue, "fixture_gene_1"),),
            interactions=(("fixture_gene_1", "fixture_gene_2"),),
        )

    @torch.inference_mode()
    def score(self, variant, tracks, network):
        graph = build_graph([VariantRecord(variant, assay_features(tracks),
                                          tuple(network.genes), tuple(network.cells))], network)
        features = graph["variant"].x
        logit = self.model(graph)[0]
        score = logit.sigmoid().item()
        effects = {}
        for index, assay in enumerate(ASSAYS):
            if assay in tracks:
                masked = features.clone()
                start = index * FEATURES_PER_ASSAY
                masked[:, start:start + FEATURES_PER_ASSAY] = 0
                graph["variant"].x = masked
                effects[assay] = (logit - self.model(graph)[0]).item()
        return score, effects
