from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.core.data_structures import VariantKey
from backend.services.alphagenome_service import Assay

Tissue = Literal["lung", "liver", "brain", "colon"]
Window = Literal["100kb", "1mb"]
SignalSource = Literal["mock", "alphagenome"]
ModelMode = Literal["demo", "research"]


class VariantQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chr: Annotated[str, Field(min_length=1, max_length=8)]
    pos: Annotated[int, Field(gt=0)]
    ref: Annotated[str, Field(min_length=1, max_length=8)]
    alt: Annotated[str, Field(min_length=1, max_length=8)]
    tissue: Tissue = "lung"
    window: Window = "100kb"
    max_points: Annotated[int, Field(ge=32, le=1024)] = 512

    @field_validator("pos", "max_points", mode="before")
    @classmethod
    def reject_bool(cls, value):
        if isinstance(value, bool):
            raise ValueError("Use a whole number.")
        return value

    @model_validator(mode="after")
    def validate_coordinates(self):
        key = self.key()
        self.chr, self.ref, self.alt = key.chromosome, key.reference, key.alternate
        return self

    def key(self) -> VariantKey:
        return VariantKey(self.chr, self.pos, self.ref, self.alt)


class ColumnResponse(BaseModel):
    name: str
    strand: str


class TrackResponse(BaseModel):
    assay: Assay
    columns: list[ColumnResponse]
    native_resolution_bp: int
    units: str
    tissue_specific: bool
    bin_edges: list[int]
    reference: list[list[float]]
    alternate: list[list[float]]
    delta: list[list[float]]


class IntervalResponse(BaseModel):
    chromosome: str
    start: int
    end: int


class TracksResponse(BaseModel):
    assembly: Literal["GRCh38"] = "GRCh38"
    variant: VariantQuery
    interval: IntervalResponse
    tissue_ontology: str
    signal_source: SignalSource
    reference_validated: bool
    tracks: list[TrackResponse]
    missing_assays: list[Assay]


class ScoreResponse(BaseModel):
    value: Annotated[float, Field(ge=0, le=1)]
    mode: ModelMode
    calibrated: bool = False
    model_version: str
    network_version: str


class AttributionResponse(BaseModel):
    assay: Assay
    mean_abs_delta: float
    max_abs_delta: float
    masked_logit_change: float


class PredictionResponse(TracksResponse):
    score: ScoreResponse
    attributions: list[AttributionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    signal_source: SignalSource
    model_mode: ModelMode


class ErrorResponse(BaseModel):
    code: str
    message: str
