import os
from threading import BoundedSemaphore
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    AttributionResponse, ColumnResponse, ErrorResponse, HealthResponse, IntervalResponse,
    PredictionResponse, ScoreResponse, TrackResponse, TracksResponse, VariantQuery,
)
from backend.core.data_structures import GenomicInterval
from backend.services.alphagenome_service import (
    ASSAYS, TISSUES, UNITS, AlphaGenomeProvider, MockProvider, ProviderBusy, ProviderError, TrackProvider,
)
from backend.services.inference import InferenceEngine, UnsupportedContext


def create_app(
    *, provider: TrackProvider | None = None, engine: InferenceEngine | None = None,
    origins: list[str] | None = None,
) -> FastAPI:
    api_key = os.getenv("ALPHAGENOME_API_KEY")
    reference_fasta = os.getenv("ALPHADRIVER_REFERENCE_FASTA")
    engine = engine if engine is not None else InferenceEngine(
        os.getenv("ALPHADRIVER_CHECKPOINT"), os.getenv("ALPHADRIVER_NETWORK"),
    )
    if provider is None:
        if engine.mode == "research" and (not api_key or not reference_fasta):
            raise ValueError("Research inference needs AlphaGenome credentials and an indexed GRCh38 FASTA.")
        provider = AlphaGenomeProvider(api_key, reference_fasta) if api_key else MockProvider()
    reference_validated = provider.name == "alphagenome" and bool(reference_fasta)
    capacity = BoundedSemaphore(1)

    app = FastAPI(title="GNN-AlphaDriver", version="0.1.0")
    if origins is None:
        origins = [
            origin.strip() for origin in os.getenv(
                "ALPHADRIVER_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",") if origin.strip()
        ]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def error(status: int, code: str, message: str, headers=None) -> JSONResponse:
        return JSONResponse(status_code=status, content=ErrorResponse(code=code, message=message).model_dump(),
                            headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        issue = exc.errors()[0]
        field = ".".join(str(part) for part in issue["loc"] if part not in {"body", "query"})
        message = issue["msg"].removeprefix("Value error, ")
        return error(422, "invalid_variant", f"{field}: {message}" if field else message)

    @app.exception_handler(UnsupportedContext)
    async def unsupported_context(_request: Request, exc: UnsupportedContext):
        return error(422, "unsupported_context", str(exc))

    @app.exception_handler(ProviderError)
    async def provider_error(_request: Request, exc: ProviderError):
        if isinstance(exc, ProviderBusy):
            return error(503, "provider_busy", str(exc), {"Retry-After": "3"})
        return error(502, "provider_error", str(exc))

    def query_tracks(query: VariantQuery, *, score: bool) -> TracksResponse:
        key = query.key()
        interval = GenomicInterval.around(key, query.window)
        network = engine.context(key, query.tissue, query.window) if score else None
        if not capacity.acquire(blocking=False):
            raise ProviderBusy("The workspace is processing another request. Try again shortly.")
        try:
            try:
                tracks = provider.predict(key, interval, query.tissue, query.max_points)
            except ValueError as exc:
                raise UnsupportedContext(str(exc)) from exc
            if not tracks:
                raise ProviderError("The provider returned no signal tracks for this request.")
            response = TracksResponse(
                variant=query, interval=IntervalResponse(chromosome=interval.chromosome,
                                                        start=interval.start, end=interval.end),
                tissue_ontology=TISSUES[query.tissue], signal_source=provider.name,
                reference_validated=reference_validated,
                missing_assays=[assay for assay in ASSAYS if assay not in tracks],
                tracks=[
                    TrackResponse(
                        assay=assay, native_resolution_bp=track.resolution, units=UNITS[assay],
                        tissue_specific=assay != "SPLICE_SITES",
                        columns=[ColumnResponse(name=column.name, strand=column.strand) for column in track.columns],
                        bin_edges=track.edges.tolist(), reference=track.reference.tolist(),
                        alternate=track.alternate.tolist(), delta=track.delta.tolist(),
                    )
                    for assay, track in tracks.items()
                ],
            )
            if network is None:
                return response
            value, effects = engine.score(key, tracks, network)
            return PredictionResponse(
                **response.model_dump(), score=ScoreResponse(
                    value=value, mode=engine.mode, model_version=engine.model_version,
                    network_version=engine.network_version,
                ),
                attributions=[
                    AttributionResponse(assay=assay, mean_abs_delta=track.mean_abs_delta,
                                        max_abs_delta=track.max_abs_delta, masked_logit_change=effects[assay])
                    for assay, track in tracks.items()
                ],
            )
        finally:
            capacity.release()

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health():
        return HealthResponse(signal_source=provider.name, model_mode=engine.mode)

    errors = {status: {"model": ErrorResponse} for status in (422, 502, 503)}

    @app.get("/api/v1/tracks", response_model=TracksResponse, responses=errors)
    def tracks(query: Annotated[VariantQuery, Query()]):
        return query_tracks(query, score=False)

    @app.post("/api/v1/predict", response_model=PredictionResponse, responses=errors)
    def predict(query: VariantQuery):
        return query_tracks(query, score=True)

    return app


app = create_app()
