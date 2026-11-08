# GNN-AlphaDriver

GNN-AlphaDriver is a research workspace for comparing allele-specific genomic signals and exploring a heterogeneous graph model for somatic driver prioritization. It accepts GRCh38 single-nucleotide variants and shows RNA-seq, ATAC-seq, CAGE, and splice-site predictions alongside an HGT score.

The default installation runs offline with deterministic synthetic signals, a small synthetic gene network, and fixed demo weights. The API and UI expose signal and model provenance independently. The displayed percentage is the sigmoid of a model logit. The bundled weights have no predictive or clinical interpretation.

## Run locally

Use Python 3.11–3.14 and Node.js 22.12 or later in the Node 22 release line. The frontend lockfile was generated with npm 11.

Start FastAPI from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start Next.js.

```sh
cd frontend
npm ci
npm run dev
```

Open [the variant workspace](http://localhost:3000). It loads the example variant, then lets you change the locus, alleles, biosample, and window. Use the assay controls to switch tracks, zoom to the variant, toggle alleles, and export the selected assay's returned bins as CSV. The CSV includes all columns for that assay and uses zero-based, half-open bin edges.

Next.js forwards prediction requests to FastAPI through a same-origin route. Set `ALPHADRIVER_API_URL` in the Next.js process if the API uses a different address. Set `ALPHADRIVER_DOCS_URL` before building the frontend to change the browser-facing API reference link. Set `ALPHADRIVER_ORIGINS` in FastAPI to a comma-separated list if calling FastAPI directly from other browser origins.

## Architecture

The provider aligns each REF and ALT column and computes native-resolution deltas. A bounded set of display bins goes to the chart. Native-resolution summary features go through a variant hash map and sparse relation-specific COO adjacency into a PyG `HeteroData` graph. A two-layer HGT model emits a variant logit. The dashboard shows the score and the logit change under one-assay-at-a-time masking.

- [Architecture and data contracts](docs/ARCHITECTURE.md)
- [Data structures, ML intuitions, and evaluation protocol](docs/DATA_STRUCTURES_AND_ML_INTUITIONS.md)

## API

Interactive documentation is available at [FastAPI Swagger UI](http://127.0.0.1:8000/docs).

| Method | Path | Returns |
|---|---|---|
| GET | `/api/v1/health` | Local process status, signal source, and model mode |
| GET | `/api/v1/tracks` | Variant, interval, assay matrices, bin edges, and source metadata |
| POST | `/api/v1/predict` | The same tracks plus the HGT score and per-assay masking effects |

```sh
curl -fsS http://127.0.0.1:8000/api/v1/health

curl -fsS 'http://127.0.0.1:8000/api/v1/tracks?chr=chr22&pos=36201698&ref=A&alt=C&tissue=lung&window=1mb&max_points=64'

curl -fsS http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  --data '{"chr":"chr22","pos":36201698,"ref":"A","alt":"C","tissue":"lung","window":"100kb","max_points":64}'
```

Coordinates use human GRCh38 chromosomes 1–22, X, and Y. Submit a 1-based SNV position and distinct A/C/G/T alleles. Biosamples are `lung`, `liver`, `brain`, and `colon` (transverse). The `100kb` and `1mb` choices resolve to 131,072 and 1,048,576 bp. Response intervals and `bin_edges` are zero-based and half-open. Each matrix is indexed by display bin, then track column. The point budget defaults to 512 and accepts 32–1,024 bins per assay. Missing assays are listed explicitly.

Errors use a `code` and `message`. HTTP 422 covers invalid coordinates and unsupported research contexts. HTTP 502 reports upstream or reference-data errors. HTTP 503 means the inference slot or upstream capacity is unavailable and includes `Retry-After`.

## Use AlphaGenome signals

Install the live extra and set an AlphaGenome API key in the FastAPI process environment before restarting the backend.

```sh
python -m pip install -e '.[live,test]'
export ALPHAGENOME_API_KEY='<your-api-key>'
# Optional for demo mode. Required for a research checkpoint.
export ALPHADRIVER_REFERENCE_FASTA='/path/to/GRCh38.fa'
```

The FASTA must have `chr`-prefixed primary contigs and an existing `.fai` index. A configured reference mismatch is rejected before requesting predictions. Live mode queries the AlphaGenome `ALL_FOLDS` model and reports real molecular signals. The graph stays in demo mode until both research artifacts are supplied. Splice-site outputs are tissue-independent. [AlphaGenome SDK documentation](https://www.alphagenomedocs.com/)

For experimental HGT inference, also set `ALPHADRIVER_CHECKPOINT` and `ALPHADRIVER_NETWORK` to compatible local artifacts. The [checkpoint and network contracts](docs/DATA_STRUCTURES_AND_ML_INTUITIONS.md#research-artifacts) describe the required features, version identifiers, and locus coverage. This repository includes no patient cohort, biological network, trained research checkpoint, or fitted calibration model.

## Verify

```sh
python -m pip install -e '.[live,test]'
OMP_NUM_THREADS=1 python -m pytest --cov=backend --cov-report=term-missing

cd frontend
npm ci
npm run typecheck
npm run coverage
npm run build
```

The Python suite exercises the real SDK data types with a local transport fixture. It requires no AlphaGenome credentials. The frontend suite exercises the input form, chart controls, score provenance, error recovery, and API proxy.
