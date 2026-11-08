import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import Workspace from "../components/Workspace";
import { EXAMPLE, type Prediction, type Track } from "../lib/contracts";

const track: Track = {
  assay: "RNA_SEQ", columns: [{ name: "Example signal", strand: "." }], native_resolution_bp: 128,
  units: "Normalized read signal", tissue_specific: true,
  bin_edges: [36_200_000, 36_202_000, 36_204_000],
  reference: [[0.1], [0.2]], alternate: [[0.2], [0.3]], delta: [[0.1], [0.1]],
};
const result: Prediction = {
  assembly: "GRCh38", variant: EXAMPLE, interval: { chromosome: "chr22", start: 36_200_000, end: 36_204_000 },
  tissue_ontology: "UBERON:0002048", signal_source: "mock", reference_validated: false,
  tracks: [track, { ...track, assay: "ATAC", units: "Fold enrichment" }],
  missing_assays: ["CAGE", "SPLICE_SITES"],
  score: { value: .72, mode: "demo", calibrated: false, model_version: "demo", network_version: "fixture" },
  attributions: [{ assay: "RNA_SEQ", max_abs_delta: .1, mean_abs_delta: .1, masked_logit_change: -.3 }],
};

describe("variant workspace", () => {
  it("loads the example and submits a new locus and window", async () => {
    const user = userEvent.setup();
    const fetcher = vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, options) =>
      Response.json({ ...result, variant: JSON.parse(String(options?.body)) }));
    render(<Workspace docsUrl="http://127.0.0.1:8000/docs" />);
    expect(screen.getByText("Analyzing the variant…")).toBeInTheDocument();
    await screen.findByRole("heading", { name: "GNN driver score" });
    expect(screen.getByText("DEMO MODEL")).toBeInTheDocument();
    expect(screen.getByText("Synthetic signals")).toBeInTheDocument();
    expect(screen.getByText("Uncalibrated score")).toBeInTheDocument();
    const position = screen.getByLabelText("Position (bp)");
    await user.clear(position);
    await user.type(position, "36201700");
    await user.selectOptions(screen.getByLabelText("Biosample"), "brain");
    await user.selectOptions(screen.getByLabelText("Model window"), "1mb");
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    await screen.findByText("chr22:36,201,700 A→C");
    expect(JSON.parse(String(fetcher.mock.lastCall?.[1]?.body))).toMatchObject({
      pos: 36_201_700, tissue: "brain", window: "1mb",
    });
    await user.click(screen.getByRole("button", { name: "Load example" }));
    await screen.findByText("chr22:36,201,698 A→C");
    expect(position).toHaveValue(String(EXAMPLE.pos));
  });

  it("keeps the last result visible after an API error and dismisses the toast", async () => {
    const user = userEvent.setup();
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(Response.json(result))
      .mockResolvedValueOnce(Response.json({ message: "The API is busy." }, { status: 503 }));
    render(<Workspace docsUrl="#" />);
    await screen.findByRole("heading", { name: "Genomic signal tracks" });
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The API is busy.");
    expect(screen.getByText("chr22:36,201,698 A→C")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Dismiss error" }));
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("validates alleles and supports assay, zoom, and allele visibility controls", async () => {
    const user = userEvent.setup();
    const fetcher = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(Response.json(result));
    render(<Workspace docsUrl="#" />);
    await screen.findByRole("heading", { name: "Genomic signal tracks" });
    await user.selectOptions(screen.getByLabelText("ALT"), "A");
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Choose different bases");
    expect(fetcher).toHaveBeenCalledTimes(1);

    expect(screen.getByRole("button", { name: "CAGE" })).toBeDisabled();
    const atac = screen.getByRole("button", { name: "ATAC-seq" });
    await user.click(atac);
    expect(atac).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Fold enrichment")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Zoom to variant" }));
    expect(screen.getByRole("button", { name: "Full window" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "ALT C" }));
    expect(screen.getByRole("button", { name: "ALT C" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "REF A" })).toBeDisabled();
    await waitFor(() => expect(screen.getByRole("heading", { name: "Assay sensitivity" })).toBeInTheDocument());
  });
});
