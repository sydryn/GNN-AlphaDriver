"use client";

import { ArrowUpRight, CircleAlert, Dna, Layers3, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import DriverScorecard from "./DriverScorecard";
import GenomicTrackViewer from "./GenomicTrackViewer";
import VariantInputForm from "./VariantInputForm";
import { EXAMPLE, TISSUES, integer, type Prediction, type VariantQuery } from "../lib/contracts";

export default function Workspace({ docsUrl }: { docsUrl: string }) {
  const [result, setResult] = useState<Prediction>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const controller = useRef<AbortController | null>(null);

  const analyze = useCallback(async (query: VariantQuery) => {
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setLoading(true);
    setError(undefined);
    try {
      const response = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(query), signal: request.signal,
      });
      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.message || "Analysis could not be completed. Try again.");
      }
      const data: Prediction = await response.json();
      if (!request.signal.aborted) setResult(data);
    } catch (cause) {
      if (!request.signal.aborted) {
        setError(cause instanceof Error ? cause.message : "Analysis could not be completed.");
      }
    } finally {
      if (!request.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initial = setTimeout(() => void analyze(EXAMPLE), 0);
    return () => {
      clearTimeout(initial);
      controller.current?.abort();
    };
  }, [analyze]);

  return (
    <>
      <header className="topbar">
        <div className="workspace-width topbar-inner">
          <div className="wordmark"><span className="brand-mark"><Dna size={22} aria-hidden /></span>
            <span>GNN<span className="brand-separator">-</span>AlphaDriver</span>
          </div>
          <a className="topbar-link" href={docsUrl} target="_blank" rel="noreferrer">API reference <ArrowUpRight size={15} aria-hidden /></a>
        </div>
      </header>
      <main className="workspace-width main-area">
        <div className="intro">
          <div>
            <p className="eyebrow"><span /> GENOMICS RESEARCH WORKSPACE</p>
            <h1>Variant analysis</h1>
            <p className="intro-description">Explore regulatory signals and graph-based scores for non-coding variants.</p>
          </div>
          <div className="source-pill"><span className={result?.signal_source === "alphagenome" ? "live-dot" : "demo-dot"} />
            {result?.signal_source === "alphagenome" ? "AlphaGenome signals" : result ? "Synthetic signals" : "GRCh38 workspace"}
          </div>
        </div>
        <VariantInputForm onAnalyze={analyze} loading={loading} />
        <div className="results-wrap" aria-busy={loading}>
          {result ? (
            <>
              <div className="result-context">
                <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                  <Layers3 size={15} aria-hidden />
                  <span className="context-label">RESULTS</span>
                  <span className="context-divider" />
                  <span className="variant-chip mono">
                    {result.variant.chr}:{integer.format(result.variant.pos)} {result.variant.ref}→{result.variant.alt}
                  </span>
                  <span className="context-tissue">{TISSUES[result.variant.tissue]}</span>
                </div>
                {loading && <span className="muted" role="status">Updating analysis…</span>}
              </div>
              <div className="result-grid">
                <GenomicTrackViewer key={JSON.stringify(result.variant)} result={result} />
                <DriverScorecard result={result} />
              </div>
            </>
          ) : (
            <div className="panel result-placeholder" role={loading ? "status" : undefined}>
              <span className="panel-icon"><Layers3 size={20} aria-hidden /></span>
              <h2 className="section-title">{loading ? "Analyzing the variant…" : "Run a variant analysis"}</h2>
              <p className="muted">{loading ? "Preparing genomic tracks and evaluating the graph."
                : "Choose a locus above to compare allele signals."}</p>
              {loading && <span className="loading-line" aria-hidden />}
            </div>
          )}
        </div>
        <footer className="workspace-footer">
          <span>GNN-AlphaDriver <span className="footer-divider">/</span> Research workspace</span>
          <span>Human GRCh38 <span className="footer-divider">·</span> v0.1.0</span>
        </footer>
      </main>
      {error && (
        <div className="toast" role="alert">
          <CircleAlert size={20} aria-hidden />
          <div><strong>Analysis unavailable</strong><p>{error}</p></div>
          <button type="button" aria-label="Dismiss error" onClick={() => setError(undefined)}><X size={17} aria-hidden /></button>
        </div>
      )}
    </>
  );
}
