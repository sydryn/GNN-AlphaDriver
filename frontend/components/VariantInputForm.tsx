"use client";

import { ArrowRight, ChevronDown, FlaskConical, LoaderCircle, RotateCcw } from "lucide-react";
import { useState, type FormEvent } from "react";
import { EXAMPLE, TISSUES, type Tissue, type VariantQuery } from "../lib/contracts";

const CHROMOSOMES = [...Array.from({ length: 22 }, (_, i) => String(i + 1)), "X", "Y"];
const BASES = ["A", "C", "G", "T"];

export default function VariantInputForm({
  onAnalyze, loading,
}: { onAnalyze: (query: VariantQuery) => void; loading: boolean }) {
  const [query, setQuery] = useState(EXAMPLE);
  const [position, setPosition] = useState(String(EXAMPLE.pos));
  const [error, setError] = useState<string>();
  const update = <K extends keyof VariantQuery>(key: K, value: VariantQuery[K]) => {
    setQuery((current) => ({ ...current, [key]: value }));
    setError(undefined);
  };

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const pos = Number(position);
    if (!Number.isSafeInteger(pos) || pos < 1) {
      setError("Enter a positive whole-number position.");
      return;
    }
    if (query.ref === query.alt) {
      setError("Choose different bases for REF and ALT.");
      return;
    }
    setError(undefined);
    onAnalyze({ ...query, pos });
  }

  function loadExample() {
    setQuery(EXAMPLE);
    setPosition(String(EXAMPLE.pos));
    setError(undefined);
    onAnalyze(EXAMPLE);
  }

  return (
    <section className="panel parameter-panel" aria-labelledby="parameter-title">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <span className="panel-icon"><FlaskConical size={17} aria-hidden /></span>
          <h2 id="parameter-title" className="section-title">Variant parameters</h2>
        </div>
        <button type="button" onClick={loadExample} disabled={loading} className="text-action">
          <RotateCcw size={14} aria-hidden /> Load example
        </button>
      </div>
      <form onSubmit={submit} aria-describedby={error ? "parameter-error" : "parameter-hint"}>
        <div className="parameter-grid">
          <label className="field">
            Chromosome
            <span className="select-wrap">
              <select value={query.chr} onChange={(event) => update("chr", event.target.value)}>
                {CHROMOSOMES.map((value) => <option key={value} value={`chr${value}`}>chr{value}</option>)}
              </select>
              <ChevronDown aria-hidden size={14} />
            </span>
          </label>
          <label className="field">
            Position (bp)
            <input
              className="mono" type="text" inputMode="numeric" required pattern="[0-9]+"
              value={position} onChange={(event) => { setPosition(event.target.value); setError(undefined); }}
            />
          </label>
          <label className="field">
            REF
            <span className="select-wrap">
              <select value={query.ref} onChange={(event) => update("ref", event.target.value)}>
                {BASES.map((base) => <option key={base}>{base}</option>)}
              </select>
              <ChevronDown aria-hidden size={14} />
            </span>
          </label>
          <label className="field">
            ALT
            <span className="select-wrap">
              <select value={query.alt} onChange={(event) => update("alt", event.target.value)}>
                {BASES.map((base) => <option key={base}>{base}</option>)}
              </select>
              <ChevronDown aria-hidden size={14} />
            </span>
          </label>
          <label className="field">
            Biosample
            <span className="select-wrap">
              <select value={query.tissue} onChange={(event) => update("tissue", event.target.value as Tissue)}>
                {Object.entries(TISSUES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
              <ChevronDown aria-hidden size={14} />
            </span>
          </label>
          <label className="field">
            Model window
            <span className="select-wrap">
              <select value={query.window} onChange={(event) => update("window", event.target.value as VariantQuery["window"])}>
                <option value="100kb">100 kb</option>
                <option value="1mb">1 Mb</option>
              </select>
              <ChevronDown aria-hidden size={14} />
            </span>
          </label>
          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? <LoaderCircle className="animate-spin motion-reduce:animate-none" size={17} aria-hidden /> : null}
            {loading ? "Analyzing…" : "Run analysis"}
            {!loading && <ArrowRight size={16} aria-hidden />}
          </button>
        </div>
        {error
          ? <p id="parameter-error" className="field-error" role="alert">{error}</p>
          : <p id="parameter-hint" className="muted form-hint">GRCh38 · 1-based coordinates · Single-nucleotide variants</p>}
      </form>
    </section>
  );
}
