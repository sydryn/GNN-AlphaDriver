"use client";

import { Activity, ChevronDown, Download, Focus, Info, Maximize2 } from "lucide-react";
import { useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { ASSAYS, ASSAY_LABEL, TISSUES, integer, type Assay, type Prediction, type Track } from "../lib/contracts";

const REF_COLOR = "#6244d7";
const ALT_COLOR = "#077d74";

function exportTrack(track: Track, result: Prediction) {
  const header = ["start_0based", "end_exclusive", ...track.columns.flatMap(({ name }) =>
    ["REF", "ALT", "delta"].map((kind) => `"${kind} ${name.replaceAll('"', '""')}"`))];
  const rows = track.reference.map((values, row) => [
    track.bin_edges[row], track.bin_edges[row + 1],
    ...values.flatMap((ref, col) => [ref, track.alternate[row][col], track.delta[row][col]]),
  ]);
  const objectUrls = URL;
  const url = objectUrls.createObjectURL(new Blob(
    [[header.join(","), ...rows.map((row) => row.join(","))].join("\n")], { type: "text/csv;charset=utf-8" },
  ));
  const link = document.createElement("a");
  const { chr, pos, ref, alt } = result.variant;
  link.href = url;
  link.download = `${chr}-${pos}-${ref}-${alt}-${track.assay}.csv`;
  link.click();
  setTimeout(() => objectUrls.revokeObjectURL(url), 1000);
}

export default function GenomicTrackViewer({ result }: { result: Prediction }) {
  const [assay, setAssay] = useState<Assay>(result.tracks[0].assay);
  const [column, setColumn] = useState(0);
  const [zoomed, setZoomed] = useState(false);
  const [showRef, setShowRef] = useState(true);
  const [showAlt, setShowAlt] = useState(true);
  const track = result.tracks.find((item) => item.assay === assay) ?? result.tracks[0];
  const { interval, variant } = result;
  const start = zoomed ? Math.max(interval.start + 1, variant.pos - 10_000) : interval.start + 1;
  const end = zoomed ? Math.min(interval.end, variant.pos + 10_000) : interval.end;
  const rows = track.reference.map((values, row) => ({
    position: (track.bin_edges[row] + track.bin_edges[row + 1] + 1) / 2,
    ref: values[column],
    alt: track.alternate[row][column],
  })).filter((row) => row.position >= start && row.position <= end);
  const [low, high] = [start, end].map((position) => integer.format(position));
  const tickFormat = (value: number) => `${(value / 1_000_000).toFixed(3)} Mb`;

  return (
    <section className="panel track-panel" aria-labelledby="track-title">
      <div className="track-header">
        <div className="flex items-start gap-3">
          <span className="panel-icon"><Activity size={18} aria-hidden /></span>
          <div>
            <h2 id="track-title" className="section-title">Genomic signal tracks</h2>
            <p className="muted track-subtitle">Compare predicted REF and ALT signals across the locus.</p>
          </div>
        </div>
        <button className="secondary-button" type="button" onClick={() => exportTrack(track, result)}>
          <Download size={15} aria-hidden /> Export CSV
        </button>
      </div>
      <div className="assay-strip" aria-label="Signal assay" role="group">
        {ASSAYS.map((item) => (
          <button type="button" key={item} aria-pressed={item === track.assay}
            disabled={result.missing_assays.includes(item)}
            title={result.missing_assays.includes(item) ? "No tracks returned for this biosample" : undefined}
            onClick={() => { setAssay(item); setColumn(0); }}>
            {ASSAY_LABEL[item]}
          </button>
        ))}
      </div>
      <div className="plot-body">
        <div className="plot-controls">
          <div className="flex flex-wrap items-center gap-3" aria-label="Visible alleles" role="group">
            <button type="button" className="legend-button" aria-pressed={showRef}
              disabled={showRef && !showAlt} onClick={() => setShowRef(!showRef)}>
              <span className="legend-swatch ref-swatch" /> REF <span className="mono">{variant.ref}</span>
            </button>
            <button type="button" className="legend-button" aria-pressed={showAlt}
              disabled={showAlt && !showRef} onClick={() => setShowAlt(!showAlt)}>
              <span className="legend-swatch alt-swatch" /> ALT <span className="mono">{variant.alt}</span>
            </button>
          </div>
          <button className="secondary-button zoom-button" type="button" onClick={() => setZoomed(!zoomed)}>
            {zoomed ? <Maximize2 size={14} aria-hidden /> : <Focus size={15} aria-hidden />}
            {zoomed ? "Full window" : "Zoom to variant"}
          </button>
        </div>
        {track.columns.length > 1 && (
          <label className="field track-select">
            Track
            <span className="select-wrap">
              <select value={column} onChange={(event) => setColumn(Number(event.target.value))}>
                {track.columns.map(({ name, strand }, i) => (
                  <option value={i} key={i}>{name}{strand !== "." ? ` (${strand})` : ""}</option>
                ))}
              </select>
              <ChevronDown size={14} aria-hidden />
            </span>
          </label>
        )}
        <div className="plot-axis-label">{track.units}</div>
        <div className="plot-frame" aria-label={`${ASSAY_LABEL[track.assay]} for ${variant.chr}, ${low} to ${high} base pairs`}>
          <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={220}>
            <LineChart data={rows} margin={{ left: 0, top: 20, right: 17, bottom: 3 }} accessibilityLayer>
              <CartesianGrid vertical={false} stroke="#ebe9f1" strokeDasharray="3 4" />
              <XAxis type="number" dataKey="position" domain={[start, end]} allowDataOverflow
                axisLine={false} tickLine={false} tickMargin={12} tickCount={5}
                tick={{ fontSize: 10, fill: "#646279", fontFamily: "var(--mono)" }} tickFormatter={tickFormat} />
              <YAxis width={40} axisLine={false} tickLine={false} tickMargin={7} tickCount={5}
                tick={{ fontSize: 10, fill: "#646279", fontFamily: "var(--mono)" }} />
              <Tooltip labelFormatter={(value) => `${variant.chr} · ${integer.format(Number(value))} bp`}
                formatter={(value) => Number(value).toPrecision(4)}
                contentStyle={{ fontSize: 12, borderRadius: 10, borderColor: "#e2deec" }} />
              <ReferenceLine x={variant.pos} stroke="#8c7bc1" strokeDasharray="4 4"
                label={{ value: "SNV", position: "insideTop", fontSize: 10, fill: "#574793", dy: -15 }} />
              <Line type="linear" dataKey="ref" name="REF" stroke={REF_COLOR} strokeWidth={2.2}
                dot={false} activeDot={{ r: 4 }} hide={!showRef} isAnimationActive={false} />
              <Line type="linear" dataKey="alt" name="ALT" stroke={ALT_COLOR} strokeWidth={2}
                strokeDasharray="5 3" dot={false} activeDot={{ r: 4 }} hide={!showAlt} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="plot-range">
          <span className="mono">{variant.chr} · {low}–{high} bp</span>
          <span>{track.tissue_specific ? TISSUES[variant.tissue] : "Tissue-independent"}</span>
        </div>
      </div>
      <p className="track-footnote">
        <Info size={15} aria-hidden />
        <span>Each plotted point is a bin mean. Source resolution is {track.native_resolution_bp} bp.</span>
      </p>
    </section>
  );
}
