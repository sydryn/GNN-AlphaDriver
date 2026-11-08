"use client";

import { GitFork, Info } from "lucide-react";
import { Pie, PieChart } from "recharts";
import { ASSAY_LABEL, type Prediction } from "../lib/contracts";

export default function DriverScorecard({ result }: { result: Prediction }) {
  const { score, attributions } = result;
  const demo = score.mode === "demo";
  const magnitude = Math.max(...attributions.map((item) => Math.abs(item.masked_logit_change)), 0);
  const ordered = [...attributions].sort((a, b) =>
    Math.abs(b.masked_logit_change) - Math.abs(a.masked_logit_change));
  const displayScore = (score.value * 100).toFixed(1);

  return (
    <aside className="result-sidebar" aria-label="Model interpretation">
      <section className="score-card" aria-labelledby="score-title">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="score-icon"><GitFork size={18} aria-hidden /></span>
          <span className="score-badge"><span />{demo ? "DEMO MODEL" : "RESEARCH MODEL"}</span>
        </div>
        <h2 id="score-title">GNN driver score</h2>
        <p className="score-subtitle">Heterogeneous Graph Transformer</p>
        <div className="gauge">
          <div aria-hidden="true">
            <PieChart width={224} height={122}>
              <Pie data={[{ value: 1 }]} dataKey="value" cx={112} cy={111}
                innerRadius={83} outerRadius={98} startAngle={180} endAngle={0}
                fill="#3d3458" stroke="none" isAnimationActive={false} />
              <Pie data={[{ value: 1 }]} dataKey="value" cx={112} cy={111}
                innerRadius={83} outerRadius={98} startAngle={180} endAngle={180 * (1 - score.value)}
                fill="#bba7ff" stroke="none" cornerRadius={6} isAnimationActive={false} />
            </PieChart>
          </div>
          <div className="gauge-readout">
            <strong>{displayScore}<span>%</span></strong>
            <span>{score.calibrated ? "Calibrated estimate" : "Uncalibrated score"}</span>
          </div>
        </div>
        <div className="gauge-scale mono" aria-hidden="true"><span>0</span><span>100</span></div>
        <p className="score-note"><Info size={15} aria-hidden />
          <span>{demo ? "Demo weights illustrate the workflow. This score has no clinical interpretation."
            : "Research score. Its calibration and predictive validity depend on the supplied checkpoint."}</span>
        </p>
      </section>
      <section className="panel evidence-panel" aria-labelledby="evidence-title">
        <div className="flex items-center justify-between gap-2">
          <h2 id="evidence-title" className="section-title">Assay sensitivity</h2>
          <span className="mini-label mono">Δ LOGIT</span>
        </div>
        <p className="muted evidence-description">Full logit minus logit with the assay masked.</p>
        <div className="evidence-list">
          {ordered.map((item) => {
            const effect = item.masked_logit_change;
            const width = magnitude ? 100 * Math.abs(effect) / magnitude : 0;
            return (
              <div className="evidence-row" key={item.assay}>
                <span>{ASSAY_LABEL[item.assay]}</span>
                <span className="evidence-bar" aria-hidden><span style={{ width: `${width}%` }} /></span>
                <span className="mono evidence-value">{effect >= 0 ? "+" : "−"}{Math.abs(effect).toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      </section>
    </aside>
  );
}
