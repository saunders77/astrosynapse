// Charts ported from astrosynapse2/app/page.tsx; retain the dashboard's scales and filters.
// Snapshot: card_acquire_bucketed_value_auto-828813dc45d24cf086e019c6537d1b02-pd9dd811c376c_20261004-180110.json
import React, { useState, useMemo, useRef, useId } from 'react';
import { createRoot } from 'react-dom/client';
import data from './home-charts-data.json';
type CardEloEntry = {
  key: string;
  cardName: string;
  source: string;
  label: string;
  elo: number;
  rawElo: number;
  uncertainty: number | null;
  rawUncertainty: number | null;
  ciLower: number | null;
  ciUpper: number | null;
  supported: boolean;
  fixedAnchor: boolean;
  decisions: number;
  comparisons: number;
  wins: number;
  losses: number;
  cardColor: "red" | "green" | "blue" | "yellow" | "neutral";
  cost: number;
};

type CardEloBucket = {
  key: string;
  label: string;
  capturedDecisions: number;
  leaderboard: CardEloEntry[];
};

type CardEloChart = {
  acquisitionValue: boolean;
  scaleLabel: string;
  key: string;
  label: string;
  unbucketedDecisions: number;
  buckets: CardEloBucket[];
};

type TurnStatChart = {
  key: string;
  label: string;
  yLabel: string;
  description: string;
  unit: string;
  series: { key: string; label: string; cost: number; cardColor: CardEloEntry["cardColor"]; source: string; points: { turn: number; value: number; lower: number; upper: number; count: number }[] }[];
};

type CardAnalysisView = {
  acquisitionValue: boolean;
  scaleLabel: string;
  acquisitionsRecorded: number;
  calibrationMessage: string;
  rulesVersion?: number;
  id: string;
  status: string;
  kind: "scrap" | "acquire" | "acquire_bucketed";
  modelId: string;
  modelLabel: string;
  progress: number;
  gamesCompleted: number;
  gamesRequested: number;
  singleCardTurns: number;
  scoredDecisions: number;
  comparisons: number;
  truncatedGames: number;
  durationSeconds: number;
  completedAt: string;
  leaderboard: CardEloEntry[];
  bucketedCharts: CardEloChart[];
  turnStatCharts: TurnStatChart[];
  error: string;
};

const numberFormatter = new Intl.NumberFormat('en-US');
const cardArtUrl = (_unused, name) => name === 'No Card' ? null : `https://www.starrealms.com/card-gallery/images/content/card-gallery/${name.toLowerCase().replaceAll(' ', '-')}.webp`;
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asNumber(value: unknown, fallback: number): number {
  const converted = Number(value);
  return Number.isFinite(converted) ? converted : fallback;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function asOptionalNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const converted = Number(value);
  return Number.isFinite(converted) ? converted : null;
}

const acquireChartContexts: Record<string, string> = {
  turn: "turn number",
  own_authority: "your authority",
  acquired_cards: "cards already acquired",
  opponent_authority: "opponent authority",
};

const legacyCardCosts = [
  0, 0, 2, 6, 2, 6, 4, 1, 3, 8, 7, 3, 5, 2, 5, 3, 8, 6, 7, 5, 2, 6, 4, 4, 3,
  1, 6, 2, 7, 8, 1, 3, 4, 6, 4, 3, 5, 4, 7, 8, 2, 5, 3, 1, 6, 4, 6, 5, 3,
];

function acquisitionScaleLabel(result: Record<string, unknown>): string {
  if (result.rating_model === "visible_bundle_acquisition_value_v1") return "Explorer = 2 at every turn (legacy)";
  const calibration = isRecord(result.calibration) ? result.calibration : {};
  return calibration.status === "calibrated" ? "Explorer at turn 3 = 2" : "Raw units · turn-3 calibration unavailable";
}

function isAcquisitionValueModel(value: unknown): boolean {
  return value === "visible_bundle_acquisition_value_v1" || value === "visible_bundle_acquisition_value_v2";
}

function normalizeCardAnalysis(raw: unknown): CardAnalysisView | null {
  if (!isRecord(raw)) return null;
  const result = isRecord(raw.result) ? raw.result : {};
  const model = isRecord(result.model) ? result.model : {};
  const config = isRecord(raw.config) ? raw.config : {};
  const gamesRequested = asNumber(result.games_requested ?? config.games, 1_000);
  const gamesCompleted = asNumber(result.games_completed, 0);
  const rawProgress = asNumber(
    result.progress,
    gamesRequested ? gamesCompleted / gamesRequested : 0,
  );
  const rawLeaderboard = Array.isArray(result.leaderboard) ? result.leaderboard : [];
  const normalizeEloEntry = (
    entry: unknown,
    index: number,
    legacyNormalizationFactor = 1,
  ): CardEloEntry => {
    const item = isRecord(entry) ? entry : {};
    const key = asString(item.key, `card-elo-${index}`);
    const cardId = Number(key.split(":").at(-1));
    const rawColor = asString(item.card_color, "neutral");
    const cardColor = ["red", "green", "blue", "yellow"].includes(rawColor)
      ? rawColor as CardEloEntry["cardColor"]
      : "neutral";
    const uncertainty = asOptionalNumber(item.uncertainty);
    const rawUncertainty = asOptionalNumber(item.raw_uncertainty)
      ?? (uncertainty === null ? null : uncertainty * Math.abs(legacyNormalizationFactor));
    return {
      key,
      cardName: asString(item.card_name, "Unknown card"),
      source: asString(item.source, ""),
      label: asString(item.label ?? item.card_name, "Unknown card"),
      elo: asNumber(item.elo, 0),
      rawElo: asNumber(item.raw_elo, asNumber(item.elo, 0)),
      uncertainty,
      rawUncertainty,
      ciLower: asOptionalNumber(item.ci_lower),
      ciUpper: asOptionalNumber(item.ci_upper),
      supported: item.supported !== false,
      fixedAnchor: item.fixed_anchor === true,
      decisions: asNumber(item.decision_count, 0),
      comparisons: asNumber(item.pairwise_comparisons, 0),
      wins: asNumber(item.wins, 0),
      losses: asNumber(item.losses, 0),
      cardColor,
      cost: asNumber(item.card_cost, legacyCardCosts[cardId] ?? 0),
    };
  };
  const rawCharts = Array.isArray(result.bucketed_charts) ? result.bucketed_charts : [];
  const rawKind = asString(raw.kind ?? result.kind, "acquire");
  return {
    acquisitionValue: isAcquisitionValueModel(result.rating_model),
    scaleLabel: acquisitionScaleLabel(result),
    calibrationMessage: isRecord(result.calibration) ? asString(result.calibration.message, "") : "",
    acquisitionsRecorded: asNumber(result.acquisitions_recorded, 0),
    id: asString(raw.id, ""),
    status: asString(raw.status, "queued"),
    rulesVersion: asNumber(config.rules_version, 1),
    kind: rawKind === "scrap" ? "scrap" : rawKind === "acquire_bucketed" ? "acquire_bucketed" : "acquire",
    modelId: asString(raw.model_id ?? model.id, ""),
    modelLabel: asString(raw.model_label ?? model.label, "Selected candidate"),
    progress: Math.max(0, Math.min(100, rawProgress <= 1 ? rawProgress * 100 : rawProgress)),
    gamesCompleted,
    gamesRequested,
    singleCardTurns: asNumber(result.eligible_turns ?? result.single_card_turns, 0),
    scoredDecisions: asNumber(result.scored_decisions ?? result.decisions_captured, 0),
    comparisons: asNumber(result.pairwise_comparisons, 0),
    truncatedGames: asNumber(result.truncated_games, 0),
    durationSeconds: asNumber(result.duration_seconds, 0),
    completedAt: asString(result.completed_at ?? raw.updated_at ?? raw.created_at, ""),
    leaderboard: rawLeaderboard.map((entry, index) => normalizeEloEntry(entry, index)),
    turnStatCharts: (Array.isArray(result.turn_stat_charts) ? result.turn_stat_charts : []).map((rawChart): TurnStatChart => {
      const chart = isRecord(rawChart) ? rawChart : {};
      return {
        key: asString(chart.key, ""), label: chart.key === "scrap_elo" ? "Scrap from hand/discard" : asString(chart.label, ""),
        yLabel: asString(chart.y_label, ""), description: asString(chart.description, ""),
        unit: asString(chart.unit, "percent"),
        series: (Array.isArray(chart.series) ? chart.series : []).map((rawSeries) => {
          const series = isRecord(rawSeries) ? rawSeries : {};
          const key = asString(series.key, "");
          const cardId = Number(chart.key === "play_scrap" ? key : asString(chart.key, "").startsWith("choice:") ? asString(chart.key, "").split(":")[1] : key.split(":")[1]);
          // Saved turn-stat reports predate card metadata; IDs share the base-set catalog.
          const cardColor: CardEloEntry["cardColor"] = cardId >= 37 ? "blue" : cardId >= 26 ? "yellow" : cardId >= 14 ? "red" : cardId >= 3 ? "green" : "neutral";
          return { key, label: asString(series.label, ""),
            cost: legacyCardCosts[cardId] ?? 0, cardColor, source: key.split(":")[2] ?? "",
            points: (Array.isArray(series.points) ? series.points : []).map((rawPoint) => {
              const point = isRecord(rawPoint) ? rawPoint : {};
              return { turn: asNumber(point.turn, 1), value: asNumber(point.value, 0),
                lower: asNumber(point.lower, 0), upper: asNumber(point.upper, 0), count: asNumber(point.count, 0) };
            }),
          };
        }),
      };
    }),
    bucketedCharts: rawCharts.filter((chart) => !isRecord(chart) || chart.key !== "opponent_top_color").map((chart, chartIndex): CardEloChart => {
      const item = isRecord(chart) ? chart : {};
      const rawBuckets = Array.isArray(item.buckets) ? item.buckets : [];
      return {
        acquisitionValue: isAcquisitionValueModel(item.rating_model),
        scaleLabel: acquisitionScaleLabel(item),
        key: asString(item.key, `bucket-chart-${chartIndex}`),
        label: `${isAcquisitionValueModel(item.rating_model) ? "Acquisition Value" : "Acquire ELO"} by ${acquireChartContexts[asString(item.key, "")] ?? asString(item.label, "bucket")}`,
        unbucketedDecisions: asNumber(item.unbucketed_decisions, 0),
        buckets: rawBuckets.map((bucket, bucketIndex): CardEloBucket => {
          const bucketItem = isRecord(bucket) ? bucket : {};
          const bucketLeaderboard = Array.isArray(bucketItem.leaderboard)
            ? bucketItem.leaderboard
            : [];
          const legacyNormalizationFactor = asNumber(bucketItem.normalization_factor, 1);
          return {
            key: asString(bucketItem.key, `bucket-${bucketIndex}`),
            label: asString(bucketItem.label, String(bucketIndex + 1)),
            capturedDecisions: asNumber(bucketItem.captured_decisions, 0),
            leaderboard: bucketLeaderboard.map((entry, index) => (
              normalizeEloEntry(entry, index, legacyNormalizationFactor)
            )),
          };
        }),
      };
    }),
    error: asString(raw.error, ""),
  };
}

const cardLineColors: Record<CardEloEntry["cardColor"], string> = {
  red: "#ff6678",
  green: "#54df91",
  blue: "#64adff",
  yellow: "#f1c75b",
  neutral: "#d4dbe5",
};

type CardPercentilePoint = {
  entry: CardEloEntry;
  percentile: number;
  lowerPercentile: number;
  upperPercentile: number;
};

const chartCostOptions = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const chartColorOptions: Array<{
  key: CardEloEntry["cardColor"];
  label: string;
}> = [
  { key: "red", label: "Red" },
  { key: "green", label: "Green" },
  { key: "blue", label: "Blue" },
  { key: "yellow", label: "Yellow" },
  { key: "neutral", label: "Neutral" },
];

function percentileRank(value: number, sortedValues: number[]): number {
  if (sortedValues.length <= 1) return 100;
  const lower = sortedValues.filter((candidate) => candidate < value).length;
  const equal = sortedValues.filter((candidate) => candidate === value).length;
  const rank = equal ? lower + (equal - 1) / 2 : lower;
  return Math.max(0, Math.min(100, rank / (sortedValues.length - 1) * 100));
}

function normalCdf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return 0.5 * (1 + sign * erf);
}

function percentileUncertainty(entry: CardEloEntry, competitors: CardEloEntry[]): number {
  if (competitors.length <= 1) return 0;
  const variance = competitors.reduce((total, competitor) => {
    if (competitor.key === entry.key) return total;
    const combinedSigma = Math.hypot(
      entry.rawUncertainty ?? 0,
      competitor.rawUncertainty ?? 0,
    );
    const probabilityAbove = combinedSigma > 0
      ? normalCdf((entry.rawElo - competitor.rawElo) / combinedSigma)
      : entry.rawElo === competitor.rawElo ? 0.5 : Number(entry.rawElo > competitor.rawElo);
    return total + probabilityAbove * (1 - probabilityAbove);
  }, 0);
  return Math.sqrt(variance) / (competitors.length - 1) * 100;
}

function BucketedEloChart({
  chart,
  selectedCosts,
  selectedColors,
}: {
  chart: CardEloChart;
  selectedCosts: number[];
  selectedColors: CardEloEntry["cardColor"][];
}) {
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const width = 1_920;
  const height = 1_080;
  const plot = { left: 112, right: 55, top: 105, bottom: 710 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = plot.bottom - plot.top;
  const series = useMemo(() => {
    const cards = new Map<string, CardEloEntry>();
    const bucketPoints = chart.buckets.map((bucket) => {
      const scored = bucket.leaderboard.filter((entry) => entry.decisions > 0 && entry.supported);
      const sortedElos = scored.map((entry) => entry.rawElo).sort((a, b) => a - b);
      return new Map(scored.map((entry): [string, CardPercentilePoint] => {
        if (!cards.has(entry.key)) cards.set(entry.key, entry);
        const percentile = chart.acquisitionValue ? entry.elo : percentileRank(entry.rawElo, sortedElos);
        const uncertainty = percentileUncertainty(entry, scored);
        return [entry.key, {
          entry,
          percentile,
          lowerPercentile: chart.acquisitionValue ? entry.ciLower ?? entry.elo : Math.max(0, percentile - uncertainty),
          upperPercentile: chart.acquisitionValue ? entry.ciUpper ?? entry.elo : Math.min(100, percentile + uncertainty),
        }];
      }));
    });
    return [...cards.values()]
      .sort((a, b) => a.label.localeCompare(b.label))
      .map((card) => ({
        key: card.key,
        label: card.label,
        cardName: card.cardName,
        artUrl: cardArtUrl(null, card.cardName),
        color: cardLineColors[card.cardColor],
        cardColor: card.cardColor,
        cost: card.cost,
        points: bucketPoints.map((points) => points.get(card.key) ?? null),
      }))
      .filter((card) => (
        selectedCosts.includes(card.cost) && selectedColors.includes(card.cardColor)
      ));
  }, [chart, selectedColors, selectedCosts]);
  const xAt = (index: number) => plot.left
    + (chart.buckets.length <= 1 ? plotWidth / 2 : index / (chart.buckets.length - 1) * plotWidth);
  const plotted = series.flatMap((item) => item.points.filter((point) => point !== null));
  const minimum = chart.acquisitionValue ? Math.min(0, ...plotted.map((point) => point.lowerPercentile)) : 0;
  const maximum = chart.acquisitionValue ? Math.max(3, ...plotted.map((point) => point.upperPercentile)) : 100;
  const yAt = (value: number) => plot.bottom - (value - minimum) / (maximum - minimum) * plotHeight;
  const xAxisY = yAt(0);
  const yTicks = Array.from({ length: 6 }, (_, row) => maximum - row / 5 * (maximum - minimum));
  if (!yTicks.some((value) => Math.abs(value) < 1e-8)) yTicks.push(0);
  const pathFor = (points: Array<CardPercentilePoint | null>) => {
    let drawing = false;
    return points.map((point, index) => {
      if (!point) {
        drawing = false;
        return "";
      }
      const command = drawing ? "L" : "M";
      drawing = true;
      return `${command}${xAt(index).toFixed(1)},${yAt(point.percentile).toFixed(1)}`;
    }).join(" ");
  };
  const resolvedActiveKey = series.some((item) => item.key === activeKey) ? activeKey : null;
  const activeSeries = series.find((item) => item.key === resolvedActiveKey) ?? null;
  const legendColumns = 5;
  const legendColumnWidth = 350;
  const legendRowHeight = 25;

  return (
    <article className="bucketed-chart-card">
      <div className="bucketed-chart-frame">
        {activeSeries?.artUrl ? <aside className="bucketed-card-art-preview" aria-live="polite"><img src={activeSeries.artUrl} alt={`${activeSeries.cardName} card`} /><span>{activeSeries.label}</span></aside> : null}
        <svg
          className="bucketed-elo-svg"
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          role="img"
          aria-label={`${chart.label} card ${chart.acquisitionValue ? "acquisition value" : "percentile"} chart`}
          onMouseLeave={() => setActiveKey(null)}
        >
          <rect width={width} height={height} rx="18" className="bucketed-chart-background" />
          <text x={plot.left} y="48" className="bucketed-chart-title">{chart.label}</text>
          <text x={plot.left} y="76" className="bucketed-chart-subtitle">
            {chart.acquisitionValue
              ? `${activeSeries?.label ?? "Acquisition Value"} · ${chart.scaleLabel} · hover for approximate 95% confidence intervals`
              : activeSeries
              ? `${activeSeries.label} · percentile uncertainty bars visible`
              : `Within-bucket percentile · hover a line or legend name to show its ±1σ rank range${chart.unbucketedDecisions ? ` · ${numberFormatter.format(chart.unbucketedDecisions)} pre-color states omitted` : ""}`}
          </text>
          {yTicks.map((value, row) => {
            const y = yAt(value);
            return <g key={`grid-${row}`}><line x1={plot.left} x2={width - plot.right} y1={y} y2={y} className="bucketed-chart-grid" /><text x={plot.left - 16} y={y + 5} textAnchor="end" className="bucketed-chart-axis">{chart.acquisitionValue ? value.toFixed(1) : `${value}th`}</text></g>;
          })}
          <text transform={`translate(31 ${(plot.top + plot.bottom) / 2}) rotate(-90)`} textAnchor="middle" className="bucketed-chart-axis-title">{chart.acquisitionValue ? `Acquisition Value (${chart.scaleLabel})` : "Percentile within bucket"}</text>
          <line x1={plot.left} x2={width - plot.right} y1={xAxisY} y2={xAxisY} stroke="#d4dbe5" strokeWidth="2" />
          {chart.buckets.map((bucket, index) => <g key={bucket.key}><line x1={xAt(index)} x2={xAt(index)} y1={plot.top} y2={plot.bottom} className="bucketed-chart-grid bucketed-chart-grid-vertical" /><text x={xAt(index)} y={xAxisY + 30} textAnchor="middle" className="bucketed-chart-axis">{bucket.label}</text><text x={xAt(index)} y={xAxisY + 50} textAnchor="middle" className="bucketed-chart-count">n={numberFormatter.format(bucket.capturedDecisions)}</text></g>)}
          {series.map((item) => {
            const active = resolvedActiveKey === item.key;
            const dimmed = resolvedActiveKey !== null && !active;
            const path = pathFor(item.points);
            return <g key={item.key} className={active ? "is-active" : dimmed ? "is-dimmed" : ""}>
              {active ? item.points.map((point, index) => {
                if (!point || point.lowerPercentile === point.upperPercentile) return null;
                const x = xAt(index);
                const upper = yAt(point.upperPercentile);
                const lower = yAt(point.lowerPercentile);
                return <g key={`${item.key}-error-${index}`} className="bucketed-error-bar" style={{ color: item.color }}><line x1={x} x2={x} y1={upper} y2={lower} /><line x1={x - 7} x2={x + 7} y1={upper} y2={upper} /><line x1={x - 7} x2={x + 7} y1={lower} y2={lower} /></g>;
              }) : null}
              <path d={path} fill="none" stroke={item.color} className="bucketed-card-line" />
              <path d={path} fill="none" stroke="transparent" strokeWidth="14" className="bucketed-card-hit" onMouseEnter={() => setActiveKey(item.key)} />
              {active ? item.points.map((point, index) => point ? <circle key={`${item.key}-point-${index}`} cx={xAt(index)} cy={yAt(point.percentile)} r="5" fill={item.color}><title>{chart.acquisitionValue
                ? `${item.label} · ${chart.buckets[index].label}: ${point.entry.elo.toFixed(2)}; 95% CI [${point.lowerPercentile.toFixed(2)}, ${point.upperPercentile.toFixed(2)}]${point.entry.fixedAnchor ? " (fixed anchor)" : ""}; ${numberFormatter.format(point.entry.decisions)} decisions`
                : `${item.label} · ${chart.buckets[index].label}: ${point.percentile.toFixed(1)}th percentile (${numberFormatter.format(point.entry.decisions)} decisions)`}</title></circle> : null) : null}
            </g>;
          })}
          <text x={plot.left} y="792" className="bucketed-chart-legend-title">Options · faction color</text>
          {series.map((item, index) => {
            const column = index % legendColumns;
            const row = Math.floor(index / legendColumns);
            const x = plot.left + column * legendColumnWidth;
            const y = 825 + row * legendRowHeight;
            const active = resolvedActiveKey === item.key;
            const dimmed = resolvedActiveKey !== null && !active;
            return <g key={`${item.key}-legend`} transform={`translate(${x} ${y})`} className={`bucketed-legend-item${active ? " is-active" : ""}${dimmed ? " is-dimmed" : ""}`} onMouseEnter={() => setActiveKey(item.key)} onFocus={() => setActiveKey(item.key)} onBlur={() => setActiveKey(null)} tabIndex={0} role="button" aria-label={`Highlight ${item.label}`}><rect x="-7" y="-17" width={legendColumnWidth - 12} height="23" rx="5" /><line x1="0" x2="31" y1="-5" y2="-5" stroke={item.color} /><text x="40" y="0" fill={active ? item.color : undefined}>{item.label}</text></g>;
          })}
        </svg>
      </div>
    </article>
  );
}

function TurnStatisticsChart({ chart }: { chart: TurnStatChart }) {
  const [active, setActive] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId();
  const width = 1920;
  const height = 900 + Math.ceil(chart.series.length / 4) * 30;
  const points = chart.series.flatMap((series) => series.points);
  const min = 0;
  const max = chart.unit === "percent" ? 100 : 3000;
  const x = (turn: number) => 120 + (turn - 1) / 29 * 1720;
  const y = (value: number) => 740 - (value - min) / (max - min) * 580;
  const color = (series: TurnStatChart["series"][number], index: number) => chart.key === "play_scrap"
    ? cardLineColors[series.cardColor]
    : chart.key === "scrap_elo" ? series.source === "hand" ? "#64adff" : "#f1c75b"
    : `hsl(${index * 137.508 % 360} 72% 64%)`;
  const download = () => {
    if (!svgRef.current) return;
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svgRef.current)], { type: "image/svg+xml" }));
    const link = document.createElement("a");
    link.href = url; link.download = `${chart.key.replaceAll(":", "-")}.svg`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <article className="bucketed-chart-card">
    <button type="button" className="button button-secondary" onClick={download}>Save {chart.label} as SVG</button>
    <div className="bucketed-chart-frame">
      <svg ref={svgRef} xmlns="http://www.w3.org/2000/svg" className="bucketed-elo-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={chart.label} onMouseLeave={() => setActive(null)}>
        <defs><clipPath id={clipId}><rect x="115" y="160" width="1730" height="580" /></clipPath></defs>
        <rect width={width} height={height} rx="18" fill="#101827" />
        <g fontFamily="sans-serif" fill="#d9e3ef">
          <text x="120" y="48" fontSize="30">{chart.label}</text>
          <text x="120" y="82" fontSize="20">{chart.key === "scrap_elo" ? "Hand = blue · Discard = yellow. Separate ratings; 95% Elo confidence intervals." : chart.description}</text>
          <text x="120" y="115" fontSize="18">Turn 30+ includes all later turns. Hover a line or legend to highlight; bars show 95% confidence intervals.</text>
          {Array.from({ length: 6 }, (_, index) => {
            const value = min + (max - min) * index / 5;
            return <g key={index}><line x1="120" x2="1840" y1={y(value)} y2={y(value)} stroke="#344155" /><text x="105" y={y(value) + 6} textAnchor="end" fontSize="18">{value.toFixed(0)}{chart.unit === "percent" ? "%" : ""}</text></g>;
          })}
          <text transform="translate(30 450) rotate(-90)" textAnchor="middle" fontSize="22">{chart.yLabel}</text>
          {Array.from({ length: 30 }, (_, index) => <text key={index} x={x(index + 1)} y="775" textAnchor="middle" fontSize="18">{index === 29 ? "30+" : index + 1}</text>)}
          <text x="980" y="815" textAnchor="middle" fontSize="22">Turn number</text>
          {!points.length ? <text x="980" y="420" textAnchor="middle" fontSize="25">No eligible observations in this run</text> : null}
          {chart.series.map((series, index) => {
            const stroke = color(series, index);
            const path = series.points.map((point, i) => `${i && series.points[i - 1].turn === point.turn - 1 ? "L" : "M"}${x(point.turn)},${y(point.value)}`).join(" ");
            return <g key={series.key} opacity={active && active !== series.key ? 0.12 : 1} onMouseEnter={() => setActive(series.key)}>
              <g clipPath={`url(#${clipId})`}>
              <path d={path} stroke={stroke} strokeWidth="3" fill="none" />
              {series.points.map((point) => <g key={point.turn}>
                <path d={`M${x(point.turn)},${y(point.lower)}V${y(point.upper)}M${x(point.turn)-5},${y(point.lower)}h10M${x(point.turn)-5},${y(point.upper)}h10`} stroke={stroke} opacity="0.55" fill="none" />
                <circle cx={x(point.turn)} cy={y(point.value)} r="5" fill={stroke}><title>{`${series.label}, turn ${point.turn === 30 ? "30+" : point.turn}: ${point.value.toFixed(2)} (95% CI ${point.lower.toFixed(2)}–${point.upper.toFixed(2)}), n=${point.count}`}</title></circle>
              </g>)}
              </g>
              <g transform={`translate(${120 + index % 4 * 440} ${875 + Math.floor(index / 4) * 30})`} tabIndex={0} onFocus={() => setActive(series.key)} onBlur={() => setActive(null)} aria-label={series.label}>
                <line x1="0" x2="28" y1="-6" y2="-6" stroke={stroke} strokeWidth="3" /><text x="38" fontSize="18">{series.label}</text>
              </g>
            </g>;
          })}
        </g>
      </svg>
    </div>
  </article>;
}

function BucketedEloCharts({ charts, turnCharts }: { charts: CardEloChart[]; turnCharts: TurnStatChart[] }) {
  const [selectedCosts, setSelectedCosts] = useState<number[]>(chartCostOptions);
  const [selectedColors, setSelectedColors] = useState<CardEloEntry["cardColor"][]>(
    chartColorOptions.map((option) => option.key),
  );
  const availableCards = useMemo(() => {
    const cards = new Map<string, Pick<CardEloEntry, "cost" | "cardColor">>();
    charts.forEach((chart) => chart.buckets.forEach((bucket) => {
      bucket.leaderboard.forEach((entry) => {
        if (entry.decisions > 0) cards.set(entry.key, entry);
      });
    }));
    turnCharts.forEach((chart) => chart.series.forEach((series) => {
      if (chart.key === "scrap_elo" && !["hand", "discard"].includes(series.source)) return;
      const cardKey = chart.key.startsWith("choice:") ? chart.key.replace("choice:", "card:")
        : chart.key === "play_scrap" ? `card:${series.key}` : series.key.split(":").slice(0, 2).join(":");
      cards.set(cardKey, series);
    }));
    return [...cards.values()];
  }, [charts, turnCharts]);
  const visibleCardCount = availableCards.filter((card) => (
    selectedCosts.includes(card.cost) && selectedColors.includes(card.cardColor)
  )).length;
  const toggleCost = (cost: number) => setSelectedCosts((current) => (
    current.includes(cost) ? current.filter((item) => item !== cost) : [...current, cost]
  ));
  const toggleColor = (color: CardEloEntry["cardColor"]) => setSelectedColors((current) => (
    current.includes(color) ? current.filter((item) => item !== color) : [...current, color]
  ));
  const resetFilters = () => {
    setSelectedCosts(chartCostOptions);
    setSelectedColors(chartColorOptions.map((option) => option.key));
  };

  const filteredTurnCharts = turnCharts.map((chart) => ({ ...chart, series: chart.series.filter((series) =>
    selectedCosts.includes(series.cost) && selectedColors.includes(series.cardColor)
    && (chart.key !== "scrap_elo" || ["hand", "discard"].includes(series.source))),
  }));
  const renderAcquire = (chart: CardEloChart) => <BucketedEloChart key={chart.key} chart={chart} selectedCosts={selectedCosts} selectedColors={selectedColors} />;
  const renderTurn = (chart: TurnStatChart) => <TurnStatisticsChart key={chart.key} chart={chart} />;

  return <div className="bucketed-elo-results">
    <div className="analysis-copy">
    <section className="bucketed-chart-filters" aria-label="Filter cards shown in all charts">
      <header><div><span>Chart filters</span><strong>{visibleCardCount} of {availableCards.length} options visible</strong></div><button type="button" onClick={resetFilters}>Reset filters</button></header>
      <fieldset><legend>Cost / special choice</legend><div>{chartCostOptions.map((cost) => <button type="button" key={cost} aria-pressed={selectedCosts.includes(cost)} onClick={() => toggleCost(cost)}>{cost === 0 ? "Cost 0 / No Card" : `Cost ${cost}`}</button>)}</div></fieldset>
      <fieldset><legend>Colour</legend><div>{chartColorOptions.map((option) => <button type="button" key={option.key} className={`filter-color-${option.key}`} aria-pressed={selectedColors.includes(option.key)} onClick={() => toggleColor(option.key)}><i style={{ background: cardLineColors[option.key] }} />{option.label}</button>)}</div></fieldset>
    </section>
    </div>
    {charts.filter((chart) => chart.key === "turn").map(renderAcquire)}
    {filteredTurnCharts.filter((chart) => !chart.key.startsWith("choice:")).map(renderTurn)}
    {charts.filter((chart) => chart.key !== "turn").map(renderAcquire)}
    {filteredTurnCharts.filter((chart) => chart.key.startsWith("choice:") && chart.series.length > 0).map(renderTurn)}
  </div>;
}


export function mountCharts() {
  const result = normalizeCardAnalysis({ result: data });
  createRoot(document.getElementById('analysis-charts')).render(<BucketedEloCharts charts={result.bucketedCharts} turnCharts={result.turnStatCharts} />);
}
