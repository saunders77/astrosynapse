"use client";

import { useEffect, useId, useRef, useState } from "react";

export type StudentTree = {
  id: number;
  score: number;
  examples?: number;
  name?: string;
  feature?: number;
  threshold?: number;
  left?: StudentTree;
  right?: StudentTree;
};

type PositionedNode = { node: StudentTree; x: number; y: number; ancestors: number[] };
const NODE_WIDTH = 244;
const NODE_HEIGHT = 142;
const GAP_X = 28;
const GAP_Y = 88;
const PAD = 24;
const number = (value: number) => value.toLocaleString("en-US", { maximumFractionDigits: 8 });

function layoutTree(tree: StudentTree) {
  const nodes: PositionedNode[] = [];
  let leaves = 0;
  let maxDepth = 0;
  function visit(node: StudentTree, depth: number, ancestors: number[]): PositionedNode {
    maxDepth = Math.max(maxDepth, depth);
    let x: number;
    if (node.left && node.right) {
      const left = visit(node.left, depth + 1, [...ancestors, node.id]);
      const right = visit(node.right, depth + 1, [...ancestors, node.id]);
      x = (left.x + right.x) / 2;
    } else {
      x = PAD + NODE_WIDTH / 2 + leaves++ * (NODE_WIDTH + GAP_X);
    }
    const placed = { node, x, y: PAD + depth * (NODE_HEIGHT + GAP_Y), ancestors };
    nodes.push(placed);
    return placed;
  }
  const root = visit(tree, 0, []);
  return { nodes, root, width: PAD * 2 + leaves * (NODE_WIDTH + GAP_X) - GAP_X,
    height: PAD * 2 + (maxDepth + 1) * NODE_HEIGHT + maxDepth * GAP_Y };
}

function wrap(text: string) {
  const lines: string[] = [];
  for (const word of text.split(" ")) {
    const last = lines.length - 1;
    if (last < 0 || `${lines[last]} ${word}`.length > 27) lines.push(word);
    else lines[last] += ` ${word}`;
  }
  return lines;
}

export default function TreeFlowchart({ tree }: { tree: StudentTree }) {
  const viewport = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [selected, setSelected] = useState<number | null>(null);
  const markerId = useId().replaceAll(":", "");
  const layout = layoutTree(tree);
  const byId = new Map(layout.nodes.map(n => [n.node.id, n]));
  const selection = selected === null ? null : byId.get(selected);
  const highlighted = new Set(selection ? [...selection.ancestors, selection.node.id] : []);
  // API polling returns a new object each time; only recenter when the tree changes.
  const rootX = layout.root.x;
  const treeKey = JSON.stringify(tree);
  useEffect(() => {
    if (viewport.current) viewport.current.scrollLeft = Math.max(0, rootX - viewport.current.clientWidth / 2);
  }, [treeKey, rootX]);

  function changeZoom(next: number) {
    const element = viewport.current;
    const x = element ? (element.scrollLeft + element.clientWidth / 2) / zoom : rootX;
    const y = element ? (element.scrollTop + element.clientHeight / 2) / zoom : 0;
    setZoom(next);
    requestAnimationFrame(() => {
      if (element) {
        element.scrollLeft = x * next - element.clientWidth / 2;
        element.scrollTop = y * next - element.clientHeight / 2;
      }
    });
  }

  return <div className="student-flowchart">
    <div className="student-flow-toolbar" role="group" aria-label="Flowchart controls">
      <span>Yes → left · No → right</span>
      <div>
        <button type="button" aria-label="Zoom out" disabled={zoom <= .2} onClick={() => changeZoom(Math.max(.2, zoom - .2))}>−</button>
        <output aria-label="Flowchart zoom">{Math.round(zoom * 100)}%</output>
        <button type="button" aria-label="Zoom in" disabled={zoom >= 2} onClick={() => changeZoom(Math.min(2, zoom + .2))}>+</button>
        <button type="button" onClick={() => { changeZoom(Math.min(1, (viewport.current?.clientWidth ?? layout.width) / layout.width)); }}>Fit width</button>
        <button type="button" onClick={() => changeZoom(1)}>100%</button>
      </div>
    </div>
    <p>Follow this tree for each available card and None, then choose the highest score. Select a node to highlight its path. Scroll to explore larger trees.</p>
    <div className="student-flow-viewport" ref={viewport} tabIndex={0} role="region" aria-label="Decision tree flowchart; scroll to explore">
      <svg width={layout.width * zoom} height={layout.height * zoom} viewBox={`0 0 ${layout.width} ${layout.height}`} role="group" aria-label="Acquire student decision tree">
        <title>Acquire student decision tree</title>
        <desc>Each question branches Yes to its left child and No to its right child. Leaves assign candidate scores. Use Tab to inspect nodes and Enter to highlight a path.</desc>
        <defs><marker id={markerId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
        {layout.nodes.flatMap(parent => [parent.node.left, parent.node.right].map((child, index) => {
          if (!child) return null;
          const target = byId.get(child.id)!;
          const y = parent.y + NODE_HEIGHT;
          const middle = y + GAP_Y / 2;
          const active = highlighted.has(child.id);
          const labelX = parent.x + (target.x - parent.x) * .6;
          return <g key={`${parent.node.id}-${child.id}`} className={`student-flow-edge ${index === 0 ? "yes" : "no"}${active ? " highlighted" : ""}${selection && !active ? " dimmed" : ""}`}>
            <path d={`M ${parent.x} ${y} C ${parent.x} ${middle}, ${target.x} ${middle}, ${target.x} ${target.y - 5}`} markerEnd={`url(#${markerId})`} />
            <rect x={labelX - 21} y={middle - 12} width="42" height="24" rx="6" />
            <text x={labelX} y={middle + 5} textAnchor="middle">{index === 0 ? "Yes" : "No"}</text>
          </g>;
        }))}
        {[...layout.nodes].sort((a, b) => a.node.id - b.node.id).map(({ node, x, y }) => {
          const question = Boolean(node.left && node.right);
          const label = question ? `Node ${node.id}: ${node.name} ≤ ${number(node.threshold!)}? Yes to ${node.left!.id}, no to ${node.right!.id}.` : `Leaf ${node.id}: candidate score ${number(node.score)}.`;
          return <g key={node.id} transform={`translate(${x - NODE_WIDTH / 2}, ${y})`} role="button" tabIndex={0} aria-label={label} aria-pressed={selected === node.id}
            className={`student-flow-node ${question ? "question" : "leaf"}${highlighted.has(node.id) ? " highlighted" : ""}${selection && !highlighted.has(node.id) ? " dimmed" : ""}`}
            onClick={() => setSelected(selected === node.id ? null : node.id)}
            onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelected(selected === node.id ? null : node.id); } }}>
            <title>{label}</title>
            <rect width={NODE_WIDTH} height={NODE_HEIGHT} rx={question ? 10 : 28} />
            <text x={NODE_WIDTH / 2} y="25" textAnchor="middle" className="student-flow-kind">{question ? `DECISION ${node.id}` : `SCORE · LEAF ${node.id}`}</text>
            {question ? <>
              <text x={NODE_WIDTH / 2} y="51" textAnchor="middle" className="student-flow-question">{wrap(node.name ?? "Feature").map((line, i) => <tspan key={i} x={NODE_WIDTH / 2} dy={i ? 19 : 0}>{line}</tspan>)}</text>
              <text x={NODE_WIDTH / 2} y="126" textAnchor="middle" className="student-flow-threshold">≤ {number(node.threshold!)}?</text>
            </> : <>
              <text x={NODE_WIDTH / 2} y="78" textAnchor="middle" className="student-flow-score">{node.score.toFixed(4)}</text>
              <text x={NODE_WIDTH / 2} y="111" textAnchor="middle" className="student-flow-caption">Higher score wins</text>
            </>}
          </g>;
        })}
      </svg>
    </div>
    <div className="student-flow-selection" aria-live="polite">
      {selection ? <><strong>{selection.node.left ? `Decision ${selected}` : `Leaf ${selected}`}</strong><span>{selection.node.left ? `${selection.node.name} ≤ ${number(selection.node.threshold!)}? Yes → ${selection.node.left.id}; No → ${selection.node.right!.id}.` : `Assign score ${number(selection.node.score)} to this candidate. This is an imitation score, not a win probability.`}</span><button type="button" onClick={() => setSelected(null)}>Clear highlight</button></> : <span>Decision boxes ask a question; green rounded leaves assign a score. Node numbers match the English tree below.</span>}
    </div>
  </div>;
}
