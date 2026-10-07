export function winRateChart(row) {
  const svg = (tag, attributes = {}, text) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  if (!row.points.length) {
    const empty = document.createElement('span'); empty.textContent = 'No decided games yet'; return empty;
  }
  const chart = svg('svg', {viewBox: '0 0 360 150', role: 'img', 'aria-label': `Level ${row.level}: cumulative win percentage across ${row.points.length} decided games, with 95% confidence intervals`, class: 'win-rate-chart'});
  const x = count => row.points.length === 1 ? 195 : 40 + (count - 1) / (row.points.length - 1) * 300;
  const y = rate => 115 - rate * 100;
  for (const rate of [0, .5, 1]) {
    chart.append(svg('path', {d: `M40 ${y(rate)}H340`, stroke: '#46556b'}), svg('text', {x: 34, y: y(rate) + 4, 'text-anchor': 'end'}, `${rate * 100}%`));
  }
  chart.append(svg('polyline', {points: row.points.map(p => `${x(p.count)},${y(p.rate)}`).join(' '), fill: 'none', stroke: '#80bfff', 'stroke-width': 2}));
  for (const point of row.points) {
    const px = x(point.count), top = y(point.interval[1]), bottom = y(point.interval[0]);
    const group = svg('g');
    group.append(svg('title', {}, `Game ${point.count}${Number.isFinite(point.completedAt) ? ` · ${new Date(point.completedAt).toLocaleString()}` : ' · approximate historical order'}: ${(point.rate * 100).toFixed(1)}% wins; 95% CI ${(point.interval[0] * 100).toFixed(1)}–${(point.interval[1] * 100).toFixed(1)}%`));
    group.append(svg('path', {d: `M${px} ${top}V${bottom}M${px - 3} ${top}h6M${px - 3} ${bottom}h6`, stroke: '#80bfff', 'stroke-opacity': .6}), svg('circle', {cx: px, cy: y(point.rate), r: 3, fill: '#b8dcff'}));
    chart.append(group);
  }
  chart.append(svg('text', {x: 40, y: 132}, '1'), svg('text', {x: 340, y: 132, 'text-anchor': 'end'}, String(row.points.length)), svg('text', {x: 190, y: 147, 'text-anchor': 'middle'}, 'Decided games (earliest → latest)'));
  return chart;
}
