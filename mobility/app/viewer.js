/* M4 viewer scaffold — data loading + three placeholder views.
   Structure only; visual polish comes after the full run. */

"use strict";

const state = { rows: [], days: [], byPid: new Map() };

const $ = (id) => document.getElementById(id);

$("file").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const text = await file.text();
  state.rows = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  index();
  render();
});

function index() {
  state.days = [...new Set(state.rows.map((r) => r.day))].sort((a, b) => a - b);
  state.byPid = new Map();
  for (const r of state.rows) {
    if (!state.byPid.has(r.pid)) state.byPid.set(r.pid, []);
    state.byPid.get(r.pid).push(r);
  }
  for (const rows of state.byPid.values()) rows.sort((a, b) => a.day - b.day);

  $("status").textContent =
    `${state.byPid.size} travelers · ${state.days.length} days`;
  const slider = $("day-slider");
  slider.max = state.days.length;
  slider.value = 1;
  slider.oninput = () => drawDay(Number(slider.value));

  const sel = $("traveler-select");
  sel.innerHTML = "";
  for (const pid of [...state.byPid.keys()].sort()) {
    sel.appendChild(new Option(pid, pid));
  }
  sel.onchange = () => drawTraveler(sel.value);
}

function render() {
  drawDay(1);
  drawHistograms();
  drawTraveler($("traveler-select").value);
}

const clock = (m) =>
  `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.round(m % 60)).padStart(2, "0")}`;

function timeScale(rows, w) {
  const lo = Math.min(...rows.map((r) => r.dep_min)) - 5;
  const hi = Math.max(...rows.map((r) => r.arrive_min)) + 5;
  return (m) => ((m - lo) / (hi - lo)) * w;
}

function drawDay(dayIndex) {
  const day = state.days[dayIndex - 1];
  $("day-label").textContent = day ? `day ${day}` : "–";
  const canvas = $("day-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rows = state.rows.filter((r) => r.day === day);
  if (!rows.length) return;
  const x = timeScale(rows, canvas.width);
  const laneY = { A: 70, B: 160 };
  const css = getComputedStyle(document.documentElement);
  const color = { A: css.getPropertyValue("--route-a"), B: css.getPropertyValue("--route-b") };
  for (const route of ["A", "B"]) {
    ctx.fillStyle = css.getPropertyValue("--muted");
    ctx.font = "12px sans-serif";
    ctx.fillText(`route ${route}`, 8, laneY[route] - 22);
  }
  for (const r of rows) {
    ctx.strokeStyle = color[r.route];
    ctx.globalAlpha = 0.35;
    ctx.beginPath();
    ctx.moveTo(x(r.dep_min), laneY[r.route]);
    ctx.lineTo(x(r.arrive_min), laneY[r.route]);
    ctx.stroke();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = color[r.route];
    ctx.beginPath();
    ctx.arc(x(r.arrive_min), laneY[r.route], 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawHistograms() {
  const canvas = $("hist-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.rows.length) return;
  const deps = state.rows.map((r) => r.dep_min);
  const lo = Math.min(...deps), hi = Math.max(...deps) + 1;
  const bins = 40;
  const css = getComputedStyle(document.documentElement);
  const colW = canvas.width / state.days.length;
  // one thin histogram column per day: the morph placeholder
  state.days.forEach((day, di) => {
    const counts = new Array(bins).fill(0);
    for (const r of state.rows) {
      if (r.day !== day) continue;
      counts[Math.min(bins - 1, Math.floor(((r.dep_min - lo) / (hi - lo)) * bins))]++;
    }
    const max = Math.max(...counts, 1);
    ctx.fillStyle = css.getPropertyValue("--route-a");
    counts.forEach((c, b) => {
      const h = (c / max) * (canvas.height - 30);
      ctx.globalAlpha = 0.25 + 0.6 * (c / max);
      ctx.fillRect(di * colW + 2, canvas.height - h - 15,
                   Math.max(colW - 4, 1), h);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = css.getPropertyValue("--muted");
    ctx.font = "10px sans-serif";
    ctx.fillText(String(day), di * colW + colW / 2 - 3, canvas.height - 2);
  });
}

function drawTraveler(pid) {
  const tbody = $("traveler-table").querySelector("tbody");
  tbody.innerHTML = "";
  const rows = state.byPid.get(pid) || [];
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${r.day}</td><td>${r.route}</td><td>${clock(r.dep_min)}</td>` +
      `<td>${Math.round(r.travel_min)} min</td><td>${clock(r.arrive_min)}</td>`;
    tbody.appendChild(tr);
  }
}
