/* GLASSBOX-Mobility M4 viewer.
   System-side trajectory log only; no truth-side data ever reaches this page. */

"use strict";

const state = {
  rows: [], days: [], byPid: new Map(), byDay: new Map(),
  peak: [], t0: 0, t1: 0,
};

const $ = (id) => document.getElementById(id);
const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clock = (m) =>
  `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.round(m % 60)).padStart(2, "0")}`;

// ---------- loading --------------------------------------------------------

$("file").addEventListener("change", (ev) => loadFile(ev.target.files[0]));
document.body.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("dragging");
});
document.body.addEventListener("dragleave", () =>
  document.body.classList.remove("dragging"));
document.body.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("dragging");
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

async function loadFile(file) {
  if (!file) return;
  const text = await file.text();
  state.rows = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  indexData();
  wire();
  drawDayStatic(1);
  drawHistogram(1);
  drawSpark();
  drawTraveler($("traveler-select").value);
  $("main").hidden = false;
}

function indexData() {
  state.days = [...new Set(state.rows.map((r) => r.day))].sort((a, b) => a - b);
  state.byPid = new Map();
  state.byDay = new Map();
  for (const r of state.rows) {
    if (!state.byPid.has(r.pid)) state.byPid.set(r.pid, []);
    state.byPid.get(r.pid).push(r);
    if (!state.byDay.has(r.day)) state.byDay.set(r.day, []);
    state.byDay.get(r.day).push(r);
  }
  for (const rows of state.byPid.values()) rows.sort((a, b) => a.day - b.day);
  // percentile clamp: a lone drifter (one traveler walked its departure
  // back 15 min a day for weeks) must not stretch the whole time axis
  const deps = state.rows.map((r) => r.dep_min).sort((a, b) => a - b);
  const arrs = state.rows.map((r) => r.arrive_min).sort((a, b) => a - b);
  state.t0 = deps[Math.floor(deps.length * 0.005)] - 4;
  state.t1 = arrs[Math.ceil(arrs.length * 0.995) - 1] + 4;
  state.peak = state.days.map((d) => peakShare(state.byDay.get(d)));
  $("status").textContent =
    `${state.byPid.size} travelers · ${state.days.length} days · ` +
    `peak share ${state.peak[0].toFixed(2)} → ${state.peak.at(-1).toFixed(2)}`;
}

function peakShare(rows) {
  const arr = rows.map((r) => r.arrive_min).sort((a, b) => a - b);
  let best = 0;
  for (let i = 0; i < arr.length; i++) {
    let j = i;
    while (j < arr.length && arr[j] < arr[i] + 15) j++;
    best = Math.max(best, j - i);
  }
  return best / arr.length;
}

// ---------- pane 1: one morning, replayed ----------------------------------

let dayAnim = null;

function wire() {
  const slider = $("day-slider");
  slider.max = state.days.length;
  slider.oninput = () => {
    stopDayAnim();
    drawDayStatic(Number(slider.value));
  };
  $("day-play").onclick = () => replayDay(Number(slider.value));

  $("hist-play").onclick = playHistogram;

  const sel = $("traveler-select");
  sel.innerHTML = "";
  for (const pid of [...state.byPid.keys()].sort()) {
    sel.appendChild(new Option(pid, pid));
  }
  sel.onchange = () => drawTraveler(sel.value);
}

const laneY = { A: 100, B: 220 };
const xOf = (m, w) => 60 + ((m - state.t0) / (state.t1 - state.t0)) * (w - 90);

function dayScaffold(ctx, w, h, day) {
  ctx.clearRect(0, 0, w, h);
  ctx.font = "13px sans-serif";
  ctx.fillStyle = css("--muted");
  for (const route of ["A", "B"]) {
    ctx.fillText(`route ${route}`, 8, laneY[route] - 34);
    ctx.strokeStyle = css("--edge");
    ctx.beginPath();
    ctx.moveTo(55, laneY[route]);
    ctx.lineTo(w - 20, laneY[route]);
    ctx.stroke();
  }
  for (let m = Math.ceil(state.t0 / 30) * 30; m <= state.t1; m += 30) {
    const x = xOf(m, w);
    ctx.fillText(clock(m), x - 16, h - 8);
    ctx.strokeStyle = css("--edge");
    ctx.beginPath(); ctx.moveTo(x, 30); ctx.lineTo(x, h - 26); ctx.stroke();
  }
  const rows = state.byDay.get(day);
  const onA = rows.filter((r) => r.route === "A").length;
  $("day-note").textContent =
    `route A ${onA}/${rows.length} · worst drive ` +
    `${Math.round(Math.max(...rows.map((r) => r.travel_min)))} min`;
}

function drawDayStatic(dayIndex) {
  const day = state.days[dayIndex - 1];
  $("day-label").textContent = `day ${day}`;
  const canvas = $("day-canvas");
  const ctx = canvas.getContext("2d");
  dayScaffold(ctx, canvas.width, canvas.height, day);
  $("clock").textContent = "—";
  for (const r of state.byDay.get(day)) {
    const color = css(r.route === "A" ? "--route-a" : "--route-b");
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.moveTo(xOf(r.dep_min, canvas.width), laneY[r.route]);
    ctx.lineTo(xOf(r.arrive_min, canvas.width), laneY[r.route]);
    ctx.stroke();
    ctx.globalAlpha = 0.95;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(xOf(r.arrive_min, canvas.width), laneY[r.route], 2.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function stopDayAnim() {
  if (dayAnim) cancelAnimationFrame(dayAnim);
  dayAnim = null;
}

function replayDay(dayIndex) {
  stopDayAnim();
  const day = state.days[dayIndex - 1];
  const rows = state.byDay.get(day);
  const canvas = $("day-canvas");
  const ctx = canvas.getContext("2d");
  const start = performance.now();
  const SPEED = 18; // simulated minutes per real second

  const frame = (now) => {
    const t = state.t0 + ((now - start) / 1000) * SPEED;
    $("clock").textContent = clock(Math.min(t, state.t1));
    dayScaffold(ctx, canvas.width, canvas.height, day);
    for (const r of rows) {
      const color = css(r.route === "A" ? "--route-a" : "--route-b");
      if (t < r.dep_min) continue;
      const done = Math.min(1, (t - r.dep_min) / (r.arrive_min - r.dep_min));
      const x0 = xOf(r.dep_min, canvas.width);
      const x1 = xOf(r.dep_min + done * (r.arrive_min - r.dep_min), canvas.width);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.25;
      ctx.beginPath();
      ctx.moveTo(x0, laneY[r.route]); ctx.lineTo(x1, laneY[r.route]);
      ctx.stroke();
      ctx.globalAlpha = done < 1 ? 1 : 0.55;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x1, laneY[r.route], done < 1 ? 3.2 : 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (t < state.t1) dayAnim = requestAnimationFrame(frame);
  };
  dayAnim = requestAnimationFrame(frame);
}

// ---------- pane 2: the peak, spreading ------------------------------------

let histTimer = null;

function histCounts(day, bins) {
  const counts = new Array(bins).fill(0);
  for (const r of state.byDay.get(day)) {
    const b = Math.floor(((r.dep_min - state.t0) / (state.t1 - state.t0)) * bins);
    counts[Math.max(0, Math.min(bins - 1, b))]++;
  }
  return counts;
}

function drawHistogram(dayIndex) {
  const day = state.days[dayIndex - 1];
  $("hist-day").textContent = `day ${day}`;
  $("peak-share").textContent =
    `peak share ${state.peak[dayIndex - 1].toFixed(3)}`;
  const canvas = $("hist-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const bins = 64;
  const day1 = histCounts(state.days[0], bins);
  const today = histCounts(day, bins);
  const max = Math.max(...day1, ...today, 1);
  const bw = canvas.width / bins;
  // day 1 ghost, for the flattening to be visible against
  ctx.fillStyle = css("--muted");
  ctx.globalAlpha = 0.18;
  day1.forEach((c, b) => {
    const h = (c / max) * (canvas.height - 20);
    ctx.fillRect(b * bw + 1, canvas.height - h - 16, bw - 2, h);
  });
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = css("--route-a");
  today.forEach((c, b) => {
    const h = (c / max) * (canvas.height - 20);
    ctx.fillRect(b * bw + 1, canvas.height - h - 16, bw - 2, h);
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = css("--muted");
  ctx.font = "12px sans-serif";
  for (let m = Math.ceil(state.t0 / 30) * 30; m <= state.t1; m += 30) {
    const x = ((m - state.t0) / (state.t1 - state.t0)) * canvas.width;
    ctx.fillText(clock(m), x - 14, canvas.height - 2);
  }
}

function drawSpark(highlight = -1) {
  const canvas = $("spark-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const n = state.peak.length;
  const lo = Math.min(...state.peak) * 0.9;
  const hi = Math.max(...state.peak) * 1.05;
  const px = (i) => 30 + (i / (n - 1)) * (canvas.width - 60);
  const py = (v) => 10 + (1 - (v - lo) / (hi - lo)) * (canvas.height - 26);
  ctx.strokeStyle = css("--route-a");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  state.peak.forEach((v, i) =>
    i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
  ctx.stroke();
  state.peak.forEach((v, i) => {
    ctx.fillStyle = i === highlight ? css("--ink") : css("--route-a");
    ctx.beginPath();
    ctx.arc(px(i), py(v), i === highlight ? 4 : 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = css("--muted");
  ctx.font = "11px sans-serif";
  ctx.fillText(state.peak[0].toFixed(2), 4, py(state.peak[0]) + 4);
  ctx.fillText(state.peak.at(-1).toFixed(2),
               canvas.width - 28, py(state.peak.at(-1)) + 4);
}

function playHistogram() {
  if (histTimer) { clearInterval(histTimer); histTimer = null; return; }
  let i = 1;
  histTimer = setInterval(() => {
    drawHistogram(i);
    drawSpark(i - 1);
    if (++i > state.days.length) { clearInterval(histTimer); histTimer = null; }
  }, 350);
}

// ---------- pane 3: one traveler -------------------------------------------

function drawTraveler(pid) {
  const rows = state.byPid.get(pid) || [];
  const tbody = $("traveler-table").querySelector("tbody");
  tbody.innerHTML = "";
  let switches = 0;
  rows.forEach((r, i) => {
    const prev = rows[i - 1];
    let delta = "—";
    let cls = "";
    if (prev) {
      const parts = [];
      if (r.route !== prev.route) { parts.push(`switched to ${r.route}`); cls = "switch"; switches++; }
      const shift = Math.round(r.dep_min - prev.dep_min);
      if (shift) parts.push(`${shift > 0 ? "+" : ""}${shift} min`);
      delta = parts.join(", ") || "same plan";
    }
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${r.day}</td><td>${r.route}</td><td>${clock(r.dep_min)}</td>` +
      `<td>${Math.round(r.travel_min)} min</td><td>${clock(r.arrive_min)}</td>` +
      `<td class="${cls}">${delta}</td>`;
    tbody.appendChild(tr);
  });
  $("traveler-summary").textContent =
    `${switches} route switch${switches === 1 ? "" : "es"} over ${rows.length} days`;

  const canvas = $("traveler-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "11px sans-serif";
  const n = rows.length;
  rows.forEach((r, i) => {
    const x = 30 + (i / Math.max(n - 1, 1)) * (canvas.width - 60);
    const y = 15 + ((r.dep_min - state.t0) / (state.t1 - state.t0)) * (canvas.height - 40);
    ctx.fillStyle = css(r.route === "A" ? "--route-a" : "--route-b");
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    if (i) {
      ctx.strokeStyle = ctx.fillStyle;
      ctx.globalAlpha = 0.4;
      const xp = 30 + ((i - 1) / Math.max(n - 1, 1)) * (canvas.width - 60);
      const yp = 15 + ((rows[i - 1].dep_min - state.t0) / (state.t1 - state.t0)) * (canvas.height - 40);
      ctx.beginPath(); ctx.moveTo(xp, yp); ctx.lineTo(x, y); ctx.stroke();
      ctx.globalAlpha = 1;
    }
  });
  ctx.fillStyle = css("--muted");
  ctx.fillText("departure time, day by day (down = later)", 30, canvas.height - 4);
}
