/* GLASSBOX-Mobility M4 viewer — final craft pass.
   System-side trajectory log only (pid, day, route, dep_min, travel_min,
   arrive_min); no truth-side data ever reaches this page.
   One day slider drives every pane; arrows scrub, space replays. */

"use strict";

const FREE_FLOW_A = 18;

const state = {
  rows: [], days: [], byPid: new Map(), byDay: new Map(),
  peak: [], t0: 0, t1: 0, day: 1, dotHits: [],
};

const $ = (id) => document.getElementById(id);
const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clock = (m) =>
  `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.round(m % 60)).padStart(2, "0")}`;
const routeColor = (r) => css(r === "A" ? "--route-a" : "--route-b");
const delayOf = (r) =>
  r.route === "A" ? Math.max(0, r.travel_min - FREE_FLOW_A) : 0;

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
  setDay(1);
  drawTraveler($("traveler-select").value);
  $("main").hidden = false;
  $("stats").hidden = false;
  $("status").textContent =
    `${state.byPid.size} travelers · ${state.days.length} days loaded`;
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
  // percentile clamp: the early-ratchet tail must not stretch the axis
  const deps = state.rows.map((r) => r.dep_min).sort((a, b) => a - b);
  const arrs = state.rows.map((r) => r.arrive_min).sort((a, b) => a - b);
  state.t0 = deps[Math.floor(deps.length * 0.005)] - 4;
  state.t1 = arrs[Math.ceil(arrs.length * 0.995) - 1] + 4;
  state.peak = state.days.map((d) => peakInfo(state.byDay.get(d)).share);

  const first = state.byDay.get(state.days[0]);
  const last = state.byDay.get(state.days.at(-1));
  const mean = (rows) =>
    rows.reduce((s, r) => s + r.travel_min, 0) / rows.length;
  $("stat-peak").textContent =
    `${state.peak[0].toFixed(2)} → ${state.peak.at(-1).toFixed(2)}`;
  $("stat-b").textContent =
    `${Math.round(100 * last.filter((r) => r.route === "B").length / last.length)}%`;
  $("stat-travel").textContent =
    `${mean(first).toFixed(0)} → ${mean(last).toFixed(0)} min`;
}

function peakInfo(rows) {
  const arr = rows.map((r) => r.arrive_min).sort((a, b) => a - b);
  let best = 0, start = arr[0];
  for (let i = 0; i < arr.length; i++) {
    let j = i;
    while (j < arr.length && arr[j] < arr[i] + 15) j++;
    if (j - i > best) { best = j - i; start = arr[i]; }
  }
  return { share: best / arr.length, windowStart: start };
}

// ---------- shared day control ---------------------------------------------

let dayAnim = null;

function wire() {
  const slider = $("day-slider");
  slider.max = state.days.length;
  slider.oninput = () => setDay(Number(slider.value));
  $("play").onclick = () => replayDay();
  $("ghost-toggle").onchange = () => drawDayStatic();

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight") setDay(Math.min(state.day + 1, state.days.length));
    else if (e.key === "ArrowLeft") setDay(Math.max(state.day - 1, 1));
    else if (e.key === " ") { e.preventDefault(); replayDay(); }
  });

  const sel = $("traveler-select");
  sel.innerHTML = "";
  for (const pid of [...state.byPid.keys()].sort()) {
    sel.appendChild(new Option(pid, pid));
  }
  sel.onchange = () => drawTraveler(sel.value);
  buildNotables();

  $("day-canvas").addEventListener("click", (e) => {
    const rect = e.target.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * e.target.width;
    const y = ((e.clientY - rect.top) / rect.height) * e.target.height;
    let bestPid = null, bestD = 144;
    for (const h of state.dotHits) {
      const d = (h.x - x) ** 2 + (h.y - y) ** 2;
      if (d < bestD) { bestD = d; bestPid = h.pid; }
    }
    if (bestPid) {
      $("traveler-select").value = bestPid;
      drawTraveler(bestPid);
      $("inspector-pane").scrollIntoView({ behavior: "smooth" });
    }
  });
}

function setDay(dayIndex) {
  stopDayAnim();
  state.day = dayIndex;
  $("day-slider").value = dayIndex;
  $("day-label").textContent = `day ${state.days[dayIndex - 1]}`;
  $("clock").textContent = "—";
  drawDayStatic();
  drawHistogram();
  drawSpark(dayIndex - 1);
}

// ---------- pane 1: one morning, replayed ----------------------------------

const laneY = { A: 120, B: 260 };
const xOf = (m, w) => 72 + ((m - state.t0) / (state.t1 - state.t0)) * (w - 104);

function drawLanes(ctx, w, h) {
  ctx.font = "13px sans-serif";
  // route B: a steady band
  ctx.fillStyle = css("--faint");
  ctx.fillText("route B — ring road", 8, laneY.B - 28);
  ctx.strokeStyle = css("--edge");
  ctx.lineWidth = 1;
  for (const dy of [-7, 7]) {
    ctx.beginPath();
    ctx.moveTo(64, laneY.B + dy);
    ctx.lineTo(w - 24, laneY.B + dy);
    ctx.stroke();
  }
  // route A: a band that pinches at the bottleneck
  ctx.fillStyle = css("--faint");
  ctx.fillText("route A — the bottleneck", 8, laneY.A - 28);
  const pinchX = 72 + 0.72 * (w - 104);
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(64, laneY.A + s * 12);
    ctx.lineTo(pinchX - 36, laneY.A + s * 12);
    ctx.quadraticCurveTo(pinchX, laneY.A + s * 3, pinchX + 36, laneY.A + s * 12);
    ctx.lineTo(w - 24, laneY.A + s * 12);
    ctx.stroke();
  }
  // time ticks
  ctx.fillStyle = css("--faint");
  for (let m = Math.ceil(state.t0 / 30) * 30; m <= state.t1; m += 30) {
    const x = xOf(m, w);
    ctx.fillText(clock(m), x - 16, h - 10);
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.beginPath(); ctx.moveTo(x, 48); ctx.lineTo(x, h - 32); ctx.stroke();
  }
}

function drawDot(ctx, x, y, r, color, delay) {
  ctx.save();
  if (delay > 2) {
    ctx.shadowColor = color;
    ctx.shadowBlur = Math.min(18, delay * 1.4);
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function ghostDay1(ctx, w) {
  if (!$("ghost-toggle").checked || state.day === 1) return;
  ctx.fillStyle = css("--faint");
  ctx.globalAlpha = 0.35;
  for (const r of state.byDay.get(state.days[0])) {
    ctx.fillRect(xOf(r.arrive_min, w) - 0.5, laneY[r.route] - 16, 1, 8);
  }
  ctx.globalAlpha = 1;
}

function drawDayStatic() {
  const day = state.days[state.day - 1];
  const canvas = $("day-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawLanes(ctx, canvas.width, canvas.height);
  ghostDay1(ctx, canvas.width);
  state.dotHits = [];
  for (const r of state.byDay.get(day)) {
    const delay = delayOf(r);
    const x = xOf(r.arrive_min, canvas.width);
    const y = laneY[r.route];
    ctx.globalAlpha = 0.28;
    ctx.strokeStyle = routeColor(r.route);
    ctx.beginPath();
    ctx.moveTo(xOf(r.dep_min, canvas.width), y);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.globalAlpha = 0.95;
    drawDot(ctx, x, y, 2.2 + Math.min(5, delay * 0.35), routeColor(r.route), delay);
    state.dotHits.push({ x, y, pid: r.pid });
  }
  ctx.globalAlpha = 1;
}

function stopDayAnim() {
  if (dayAnim) cancelAnimationFrame(dayAnim);
  dayAnim = null;
}

function replayDay() {
  stopDayAnim();
  const day = state.days[state.day - 1];
  const rows = state.byDay.get(day);
  const canvas = $("day-canvas");
  const ctx = canvas.getContext("2d");
  const start = performance.now();
  const SPEED = 20; // simulated minutes per real second

  const frame = (now) => {
    const t = state.t0 + ((now - start) / 1000) * SPEED;
    $("clock").textContent = clock(Math.min(t, state.t1));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawLanes(ctx, canvas.width, canvas.height);
    ghostDay1(ctx, canvas.width);
    // time cursor
    const cx = xOf(Math.min(t, state.t1), canvas.width);
    ctx.strokeStyle = "rgba(237,235,229,0.25)";
    ctx.beginPath(); ctx.moveTo(cx, 48); ctx.lineTo(cx, canvas.height - 32);
    ctx.stroke();
    for (const r of rows) {
      if (t < r.dep_min) continue;
      const delay = delayOf(r);
      const done = Math.min(1, (t - r.dep_min) / (r.arrive_min - r.dep_min));
      const x = xOf(r.dep_min + done * (r.arrive_min - r.dep_min), canvas.width);
      const y = laneY[r.route];
      ctx.globalAlpha = 0.22;
      ctx.strokeStyle = routeColor(r.route);
      ctx.beginPath();
      ctx.moveTo(xOf(r.dep_min, canvas.width), y); ctx.lineTo(x, y);
      ctx.stroke();
      ctx.globalAlpha = done < 1 ? 1 : 0.55;
      drawDot(ctx, x, y, done < 1 ? 3 + Math.min(5, delay * 0.35)
                                  : 2.2 + Math.min(5, delay * 0.35),
              routeColor(r.route), done < 1 ? delay : 0);
    }
    ctx.globalAlpha = 1;
    if (t < state.t1) dayAnim = requestAnimationFrame(frame);
    else drawDayStatic();
  };
  dayAnim = requestAnimationFrame(frame);
}

// ---------- pane 2: the peak, spreading ------------------------------------

let peakShown = 0;

function histCounts(day, bins) {
  const counts = new Array(bins).fill(0);
  for (const r of state.byDay.get(day)) {
    const b = Math.floor(((r.dep_min - state.t0) / (state.t1 - state.t0)) * bins);
    if (b >= 0 && b < bins) counts[b]++;
  }
  return counts;
}

function drawHistogram() {
  const day = state.days[state.day - 1];
  const canvas = $("hist-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const bins = 72;
  const day1 = histCounts(state.days[0], bins);
  const today = histCounts(day, bins);
  const max = Math.max(...day1, ...today, 1);
  const bw = canvas.width / bins;
  const hOf = (c) => (c / max) * (canvas.height - 44);

  // the day's busiest 15 arrival minutes, as a translucent band
  const info = peakInfo(state.byDay.get(day));
  const bx0 = ((info.windowStart - state.t0) / (state.t1 - state.t0)) * canvas.width;
  const bx1 = ((info.windowStart + 15 - state.t0) / (state.t1 - state.t0)) * canvas.width;
  ctx.fillStyle = css("--band");
  ctx.fillRect(bx0, 8, bx1 - bx0, canvas.height - 30);

  // today's departures
  ctx.fillStyle = css("--route-a");
  today.forEach((c, b) => {
    if (!c) return;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(b * bw + 1, canvas.height - hOf(c) - 22, Math.max(bw - 2, 1), hOf(c));
  });
  ctx.globalAlpha = 1;

  // day-1 outline, permanently ghosted
  ctx.strokeStyle = css("--muted");
  ctx.globalAlpha = 0.55;
  ctx.beginPath();
  day1.forEach((c, b) => {
    const y = canvas.height - hOf(c) - 22;
    if (b === 0) ctx.moveTo(0, y);
    ctx.lineTo(b * bw, y);
    ctx.lineTo((b + 1) * bw, y);
  });
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.fillStyle = css("--faint");
  ctx.font = "12px sans-serif";
  for (let m = Math.ceil(state.t0 / 30) * 30; m <= state.t1; m += 30) {
    const x = ((m - state.t0) / (state.t1 - state.t0)) * canvas.width;
    ctx.fillText(clock(m), x - 14, canvas.height - 6);
  }
  animatePeakNumeral(state.peak[state.day - 1]);
}

function animatePeakNumeral(target) {
  const el = $("peak-share");
  const from = peakShown || target;
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - start) / 200);
    peakShown = from + (target - from) * p;
    el.textContent = `peak share ${peakShown.toFixed(3)}`;
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function drawSpark(highlight) {
  const canvas = $("spark-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const n = state.peak.length;
  const lo = Math.min(...state.peak) * 0.92;
  const hi = Math.max(...state.peak) * 1.05;
  const px = (i) => 40 + (i / (n - 1)) * (canvas.width - 220);
  const py = (v) => 12 + (1 - (v - lo) / (hi - lo)) * (canvas.height - 40);

  // S1 reference line
  ctx.strokeStyle = css("--faint");
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.moveTo(40, py(state.peak[0]));
  ctx.lineTo(canvas.width - 180, py(state.peak[0]));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = css("--faint");
  ctx.font = "11px sans-serif";
  ctx.fillText(`S1 ${state.peak[0].toFixed(3)}`, canvas.width - 172, py(state.peak[0]) + 4);

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
  // day-2 rebase annotation (exploratory context, Ruling 32 footnote)
  if (n > 2) {
    const drop = 100 * (state.peak.at(-1) - state.peak[1]) / state.peak[1];
    ctx.fillStyle = css("--faint");
    ctx.fillText(`${drop.toFixed(0)}% from day 2 (exploratory)`,
                 px(1) + 8, py(state.peak[1]) - 8);
  }
  ctx.fillStyle = css("--muted");
  ctx.fillText("day 1", 40, canvas.height - 4);
  ctx.fillText("day 20", canvas.width - 220, canvas.height - 4);
}

// ---------- pane 3: one traveler -------------------------------------------

function buildNotables() {
  const chips = [];
  for (const [pid, rows] of state.byPid) {
    const earliest = Math.min(...rows.map((r) => r.dep_min));
    const switches = rows.filter((r, i) =>
      i && r.route !== rows[i - 1].route).length;
    chips.push({ pid, earliest, switches });
  }
  // the extreme tail: travelers who ratcheted to pre-06:30 departures
  const ratchet = chips.filter((c) => c.earliest < 6.5 * 60)
    .sort((a, b) => a.earliest - b.earliest);
  const switchy = chips.sort((a, b) => b.switches - a.switches).slice(0, 3);
  const host = $("notable-chips");
  host.innerHTML = "";
  const add = (pid, label) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = label;
    b.onclick = () => {
      $("traveler-select").value = pid;
      drawTraveler(pid);
      [...host.children].forEach((c) => c.classList.toggle("active", c === b));
    };
    host.appendChild(b);
  };
  for (const c of ratchet) add(c.pid, `${c.pid} · early ratchet`);
  for (const c of switchy) add(c.pid, `${c.pid} · ${c.switches} switches`);
}

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
      if (r.route !== prev.route) {
        parts.push(`switched to ${r.route}`); cls = "switch"; switches++;
      }
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
    `${switches} route switch${switches === 1 ? "" : "es"} · ` +
    `${Math.round(rows.at(-1).dep_min - rows[0].dep_min)} min drift over ` +
    `${rows.length} days`;

  const canvas = $("traveler-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "11px sans-serif";
  const n = rows.length;
  const X = (i) => 40 + (i / Math.max(n - 1, 1)) * (canvas.width - 80);
  // departure line (top two thirds)
  const dLo = Math.min(...rows.map((r) => r.dep_min));
  const dHi = Math.max(...rows.map((r) => r.dep_min)) + 1;
  const Y = (v) => 14 + ((v - dLo) / (dHi - dLo)) * (canvas.height - 84);
  rows.forEach((r, i) => {
    if (i) {
      ctx.strokeStyle = routeColor(r.route);
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.moveTo(X(i - 1), Y(rows[i - 1].dep_min));
      ctx.lineTo(X(i), Y(r.dep_min));
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = routeColor(r.route);
    ctx.beginPath();
    ctx.arc(X(i), Y(r.dep_min), 3, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = css("--faint");
  ctx.fillText(`departures ${clock(dLo)} – ${clock(dHi)}`, 40, 10);
  // queue-delay sparkbars (bottom strip)
  const base = canvas.height - 16;
  rows.forEach((r, i) => {
    const d = delayOf(r);
    ctx.fillStyle = css("--route-a");
    ctx.globalAlpha = 0.8;
    ctx.fillRect(X(i) - 2, base - d * 2.5, 4, Math.max(d * 2.5, 1));
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = css("--faint");
  ctx.fillText("queue delay, day by day", 40, canvas.height - 2);
}
