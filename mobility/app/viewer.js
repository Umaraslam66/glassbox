/* GLASSBOX-Mobility viewer V2 — the living corridor.

   Every moving element is interpolated from the recorded system-side log
   {pid, day, route, dep_min, travel_min, arrive_min} and nothing else:
   a traveler departs at dep_min, spends (travel_min - 18) of route A's
   journey queued at the bridge (their own recorded delay), and arrives at
   arrive_min. Queue length, glow, arrival pooling, captions and story
   beats are all computed from those five recorded fields. */

"use strict";

const FF_A = 18, T_B = 26;
const W = 1600, H = 900;
const S_PINCH = 0.60;          // path fraction of route A at the bridge
const QRATE = 0.02;            // queue extent per minute of recorded delay

const state = {
  rows: [], days: [], byPid: new Map(), byDay: new Map(),
  peak: [], triedB: [], bShare: [], maxDelay: [],
  day: 1, simT: 0, playing: false, mode: "free", beat: 0,
  spotlight: null, share: false, recording: false,
};

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const clock = (m) =>
  `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.round(m % 60)).padStart(2, "0")}`;
const lerp = (a, b, t) => a + (b - a) * t;

// ---------------------------------------------------------------- loading

$("file").addEventListener("change", (ev) => loadFile(ev.target.files[0]));
document.body.addEventListener("dragover", (e) => {
  e.preventDefault(); document.body.classList.add("dragging");
});
document.body.addEventListener("dragleave", () =>
  document.body.classList.remove("dragging"));
document.body.addEventListener("drop", (e) => {
  e.preventDefault(); document.body.classList.remove("dragging");
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

async function loadFile(file) {
  if (!file) return;
  state.rows = (await file.text()).split("\n").filter(Boolean).map(JSON.parse);
  indexData();
  paintMap();
  wire();
  setDay(1, { autoplay: false });
  buildNumbers();
  $("main").hidden = false;
  $("stats").hidden = false;
  $("status").textContent =
    `${state.byPid.size} travelers · ${state.days.length} days loaded`;
}

function indexData() {
  state.days = [...new Set(state.rows.map((r) => r.day))].sort((a, b) => a - b);
  state.byPid = new Map(); state.byDay = new Map();
  for (const r of state.rows) {
    if (!state.byPid.has(r.pid)) state.byPid.set(r.pid, []);
    state.byPid.get(r.pid).push(r);
    if (!state.byDay.has(r.day)) state.byDay.set(r.day, []);
    state.byDay.get(r.day).push(r);
  }
  for (const v of state.byPid.values()) v.sort((a, b) => a.day - b.day);

  state.peak = state.days.map((d) => peakInfo(state.byDay.get(d)).share);
  const tried = new Set();
  state.triedB = []; state.bShare = []; state.maxDelay = [];
  for (const d of state.days) {
    const rows = state.byDay.get(d);
    for (const r of rows) if (r.route === "B") tried.add(r.pid);
    state.triedB.push(tried.size);
    state.bShare.push(rows.filter((r) => r.route === "B").length / rows.length);
    state.maxDelay.push(Math.max(...rows.map((r) =>
      r.route === "A" ? r.travel_min - FF_A : 0)));
  }
  const first = state.byDay.get(state.days[0]);
  const last = state.byDay.get(state.days.at(-1));
  const mean = (rows) => rows.reduce((s, r) => s + r.travel_min, 0) / rows.length;
  $("stat-peak").textContent =
    `${state.peak[0].toFixed(2)} → ${state.peak.at(-1).toFixed(2)}`;
  $("stat-b").textContent = `${Math.round(100 * state.bShare.at(-1))}%`;
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

function dayDomain(dayIndex) {
  const rows = state.byDay.get(state.days[dayIndex - 1]);
  return [Math.min(...rows.map((r) => r.dep_min)) - 6,
          Math.max(...rows.map((r) => r.arrive_min)) + 6];
}

// ------------------------------------------------------------ map geometry

function buildPath(points) {
  // dense resample of a polyline smoothed by quadratic midpoint curves
  const pts = [];
  const q = (p0, p1, p2, t) => ({
    x: lerp(lerp(p0.x, p1.x, t), lerp(p1.x, p2.x, t), t),
    y: lerp(lerp(p0.y, p1.y, t), lerp(p1.y, p2.y, t), t),
  });
  const mids = points.map((p, i) =>
    i < points.length - 1
      ? { x: (p.x + points[i + 1].x) / 2, y: (p.y + points[i + 1].y) / 2 }
      : p);
  pts.push(points[0]);
  for (let i = 1; i < points.length - 1; i++) {
    for (let t = 0.05; t <= 1.0001; t += 0.05) {
      pts.push(q(mids[i - 1], points[i], mids[i], t));
    }
  }
  pts.push(points.at(-1));
  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x,
                                     pts[i].y - pts[i - 1].y));
  }
  const total = cum.at(-1);
  return {
    pts, cum, total,
    at(s) {
      const target = Math.max(0, Math.min(1, s)) * total;
      let lo = 0, hi = cum.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cum[mid] < target) lo = mid + 1; else hi = mid;
      }
      const i = Math.max(1, lo);
      const t = (target - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
      return { x: lerp(pts[i - 1].x, pts[i].x, t),
               y: lerp(pts[i - 1].y, pts[i].y, t) };
    },
  };
}

const ROUTE_A = buildPath([
  { x: 195, y: 470 }, { x: 360, y: 445 }, { x: 560, y: 415 },
  { x: 760, y: 395 }, { x: 900, y: 388 }, { x: 1000, y: 388 },
  { x: 1140, y: 398 }, { x: 1290, y: 420 }, { x: 1410, y: 445 },
]);
const ROUTE_B = buildPath([
  { x: 200, y: 495 }, { x: 330, y: 590 }, { x: 520, y: 690 },
  { x: 780, y: 745 }, { x: 1040, y: 700 }, { x: 1240, y: 590 },
  { x: 1400, y: 480 },
]);
const ORIGIN = { x: 150, y: 470 };
const DEST = { x: 1455, y: 430 };

// ------------------------------------------------------------- static map

function hash(i) { return ((i * 2654435761) >>> 8) % 1000 / 1000; }

function buildings(ctx, cx, cy, n, spread, seed) {
  for (let i = 0; i < n; i++) {
    const bx = cx + (hash(seed + i) - 0.5) * spread;
    const by = cy + (hash(seed + i + 50) - 0.5) * spread * 0.62;
    const bw = 22 + hash(seed + i + 100) * 40;
    const bh = 16 + hash(seed + i + 150) * 34;
    ctx.fillStyle = `rgba(${26 + hash(seed + i + 200) * 10}, ${27 +
      hash(seed + i + 250) * 10}, ${36 + hash(seed + i + 300) * 12}, 1)`;
    ctx.beginPath();
    ctx.roundRect(bx - bw / 2, by - bh / 2, bw, bh, 3);
    ctx.fill();
    // a few lit windows
    for (let wgt = 0; wgt < 3; wgt++) {
      if (hash(seed + i * 7 + wgt) > 0.45) continue;
      ctx.fillStyle = hash(seed + i * 11 + wgt) > 0.5
        ? "rgba(232, 179, 102, 0.5)" : "rgba(160, 190, 210, 0.35)";
      ctx.fillRect(bx - bw / 2 + 4 + wgt * 7, by - 3, 3, 4);
    }
  }
}

function roadStroke(ctx, path, width, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  path.pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.stroke();
}

function paintMap() {
  const ctx = $("map-canvas").getContext("2d");
  // ground
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, "#0a0c15"); g.addColorStop(0.6, "#0b0d16");
  g.addColorStop(1, "#0d0e14");
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  // the river under the bridge
  const river = ctx.createLinearGradient(880, 0, 1020, 0);
  river.addColorStop(0, "rgba(20, 34, 48, 0)");
  river.addColorStop(0.5, "rgba(24, 44, 62, 0.85)");
  river.addColorStop(1, "rgba(20, 34, 48, 0)");
  ctx.fillStyle = river;
  ctx.save();
  ctx.translate(950, 450); ctx.rotate(0.08); ctx.fillRect(-90, -460, 180, 920);
  ctx.restore();
  // districts
  buildings(ctx, 128, 452, 16, 220, 3);
  buildings(ctx, 1462, 420, 20, 240, 77);
  // roads: casing then surface; B narrower
  roadStroke(ctx, ROUTE_B, 20, "#141720");
  roadStroke(ctx, ROUTE_B, 13, "#1a1e2a");
  roadStroke(ctx, ROUTE_A, 26, "#161821");
  roadStroke(ctx, ROUTE_A, 17, "#1e2029");
  // the pinch: converging guard walls narrowing route A at the bridge
  const p = ROUTE_A.at(S_PINCH);
  ctx.fillStyle = "#2a2c38";
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(p.x - 96, p.y + s * 26);
    ctx.lineTo(p.x - 12, p.y + s * 9);
    ctx.lineTo(p.x + 46, p.y + s * 9);
    ctx.lineTo(p.x + 96, p.y + s * 24);
    ctx.lineTo(p.x + 96, p.y + s * 30);
    ctx.lineTo(p.x - 96, p.y + s * 32);
    ctx.closePath();
    ctx.fill();
  }
  ctx.strokeStyle = "rgba(232, 179, 102, 0.25)";
  ctx.lineWidth = 1;
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(p.x - 90, p.y + s * 25);
    ctx.lineTo(p.x - 12, p.y + s * 8.5);
    ctx.lineTo(p.x + 46, p.y + s * 8.5);
    ctx.lineTo(p.x + 90, p.y + s * 23);
    ctx.stroke();
  }
  // labels
  ctx.font = "600 15px 'Avenir Next', sans-serif";
  ctx.fillStyle = "rgba(142, 140, 152, 0.85)";
  ctx.fillText("route A · the bottleneck", p.x - 78, p.y - 44);
  const bMid = ROUTE_B.at(0.52);
  ctx.fillText("route B · ring road", bMid.x - 60, bMid.y + 38);
  ctx.font = "500 13px 'Avenir Next', sans-serif";
  ctx.fillStyle = "rgba(86, 84, 96, 1)";
  ctx.fillText("origin", 100, 560);
  ctx.fillText("the district", 1400, 528);
}

// -------------------------------------------------------------- kinematics

function positionAt(r, t) {
  // returns {s, path, queued} or null (not on the road at t)
  const elapsed = t - r.dep_min;
  if (elapsed < 0 || elapsed >= r.travel_min) return null;
  if (r.route === "B") {
    return { s: elapsed / r.travel_min, path: ROUTE_B, queued: false };
  }
  const d = Math.max(0, r.travel_min - FF_A);
  const sJoin = Math.max(0.1, S_PINCH - d * QRATE);
  const t1 = sJoin * FF_A;
  const t3 = (1 - S_PINCH) * FF_A;
  const t2 = r.travel_min - t1 - t3;
  if (elapsed < t1) {
    return { s: (elapsed / t1) * sJoin, path: ROUTE_A, queued: false };
  }
  if (elapsed < t1 + t2) {
    const f = (elapsed - t1) / t2;
    return { s: sJoin + f * (S_PINCH - sJoin), path: ROUTE_A, queued: d > 1 };
  }
  return { s: S_PINCH + ((elapsed - t1 - t2) / t3) * (1 - S_PINCH),
           path: ROUTE_A, queued: false };
}

// ---------------------------------------------------------- the animation

let rafId = null, lastNow = 0;
const animCtx = () => $("anim-canvas").getContext("2d");

// pre-rendered glow sprites: drawImage is cheap where shadowBlur is not
function makeSprite(color, core) {
  const c = Object.assign(document.createElement("canvas"),
                          { width: 64, height: 64 });
  const g = c.getContext("2d").createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, color);
  g.addColorStop(core, color);
  g.addColorStop(1, "rgba(0,0,0,0)");
  const ctx = c.getContext("2d");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  return c;
}
let SPRITES = null;
function sprites() {
  if (!SPRITES) {
    SPRITES = {
      A: makeSprite("rgba(232, 179, 102, 0.9)", 0.16),
      B: makeSprite("rgba(111, 192, 216, 0.9)", 0.16),
      pool: makeSprite("rgba(238, 214, 170, 0.5)", 0.05),
    };
  }
  return SPRITES;
}

function dawnLevel(t) {
  return Math.max(0, Math.min(1, (t - 6.5 * 60) / (3 * 60)));
}

function drawFrame(trails) {
  const day = state.days[state.day - 1];
  const rows = state.byDay.get(day);
  const ctx = animCtx();
  const spr = sprites();
  if (trails) {
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0, 0, 0, 0.30)";
    ctx.fillRect(0, 0, W, H);
  } else {
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, W, H);
  }
  ctx.globalCompositeOperation = "lighter";
  $("dawn").style.opacity = (0.45 * dawnLevel(state.simT)).toFixed(3);
  // arrival pool: soft light over the destination district
  const arrived = rows.filter((r) => state.simT >= r.arrive_min).length;
  if (arrived) {
    ctx.globalAlpha = 0.10 + 0.5 * (arrived / rows.length);
    ctx.drawImage(spr.pool, DEST.x - 150, DEST.y - 150, 300, 300);
  }
  // vehicles
  const spot = state.spotlight;
  let spotXY = null;
  for (const r of rows) {
    const pos = positionAt(r, state.simT);
    if (!pos) continue;
    const isSpot = spot && r.pid === spot;
    const dim = spot && !isSpot;
    const d = r.route === "A" ? Math.max(0, r.travel_min - FF_A) : 0;
    const { x, y } = pos.path.at(pos.s);
    // light physics: glow radius grows with that traveler's recorded delay
    const rad = (7 + Math.min(15, d * 1.1)) * (isSpot ? 1.6 : 1)
      * (pos.queued ? 1.2 : 1);
    ctx.globalAlpha = dim ? 0.12 : pos.queued ? 0.72 : 0.58;
    ctx.drawImage(spr[r.route], x - rad, y - rad, rad * 2, rad * 2);
    if (isSpot) spotXY = { x, y, pid: r.pid };
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  if (spotXY) {
    ctx.strokeStyle = css("--ink");
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(spotXY.x, spotXY.y, 13, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = css("--ink");
    ctx.font = "600 15px 'Avenir Next', sans-serif";
    ctx.fillText(spotXY.pid, spotXY.x + 18, spotXY.y - 12);
  }
  $("clock").textContent = clock(state.simT);
  if (state.recording) blitCapture();
}

function loop(now) {
  const [d0, d1] = dayDomain(state.day);
  const dur = state.mode === "mornings" ? 2.2 : 14;
  const rate = (d1 - d0) / dur;               // sim-minutes per real second
  // clamp survives throttled/occluded windows without giant jumps
  const dt = Math.min(0.5, (now - lastNow) / 1000);
  lastNow = now;
  state.simT += rate * dt;
  if (state.simT >= d1) {
    if (state.mode === "mornings" && state.day < state.days.length) {
      setDay(state.day + 1, { keepMode: true, autoplay: true });
    } else {
      state.playing = false;
      state.mode = state.mode === "mornings" ? "free" : state.mode;
      $("play").textContent = "▶";
      drawFrame(false);
      return;
    }
  }
  syncScrub();
  drawFrame(true);
  if (state.playing) rafId = requestAnimationFrame(loop);
}

function play() {
  if (state.playing) { pause(); return; }
  const [d0, d1] = dayDomain(state.day);
  if (state.simT >= d1 - 1) state.simT = d0;
  state.playing = true;
  $("play").textContent = "❙❙";
  lastNow = performance.now();
  rafId = requestAnimationFrame(loop);
}

function pause() {
  state.playing = false;
  if (rafId) cancelAnimationFrame(rafId);
  $("play").textContent = "▶";
  drawFrame(false);
}

function syncScrub() {
  const [d0, d1] = dayDomain(state.day);
  $("time-scrub").value = Math.round(1000 * (state.simT - d0) / (d1 - d0));
}

function setDay(dayIndex, opts = {}) {
  if (!opts.keepMode) state.mode = opts.mode || "free";
  state.day = dayIndex;
  const [d0] = dayDomain(dayIndex);
  state.simT = d0;
  $("day-slider").value = dayIndex;
  $("day-chip").textContent = `day ${state.days[dayIndex - 1]}`;
  updateCaption();
  syncScrub();
  updateNumbers();
  if (opts.autoplay) {
    state.playing = true;
    $("play").textContent = "❙❙";
    lastNow = performance.now();
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(loop);
  } else {
    // land the scrub mid-morning so the static frame isn't empty
    const [a, b] = dayDomain(dayIndex);
    state.simT = lerp(a, b, 0.55);
    syncScrub();
    drawFrame(false);
  }
}

function updateCaption(custom) {
  const i = state.day - 1;
  if (custom !== undefined) { $("caption").innerHTML = custom; return; }
  if (state.share) {
    $("caption").innerHTML =
      `peak share ${state.peak[i].toFixed(2)} · ` +
      `${Math.round(100 * state.bShare[i])}% on the ring road · ` +
      `worst wait ${Math.round(state.maxDelay[i])} min`;
    return;
  }
  $("caption").innerHTML =
    `day ${state.days[i]} <span class="dim">· ${state.triedB[i]} travelers ` +
    `have tried the ring road · worst wait ` +
    `${Math.round(state.maxDelay[i])} min · peak share ` +
    `${state.peak[i].toFixed(2)}</span>`;
}

// ------------------------------------------------------------- story mode

const BEATS = [
  { day: 1, cap: () =>
      "Day one. Three hundred commuters, one bridge. The queue burns amber." },
  { day: 3, cap: () =>
      `Within days the ring road is discovered — six minutes longer, never ` +
      `jammed. ${state.triedB[2]} travelers have tried it.` },
  { day: 12, spotlight: "t0287", cap: () => {
      const t = state.byPid.get("t0287");
      const today = t.find((r) => r.day === 12);
      const last = t.at(-1);
      return `A few cannot stop leaving earlier. t0287 departs ` +
        `${clock(today.dep_min)} today; by day twenty, ${clock(last.dep_min)}.`;
    } },
  { day: 20, cap: () =>
      `Day twenty. ${Math.round(100 * state.bShare.at(-1))}% ride the ring ` +
      `road; the peak has spread; the bridge breathes.` },
];

function enterBeat(i) {
  state.beat = i;
  const b = BEATS[i];
  state.spotlight = b.spotlight || null;
  setDay(b.day, { mode: "story", autoplay: true, keepMode: false });
  state.mode = "story";
  updateCaption(`${b.cap()} <span class="dim">(${i + 1}/${BEATS.length})</span>`);
}

// ---------------------------------------------------------------- capture

function compositeTo(ctx) {
  ctx.drawImage($("map-canvas"), 0, 0);
  ctx.drawImage($("anim-canvas"), 0, 0);
  ctx.font = "600 34px 'Avenir Next', sans-serif";
  ctx.fillStyle = css("--ink");
  ctx.textAlign = "right";
  ctx.fillText(clock(state.simT), W - 32, 56);
  ctx.textAlign = "left";
  if (state.share) {
    ctx.font = "700 40px 'Avenir Next', sans-serif";
    ctx.fillText("Twenty mornings", 40, 64);
    ctx.font = "400 17px 'Avenir Next', sans-serif";
    ctx.fillStyle = css("--muted");
    ctx.fillText("300 LLM travelers learn a congested corridor · " +
                 "GLASSBOX-Mobility", 40, 92);
  } else {
    ctx.font = "500 15px 'Avenir Next', sans-serif";
    ctx.fillStyle = css("--ink");
    ctx.fillText($("day-chip").textContent, 34, 46);
  }
  ctx.font = "400 16px 'Avenir Next', sans-serif";
  ctx.fillStyle = css("--ink");
  ctx.fillText($("caption").textContent, 32, H - 28);
}

function savePng() {
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  compositeTo(c.getContext("2d"));
  c.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `glassbox_mobility_day${state.days[state.day - 1]}.png`;
    a.click();
  });
}

let captureCanvas = null, recorder = null;

function blitCapture() {
  const ctx = captureCanvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  compositeTo(ctx);
}

function recordWebm() {
  if (state.recording || !("MediaRecorder" in window)) {
    $("capture-note").textContent =
      "MediaRecorder" in window ? "already recording" : "capture unsupported";
    return;
  }
  captureCanvas = captureCanvas || Object.assign(
    document.createElement("canvas"), { width: W, height: H });
  const stream = captureCanvas.captureStream(60);
  recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  const chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
    a.download = `glassbox_mobility_day${state.days[state.day - 1]}.webm`;
    a.click();
    state.recording = false;
    $("capture-note").textContent = "webm saved";
  };
  state.recording = true;
  $("capture-note").textContent = "recording 10 s…";
  recorder.start();
  if (!state.playing) play();
  setTimeout(() => recorder.state !== "inactive" && recorder.stop(), 10000);
}

// ---------------------------------------------------------------- wiring

function wire() {
  $("play").onclick = play;
  $("day-slider").max = state.days.length;
  $("day-slider").oninput = (e) => setDay(Number(e.target.value));
  $("time-scrub").oninput = (e) => {
    const [d0, d1] = dayDomain(state.day);
    state.simT = lerp(d0, d1, Number(e.target.value) / 1000);
    if (!state.playing) drawFrame(false);
  };
  $("mornings").onclick = () => {
    state.mode = "mornings"; state.spotlight = null;
    setDay(1, { keepMode: true, autoplay: true });
  };
  $("story").onclick = () => {
    const on = state.mode !== "story";
    $("story").classList.toggle("active", on);
    $("story-next").hidden = !on;
    if (on) enterBeat(0);
    else { state.spotlight = null; state.mode = "free"; updateCaption(); }
  };
  $("story-next").onclick = () => enterBeat((state.beat + 1) % BEATS.length);
  $("share").onclick = () => {
    state.share = !state.share;
    document.body.classList.toggle("share", state.share);
    $("share-title").hidden = !state.share;
    $("share").classList.toggle("active", state.share);
    updateCaption();
    drawFrame(false);
  };
  document.addEventListener("keydown", (e) => {
    if (["SELECT", "INPUT"].includes(e.target.tagName)) return;
    if (e.key === " ") { e.preventDefault(); play(); }
    else if (e.key === "ArrowRight") setDay(Math.min(state.day + 1, state.days.length));
    else if (e.key === "ArrowLeft") setDay(Math.max(state.day - 1, 1));
    else if (e.key === "p") savePng();
    else if (e.key === "r") recordWebm();
  });
}

// ------------------------------------------------- the numbers (below fold)

function buildNumbers() {
  const sel = $("traveler-select");
  sel.innerHTML = "";
  for (const pid of [...state.byPid.keys()].sort()) {
    sel.appendChild(new Option(pid, pid));
  }
  sel.onchange = () => drawTable(sel.value);
  const chips = [];
  for (const [pid, rows] of state.byPid) {
    const earliest = Math.min(...rows.map((r) => r.dep_min));
    const switches = rows.filter((r, i) =>
      i && r.route !== rows[i - 1].route).length;
    chips.push({ pid, earliest, switches });
  }
  const host = $("notable-chips");
  host.innerHTML = "";
  const add = (pid, label) => {
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = label;
    b.onclick = () => {
      sel.value = pid; drawTable(pid);
      [...host.children].forEach((c) => c.classList.toggle("active", c === b));
    };
    host.appendChild(b);
  };
  for (const c of chips.filter((c) => c.earliest < 6.5 * 60)
      .sort((a, b) => a.earliest - b.earliest)) {
    add(c.pid, `${c.pid} · early ratchet`);
  }
  for (const c of chips.sort((a, b) => b.switches - a.switches).slice(0, 3)) {
    add(c.pid, `${c.pid} · ${c.switches} switches`);
  }
  drawTable(sel.value);
  updateNumbers();
}

function updateNumbers() {
  if (!state.rows.length || !$("hist-canvas")) return;
  drawHistogram();
  drawSpark(state.day - 1);
}

function drawHistogram() {
  const day = state.days[state.day - 1];
  $("peak-share").textContent =
    `peak share ${state.peak[state.day - 1].toFixed(3)}`;
  const canvas = $("hist-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const deps = state.rows.map((r) => r.dep_min).sort((a, b) => a - b);
  const t0 = deps[Math.floor(deps.length * 0.005)] - 4;
  const t1 = Math.max(...state.rows.map((r) => r.arrive_min)) + 4;
  const bins = 72, bw = canvas.width / bins;
  const count = (d) => {
    const c = new Array(bins).fill(0);
    for (const r of state.byDay.get(d)) {
      const b = Math.floor(((r.dep_min - t0) / (t1 - t0)) * bins);
      if (b >= 0 && b < bins) c[b]++;
    }
    return c;
  };
  const day1 = count(state.days[0]);
  const today = count(day);
  const max = Math.max(...day1, ...today, 1);
  const hOf = (c) => (c / max) * (canvas.height - 40);
  ctx.fillStyle = css("--route-a");
  ctx.globalAlpha = 0.85;
  today.forEach((c, b) => c && ctx.fillRect(
    b * bw + 1, canvas.height - hOf(c) - 20, Math.max(bw - 2, 1), hOf(c)));
  ctx.globalAlpha = 0.5;
  ctx.strokeStyle = css("--muted");
  ctx.beginPath();
  day1.forEach((c, b) => {
    const y = canvas.height - hOf(c) - 20;
    if (!b) ctx.moveTo(0, y);
    ctx.lineTo(b * bw, y); ctx.lineTo((b + 1) * bw, y);
  });
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = css("--faint");
  ctx.font = "12px sans-serif";
  for (let m = Math.ceil(t0 / 30) * 30; m <= t1; m += 30) {
    ctx.fillText(clock(m), ((m - t0) / (t1 - t0)) * canvas.width - 14,
                 canvas.height - 4);
  }
}

function drawSpark(highlight) {
  const canvas = $("spark-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const n = state.peak.length;
  const lo = Math.min(...state.peak) * 0.92;
  const hi = Math.max(...state.peak) * 1.05;
  const px = (i) => 40 + (i / (n - 1)) * (canvas.width - 240);
  const py = (v) => 10 + (1 - (v - lo) / (hi - lo)) * (canvas.height - 34);
  ctx.strokeStyle = css("--faint");
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.moveTo(40, py(state.peak[0]));
  ctx.lineTo(canvas.width - 200, py(state.peak[0]));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = css("--faint");
  ctx.fillText(`S1 ${state.peak[0].toFixed(3)}`,
               canvas.width - 192, py(state.peak[0]) + 4);
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
  if (n > 2) {
    const drop = 100 * (state.peak.at(-1) - state.peak[1]) / state.peak[1];
    ctx.fillStyle = css("--faint");
    ctx.fillText(`${drop.toFixed(0)}% from day 2 (exploratory)`,
                 px(1) + 8, py(state.peak[1]) - 6);
  }
}

function drawTable(pid) {
  const rows = state.byPid.get(pid) || [];
  const tbody = $("traveler-table").querySelector("tbody");
  tbody.innerHTML = "";
  let switches = 0;
  rows.forEach((r, i) => {
    const prev = rows[i - 1];
    let delta = "—", cls = "";
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
    `${Math.round(rows.at(-1).dep_min - rows[0].dep_min)} min drift`;
}
