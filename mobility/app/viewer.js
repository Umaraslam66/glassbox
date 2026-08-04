/* viewer.js — GLASSBOX-Mobility viewer V3, the app shell.
 *
 * town.js does all the drawing and all the arithmetic. This file only:
 *   · loads the three published data files (fetch first, bundle fallback)
 *   · runs the rAF loop, the scrubbers, play/pause, speed, keyboard
 *   · runs the guided story
 *   · opens the traveler dossier on a click
 *   · draws the numbers section below the fold
 *
 * No numbers are written down in here either — everything comes off the
 * world object that town.js builds from the data.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  world: null,
  day: 1,
  clock: 0,
  domain: [0, 1],
  playing: false,
  speed: 1,
  allDays: false,
  storyOn: false,
  beat: 0,
  beats: [],
  inspect: null,
  share: false,
  cam: null,
  camFrom: null,
  camT: 1,
  camStart: 0,
  hover: null,
  lastNow: 0,
};

const BASE_RATE = 26;      // simulated minutes per real second at 1x

/* ============================================================== loading */

function parseJsonl(text) {
  const out = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (s) out.push(JSON.parse(s));
  }
  return out;
}

async function loadOne(file, parse) {
  // 1. try the network. This is the path used over http(s).
  try {
    const res = await fetch("data/" + file, { cache: "no-store" });
    if (res.ok) return { via: "fetch", value: parse(await res.text()) };
  } catch (e) { /* file:// throws — fall through */ }
  // 2. fall back to the generated bundle (make_bundle.mjs)
  const b = window.__GBM_BUNDLE__;
  const key = file.split(".")[0];
  if (b && typeof b[key] === "string") return { via: "bundle", value: parse(b[key]) };
  throw new Error("could not load data/" + file);
}

async function loadData() {
  const [t, p, c] = await Promise.all([
    loadOne("trajectories.jsonl", parseJsonl),
    loadOne("profiles.json", JSON.parse),
    loadOne("cards.jsonl", parseJsonl),
  ]);
  return {
    via: [t.via, p.via, c.via].every((v) => v === "fetch") ? "fetch" : "bundle",
    data: { trajectories: t.value, profiles: p.value, cards: c.value },
  };
}

/* ================================================================ boot */

async function boot() {
  const status = $("status");
  let loaded;
  try {
    status.textContent = "reading the published log…";
    loaded = await loadData();
  } catch (err) {
    status.textContent = "could not load the data files: " + err.message;
    status.className = "status err";
    return;
  }
  const world = Town.buildWorld(loaded.data);
  state.world = world;

  // build the two static passes, yielding so the browser can paint progress
  const steps = Town.buildStaticSteps(world);
  for (let i = 0; i < steps.length; i++) {
    status.textContent = "painting the town — " + steps[i].label +
      " (" + (i + 1) + "/" + steps.length + ")";
    await new Promise((r) => setTimeout(r, 0));
    steps[i].run();
  }

  state.beats = Town.beats(world);
  state.day = world.days[0];
  setDomain();
  state.clock = state.domain[0];

  fillHeader(loaded.via);
  $("main").hidden = false;
  buildNumbers();
  wire();
  sizeCanvas();
  requestAnimationFrame(sizeCanvas);
  status.textContent = world.pids.length + " travelers · " + world.days.length +
    " mornings · " + world.checks.nRows.toLocaleString() + " recorded trips" +
    (loaded.via === "bundle" ? " · loaded from the offline bundle" : "");
  requestAnimationFrame(loop);
  setTimeout(() => $("hint").classList.add("gone"), 9000);
}

function fillHeader(via) {
  const w = state.world;
  const d1 = w.dayStats[w.days[0]], dN = w.dayStats[w.days[w.days.length - 1]];
  const items = [
    [d1.peakQueue.n + " → " + dN.peakQueue.n, "worst queue, day " + d1.day + " → " + dN.day],
    [d1.nB + " → " + dN.nB, "on the ring road"],
    [d1.meanTravel.toFixed(1) + " → " + dN.meanTravel.toFixed(1) + " min", "mean drive"],
    [d1.maxDelayA.toFixed(1) + " → " + dN.maxDelayA.toFixed(1) + " min", "longest wait"],
  ];
  $("head-stats").innerHTML = items.map(
    (i) => '<div class="head-stat"><b>' + i[0] + "</b><span>" + i[1] + "</span></div>").join("");
  $("strap").textContent = w.pids.length + " travelers · two routes · one bridge · " +
    w.days.length + " recorded mornings";
  $("provenance-text").textContent = Town.PROVENANCE;
  const man = (window.__GBM_BUNDLE__ && window.__GBM_BUNDLE__.manifest) || [];
  $("provenance-files").textContent =
    "Sources (" + (via === "bundle" ? "offline bundle" : "app/data/") + "): " +
    (man.length
      ? man.map((m) => m.file + " · " + m.bytes.toLocaleString() + " B · sha256 " + m.sha256.slice(0, 12)).join("   ")
      : "trajectories.jsonl · profiles.json · cards.jsonl") +
    "   ·   route A free-flows in " + w.checks.routeAFreeFlow +
    " min and the ring road is a flat " + w.checks.ringMinutes +
    " min in every recorded row — both re-checked against the data at load time.";
}

/* ============================================================= plumbing */

function setDomain() {
  const rows = state.world.byDay[state.day];
  let lo = Infinity, hi = -Infinity;
  for (const r of rows) {
    if (r.dep_min < lo) lo = r.dep_min;
    if (r.arrive_min > hi) hi = r.arrive_min;
  }
  state.domain = [Math.floor(lo) - 8, Math.ceil(hi) + 8];
}

function sizeCanvas() {
  const c = $("town");
  const rect = c.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const w = Math.max(960, Math.min(1920, Math.round(rect.width * dpr)));
  const h = Math.round(w * 9 / 16);
  if (c.width !== w) { c.width = w; c.height = h; }
}

function currentBeat() {
  return state.storyOn ? state.beats[state.beat] : null;
}

function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

function lerpCam(a, b, t) {
  const A = a || { fx: Town.W / 2, fy: Town.H / 2, zoom: 1 };
  const B = b || { fx: Town.W / 2, fy: Town.H / 2, zoom: 1 };
  return {
    fx: A.fx + (B.fx - A.fx) * t,
    fy: A.fy + (B.fy - A.fy) * t,
    zoom: A.zoom + (B.zoom - A.zoom) * t,
  };
}

function frameOptions() {
  const w = state.world;
  const beat = currentBeat();
  let opts;
  if (beat) {
    opts = Town.beatOptions(w, beat);
  } else {
    opts = { title: "Day " + state.day, clockText: Town.hhmm(state.clock) };
  }
  // camera easing between beats / back to the wide shot
  const t = state.camT >= 1 ? 1 : easeInOut(state.camT);
  opts.cam = lerpCam(state.camFrom, beat ? beat.cam : null, t);
  if (opts.cam.zoom < 1.002) opts.cam = null;
  if (t < 1) opts.focus = null;
  if (state.inspect) {
    opts.inspect = state.inspect;
    if (!beat) {
      // nudge the subject clear of the panel
      const loc = Town.locate(w, state.day, state.clock, state.inspect);
      if (loc) opts.cam = { fx: loc.x, fy: loc.y, zoom: 1.10, sx: 560, sy: 470 };
      opts.focus = loc ? { x: loc.x, y: loc.y, r: 300, a: 0.30 } : null;
    }
  }
  if (state.share) { opts.share = true; opts.chips = null; opts.focus = null; }
  return opts;
}

function draw() {
  const c = $("town");
  Town.render(c.getContext("2d"), state.world, state.day, state.clock, frameOptions());
}

function loop(now) {
  const dt = state.lastNow ? Math.min(0.1, (now - state.lastNow) / 1000) : 0;
  state.lastNow = now;
  if (state.camT < 1) state.camT = Math.min(1, state.camT + dt / 0.85);
  if (state.playing) {
    state.clock += dt * BASE_RATE * state.speed;
    if (state.clock >= state.domain[1]) {
      if (state.allDays && state.day < state.world.days[state.world.days.length - 1]) {
        setDay(state.day + 1, true);
      } else {
        state.clock = state.domain[1];
        setPlaying(false);
      }
    }
    syncScrub();
  }
  draw();
  requestAnimationFrame(loop);
}

function setPlaying(v) {
  state.playing = v;
  $("play").innerHTML = v ? "&#10074;&#10074;" : "&#9654;";
  $("play").classList.toggle("active", v);
}

function setDay(d, keepPlaying) {
  state.day = Math.max(state.world.days[0],
    Math.min(state.world.days[state.world.days.length - 1], d));
  setDomain();
  state.clock = state.domain[0];
  $("day-slider").value = state.day;
  $("day-read").textContent = state.day;
  if (!keepPlaying) setPlaying(false);
  syncScrub();
  updateNumbers();
}

function syncScrub() {
  const f = (state.clock - state.domain[0]) / (state.domain[1] - state.domain[0]);
  $("time-scrub").value = Math.round(Math.max(0, Math.min(1, f)) * 1000);
}

function scrubTo(f) {
  state.clock = state.domain[0] + f * (state.domain[1] - state.domain[0]);
}

/* =============================================================== story */

function enterStory(on) {
  const from = frameOptions().cam;          // capture before the state moves
  state.storyOn = on;
  $("story").classList.toggle("active", on);
  $("prev-beat").hidden = !on;
  $("next-beat").hidden = !on;
  $("beat-line").hidden = !on;
  if (on) { goBeat(0, from); }
  else {
    state.inspect = null;
    state.camFrom = from;
    state.camT = 0;
    $("beat-line").textContent = "";
  }
}

function goBeat(i, camFrom) {
  const from = camFrom === undefined ? frameOptions().cam : camFrom;
  const b = state.beats[(i + state.beats.length) % state.beats.length];
  state.beat = (i + state.beats.length) % state.beats.length;
  state.camFrom = from;
  state.camT = 0;
  state.inspect = b.inspect || null;
  setPlaying(false);
  state.day = b.day;
  setDomain();
  state.clock = b.clock;
  $("day-slider").value = state.day;
  $("day-read").textContent = state.day;
  syncScrub();
  updateNumbers();
  $("beat-line").innerHTML =
    "<b>Beat " + (state.beat + 1) + " of " + state.beats.length + " — " + b.title +
    ", " + Town.hhmm(b.clock) + "</b> · " + b.sub +
    " &nbsp;·&nbsp; press <b>n</b> for the next beat";
}

/* ============================================================ dossier */

function canvasPoint(ev) {
  const c = $("town");
  const r = c.getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width * c.width;
  const y = (ev.clientY - r.top) / r.height * c.height;
  return Town.screenToDesign(state.world, x, y, c.width);
}

function onCanvasClick(ev) {
  const p = canvasPoint(ev);
  // clicking inside the open panel does nothing; clicking outside closes it
  if (state.inspect) {
    const c = $("town"), r = c.getBoundingClientRect();
    const sx = (ev.clientX - r.left) / r.width * Town.W;
    if (sx > Town.W - 700) return;
  }
  const hit = Town.hitTest(state.world, p.x, p.y);
  if (hit) {
    state.inspect = hit.pid;
    state.camFrom = frameOptions().cam;
    state.camT = 0;
    $("hint").classList.add("gone");
    const sel = $("traveler-select");
    if (sel) { sel.value = hit.pid; drawTable(hit.pid); }
  } else if (state.inspect) {
    state.inspect = null;
    state.camFrom = frameOptions().cam;
    state.camT = 0;
  }
}

function onCanvasMove(ev) {
  const p = canvasPoint(ev);
  const hit = Town.hitTest(state.world, p.x, p.y);
  $("town").classList.toggle("pointing", !!hit);
}

/* ============================================================== export */

function exportPng() {
  const c = $("town");
  const link = document.createElement("a");
  link.download = "glassbox-mobility-day" + state.day + "-" +
    Town.hhmm(state.clock).replace(":", "") + ".png";
  link.href = c.toDataURL("image/png");
  link.click();
}

/* ================================================================ wire */

function wire() {
  $("play").onclick = () => setPlaying(!state.playing);
  $("time-scrub").oninput = (e) => { setPlaying(false); scrubTo(e.target.value / 1000); };
  $("day-slider").oninput = (e) => { if (state.storyOn) enterStory(false); setDay(+e.target.value); };
  $("speed").onchange = (e) => { state.speed = +e.target.value; };
  $("all-days").onclick = function () {
    state.allDays = !state.allDays;
    this.classList.toggle("active", state.allDays);
    if (state.allDays) { if (state.storyOn) enterStory(false); setDay(state.world.days[0], true); setPlaying(true); }
  };
  $("story").onclick = () => enterStory(!state.storyOn);
  $("next-beat").onclick = () => goBeat(state.beat + 1);
  $("prev-beat").onclick = () => goBeat(state.beat - 1);
  $("share").onclick = function () {
    state.share = !state.share;
    document.body.classList.toggle("share", state.share);
    this.classList.toggle("active", state.share);
    setTimeout(sizeCanvas, 60);
  };
  $("png").onclick = exportPng;

  const c = $("town");
  c.addEventListener("click", onCanvasClick);
  c.addEventListener("mousemove", onCanvasMove);
  window.addEventListener("resize", sizeCanvas);

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    const step = (state.domain[1] - state.domain[0]) / 90;
    if (e.key === " ") { e.preventDefault(); setPlaying(!state.playing); }
    else if (e.key === "ArrowRight") { setPlaying(false); state.clock = Math.min(state.domain[1], state.clock + step); syncScrub(); }
    else if (e.key === "ArrowLeft") { setPlaying(false); state.clock = Math.max(state.domain[0], state.clock - step); syncScrub(); }
    else if (e.key === "ArrowUp") { if (state.storyOn) enterStory(false); setDay(state.day + 1); }
    else if (e.key === "ArrowDown") { if (state.storyOn) enterStory(false); setDay(state.day - 1); }
    else if (e.key === "s") enterStory(!state.storyOn);
    else if (e.key === "n") { if (state.storyOn) goBeat(state.beat + 1); else enterStory(true); }
    else if (e.key === "p") exportPng();
    else if (e.key === "Escape") {
      if (state.share) $("share").onclick.call($("share"));
      else if (state.inspect) { state.inspect = null; state.camFrom = frameOptions().cam; state.camT = 0; }
    }
  });
}

/* ====================================================== the numbers */

const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function buildNumbers() {
  const w = state.world;
  const sel = $("traveler-select");
  sel.innerHTML = "";
  for (const pid of w.pids) {
    sel.appendChild(new Option(pid + " · " + (w.name[pid] || ""), pid));
  }
  sel.onchange = () => { drawTable(sel.value); state.inspect = sel.value; state.camT = 0; };

  const host = $("notable-chips");
  host.innerHTML = "";
  const add = (pid, label) => {
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = label;
    b.onclick = () => {
      sel.value = pid; drawTable(pid);
      state.inspect = pid; state.camFrom = frameOptions().cam; state.camT = 0;
      [...host.children].forEach((x) => x.classList.toggle("active", x === b));
      $("stage").scrollIntoView({ behavior: "smooth", block: "start" });
    };
    host.appendChild(b);
  };
  const byDrift = w.pids.slice().sort((a, b) => w.trav[a].drift - w.trav[b].drift);
  for (const pid of byDrift.slice(0, 3)) {
    add(pid, pid + " · " + Town.durHM(w.trav[pid].drift) + " earlier by day " + w.trav[pid].last.day);
  }
  const bySw = w.pids.slice().sort((a, b) => w.trav[b].switches - w.trav[a].switches);
  for (const pid of bySw.slice(0, 3)) add(pid, pid + " · " + w.trav[pid].switches + " switches");

  sel.value = byDrift[0];
  drawTable(byDrift[0]);
  updateNumbers();
}

function updateNumbers() {
  if (!state.world) return;
  drawHistogram();
  drawSpark();
}

function drawHistogram() {
  const w = state.world;
  const cv = $("hist-canvas"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const ds = w.dayStats[state.day], d1 = w.dayStats[w.days[0]];
  $("hist-badge").textContent =
    "spread " + ds.depSD.toFixed(1) + " min · day " + w.days[0] + " was " + d1.depSD.toFixed(1) + " min";

  let t0 = Infinity, t1 = -Infinity;
  for (const d of w.days) {
    const s = w.dayStats[d];
    if (s.depMin < t0) t0 = s.depMin;
    if (s.depMax > t1) t1 = s.depMax;
  }
  t0 = Math.floor(t0) - 5; t1 = Math.ceil(t1) + 5;
  const bins = 84, bw = cv.width / bins;
  const count = (day) => {
    const c = new Array(bins).fill(0);
    for (const r of w.byDay[day]) {
      const b = Math.floor(((r.dep_min - t0) / (t1 - t0)) * bins);
      if (b >= 0 && b < bins) c[b]++;
    }
    return c;
  };
  const first = count(w.days[0]), today = count(state.day);
  const max = Math.max.apply(null, first.concat(today).concat([1]));
  const hOf = (c) => (c / max) * (cv.height - 42);

  ctx.fillStyle = cssVar("--route-a");
  ctx.globalAlpha = 0.9;
  today.forEach((c, b) => c && ctx.fillRect(b * bw + 1, cv.height - hOf(c) - 22, Math.max(bw - 2, 1), hOf(c)));
  ctx.globalAlpha = 0.55;
  ctx.strokeStyle = cssVar("--muted");
  ctx.lineWidth = 1;
  ctx.beginPath();
  first.forEach((c, b) => {
    const y = cv.height - hOf(c) - 22;
    if (!b) ctx.moveTo(0, y);
    ctx.lineTo(b * bw, y); ctx.lineTo((b + 1) * bw, y);
  });
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = cssVar("--faint");
  ctx.font = "12px " + Town.SANS;
  for (let m = Math.ceil(t0 / 30) * 30; m <= t1; m += 30) {
    const x = ((m - t0) / (t1 - t0)) * cv.width;
    ctx.fillRect(x, cv.height - 20, 1, 5);
    ctx.fillText(Town.hhmm(m), x - 15, cv.height - 5);
  }
}

function drawSpark() {
  const w = state.world;
  const cv = $("spark-canvas"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const peak = w.days.map((d) => w.dayStats[d].peakShare);
  const n = peak.length;
  $("peak-badge").textContent = "day " + state.day + " · peak share " +
    w.dayStats[state.day].peakShare.toFixed(3);

  const lo = Math.min.apply(null, peak) * 0.94, hi = Math.max.apply(null, peak) * 1.06;
  const px = (i) => 54 + (i / (n - 1)) * (cv.width - 300);
  const py = (v) => 26 + (1 - (v - lo) / (hi - lo)) * (cv.height - 62);

  ctx.strokeStyle = cssVar("--faint");
  ctx.setLineDash([3, 5]);
  ctx.beginPath(); ctx.moveTo(54, py(peak[0])); ctx.lineTo(cv.width - 250, py(peak[0])); ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = "12px " + Town.SANS;
  ctx.fillStyle = cssVar("--faint");
  ctx.fillText("day " + w.days[0] + " · " + peak[0].toFixed(3), cv.width - 242, py(peak[0]) + 4);

  ctx.strokeStyle = cssVar("--route-a");
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  peak.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
  ctx.stroke();
  peak.forEach((v, i) => {
    const on = w.days[i] === state.day;
    ctx.fillStyle = on ? cssVar("--ink") : cssVar("--route-a");
    ctx.beginPath(); ctx.arc(px(i), py(v), on ? 4.5 : 2.4, 0, Math.PI * 2); ctx.fill();
  });
  if (n > 2) {
    const drop = 100 * (peak[n - 1] - peak[1]) / peak[1];
    ctx.strokeStyle = cssVar("--edge-hi");
    ctx.beginPath(); ctx.moveTo(px(1), py(peak[1]) - 8); ctx.lineTo(px(1), py(peak[1]) - 26); ctx.stroke();
    ctx.fillStyle = cssVar("--muted");
    ctx.fillText(drop.toFixed(0) + "% from day " + w.days[1] + " to day " + w.days[n - 1] +
      "  (exploratory)", px(1) + 8, py(peak[1]) - 22);
  }
  ctx.fillStyle = cssVar("--faint");
  for (let i = 0; i < n; i += 3) ctx.fillText("d" + w.days[i], px(i) - 8, cv.height - 8);
}

function drawTable(pid) {
  const w = state.world, t = w.trav[pid];
  if (!t) return;
  const tbody = $("traveler-table").querySelector("tbody");
  tbody.innerHTML = "";
  t.list.forEach(function (r, i) {
    const prev = t.list[i - 1];
    let delta = "—", cls = "";
    if (prev) {
      const parts = [];
      if (r.route !== prev.route) { parts.push("switched to " + (r.route === "B" ? "the ring road" : "route A")); cls = "switch"; }
      const shift = Math.round(r.dep - prev.dep);
      if (shift) parts.push((shift > 0 ? "+" : "") + shift + " min");
      delta = parts.join(", ") || "same plan";
    }
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + r.day + '</td><td class="' + (r.route === "B" ? "ring" : "") + '">' +
      (r.route === "B" ? "ring" : "A") + "</td>" +
      "<td>" + Town.hhmm(r.dep) + "</td><td>" + r.travel.toFixed(1) + " min</td>" +
      '<td class="' + (r.delay >= 1 ? "queued" : "") + '">' +
      (r.delay >= 0.05 ? r.delay.toFixed(1) + " min" : "—") + "</td>" +
      "<td>" + Town.hhmm(r.arrive) + '</td><td class="' + cls + '">' + delta + "</td>";
    tbody.appendChild(tr);
  });
  $("traveler-summary").textContent =
    (w.name[pid] || pid) + " · " + t.switches + (t.switches === 1 ? " switch · " : " switches · ") +
    (t.daysOnB ? t.daysOnB + "/" + t.list.length + " on the ring · " : "never on the ring · ") +
    Town.durHM(t.drift) + (t.drift <= 0 ? " earlier" : " later") + " by day " + t.last.day;
}

boot();
