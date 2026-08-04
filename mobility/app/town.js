/* town.js — GLASSBOX-Mobility viewer V3, the shared scene renderer.
 *
 * One module, no DOM dependency, loadable three ways:
 *   browser   <script src="town.js"></script>   ->  window.Town
 *   node      const Town = require("./town.js")
 *   bundler   import Town from "./town.js"      (CommonJS interop)
 *
 * It draws one frame of the living miniature town given
 *   Town.render(ctx, world, day, clockMin, options)
 * where `world` comes from Town.buildWorld(data) and `data` is the three
 * published files parsed:  {trajectories, profiles, cards}.
 *
 * HONESTY CONTRACT
 * ----------------
 * Nothing on screen is invented. Every house, window, car, queue position,
 * name, caption number and diary line is computed here, at run time, from
 *   trajectories.jsonl  {pid, day, route, dep_min, travel_min, arrive_min}
 *   profiles.json       {pid, age_band, occupation_type, household, area_type}
 *   cards.jsonl         {pid, name, card}
 * The only quantities NOT read from those files are the study's two published
 * world constants — route A free-flows in 18 minutes, the ring road is a flat
 * 26 — and both are re-checked against the data at build time (see
 * world.checks). Everything else that looks like a constant in this file is a
 * drawing convention (where a road bends, how far apart queued cars are
 * spaced), never a claim.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Town = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /* ============================================================ constants */

  const W = 1920, H = 1080, HORIZON = 172;
  const FF_A = 18;            // route A free-flow minutes (study world constant)
  const T_B = 26;             // ring road minutes (study world constant)
  const PINCH_FRAC = 0.62;    // the bridge sits at 62% along route A
  const LIT_LEAD = 20;        // a house is lit from 20 min before departure

  // NOTE: keep these stacks to quoted families + plain identifiers. node-canvas
  // silently rejects a whole font string containing `system-ui`/`-apple-system`
  // and leaves the previous font in place, which is very hard to spot.
  const SANS = '"Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif';
  const SERIF = '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif';

  const PROVENANCE =
    "Every window, car, name, queue and diary line is computed from the study's " +
    "recorded 20-day trajectory log and its real traveler biography cards. " +
    "Monograms derive from traveler id — no faces generated.";

  /* ====================================================== canvas plumbing */

  let makeCanvas = null;
  if (typeof document !== "undefined" && document.createElement) {
    makeCanvas = function (w, h) {
      const c = document.createElement("canvas");
      c.width = w; c.height = h; return c;
    };
  }
  function setCanvasFactory(fn) { makeCanvas = fn; }
  function nc(w, h) {
    if (!makeCanvas) throw new Error("Town: call Town.setCanvasFactory() first");
    return makeCanvas(Math.max(1, Math.round(w)), Math.max(1, Math.round(h)));
  }

  /* ============================================================== helpers */

  function hash(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rngFor = (pid, salt) => mulberry32(hash(pid + "|" + (salt || "")));
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const rgb = (c, a) => "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + (a === undefined ? 1 : a) + ")";
  const hexToRgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);

  function hhmm(m) {
    const t = Math.floor(m);
    return String(Math.floor(t / 60) % 24).padStart(2, "0") + ":" + String(((t % 60) + 60) % 60).padStart(2, "0");
  }
  function durHM(m) {
    const a = Math.abs(Math.round(m)), h = Math.floor(a / 60), mm = a % 60;
    return h ? h + " h " + String(mm).padStart(2, "0") + " min" : mm + " min";
  }

  /* Text metrics are memoised. measureText is one of the most expensive calls
   * in a 2D context and the chrome re-measures the same strings every frame;
   * the cache is keyed on the font as well as the string, so it is safe. */
  const _memo = new Map();
  function memo(key, make) {
    let v = _memo.get(key);
    if (v === undefined) {
      if (_memo.size > 2000) _memo.clear();
      v = make(); _memo.set(key, v);
    }
    return v;
  }
  function charWidths(ctx, text) {
    return memo("cw|" + ctx.font + "|" + text, function () {
      const a = [];
      for (const ch of text) a.push(ctx.measureText(ch).width);
      return a;
    });
  }
  function textW(ctx, text) {
    return memo("tw|" + ctx.font + "|" + text, () => ctx.measureText(text).width);
  }
  function tracked(ctx, text, x, y, sp) {
    const ws = charWidths(ctx, text);
    let cx = x, i = 0;
    for (const ch of text) { ctx.fillText(ch, cx, y); cx += ws[i++] + sp; }
    return cx - sp - x;
  }
  function trackedWidth(ctx, text, sp) {
    const ws = charWidths(ctx, text);
    let w = 0;
    for (const v of ws) w += v + sp;
    return w - sp;
  }
  function wrapText(ctx, text, maxW) {
    return memo("wr|" + ctx.font + "|" + maxW + "|" + text, function () {
      const words = String(text).split(" ");
      const out = []; let line = "";
      for (const w of words) {
        const t = line ? line + " " + w : w;
        if (ctx.measureText(t).width > maxW && line) { out.push(line); line = w; }
        else line = t;
      }
      if (line) out.push(line);
      return out;
    });
  }
  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }
  function blurredCopy(src, radius) {
    const c = nc(src.width, src.height);
    const x = c.getContext("2d");
    x.filter = "blur(" + radius + "px)";
    x.drawImage(src, 0, 0);
    return c;
  }

  /* --------------------------------------------------------- curves/paths */

  function catmull(pts, samples) {
    const P = [pts[0]].concat(pts, [pts[pts.length - 1]]);
    const out = [];
    for (let i = 1; i < P.length - 2; i++) {
      const p0 = P[i - 1], p1 = P[i], p2 = P[i + 1], p3 = P[i + 2];
      for (let j = 0; j < samples; j++) {
        const t = j / samples, t2 = t * t, t3 = t2 * t;
        out.push([
          0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
          0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
        ]);
      }
    }
    out.push(pts[pts.length - 1]);
    return out;
  }
  function makePath(ctrl, samples) {
    const pts = catmull(ctrl, samples || 60);
    const cum = [0];
    for (let i = 1; i < pts.length; i++) {
      cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    }
    const total = cum[cum.length - 1];
    function at(t) {
      const d = clamp(t, 0, 1) * total;
      let lo = 0, hi = cum.length - 1;
      while (lo < hi - 1) { const m = (lo + hi) >> 1; if (cum[m] <= d) lo = m; else hi = m; }
      const seg = cum[hi] - cum[lo] || 1;
      const f = (d - cum[lo]) / seg;
      return {
        x: pts[lo][0] + (pts[hi][0] - pts[lo][0]) * f,
        y: pts[lo][1] + (pts[hi][1] - pts[lo][1]) * f,
        ang: Math.atan2(pts[hi][1] - pts[lo][1], pts[hi][0] - pts[lo][0]),
      };
    }
    return { pts: pts, total: total, at: at };
  }

  /* ====================================================== town geography
   * Authored screen-space layout. These coordinates are a drawing choice —
   * the study's corridor is abstract; only the topology (two routes, one
   * pinch, one work district) and every quantity above are from the data. */

  const depth = (y) => clamp((y - HORIZON) / (H - HORIZON), 0, 1);
  const scaleAt = (y) => 0.32 + 1.18 * Math.pow(depth(y), 0.95);

  const HOME_HUB = [524, 686];
  const DISTRICT_GATE = [1556, 486];
  const PLAZA = { x: 1652, y: 520 };

  const ROUTE_A = makePath([
    HOME_HUB, [644, 668], [768, 634], [886, 592], [1004, 552],
    [1130, 518], [1268, 492], [1400, 480], [1494, 482], DISTRICT_GATE,
  ]);
  const PINCH = ROUTE_A.at(PINCH_FRAC);

  const ROUTE_B = makePath([
    [518, 706], [548, 796], [648, 866], [806, 906], [986, 912],
    [1160, 886], [1310, 830], [1428, 742], [1512, 640], [1552, 546], DISTRICT_GATE,
  ]);

  const RIVER = makePath([
    [1272, 30], [1214, 170], [1162, 330], [PINCH.x, PINCH.y],
    [1178, 636], [1238, 776], [1330, 906], [1450, 1024], [1610, 1140],
  ], 40);
  const riverWidth = (y) => 26 + 92 * Math.pow(depth(y), 1.2);

  const POND = { x: 802, y: 748, rx: 122, ry: 44 };
  const WOODS = [
    [700, 604, 72], [962, 776, 98], [1424, 676, 84], [138, 912, 64],
    [1128, 316, 54], [786, 462, 56], [1748, 764, 98], [392, 392, 46],
    [1006, 398, 50], [610, 312, 40], [1332, 366, 44], [246, 684, 38],
    [880, 352, 44], [1240, 232, 34], [520, 252, 30], [1560, 300, 36],
    [1680, 940, 88], [420, 980, 74], [860, 1000, 66], [1180, 700, 40],
    [640, 760, 34], [1520, 848, 52], [960, 560, 30], [300, 448, 26],
  ];

  const RING_BRIDGE = (function () {
    let best = null;
    for (let i = 1; i < ROUTE_B.pts.length; i++) {
      const p = ROUTE_B.pts[i];
      let dm = 1e9, near = null;
      for (const q of RIVER.pts) {
        const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d < dm) { dm = d; near = q; }
      }
      if (!best || dm < best.d) best = { d: dm, i: i };
    }
    const t = best.i / (ROUTE_B.pts.length - 1);
    const pos = ROUTE_B.at(t);
    return { x: pos.x, y: pos.y, ang: pos.ang, t: t };
  })();

  const NEIGHBOURHOODS = {
    "inner city": { cx: 296, cy: 322, rx: 150, ry: 50, tight: 0.88, tagY: 246 },
    "small town": { cx: 224, cy: 540, rx: 158, ry: 62, tight: 0.52, tagY: 448 },
    "suburb": { cx: 452, cy: 758, rx: 214, ry: 74, tight: 0.44, tagY: 664 },
  };
  // house form is chosen by the recorded `household` field
  const FORM = {
    "single": { w: 15, h: 10, storeys: 1, windows: 1, flat: false },
    "couple": { w: 18, h: 11, storeys: 1, windows: 2, flat: false },
    "couple with children": { w: 22, h: 13, storeys: 1, windows: 3, flat: false },
    "single parent": { w: 18, h: 12, storeys: 1, windows: 2, flat: false },
    "shared flat": { w: 16, h: 20, storeys: 3, windows: 4, flat: true },
  };
  const FORM_FALLBACK = { w: 17, h: 11, storeys: 1, windows: 2, flat: false };
  const ROOFS = ["#b8603b", "#a04a2e", "#8d5b46", "#4f5d6b", "#5d6b4c", "#8a7a5c", "#a95a3c", "#6a5f52"];
  const CAR_HUES = ["#c8563a", "#d8a04a", "#8e9aa8", "#cfc3ac", "#7d6a52", "#a8503f", "#5f6f7d", "#b9884f"];

  const SLABS = [
    { x: 1436, y: 452, w: 82, h: 156, cols: 5, rows: 10 },
    { x: 1540, y: 432, w: 74, h: 196, cols: 5, rows: 12 },
    { x: 1642, y: 452, w: 90, h: 146, cols: 5, rows: 9 },
    { x: 1758, y: 438, w: 78, h: 178, cols: 5, rows: 11 },
    { x: 1866, y: 458, w: 68, h: 134, cols: 4, rows: 9 },
    { x: 1462, y: 556, w: 112, h: 88, cols: 6, rows: 6 },
    { x: 1620, y: 578, w: 104, h: 72, cols: 6, rows: 5 },
    { x: 1770, y: 562, w: 94, h: 96, cols: 5, rows: 6 },
    { x: 1888, y: 552, w: 72, h: 82, cols: 4, rows: 5 },
  ];

  /* ================================================= palette / light model */

  // 0 = night, 1 = full morning. A rendering choice, keyed to clock time only.
  function lightOf(clock) {
    const L = clamp((clock - 292) / 176, 0, 1);
    return L * L * (3 - 2 * L);
  }
  function palette(L) {
    return {
      L: L, night: 1 - L,
      skyTop: mix([10, 15, 34], [74, 118, 178], L),
      skyMid: mix([20, 27, 56], [146, 180, 214], L),
      skyLow: mix([52, 44, 74], [244, 214, 186], L),
      skyGlow: mix([146, 88, 68], [255, 218, 162], L),
      ridge: mix([13, 19, 36], [118, 136, 148], L),
      ridge2: mix([18, 25, 46], [148, 162, 160], L),
      ground: mix([21, 28, 38], [124, 132, 92], L),
      groundNear: mix([12, 17, 25], [76, 84, 60], L),
      field: mix([27, 35, 46], [154, 156, 104], L),
      fieldWarm: mix([34, 34, 44], [200, 180, 116], L),
      water: mix([10, 20, 30], [56, 100, 106], L),
      waterHi: mix([38, 56, 72], [206, 226, 214], L),
      asphalt: mix([28, 34, 48], [106, 106, 110], L),
      asphaltEdge: mix([44, 52, 70], [156, 154, 150], L),
      wallSun: mix([48, 52, 68], [248, 234, 210], L),
      wallShade: mix([26, 31, 44], [166, 156, 142], L),
      roofMul: 0.22 + 0.78 * L,
      shadow: mix([4, 6, 16], [48, 46, 74], L * 0.7),
      haze: mix([18, 25, 48], [206, 218, 228], L),
      amber: [255, 176, 78],
      amberCore: [255, 244, 214],
      cyan: [64, 208, 230],
      cyanCore: [178, 246, 255],
      brake: [255, 84, 56],
    };
  }

  const INK = [247, 244, 239];
  const GOLD = [255, 206, 146];
  const CY = [150, 234, 248];
  const HOT = [255, 158, 120];
  const PAPER = [244, 236, 223];
  const PAPER_INK = [25, 20, 16];
  const PAPER_MUTE = [122, 107, 92];
  const RIBBON_A = [184, 122, 60];
  const RIBBON_B = [18, 114, 122];
  const MONO_HUES = ["#B8543F", "#8E5A8C", "#3E6E7A", "#9A6B2F", "#5B6E3A", "#7A4A5E",
                     "#3C5A86", "#A45C34", "#4E6E5E", "#6E4E8A", "#8A4A46", "#43607A"];

  /* ========================================================== world build */

  function buildWorld(data) {
    const traj = data.trajectories, profiles = data.profiles, cards = data.cards;
    if (!traj || !traj.length) throw new Error("Town: no trajectories");

    const prof = Object.create(null);
    for (const p of profiles) prof[p.pid] = p;
    const name = Object.create(null), card = Object.create(null);
    for (const c of cards) { name[c.pid] = c.name; card[c.pid] = c.card; }

    const pids = Object.keys(prof).sort();
    const days = Array.from(new Set(traj.map((r) => r.day))).sort((a, b) => a - b);

    // ---- per traveler, per day: reconstruct the drive from the five fields
    const trav = Object.create(null);
    for (const pid of pids) trav[pid] = { pid: pid, days: Object.create(null), list: [] };
    // The published minutes are decimal; subtracting them in binary floating
    // point can turn 14.65 into 14.649999999999999, which then prints as 14.6
    // instead of 14.7. Round every derived duration back to the data's own
    // precision before anything is displayed or ranked.
    const r6 = (v) => Math.round(v * 1e6) / 1e6;
    for (const r of traj) {
      const span = r6(r.arrive_min - r.dep_min);             // recorded door-to-door
      const delay = r.route === "A" ? Math.max(0, r6(r.travel_min - FF_A)) : 0;
      const rec = {
        pid: r.pid, day: r.day, route: r.route,
        dep: r.dep_min, travel: r.travel_min, arrive: r.arrive_min,
        delay: delay,
        free: Math.max(0.001, r6(span - delay)),             // moving minutes
        span: span,
      };
      // the minute this car reaches the bridge (route A only)
      rec.tPinch = rec.dep + rec.free * PINCH_FRAC;
      rec.tRelease = rec.tPinch + delay;
      const t = trav[r.pid];
      if (t) { t.days[r.day] = rec; t.list.push(rec); }
    }
    for (const pid of pids) trav[pid].list.sort((a, b) => a.day - b.day);

    // ---- world-constant checks, re-derived from the published rows
    const bTimes = traj.filter((r) => r.route === "B").map((r) => r.travel_min);
    const aTimes = traj.filter((r) => r.route === "A").map((r) => r.travel_min);
    const checks = {
      ringConstant: bTimes.length ? (Math.min.apply(null, bTimes) === Math.max.apply(null, bTimes)) : false,
      ringMinutes: bTimes.length ? bTimes[0] : T_B,
      routeAFreeFlow: aTimes.length ? Math.min.apply(null, aTimes) : FF_A,
      nTravelers: pids.length, nDays: days.length, nRows: traj.length,
    };

    // ---- per traveler summary (all computed)
    for (const pid of pids) {
      const t = trav[pid], L = t.list;
      let switches = 0, onB = 0, firstB = null, worst = null, totalQ = 0;
      for (let i = 0; i < L.length; i++) {
        if (i && L[i].route !== L[i - 1].route) switches++;
        if (L[i].route === "B") { onB++; if (firstB === null) firstB = L[i].day; }
        totalQ += L[i].delay;
        if (!worst || L[i].delay > worst.delay) worst = L[i];
      }
      t.switches = switches;
      t.daysOnB = onB;
      t.firstDayOnB = firstB;
      t.worst = worst;
      t.totalQueue = totalQ;
      t.first = L[0];
      t.last = L[L.length - 1];
      t.drift = L.length > 1 ? L[L.length - 1].dep - L[0].dep : 0;
    }

    // ---- per day stats
    const byDay = Object.create(null);
    for (const r of traj) (byDay[r.day] || (byDay[r.day] = [])).push(r);

    function queueCountAt(day, clock) {
      let n = 0;
      const list = trav; // walk the reconstructed records
      for (const pid of pids) {
        const rec = list[pid].days[day];
        if (!rec || rec.route !== "A" || rec.delay <= 0) continue;
        if (clock >= rec.tPinch && clock < rec.tRelease) n++;
      }
      return n;
    }

    const dayStats = Object.create(null);
    const triedB = new Set();
    for (const d of days) {
      const rows = byDay[d];
      const deps = rows.map((r) => r.dep_min);
      const mean = deps.reduce((a, b) => a + b, 0) / deps.length;
      const sd = Math.sqrt(deps.reduce((a, b) => a + (b - mean) * (b - mean), 0) / deps.length);
      const aRows = rows.filter((r) => r.route === "A");
      const bRows = rows.filter((r) => r.route === "B");
      for (const r of bRows) triedB.add(r.pid);

      // queue peak, scanned at whole-minute resolution across the morning
      const lo = Math.floor(Math.min.apply(null, deps));
      const hi = Math.ceil(Math.max.apply(null, rows.map((r) => r.arrive_min)));
      let peak = { n: 0, minute: lo };
      for (let m = lo; m <= hi; m++) {
        const n = queueCountAt(d, m);
        if (n > peak.n) peak = { n: n, minute: m };
      }
      // observed bridge throughput: crossings per minute over congested minutes
      const crossings = Object.create(null);
      for (const pid of pids) {
        const rec = trav[pid].days[d];
        if (!rec || rec.route !== "A") continue;
        const k = Math.floor(rec.tRelease);
        crossings[k] = (crossings[k] || 0) + 1;
      }
      const busy = Object.keys(crossings).filter((k) => queueCountAt(d, +k) > 3);
      const thr = busy.length ? busy.reduce((a, k) => a + crossings[k], 0) / busy.length : null;

      // arrival peak share in the tightest 15-minute window (V2's frozen measure)
      const arr = rows.map((r) => r.arrive_min).sort((a, b) => a - b);
      let best = 0;
      for (let i = 0; i < arr.length; i++) {
        let j = i; while (j < arr.length && arr[j] < arr[i] + 15) j++;
        if (j - i > best) best = j - i;
      }

      dayStats[d] = {
        day: d, n: rows.length, nA: aRows.length, nB: bRows.length,
        triedBCumulative: triedB.size,
        meanTravel: rows.reduce((a, r) => a + r.travel_min, 0) / rows.length,
        maxDelayA: aRows.length ? Math.max.apply(null, aRows.map((r) => trav[r.pid].days[d].delay)) : 0,
        depSD: sd, depMin: Math.min.apply(null, deps), depMax: Math.max.apply(null, deps),
        peakQueue: peak,
        throughput: thr,
        peakShare: best / rows.length,
        firstOntoB: bRows.slice().sort((a, b) => a.dep_min - b.dep_min).map((r) => r.pid),
        // dense ranking of the day's bridge waits (ties share a rank)
        delayRank: (function () {
          const vals = Array.from(new Set(aRows.map((r) => trav[r.pid].days[d].delay)))
            .sort((a, b) => b - a);
          const idx = Object.create(null);
          vals.forEach((v, i) => { idx[v] = i + 1; });
          const rank = Object.create(null), share = Object.create(null);
          for (const r of aRows) {
            const v = trav[r.pid].days[d].delay;
            rank[r.pid] = idx[v];
            share[idx[v]] = (share[idx[v]] || 0) + 1;
          }
          return { rank: rank, tie: share };
        })(),
      };
    }

    const areaCounts = { "inner city": 0, "small town": 0, "suburb": 0 };
    for (const pid of pids) if (areaCounts[prof[pid].area_type] !== undefined) areaCounts[prof[pid].area_type]++;

    const world = {
      pids: pids, days: days, prof: prof, name: name, card: card, trav: trav,
      dayStats: dayStats, areaCounts: areaCounts, checks: checks,
      byDay: byDay, queueCountAt: queueCountAt,
      FF_A: FF_A, T_B: T_B, PINCH_FRAC: PINCH_FRAC, LIT_LEAD: LIT_LEAD,
      _static: null, _sprites: null, _dossier: Object.create(null), _cars: [],
    };

    buildPlots(world);
    buildDistrict(world);
    return world;
  }

  /* ---- one plot per traveler, grouped by the recorded area_type --------- */

  function buildPlots(world) {
    const groups = { "inner city": [], "small town": [], "suburb": [] };
    for (const pid of world.pids) {
      const a = world.prof[pid].area_type;
      (groups[a] || (groups[a] = [])).push(pid);
    }
    const plots = Object.create(null);
    for (const k of Object.keys(groups)) {
      const N = NEIGHBOURHOODS[k] || NEIGHBOURHOODS["suburb"];
      const list = groups[k], n = list.length;
      if (!n) continue;
      const cols = Math.ceil(Math.sqrt(n * (N.rx / N.ry)));
      const rows = Math.ceil(n / cols);
      const cells = [];
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cells.push([c, r]);
      cells.sort(function (a, b) {
        const da = Math.hypot((a[0] + 0.5) / cols - 0.5, (a[1] + 0.5) / rows - 0.5);
        const db = Math.hypot((b[0] + 0.5) / cols - 0.5, (b[1] + 0.5) / rows - 0.5);
        return da - db;
      });
      let i = 0;
      for (const pid of list) {
        const cell = cells[i++];
        const rnd = rngFor(pid, "plot");
        const jx = (rnd() - 0.5) * (1 - N.tight) * 1.9;
        const jy = (rnd() - 0.5) * (1 - N.tight) * 1.9;
        const u = (cell[0] + 0.5) / cols - 0.5 + jx / cols;
        const v = (cell[1] + 0.5) / rows - 0.5 + jy / rows;
        const x = N.cx + u * N.rx * 2.0;
        const y = N.cy + v * N.ry * 2.0;
        const form = FORM[world.prof[pid].household] || FORM_FALLBACK;
        const plot = {
          pid: pid, x: x, y: y, area: k, form: form,
          s: scaleAt(y) * (0.92 + rnd() * 0.22),
          roof: ROOFS[Math.floor(rnd() * ROOFS.length)],
          rot: rnd() < 0.5 ? -1 : 1,
          chim: rnd() < 0.45,
          tree: rnd() < (k === "inner city" ? 0.10 : 0.42),
          tj: [rnd() - 0.5, rnd() - 0.5],
        };
        houseGeometry(plot);
        plots[pid] = plot;
      }
    }
    world.plots = plots;
    world.plotList = world.pids.map((p) => plots[p]).filter(Boolean).sort((a, b) => a.y - b.y);
  }

  // cache the isometric shell + window rectangles once, so the per-frame
  // "light this house" pass is a handful of fillRects
  function houseGeometry(plot) {
    const s = plot.s, form = plot.form;
    const w = form.w * s, d = w * 0.60;
    const hh = form.flat ? form.h * s : form.h * s * 0.66;
    const hw = w / 2, hd = d / 2, x = plot.x, y = plot.y;
    const P0 = [x, y - hd], P1 = [x + hw, y], P2 = [x, y + hd], P3 = [x - hw, y];
    const up = (p, k) => [p[0], p[1] - k];
    plot.g = {
      w: w, d: d, hh: hh, hw: hw, hd: hd,
      P0: P0, P1: P1, P2: P2, P3: P3,
      Q0: up(P0, hh), Q1: up(P1, hh), Q2: up(P2, hh), Q3: up(P3, hh),
    };
    const wins = [];
    const rowsN = form.flat ? form.storeys : 1;
    const cnt = form.flat ? 2 : (form.windows >= 3 ? 2 : 1);
    for (let r = 0; r < rowsN; r++) {
      for (let i = 0; i < cnt; i++) {
        const fx = (i + 1) / (cnt + 1);
        const yOff = hh * ((r + 0.62) / rowsN);
        wins.push([P2[0] + (P1[0] - P2[0]) * fx, P2[1] + (P1[1] - P2[1]) * fx - yOff, 1]);
        wins.push([P3[0] + (P2[0] - P3[0]) * fx, P3[1] + (P2[1] - P3[1]) * fx - yOff, 0]);
      }
    }
    plot.wins = wins;
    plot.ww = Math.max(1.5, 2.4 * s);
    plot.wh = Math.max(1.9, 3.0 * s);
    plot.top = y - hh - (form.flat ? 0 : hh * 0.72);
  }

  /* ---- the work district: exactly one window cell per traveler --------- */

  function buildDistrict(world) {
    const cells = [];
    for (const s of SLABS) {
      for (let r = 0; r < s.rows; r++) for (let c = 0; c < s.cols; c++) cells.push({ s: s, c: c, r: r });
    }
    const order = cells.map(function (c, i) {
      return { c: c, k: mulberry32((0x9e37 + i * 2654435761) % 2147483647)() };
    });
    order.sort((a, b) => a.k - b.k);
    const map = Object.create(null);
    world.pids.forEach(function (pid, i) {
      const cell = order[i % order.length].c, s = cell.s;
      const hw = s.w / 2, hd = hw * 0.44, y = s.y, x = s.x;
      const P1 = [x + hw, y], P2 = [x, y + hd], P3 = [x - hw, y];
      const fx = (cell.c + 0.5) / s.cols;
      const sunFace = cell.c % 2 === 0;
      const ax = sunFace ? P2[0] + (P1[0] - P2[0]) * fx : P3[0] + (P2[0] - P3[0]) * fx;
      const ay = sunFace ? P2[1] + (P1[1] - P2[1]) * fx : P3[1] + (P2[1] - P3[1]) * fx;
      map[pid] = {
        x: ax, y: ay - s.h * ((cell.r + 0.5) / s.rows),
        w: Math.max(2.4, (s.w / 2 / s.cols) * 0.66),
        h: Math.max(2.8, (s.h / s.rows) * 0.48),
      };
    });
    world.district = map;
  }

  /* ============================================== per-minute world state */

  function stateAt(world, day, clock) {
    const home = [], driving = [], queued = [], arrived = [], lit = [];
    for (const pid of world.pids) {
      const rec = world.trav[pid].days[day];
      if (!rec) continue;
      if (clock >= rec.dep - LIT_LEAD && clock < rec.arrive) lit.push(pid);
      if (clock < rec.dep) { home.push(rec); continue; }
      if (clock >= rec.arrive) { arrived.push(rec); continue; }
      if (rec.route === "B") {
        driving.push({ rec: rec, t: (clock - rec.dep) / rec.free, route: "B" });
      } else if (clock < rec.tPinch) {
        driving.push({ rec: rec, t: (clock - rec.dep) / rec.free, route: "A" });
      } else if (clock < rec.tRelease) {
        queued.push({ rec: rec });
      } else {
        driving.push({ rec: rec, t: (clock - rec.dep - rec.delay) / rec.free, route: "A" });
      }
    }
    // head of the queue = the car that reached the bridge first
    queued.sort((a, b) => a.rec.tPinch - b.rec.tPinch);
    queued.forEach(function (q, i) { q.rank = i; q.lane = i % 2; q.slot = i >> 1; });
    return { day: day, clock: clock, home: home, driving: driving, queued: queued, arrived: arrived, lit: lit };
  }

  // where a queued car stands. QSLOT is a drawing convention (spacing), not data.
  const QSLOT = 0.0068;
  function queuePos(q) {
    const t = Math.max(0.02, PINCH_FRAC - q.slot * QSLOT);
    const p = ROUTE_A.at(t), s = scaleAt(p.y);
    const nx = -Math.sin(p.ang), ny = Math.cos(p.ang);
    const off = (q.lane === 0 ? -1 : 1) * 5.6 * s;
    return { x: p.x + nx * off, y: p.y + ny * off, ang: p.ang, s: s };
  }
  function drivePos(d) {
    const path = d.route === "B" ? ROUTE_B : ROUTE_A;
    const p = path.at(clamp(d.t, 0, 1));
    const s = scaleAt(p.y);
    const nx = -Math.sin(p.ang), ny = Math.cos(p.ang);
    const bit = (hash(d.rec.pid) & 1) ? 1 : -1;
    const off = bit * (d.route === "B" ? 4.4 : 5.2) * s;
    return { x: p.x + nx * off, y: p.y + ny * off, ang: p.ang, s: s };
  }
  // where is one traveler on screen right now
  function locate(world, day, clock, pid) {
    const rec = world.trav[pid] && world.trav[pid].days[day];
    const plot = world.plots[pid];
    if (!rec) return plot ? { kind: "home", x: plot.x, y: plot.top, s: plot.s } : null;
    if (clock < rec.dep) return { kind: "home", x: plot.x, y: plot.top, s: plot.s };
    if (clock >= rec.arrive) {
      const c = world.district[pid];
      return { kind: "arrived", x: c.x, y: c.y, s: 1 };
    }
    if (rec.route === "A" && clock >= rec.tPinch && clock < rec.tRelease) {
      const st = stateAt(world, day, clock);
      const q = st.queued.find((x) => x.rec.pid === pid);
      if (q) return Object.assign({ kind: "queued" }, queuePos(q));
    }
    const t = rec.route === "B" || clock < rec.tPinch
      ? (clock - rec.dep) / rec.free
      : (clock - rec.dep - rec.delay) / rec.free;
    return Object.assign({ kind: "driving" }, drivePos({ rec: rec, t: t, route: rec.route }));
  }

  /* ============================================================= sprites */

  /* Glow sprites are pre-rendered at a ladder of fixed diameters and always
   * blitted 1:1 (three-argument drawImage). This matters a lot: any resampling
   * — even a 1.1x one, and even a 1:1 one written as the nine-argument call —
   * drops the rasteriser onto its filtered path, which is 10-80x slower per
   * blit. Quantising a soft blob's radius to the nearest ladder step is
   * invisible; paying for the filter 1,200 times a frame is not. */
  const MIP = [8, 11, 15, 20, 27, 36, 48, 64, 86, 114, 152, 202, 256];

  // plain soft blob
  const GLOW_STOPS = [[0, 1], [0.45, 0.42], [1, 0]];
  // bright core plus a long tail — one blit of this reads like a core blob and
  // a wide bloom stacked, which is what a lit window actually looks like and
  // saves two thirds of the per-frame blits
  const LAMP_STOPS = [[0, 1], [0.07, 0.86], [0.17, 0.44], [0.36, 0.17], [0.66, 0.05], [1, 0]];

  function glowSet(col, stops) {
    const S = stops || GLOW_STOPS;
    return MIP.map(function (n) {
      const c = nc(n, n), x = c.getContext("2d");
      const g = x.createRadialGradient(n / 2, n / 2, 0, n / 2, n / 2, n / 2);
      for (const s of S) g.addColorStop(s[0], rgb(col, s[1]));
      x.fillStyle = g; x.fillRect(0, 0, n, n);
      return c;
    });
  }
  function shadowSet() {
    return MIP.map(function (n) {
      const c = nc(n, n), x = c.getContext("2d");
      const g = x.createRadialGradient(n / 2, n / 2, 0, n / 2, n / 2, n / 2);
      g.addColorStop(0, "rgba(0,0,0,0.66)");
      g.addColorStop(0.5, "rgba(0,0,0,0.30)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = g; x.fillRect(0, 0, n, n);
      return c;
    });
  }
  function buildSprites() {
    return {
      amber: glowSet([255, 176, 78]),
      lamp: glowSet([255, 190, 106], LAMP_STOPS),
      deskLamp: glowSet([255, 200, 126], LAMP_STOPS),
      amberCore: glowSet([255, 244, 214]),
      cyan: glowSet([64, 208, 230]),
      cyanCore: glowSet([178, 246, 255]),
      brake: glowSet([255, 84, 56]),
      hot: glowSet([255, 104, 62]),
      white: glowSet([255, 236, 206]),
      shadow: shadowSet(),
    };
  }
  // nearest ladder step for a wanted diameter
  function mipOf(d) {
    let i = 0;
    while (i < MIP.length - 1 && MIP[i + 1] < d) i++;
    if (i < MIP.length - 1 && (d - MIP[i]) > (MIP[i + 1] - d)) i++;
    return i;
  }
  function blitGlow(ctx, set, x, y, r, a) {
    if (a <= 0.006 || r <= 1.5) return;
    ctx.globalAlpha = a > 1 ? 1 : a;
    const d = r * 2;
    if (d > 256) { ctx.drawImage(set[MIP.length - 1], x - r, y - r, d, d); return; }
    const i = mipOf(d), n = MIP[i];
    ctx.drawImage(set[i], x - n / 2, y - n / 2);
  }

  /* ===================================================== static scenery
   * Drawn twice — once at full night, once at full morning — then cross-faded
   * at render time. The palette is linear in L, so the cross-fade reproduces
   * the intermediate light almost exactly at a fraction of the cost. */

  function drawSky(ctx, P) {
    const g = ctx.createLinearGradient(0, -60, 0, HORIZON + 210);
    g.addColorStop(0.00, rgb(P.skyTop));
    g.addColorStop(0.34, rgb(mix(P.skyTop, P.skyMid, 0.55)));
    g.addColorStop(0.62, rgb(P.skyMid));
    g.addColorStop(0.86, rgb(mix(P.skyMid, P.skyLow, 0.7)));
    g.addColorStop(1.00, rgb(P.skyLow));
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, HORIZON + 210);

    const sg = ctx.createRadialGradient(1610, HORIZON - 4, 0, 1610, HORIZON - 4, 720);
    sg.addColorStop(0, rgb(P.skyGlow, 0.85 * (0.35 + 0.65 * P.L)));
    sg.addColorStop(0.35, rgb(P.skyGlow, 0.22 * (0.3 + 0.7 * P.L)));
    sg.addColorStop(1, rgb(P.skyGlow, 0));
    ctx.fillStyle = sg; ctx.fillRect(0, 0, W, HORIZON + 210);

    if (P.night > 0.5) {
      for (let i = 0; i < 190; i++) {
        const r = mulberry32(0x77 + i * 9176);
        const x = r() * W, y = r() * (HORIZON - 10);
        ctx.fillStyle = rgb([220, 230, 255], (0.10 + r() * 0.62) * (1 - y / HORIZON) * P.night);
        ctx.fillRect(x, y, 1.4, 1.4);
      }
    }
    for (let i = 0; i < 7; i++) {
      const r = mulberry32(0x51f0 + i * 977)();
      const y = 26 + r * 118, w = 260 + r * 620, h = 8 + r * 16, x = -120 + ((i * 331) % 2100);
      const cg = ctx.createLinearGradient(x, y, x + w, y);
      const cc = mix(P.skyMid, [255, 226, 198], 0.5 + 0.4 * P.L);
      cg.addColorStop(0, rgb(cc, 0));
      cg.addColorStop(0.5, rgb(cc, 0.10 + 0.13 * P.L));
      cg.addColorStop(1, rgb(cc, 0));
      ctx.fillStyle = cg;
      ctx.beginPath(); ctx.ellipse(x + w / 2, y, w / 2, h, 0, 0, 7); ctx.fill();
    }
  }

  function drawTerrain(ctx, P) {
    const ridges = [[0, 26, -6, P.ridge, 1], [1, 18, 16, P.ridge2, 0.85]];
    for (const rd of ridges) {
      ctx.beginPath(); ctx.moveTo(-10, HORIZON + 60);
      for (let x = -10; x <= W + 10; x += 12) {
        const y = HORIZON + rd[2] - rd[1] * (Math.sin(x / 340 + rd[0] * 2.1) * 0.6 + Math.sin(x / 137 + rd[0] * 5.7) * 0.4);
        ctx.lineTo(x, y);
      }
      ctx.lineTo(W + 10, HORIZON + 90); ctx.closePath();
      ctx.fillStyle = rgb(rd[3], rd[4]); ctx.fill();
    }
    const g = ctx.createLinearGradient(0, HORIZON, 0, H);
    g.addColorStop(0, rgb(mix(P.ground, P.haze, 0.42)));
    g.addColorStop(0.22, rgb(P.ground));
    g.addColorStop(0.68, rgb(mix(P.ground, P.groundNear, 0.55)));
    g.addColorStop(1, rgb(P.groundNear));
    ctx.fillStyle = g; ctx.fillRect(0, HORIZON - 2, W, H - HORIZON + 2);

    const sw = ctx.createLinearGradient(W * 0.35, HORIZON, W * 1.05, H * 0.9);
    sw.addColorStop(0, rgb(P.fieldWarm, 0));
    sw.addColorStop(0.55, rgb(P.fieldWarm, 0.13 * (0.3 + 0.7 * P.L)));
    sw.addColorStop(1, rgb(P.fieldWarm, 0));
    ctx.fillStyle = sw; ctx.fillRect(0, HORIZON, W, H - HORIZON);

    ctx.save();
    for (let i = 0; i < 54; i++) {
      const r = mulberry32(0x2ab1 + i * 7919);
      const y = HORIZON + 14 + Math.pow(r(), 0.72) * (H - HORIZON - 4);
      const x = -60 + r() * (W + 120);
      const s = scaleAt(y);
      const rx = (54 + r() * 176) * s, ry = rx * (0.22 + r() * 0.18);
      const warm = r();
      const col = warm > 0.62 ? mix(P.fieldWarm, [216, 186, 108], 0.5 * P.L)
        : (warm > 0.42 ? P.fieldWarm : (warm > 0.20 ? P.field : mix(P.field, [22, 46, 30], 0.55)));
      ctx.save();
      ctx.translate(x, y); ctx.rotate((r() - 0.5) * 0.36); ctx.scale(1, ry / rx);
      ctx.beginPath();
      const n = 6 + Math.floor(r() * 3);
      for (let k = 0; k < n; k++) {
        const a = (k / n) * Math.PI * 2, rr = rx * (0.72 + r() * 0.46);
        const px = Math.cos(a) * rr, py = Math.sin(a) * rr;
        if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = rgb(col, 0.14 + 0.17 * r());
      ctx.filter = "blur(" + (10 + 22 * r()) + "px)";
      ctx.fill();
      ctx.restore();
    }
    ctx.lineCap = "round";
    for (let i = 0; i < 78; i++) {
      const r = mulberry32(0x7de3 + i * 3571);
      const y = HORIZON + 18 + Math.pow(r(), 0.66) * (H - HORIZON - 24);
      const x = -40 + r() * (W + 80);
      const sc = scaleAt(y);
      const len = (60 + r() * 230) * sc;
      const ang = Math.sin(x / 620) * 0.30 + Math.cos(y / 380) * 0.22 + (r() - 0.5) * 0.16;
      const x2 = x + Math.cos(ang) * len, y2 = y + Math.sin(ang) * len;
      ctx.strokeStyle = rgb(mix(P.ground, [8, 20, 12], 0.60), 0.13 + 0.12 * r());
      ctx.lineWidth = (1.4 + 1.6 * r()) * sc;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x2, y2); ctx.stroke();
      if (r() > 0.55) {
        ctx.strokeStyle = rgb(mix(P.field, [255, 232, 190], 0.45 * P.L), 0.10 + 0.10 * P.L);
        ctx.lineWidth = 1.1 * sc;
        ctx.beginPath(); ctx.moveTo(x, y - 1.6 * sc); ctx.lineTo(x2, y2 - 1.6 * sc); ctx.stroke();
      }
    }
    ctx.restore();

    const hz = ctx.createLinearGradient(0, HORIZON - 6, 0, HORIZON + 210);
    hz.addColorStop(0, rgb(P.haze, 0.34 + 0.08 * P.L));
    hz.addColorStop(0.5, rgb(P.haze, 0.10 + 0.04 * P.L));
    hz.addColorStop(1, rgb(P.haze, 0));
    ctx.fillStyle = hz; ctx.fillRect(0, HORIZON - 6, W, 216);
  }

  function drawTree(ctx, P, x, y, s, seed) {
    const r = mulberry32(seed);
    const h = (10 + r() * 8) * s, w = (5 + r() * 4) * s;
    ctx.save();
    ctx.fillStyle = rgb(P.shadow, 0.30);
    ctx.filter = "blur(3px)";
    ctx.beginPath(); ctx.ellipse(x - w * 1.6, y + 1.2 * s, w * 1.9, w * 0.7, 0, 0, 7); ctx.fill();
    ctx.restore();
    ctx.strokeStyle = rgb(mix(P.ground, [30, 20, 14], 0.7), 0.9);
    ctx.lineWidth = 1.4 * s; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - h * 0.45); ctx.stroke();
    ctx.fillStyle = rgb(mix([26, 42, 30], [56, 78, 48], P.L));
    ctx.beginPath(); ctx.ellipse(x, y - h * 0.72, w, h * 0.42, 0, 0, 7); ctx.fill();
    ctx.fillStyle = rgb(mix([40, 56, 44], [128, 146, 84], P.L), 0.85);
    ctx.beginPath(); ctx.ellipse(x + w * 0.30, y - h * 0.82, w * 0.62, h * 0.28, 0, 0, 7); ctx.fill();
  }

  function drawPond(ctx, P) {
    const x = POND.x, y = POND.y, rx = POND.rx, ry = POND.ry;
    ctx.save();
    ctx.filter = "blur(5px)";
    ctx.fillStyle = rgb(P.shadow, 0.30);
    ctx.beginPath(); ctx.ellipse(x - 6, y + 5, rx * 1.06, ry * 1.12, 0.06, 0, 7); ctx.fill();
    ctx.restore();
    const g = ctx.createLinearGradient(x, y - ry, x, y + ry);
    g.addColorStop(0, rgb(mix(P.water, P.haze, 0.24)));
    g.addColorStop(1, rgb(mix(P.water, [0, 0, 0], 0.28)));
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.ellipse(x, y, rx, ry, 0.06, 0, 7); ctx.fill();
    ctx.strokeStyle = rgb(mix(P.ground, [255, 232, 196], 0.4 * P.L + 0.06), 0.4);
    ctx.lineWidth = 1.4; ctx.stroke();
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < 22; i++) {
      const r = mulberry32(0x66aa + i * 1531);
      const px = x + (r() - 0.5) * rx * 1.7, py = y + (r() - 0.5) * ry * 1.5;
      ctx.strokeStyle = rgb(P.waterHi, 0.05 + 0.10 * r() * (0.2 + 0.8 * P.L));
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(px - 10 - r() * 14, py); ctx.lineTo(px + 10 + r() * 14, py); ctx.stroke();
    }
    ctx.restore();
  }

  function riverPolys(k) {
    const left = [], right = [];
    for (const p of RIVER.pts) {
      const w = riverWidth(p[1]) * (k || 1) / 2;
      left.push([p[0] - w, p[1]]); right.push([p[0] + w, p[1]]);
    }
    return { left: left, right: right };
  }
  function riverShape(ctx, k) {
    const s = riverPolys(k);
    ctx.beginPath();
    ctx.moveTo(s.left[0][0], s.left[0][1]);
    for (const p of s.left) ctx.lineTo(p[0], p[1]);
    for (let i = s.right.length - 1; i >= 0; i--) ctx.lineTo(s.right[i][0], s.right[i][1]);
    ctx.closePath();
  }
  function drawRiver(ctx, P) {
    ctx.save();
    ctx.filter = "blur(13px)";
    riverShape(ctx, 1.75);
    ctx.fillStyle = rgb(mix(P.ground, [196, 180, 138], 0.02 + 0.56 * P.L), 0.72); ctx.fill();
    riverShape(ctx, 1.30);
    ctx.fillStyle = rgb(mix(P.ground, [210, 196, 156], 0.03 + 0.58 * P.L), 0.55); ctx.fill();
    ctx.restore();

    ctx.save();
    riverShape(ctx, 1.0);
    const g = ctx.createLinearGradient(0, HORIZON, 0, H);
    g.addColorStop(0, rgb(mix(P.water, P.haze, 0.44)));
    g.addColorStop(0.45, rgb(mix(P.water, [40, 96, 92], 0.34)));
    g.addColorStop(1, rgb(mix(P.water, [0, 8, 18], 0.42)));
    ctx.fillStyle = g; ctx.fill();
    ctx.clip();
    ctx.strokeStyle = rgb(mix(P.water, [0, 10, 22], 0.60), 0.75);
    ctx.lineWidth = 14; ctx.stroke();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < 240; i++) {
      const r = mulberry32(0x1cd9 + i * 6151);
      const p = RIVER.at(r());
      const w = riverWidth(p.y) / 2;
      const x = p.x + (r() - 0.5) * w * 1.85;
      const len = (5 + r() * 22) * scaleAt(p.y);
      ctx.strokeStyle = rgb(P.waterHi, 0.05 + 0.17 * r() * (0.28 + 0.72 * P.L));
      ctx.lineWidth = 1.1 * scaleAt(p.y);
      ctx.beginPath(); ctx.moveTo(x - len / 2, p.y); ctx.lineTo(x + len / 2, p.y + (r() - 0.5) * 2.4); ctx.stroke();
    }
    ctx.restore();

    const s = riverPolys(1);
    ctx.save();
    ctx.strokeStyle = rgb(mix([255, 236, 200], P.waterHi, 0.35), 0.34 + 0.24 * P.L);
    ctx.lineWidth = 1.6;
    ctx.beginPath(); s.right.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
    ctx.strokeStyle = rgb(mix(P.ground, [0, 0, 0], 0.4), 0.35);
    ctx.beginPath(); s.left.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
    ctx.restore();
  }

  const A_W = function (t, p) {
    const s = scaleAt(p.y);
    const d = Math.abs(t - PINCH_FRAC);
    return 36 * s * (1 - 0.56 * Math.exp(-Math.pow(d / 0.085, 2)));
  };
  const B_W = (t, p) => 26 * scaleAt(p.y);

  function drawRoadVariable(ctx, pathObj, wFn, fill, steps) {
    const n = steps || 260, L = [], R = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n, p = pathObj.at(t), w = wFn(t, p) / 2;
      const nx = -Math.sin(p.ang), ny = Math.cos(p.ang);
      L.push([p.x + nx * w, p.y + ny * w]); R.push([p.x - nx * w, p.y - ny * w]);
    }
    ctx.beginPath();
    L.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    for (let i = R.length - 1; i >= 0; i--) ctx.lineTo(R[i][0], R[i][1]);
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
  }

  function drawRoads(ctx, P) {
    ctx.save();
    ctx.filter = "blur(7px)";
    drawRoadVariable(ctx, ROUTE_A, (t, p) => A_W(t, p) * 1.7, rgb(P.shadow, 0.42));
    drawRoadVariable(ctx, ROUTE_B, (t, p) => B_W(t, p) * 1.7, rgb(P.shadow, 0.40));
    ctx.restore();
    drawRoadVariable(ctx, ROUTE_A, (t, p) => A_W(t, p) + 6 * scaleAt(p.y), rgb(mix(P.asphalt, [0, 0, 0], 0.45)));
    drawRoadVariable(ctx, ROUTE_B, (t, p) => B_W(t, p) + 7 * scaleAt(p.y), rgb(mix(P.asphalt, [4, 30, 40], 0.62)));
    drawRoadVariable(ctx, ROUTE_A, A_W, rgb(P.asphalt));
    drawRoadVariable(ctx, ROUTE_B, B_W, rgb(mix(P.asphalt, P.cyan, 0.10 + 0.14 * P.L)));

    const dash = function (pathObj, col, alpha, phase) {
      ctx.save(); ctx.lineCap = "butt";
      const steps = 420;
      for (let i = 0; i < steps; i++) {
        if ((i + phase) % 4 > 1.6) continue;
        const a = pathObj.at(i / steps), b = pathObj.at((i + 1) / steps);
        ctx.strokeStyle = rgb(col, alpha);
        ctx.lineWidth = 1.5 * scaleAt(a.y);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.restore();
    };
    dash(ROUTE_A, mix(P.asphaltEdge, [255, 240, 210], 0.5), 0.10 + 0.22 * P.L, 0);
    dash(ROUTE_B, mix(P.cyanCore, [255, 255, 255], 0.2), 0.08 + 0.18 * P.L, 2);

    const rimSide = function (pathObj, wFn, col, alpha, sign) {
      ctx.save(); ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const t = i / 300, p = pathObj.at(t), w = wFn(t, p) / 2;
        const nx = -Math.sin(p.ang), ny = Math.cos(p.ang);
        const x = p.x + sign * nx * w, y = p.y + sign * ny * w;
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
      ctx.strokeStyle = rgb(col, alpha); ctx.lineWidth = 1.7; ctx.stroke();
      ctx.restore();
    };
    rimSide(ROUTE_A, A_W, mix([255, 226, 186], P.asphaltEdge, 1 - P.L * 0.8), 0.42, -1);
    rimSide(ROUTE_B, B_W, P.cyanCore, 0.22 + 0.34 * P.L, -1);
    rimSide(ROUTE_B, B_W, P.cyanCore, 0.12 + 0.24 * P.L, 1);
  }

  function drawJunction(ctx, P) {
    const x = HOME_HUB[0], y = HOME_HUB[1], s = scaleAt(y);
    ctx.save();
    ctx.filter = "blur(6px)";
    ctx.fillStyle = rgb(P.shadow, 0.34);
    ctx.beginPath(); ctx.ellipse(x - 6 * s, y + 6 * s, 30 * s, 15 * s, 0, 0, 7); ctx.fill();
    ctx.restore();
    ctx.fillStyle = rgb(mix(P.asphalt, [0, 0, 0], 0.34));
    ctx.beginPath(); ctx.ellipse(x, y, 30 * s, 15 * s, 0, 0, 7); ctx.fill();
    ctx.fillStyle = rgb(P.asphalt);
    ctx.beginPath(); ctx.ellipse(x, y, 26 * s, 13 * s, 0, 0, 7); ctx.fill();
    ctx.fillStyle = rgb(mix(P.field, [120, 140, 90], 0.4), 0.9);
    ctx.beginPath(); ctx.ellipse(x, y, 11 * s, 5.5 * s, 0, 0, 7); ctx.fill();
    ctx.strokeStyle = rgb(mix([255, 232, 196], P.asphaltEdge, 1 - P.L * 0.7), 0.34);
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.ellipse(x, y, 26 * s, 13 * s, 0, 0, 7); ctx.stroke();
    drawTree(ctx, P, x + 2 * s, y - 1 * s, s * 0.8, 991);
  }

  function drawBridge(ctx, P, pathObj, tc, halfT, wFn, cyanTint, lamps) {
    const N = 26, samples = [];
    for (let i = 0; i <= N; i++) {
      const t = tc - halfT + (2 * halfT) * (i / N);
      const p = pathObj.at(t);
      samples.push({ t: t, p: p, s: scaleAt(p.y), w: wFn(t, p) / 2 + 4 * scaleAt(p.y) });
    }
    const edge = (k, extra) => samples.map(function (m) {
      const nx = -Math.sin(m.p.ang), ny = Math.cos(m.p.ang);
      return [m.p.x + k * nx * (m.w + (extra || 0) * m.s), m.p.y + k * ny * (m.w + (extra || 0) * m.s)];
    });
    const L = edge(1, 0), R = edge(-1, 0);
    const ribbon = function (Le, Re, fill) {
      ctx.beginPath();
      Le.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      for (let i = Re.length - 1; i >= 0; i--) ctx.lineTo(Re[i][0], Re[i][1]);
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    };
    for (const k of [0.3, 0.7]) {
      const m = samples[Math.round(N * k)];
      const nx = -Math.sin(m.p.ang), ny = Math.cos(m.p.ang);
      ctx.fillStyle = rgb(mix(P.wallShade, [8, 12, 22], 0.68), 0.95);
      ctx.beginPath();
      ctx.ellipse(m.p.x + nx * m.w * 0.9, m.p.y + ny * m.w * 0.9, m.w * 0.55, m.w * 0.26, m.p.ang, 0, 7);
      ctx.fill();
    }
    ctx.save(); ctx.filter = "blur(6px)";
    ribbon(edge(1, 5.5), edge(-1, -1), "rgba(0,0,0,0.45)");
    ctx.restore();
    ribbon(L, R, rgb(mix(P.asphalt, [198, 188, 170], 0.22 + 0.14 * P.L)));
    ctx.save();
    for (const k of [0.16, 0.42, 0.58, 0.84]) {
      const idx = Math.round(N * k);
      ctx.strokeStyle = rgb(mix(P.asphalt, [0, 0, 0], 0.5), 0.45);
      ctx.lineWidth = 1.2 * samples[idx].s;
      ctx.beginPath(); ctx.moveTo(L[idx][0], L[idx][1]); ctx.lineTo(R[idx][0], R[idx][1]); ctx.stroke();
    }
    ctx.restore();
    const railCol = cyanTint ? mix(P.wallSun, P.cyan, 0.55) : mix([255, 240, 212], P.wallSun, 0.28);
    const rail = function (E, alpha) {
      ctx.save(); ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.beginPath(); E.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.strokeStyle = rgb(railCol, alpha); ctx.lineWidth = 2.4 * samples[N >> 1].s; ctx.stroke();
      ctx.restore();
      for (let i = 1; i < N; i += 3) {
        ctx.fillStyle = rgb(railCol, alpha * 0.92);
        ctx.fillRect(E[i][0] - samples[i].s, E[i][1] - 4.6 * samples[i].s, 2 * samples[i].s, 4.8 * samples[i].s);
      }
    };
    rail(R, 0.95); rail(L, 0.55);
    for (const idx of [0, N]) for (const E of [L, R]) {
      const sc = samples[Math.min(idx, N)].s;
      ctx.fillStyle = rgb(cyanTint ? P.cyanCore : P.amberCore, 0.5 + 0.45 * P.night);
      ctx.beginPath(); ctx.arc(E[idx][0], E[idx][1] - 6.4 * sc, 1.5 * sc, 0, 7); ctx.fill();
      lamps.push({ x: E[idx][0], y: E[idx][1] - 6.4 * sc, r: 13 * sc, cyan: !!cyanTint });
    }
  }

  function drawHouseShell(ctx, P, plot) {
    const g = plot.g, s = plot.s;
    if (plot.area !== "inner city") {
      ctx.save();
      ctx.filter = "blur(" + Math.max(2.5, 4 * s) + "px)";
      ctx.fillStyle = rgb(mix(P.field, [150, 156, 104], 0.35), 0.30);
      ctx.beginPath(); ctx.ellipse(plot.x, plot.y + g.d * 0.25, g.w * 1.05, g.d * 0.95, 0, 0, 7); ctx.fill();
      ctx.restore();
    }
    ctx.save();
    ctx.filter = "blur(" + Math.max(2, 3.0 * s) + "px)";
    ctx.fillStyle = rgb(P.shadow, 0.26 + 0.20 * P.L);
    const sh = g.hh * 1.05;
    ctx.beginPath();
    ctx.moveTo(plot.x - g.hw, plot.y);
    ctx.lineTo(plot.x - g.hw - sh, plot.y + sh * 0.40);
    ctx.lineTo(plot.x - sh, plot.y + g.hd + sh * 0.40);
    ctx.lineTo(plot.x + g.hw * 0.35, plot.y + g.hd * 1.02);
    ctx.closePath(); ctx.fill();
    ctx.restore();

    const roofRGB = hexToRgb(plot.roof);
    const roofLit = mix(mix([15, 18, 27], roofRGB, P.roofMul), [255, 218, 176], 0.16 * P.L);
    const roofShd = mix(roofLit, [8, 10, 20], 0.40);
    const poly = function (pts, fill, stroke, lw) {
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
    };
    poly([g.P3, g.P2, g.Q2, g.Q3], rgb(P.wallShade));
    poly([g.P2, g.P1, g.Q1, g.Q2], rgb(P.wallSun));

    if (plot.form.flat) {
      poly([g.Q0, g.Q1, g.Q2, g.Q3], rgb(mix(roofLit, [86, 92, 100], 0.55)));
      poly([g.Q0, g.Q1, g.Q2, g.Q3], null, rgb(mix(roofLit, [255, 250, 240], 0.35 * P.L + 0.1), 0.55), 1);
    } else {
      const rh = g.hh * 0.72 + 2.6 * s;
      const midp = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - rh];
      let A, B, gab1, gab2, planeFar, planeNear, farLit;
      if (plot.rot > 0) {
        A = midp(g.Q0, g.Q3); B = midp(g.Q1, g.Q2);
        gab1 = [g.Q0, g.Q3, A]; gab2 = [g.Q1, g.Q2, B];
        planeFar = [g.Q0, g.Q1, B, A]; planeNear = [g.Q3, g.Q2, B, A]; farLit = true;
      } else {
        A = midp(g.Q0, g.Q1); B = midp(g.Q3, g.Q2);
        gab1 = [g.Q0, g.Q1, A]; gab2 = [g.Q3, g.Q2, B];
        planeFar = [g.Q0, g.Q3, B, A]; planeNear = [g.Q1, g.Q2, B, A]; farLit = false;
      }
      poly(gab1, rgb(mix(P.wallSun, [0, 0, 0], 0.10)));
      poly(gab2, rgb(mix(P.wallShade, [255, 236, 206], 0.10 * P.L)));
      poly(planeFar, rgb(farLit ? roofLit : roofShd));
      poly(planeNear, rgb(farLit ? roofShd : roofLit));
      ctx.strokeStyle = rgb(mix(roofLit, P.night > 0.5 ? [150, 176, 220] : [255, 238, 206], 0.5 + 0.3 * P.night), 0.5 + 0.30 * P.night);
      ctx.lineWidth = 1.0 + 0.3 * P.night;
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
      if (plot.chim && s > 0.6) {
        ctx.fillStyle = rgb(mix(roofShd, [0, 0, 0], 0.25));
        const cxp = A[0] + (B[0] - A[0]) * 0.7, cyp = A[1] + (B[1] - A[1]) * 0.7;
        ctx.fillRect(cxp - 1.1 * s, cyp - 5.4 * s, 2.2 * s, 5.6 * s);
      }
    }
    for (const w of plot.wins) {
      ctx.fillStyle = rgb(mix(w[2] ? P.wallSun : P.wallShade, [9, 13, 24], 0.70), 0.88);
      ctx.fillRect(w[0] - plot.ww / 2, w[1] - plot.wh / 2, plot.ww, plot.wh);
    }
    if (plot.tree) {
      drawTree(ctx, P, plot.x + g.hw * (1.12 + plot.tj[0] * 0.30),
        plot.y + g.hd * (0.85 + plot.tj[1] * 0.5), s * 0.9, hash(plot.pid + "t"));
    }
  }

  function drawDistrictShell(ctx, P, world) {
    ctx.save();
    ctx.filter = "blur(22px)";
    ctx.fillStyle = rgb(P.shadow, 0.40);
    ctx.beginPath(); ctx.ellipse(PLAZA.x, PLAZA.y + 30, 340, 106, 0, 0, 7); ctx.fill();
    ctx.restore();
    const pg = ctx.createRadialGradient(PLAZA.x, PLAZA.y, 20, PLAZA.x, PLAZA.y, 350);
    pg.addColorStop(0, rgb(mix(P.asphalt, [200, 190, 170], 0.30 * P.L + 0.05), 0.55));
    pg.addColorStop(1, rgb(mix(P.asphalt, [200, 190, 170], 0.2), 0));
    ctx.fillStyle = pg;
    ctx.beginPath(); ctx.ellipse(PLAZA.x, PLAZA.y, 348, 112, 0, 0, 7); ctx.fill();

    const order = SLABS.map((s, i) => ({ s: s, i: i })).sort((a, b) => a.s.y - b.s.y);
    for (const o of order) {
      const s = o.s, br = mulberry32(0x77c1 + o.i * 8191);
      const tintK = br() * 0.55;
      const hw = s.w / 2, hd = hw * 0.44, hh = s.h, x = s.x, y = s.y;
      const P0 = [x, y - hd], P1 = [x + hw, y], P2 = [x, y + hd], P3 = [x - hw, y];
      ctx.save(); ctx.filter = "blur(9px)";
      ctx.fillStyle = rgb(P.shadow, 0.46);
      ctx.beginPath();
      ctx.moveTo(P3[0], P3[1]); ctx.lineTo(P3[0] - hh * 0.95, P3[1] + hh * 0.36);
      ctx.lineTo(P2[0] - hh * 0.95 + hw * 0.5, P2[1] + hh * 0.36); ctx.lineTo(P1[0], P1[1]);
      ctx.closePath(); ctx.fill();
      ctx.restore();
      const face = function (a, b, fill) {
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.lineTo(b[0], b[1] - hh); ctx.lineTo(a[0], a[1] - hh); ctx.closePath();
        ctx.fillStyle = fill; ctx.fill();
      };
      face(P3, P2, rgb(mix(mix([18, 22, 34], [30, 36, 52], P.L), [46, 40, 44], tintK)));
      face(P2, P1, rgb(mix(mix([26, 31, 44], [54, 60, 76], P.L), [74, 62, 58], tintK)));
      ctx.beginPath();
      ctx.moveTo(P0[0], P0[1] - hh); ctx.lineTo(P1[0], P1[1] - hh);
      ctx.lineTo(P2[0], P2[1] - hh); ctx.lineTo(P3[0], P3[1] - hh);
      ctx.closePath();
      ctx.fillStyle = rgb(mix([34, 40, 54], [86, 92, 106], P.L)); ctx.fill();
      ctx.strokeStyle = rgb(mix([120, 128, 142], [230, 216, 194], P.L), 0.50); ctx.lineWidth = 1; ctx.stroke();
      const rc = [x, y - hh];
      ctx.fillStyle = rgb(mix([30, 35, 46], [74, 78, 90], P.L), 0.98);
      const bwr = hw * (0.22 + br() * 0.18), bhr = 4 + br() * 7;
      ctx.beginPath();
      ctx.moveTo(rc[0] - bwr, rc[1] - hd * 0.2); ctx.lineTo(rc[0], rc[1] - hd * 0.2 - hd * 0.5);
      ctx.lineTo(rc[0] + bwr, rc[1] - hd * 0.2); ctx.lineTo(rc[0], rc[1] - hd * 0.2 + hd * 0.5);
      ctx.closePath(); ctx.fill();
      ctx.fillStyle = rgb(mix([38, 44, 56], [104, 108, 120], P.L), 0.98);
      ctx.fillRect(rc[0] - bwr, rc[1] - hd * 0.2 - bhr, bwr * 2, bhr);
      ctx.fillStyle = rgb(mix([54, 60, 74], [150, 152, 160], P.L), 0.9);
      ctx.fillRect(rc[0] - bwr, rc[1] - hd * 0.2 - bhr - 1.5, bwr * 2, 1.5);
      if (br() > 0.55) {
        ctx.strokeStyle = rgb(mix([70, 76, 90], [150, 156, 168], P.L), 0.9); ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.moveTo(rc[0] + bwr * 0.4, rc[1] - hd * 0.2 - bhr);
        ctx.lineTo(rc[0] + bwr * 0.4, rc[1] - hd * 0.2 - bhr - 16); ctx.stroke();
      }
      ctx.strokeStyle = rgb(mix([255, 226, 186], P.wallSun, 0.3), 0.34 + 0.34 * P.L);
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(P1[0], P1[1]); ctx.lineTo(P1[0], P1[1] - hh); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(P2[0], P2[1]); ctx.lineTo(P2[0], P2[1] - hh); ctx.stroke();
    }
    for (const pid of world.pids) {
      const c = world.district[pid];
      ctx.fillStyle = "rgba(132,148,178,0.15)";
      ctx.fillRect(c.x - c.w / 2, c.y - c.h / 2, c.w, c.h);
    }
    for (let i = 0; i < 24; i++) {
      const r = mulberry32(0x3f7a + i * 2609);
      const a = -0.55 + r() * 2.1, rad = 250 + r() * 130;
      const x = PLAZA.x + Math.cos(a) * rad, y = PLAZA.y + 40 + Math.sin(a) * (rad * 0.30);
      drawTree(ctx, P, x, y, scaleAt(y) * (0.65 + r() * 0.5), hash("d" + i));
    }
  }

  function lightWrap(ctx, P) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(1690, 190, 0, 1690, 190, 1560);
    g.addColorStop(0, rgb([255, 208, 148], 0.20 * (0.25 + 0.75 * P.L)));
    g.addColorStop(0.45, rgb([255, 190, 132], 0.07 * (0.2 + 0.8 * P.L)));
    g.addColorStop(1, rgb([255, 190, 132], 0));
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.restore();
    ctx.save();
    ctx.globalCompositeOperation = "multiply";
    const c = ctx.createLinearGradient(0, 0, 620, H);
    c.addColorStop(0, "rgba(150,164,200," + (0.20 + 0.10 * P.night) + ")");
    c.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = c; ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }

  function tiltShift(src) {
    const out = nc(W, H), g = out.getContext("2d");
    g.drawImage(src, 0, 0);
    const bands = [
      { r: 2.2, from: 330, to: 190 }, { r: 5.0, from: 230, to: 40 },
      { r: 2.6, from: 700, to: 830 }, { r: 7.0, from: 830, to: 1080 },
    ];
    for (const b of bands) {
      const bl = blurredCopy(src, b.r);
      const layer = nc(W, H), lc = layer.getContext("2d");
      lc.drawImage(bl, 0, 0);
      lc.globalCompositeOperation = "destination-in";
      const grad = lc.createLinearGradient(0, b.from, 0, b.to);
      grad.addColorStop(0, "rgba(255,255,255,0)");
      grad.addColorStop(1, "rgba(255,255,255,1)");
      lc.fillStyle = grad;
      const y0 = Math.min(b.from, b.to), y1 = Math.max(b.from, b.to);
      lc.fillRect(0, y0, W, y1 - y0);
      g.drawImage(layer, 0, 0);
    }
    return out;
  }

  function gradeAndGrain(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const img = ctx.getImageData(0, 0, W, H), d = img.data;
    let seed = 0x1f2e3d4c;
    const con = opts.contrast, sat = opts.saturation, lift = opts.lift, amount = opts.grain;
    for (let i = 0; i < d.length; i += 4) {
      let r = (d[i] - 128) * con + 128 + lift;
      let g = (d[i + 1] - 128) * con + 128 + lift;
      let b = (d[i + 2] - 128) * con + 128 + lift;
      const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      r = l + (r - l) * sat; g = l + (g - l) * sat; b = l + (b - l) * sat;
      const t = clamp(l / 255, 0, 1);
      r += (t - 0.5) * 9; b -= (t - 0.5) * 11; g += (t - 0.5) * 2;
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const n = (((seed >>> 16) & 255) / 255 - 0.5) * amount;
      d[i] = clamp(r + n, 0, 255);
      d[i + 1] = clamp(g + n, 0, 255);
      d[i + 2] = clamp(b + n, 0, 255);
    }
    ctx.putImageData(img, 0, 0);
  }

  /* ---- build one static base at a fixed light level -------------------- */

  function buildBase(world, L) {
    const P = palette(L);
    const base = nc(W, H), ctx = base.getContext("2d");
    const lamps = [];
    drawSky(ctx, P);
    drawTerrain(ctx, P);
    drawPond(ctx, P);
    drawRiver(ctx, P);
    drawRoads(ctx, P);
    drawJunction(ctx, P);
    for (const wd of WOODS) {
      const r = mulberry32(hash(wd[0] + "_" + wd[1]));
      const n = Math.round(wd[2] / 5), pts = [];
      for (let i = 0; i < n; i++) {
        const a = r() * Math.PI * 2, dd = Math.sqrt(r()) * wd[2];
        pts.push([wd[0] + Math.cos(a) * dd, wd[1] + Math.sin(a) * dd * 0.42]);
      }
      pts.sort((a, b) => a[1] - b[1]);
      for (const p of pts) drawTree(ctx, P, p[0], p[1], scaleAt(p[1]) * (0.8 + r() * 0.5), hash((p[0] | 0) + "_" + (p[1] | 0)));
    }
    drawBridge(ctx, P, ROUTE_B, RING_BRIDGE.t, 0.030, B_W, true, lamps);
    drawBridge(ctx, P, ROUTE_A, PINCH_FRAC, 0.034, A_W, false, lamps);
    drawDistrictShell(ctx, P, world);
    for (const plot of world.plotList) drawHouseShell(ctx, P, plot);
    lightWrap(ctx, P);

    // baked bridge lamps (static light sources)
    const glow = nc(W, H), gx = glow.getContext("2d");
    gx.globalCompositeOperation = "lighter";
    for (const l of lamps) {
      const g = gx.createRadialGradient(l.x, l.y, 0, l.x, l.y, l.r);
      const col = l.cyan ? [178, 246, 255] : [255, 244, 214];
      const a = 0.07 + 0.44 * P.night;
      g.addColorStop(0, rgb(col, a)); g.addColorStop(0.45, rgb(col, a * 0.42)); g.addColorStop(1, rgb(col, 0));
      gx.fillStyle = g; gx.beginPath(); gx.arc(l.x, l.y, l.r, 0, 7); gx.fill();
    }
    const b1 = blurredCopy(glow, 5);
    ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 0.62;
    ctx.drawImage(b1, 0, 0); ctx.restore();

    const out = tiltShift(base);
    const g2 = out.getContext("2d");
    applyPost(g2);
    if (P.night > 0.001) {
      // the hour-dependent extra darkening, night only
      const vg = g2.createRadialGradient(VIG_CX, VIG_CY, VIG_R0, VIG_CX, VIG_CY, VIG_R1);
      vg.addColorStop(0, "rgba(0,0,0,0)");
      vg.addColorStop(0.56, "rgba(0,0,0,0)");
      vg.addColorStop(1, "rgba(4,7,16," + (0.16 * P.night) + ")");
      g2.fillStyle = vg; g2.fillRect(0, 0, W, H);
      const ts = g2.createLinearGradient(0, 0, 0, TOP_H);
      ts.addColorStop(0, "rgba(6,9,18," + (0.20 * P.night) + ")");
      ts.addColorStop(1, "rgba(6,9,18,0)");
      g2.fillStyle = ts; g2.fillRect(0, 0, W, TOP_H);
    }
    gradeAndGrain(out, { contrast: 1.16, saturation: 1.18, lift: 2, grain: 6.5 });
    return out;
  }

  /* ---- vignette + scrims -----------------------------------------------
   * These are baked into the two static bases, in concept-A's order: scrims
   * first, colour grade last, so the grade re-saturates the darkened corners.
   * The moving parts (cars, lit windows, glow) are drawn on top afterwards, so
   * they are dimmed analytically by postDim() with the same numbers instead of
   * being covered by a second full-screen layer. Same picture, one blit less. */

  const VIG_CX = W * 0.54, VIG_CY = H * 0.46, VIG_R0 = H * 0.26, VIG_R1 = H * 1.02;
  const TOP_H = 250, BOT_Y = H - 340, BOT_H = 340;

  function applyPost(ctx) {
    const vg = ctx.createRadialGradient(VIG_CX, VIG_CY, VIG_R0, VIG_CX, VIG_CY, VIG_R1);
    vg.addColorStop(0, "rgba(0,0,0,0)");
    vg.addColorStop(0.56, "rgba(0,0,0,0.20)");
    vg.addColorStop(1, "rgba(4,7,16,0.62)");
    ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);
    const ts = ctx.createLinearGradient(0, 0, 0, TOP_H);
    ts.addColorStop(0, "rgba(6,9,18,0.50)"); ts.addColorStop(1, "rgba(6,9,18,0)");
    ctx.fillStyle = ts; ctx.fillRect(0, 0, W, TOP_H);
    const bs = ctx.createLinearGradient(0, BOT_Y, 0, H);
    bs.addColorStop(0, "rgba(6,9,18,0)");
    bs.addColorStop(0.5, "rgba(6,9,18,0.32)");
    bs.addColorStop(1, "rgba(6,9,18,0.84)");
    ctx.fillStyle = bs; ctx.fillRect(0, BOT_Y, W, BOT_H);
  }

  // how much light survives the baked vignette/scrims at (x, y)
  function postDim(x, y) {
    const r = Math.hypot(x - VIG_CX, y - VIG_CY);
    const t = clamp((r - VIG_R0) / (VIG_R1 - VIG_R0), 0, 1);
    let v = t <= 0.56 ? t / 0.56 * 0.20 : 0.20 + (t - 0.56) / 0.44 * 0.42;
    if (y < TOP_H) v += 0.50 * (1 - y / TOP_H);
    if (y > BOT_Y) {
      const u = (y - BOT_Y) / BOT_H;
      v += u < 0.5 ? u * 0.64 : 0.32 + (u - 0.5) * 1.04;
    }
    return clamp(1 - v, 0.05, 1);
  }

  /* ---- staged build so a browser can show progress --------------------- */

  function buildStaticSteps(world) {
    const steps = [
      { label: "night pass", run: function () { world._staticNight = buildBase(world, 0); } },
      { label: "morning pass", run: function () { world._staticDay = buildBase(world, 1); } },
      { label: "light and lenses", run: function () {
        world._sprites = buildSprites();
        world._static = true;
      } },
    ];
    return steps;
  }
  function buildStatic(world) {
    for (const s of buildStaticSteps(world)) s.run();
    return world;
  }

  /* ============================================================ the frame */

  const NIGHT_INK = [6, 9, 18];
  function drawCar(ctx, sp, P, c) {
    const s = c.s, ang = c.ang, x = c.x, y = c.y;
    const L = 13.0 * s, Wd = 6.6 * s;
    const dim = c.dim === undefined ? 1 : c.dim, sh = 1 - dim;
    const tx = Math.cos(ang), ty = Math.sin(ang), nx = -Math.sin(ang), ny = Math.cos(ang);
    const pt = (a, b) => [x + tx * a + nx * b, y + ty * a + ny * b];

    blitGlow(ctx, sp.shadow, x - 2.4 * s, y + 2.2 * s, L * 0.62, (0.40 + 0.16 * P.L) * dim);
    ctx.globalAlpha = 1;

    const body = c.route === "B" ? mix(P.cyan, [18, 40, 52], 0.55) : c.hue;
    const a = pt(-L / 2, -Wd / 2), b = pt(L / 2, -Wd / 2), cc = pt(L / 2, Wd / 2), d = pt(-L / 2, Wd / 2);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(cc[0], cc[1]); ctx.lineTo(d[0], d[1]);
    ctx.closePath();
    ctx.fillStyle = rgb(mix(mix(body, [10, 12, 20], 0.50 - 0.42 * P.L), NIGHT_INK, sh)); ctx.fill();
    const e = pt(-L * 0.22, -Wd * 0.30), f = pt(L * 0.30, -Wd * 0.30);
    const g2 = pt(L * 0.30, Wd * 0.30), h2 = pt(-L * 0.22, Wd * 0.30);
    ctx.beginPath();
    ctx.moveTo(e[0], e[1]); ctx.lineTo(f[0], f[1]); ctx.lineTo(g2[0], g2[1]); ctx.lineTo(h2[0], h2[1]);
    ctx.closePath();
    ctx.fillStyle = rgb(mix(mix(body, [255, 238, 214], 0.14 + 0.22 * P.L), NIGHT_INK, sh), 0.95); ctx.fill();

    const hp = pt(L * 0.52, 0);
    ctx.fillStyle = rgb(mix(c.route === "B" ? P.cyanCore : P.amberCore, NIGHT_INK, sh * 0.7), 0.95);
    ctx.beginPath(); ctx.arc(hp[0], hp[1], Math.max(0.8, 1.25 * s), 0, 7); ctx.fill();
    if (c.brake) {
      const tp = pt(-L * 0.52, 0);
      ctx.fillStyle = rgb(mix(P.brake, NIGHT_INK, sh * 0.5), 0.95);
      ctx.beginPath(); ctx.arc(tp[0], tp[1], Math.max(0.8, 1.3 * s), 0, 7); ctx.fill();
    }
  }

  function drawCarGlow(ctx, sp, P, c) {
    const s = c.s, ang = c.ang, d = c.dim === undefined ? 1 : c.dim;
    const tx = Math.cos(ang), ty = Math.sin(ang);
    const head = c.route === "B" ? sp.cyanCore : sp.amberCore;
    const ha = (c.hero ? 0.9 : 0.14 + 0.62 * P.night) * d;
    if (ha > 0.17 || c.hero) {
      blitGlow(ctx, head, c.x + tx * 6.8 * s, c.y + ty * 6.8 * s, (7 + 12 * P.night) * s, ha);
    }
    if (P.night > 0.35) {
      blitGlow(ctx, head, c.x + tx * 19 * s, c.y + ty * 19 * s, 16 * s, 0.24 * P.night * d);
      blitGlow(ctx, head, c.x + tx * 39 * s, c.y + ty * 39 * s, 22 * s, 0.13 * P.night * d);
    }
    if (c.brake) {
      blitGlow(ctx, sp.brake, c.x - tx * 6.8 * s, c.y - ty * 6.8 * s,
        (3.8 + 6 * P.night) * s, (0.34 + 0.34 * P.night) * d);
    }
  }

  function frameCars(world, st) {
    const cars = [];
    for (const q of st.queued) {
      const g = queuePos(q);
      const rnd = rngFor(q.rec.pid, "car");
      cars.push({
        pid: q.rec.pid, x: g.x, y: g.y, ang: g.ang, s: g.s,
        route: "A", brake: true, hue: hexToRgb(CAR_HUES[Math.floor(rnd() * CAR_HUES.length)]),
      });
    }
    for (const d of st.driving) {
      const g = drivePos(d);
      const rnd = rngFor(d.rec.pid, "car");
      cars.push({
        pid: d.rec.pid, x: g.x, y: g.y, ang: g.ang, s: g.s,
        route: d.route, brake: false, hue: hexToRgb(CAR_HUES[Math.floor(rnd() * CAR_HUES.length)]),
      });
    }
    cars.sort((a, b) => a.y - b.y);
    return cars;
  }

  /* ------------------------------------------------------------ world tags */

  // tags register the box they occupy so the traveler chips can avoid them
  let TAG_BOXES = [];
  function tagBox(x, y, w, h) { TAG_BOXES.push({ x0: x, y0: y, x1: x + w, y1: y + h }); }

  function roadTag(ctx, txt, sub, x, y, col, align) {
    ctx.font = '600 13px ' + SANS;
    const w1 = trackedWidth(ctx, txt, 2.8);
    ctx.font = "400 12.5px " + SANS;
    const w2 = sub ? textW(ctx, sub) : 0;
    const bw = Math.max(w1, w2);
    const ox = align === "r" ? -bw : (align === "c" ? -bw / 2 : 0);
    ctx.save();
    ctx.filter = "blur(15px)";
    ctx.fillStyle = "rgba(8,11,20,0.40)";
    roundRect(ctx, x + ox - 16, y - 22, bw + 32, sub ? 48 : 30, 16); ctx.fill();
    ctx.restore();
    tagBox(x + ox - 20, y - 26, bw + 40, sub ? 56 : 38);
    ctx.font = "600 13px " + SANS;
    ctx.fillStyle = rgb(col, 0.98);
    tracked(ctx, txt, x + ox, y, 2.8);
    if (sub) {
      ctx.font = "400 12.5px " + SANS;
      ctx.fillStyle = rgb(INK, 0.60);
      ctx.fillText(sub, x + ox, y + 18);
    }
  }
  function placeTag(ctx, txt, x, y) {
    ctx.font = "600 12px " + SANS;
    const w = trackedWidth(ctx, txt, 2.6);
    ctx.save();
    ctx.filter = "blur(10px)";
    ctx.fillStyle = "rgba(8,11,20,0.42)";
    roundRect(ctx, x - w / 2 - 14, y - 18, w + 28, 26, 13); ctx.fill();
    ctx.restore();
    tagBox(x - w / 2 - 18, y - 22, w + 36, 34);
    ctx.fillStyle = rgb(INK, 0.62);
    ctx.font = "600 12px " + SANS;
    tracked(ctx, txt, x - w / 2, y, 2.6);
  }
  function tickLine(ctx, x1, y1, x2, y2, col, a) {
    ctx.save();
    ctx.strokeStyle = rgb(col, a); ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.restore();
  }

  function worldAnnotations(ctx, world, st, ok) {
    const ds = world.dayStats[st.day];
    // try a few points along each road; use the first that clears the chrome
    const roadTagAt = function (path, ts, dx, txt, sub, col, bw) {
      for (const t of ts) {
        const p = path.at(t);
        if (!ok(p.x + dx, p.y - 72, bw, 70)) continue;
        tickLine(ctx, p.x + dx, p.y - 58, p.x + dx, p.y - 16, col, 0.40);
        roadTag(ctx, txt, sub, p.x + dx, p.y - 72, col, "c");
        return;
      }
    };
    roadTagAt(ROUTE_A, [0.105, 0.22, 0.34, 0.05], 6, "ROUTE A",
      "one bridge · " + world.checks.routeAFreeFlow + " min when nobody is queueing", GOLD, 330);
    roadTagAt(ROUTE_B, [0.46, 0.34, 0.60, 0.24, 0.72], 60, "ROUTE B — THE RING ROAD",
      world.checks.ringMinutes + " min every single morning · never congested", CY, 380);
    const places = [
      ["INNER CITY · " + world.areaCounts["inner city"] + " HOMES", NEIGHBOURHOODS["inner city"].cx - 40, NEIGHBOURHOODS["inner city"].tagY],
      ["SMALL TOWN · " + world.areaCounts["small town"] + " HOMES", NEIGHBOURHOODS["small town"].cx + 30, NEIGHBOURHOODS["small town"].tagY],
      ["SUBURB · " + world.areaCounts["suburb"] + " HOMES", NEIGHBOURHOODS["suburb"].cx - 20, NEIGHBOURHOODS["suburb"].tagY],
    ];
    for (const p of places) if (ok(p[1], p[2], 250, 40)) placeTag(ctx, p[0], p[1], p[2]);
    if (ok(PLAZA.x + 30, PLAZA.y + 176, 400, 70)) {
      roadTag(ctx, "WORK DISTRICT",
        st.arrived.length + " of the " + world.pids.length + " are already at their desk",
        PLAZA.x + 30, PLAZA.y + 176, GOLD, "c");
    }
    if (st.queued.length > 0 && ds.throughput && ok(PINCH.x - 30, PINCH.y + 96, 340, 70)) {
      roadTag(ctx, "THE BRIDGE",
        "the log clears " + ds.throughput.toFixed(1) + " cars a minute here",
        PINCH.x - 30, PINCH.y + 96, HOT, "c");
    }
  }

  /* --------------------------------------------------------------- chips */

  function makeChipPlacer(obstacles) {
    const placed = (obstacles || []).slice();
    return function chip(ctx, cfg) {
      ctx.font = "600 17px " + SANS;
      const nw = textW(ctx, cfg.name);
      ctx.font = "400 13px " + SANS;
      let lw = 0; for (const l of cfg.lines) lw = Math.max(lw, textW(ctx, l));
      const pad = 14, mono = 13;
      const bw = Math.max(nw, lw) + pad * 2 + mono * 2 + 12;
      const bh = 26 + cfg.lines.length * 17;
      const bx0 = cfg.side === "r" ? cfg.x : cfg.x - bw;
      const by0 = cfg.y - bh / 2;
      const lo = cfg.minX === undefined ? 96 : cfg.minX;
      const hi = (cfg.maxX || W - 96) - bw;
      const clampX = (v) => Math.max(lo, Math.min(Math.max(lo, hi), v));
      const clampY = (v) => Math.max(cfg.minY === undefined ? 186 : cfg.minY,
        Math.min((cfg.maxY === undefined ? H - 300 : cfg.maxY) - bh, v));
      const area = function (a, b) {
        const ow = Math.min(a.x1 + 14, b.x1) - Math.max(a.x0 - 14, b.x0);
        const oh = Math.min(a.y1 + 12, b.y1) - Math.max(a.y0 - 12, b.y0);
        return (ow > 0 && oh > 0) ? ow * oh : 0;
      };
      const cands = [];
      const dys = [0, -46, 46, -92, 92, -140, 140, -190, 190, -240, 240, -300, 300];
      // anchors on the right half prefer a chip placed to their left
      const dxs = cfg.ax > (cfg.pivot || W * 0.60)
        ? [-(bw + 40), 0, bw + 40] : [0, -(bw + 40), bw + 40];
      dxs.forEach(function (dx, di) {
        for (const dy of dys) {
          const x = clampX(bx0 + dx), y = clampY(by0 + dy);
          const box = { x0: x, y0: y, x1: x + bw, y1: y + bh };
          let cost = 0;
          for (const q of placed) cost += area(box, q);
          cost += Math.abs(dy) * 6 + di * 260;
          cands.push({ x: x, y: y, cost: cost });
        }
      });
      cands.sort((a, b) => a.cost - b.cost);
      const bx = cands[0].x, by = cands[0].y;
      placed.push({ x0: bx, y0: by, x1: bx + bw, y1: by + bh });
      const cy2 = by + bh / 2;
      const ex = (cfg.ax < bx) ? bx : (cfg.ax > bx + bw ? bx + bw : cfg.ax);

      ctx.save();
      ctx.strokeStyle = rgb(cfg.tint, 0.50); ctx.lineWidth = 1.1;
      ctx.beginPath(); ctx.moveTo(cfg.ax, cfg.ay); ctx.lineTo(ex, cy2); ctx.stroke();
      ctx.fillStyle = rgb(cfg.tint, 0.9);
      ctx.beginPath(); ctx.arc(cfg.ax, cfg.ay, 2.4, 0, 7); ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.filter = "blur(16px)";
      ctx.fillStyle = "rgba(7,10,19,0.60)";
      roundRect(ctx, bx - 8, by - 8, bw + 16, bh + 16, 20); ctx.fill();
      ctx.restore();
      roundRect(ctx, bx, by, bw, bh, 10);
      const pg = ctx.createLinearGradient(0, by, 0, by + bh);
      pg.addColorStop(0, "rgba(20,26,40,0.62)");
      pg.addColorStop(1, "rgba(8,11,20,0.72)");
      ctx.fillStyle = pg; ctx.fill();
      ctx.strokeStyle = rgb(cfg.tint, 0.30); ctx.lineWidth = 1; ctx.stroke();

      const mcx = bx + pad + mono, mcy = by + bh / 2;
      ctx.beginPath(); ctx.arc(mcx, mcy, mono, 0, 7);
      ctx.fillStyle = rgb(cfg.tint, 0.20); ctx.fill();
      ctx.strokeStyle = rgb(cfg.tint, 0.72); ctx.lineWidth = 1.1; ctx.stroke();
      ctx.font = "600 14px " + SANS;
      ctx.fillStyle = rgb(cfg.tint, 1);
      const ini = cfg.name[0].toUpperCase();
      ctx.fillText(ini, mcx - textW(ctx, ini) / 2, mcy + 5);

      const tx = mcx + mono + 12;
      let ty = by + 19;
      ctx.font = "600 17px " + SANS;
      ctx.fillStyle = rgb(INK, 0.97); ctx.fillText(cfg.name, tx, ty);
      ctx.font = "400 13px " + SANS;
      ctx.fillStyle = rgb(INK, 0.64);
      for (const l of cfg.lines) { ty += 17; ctx.fillText(l, tx, ty); }
    };
  }

  function heroRing(ctx, x, y, s, tint) {
    ctx.save();
    const r = 20 * Math.max(0.85, s);
    ctx.strokeStyle = rgb(tint, 0.20); ctx.lineWidth = 7;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.strokeStyle = rgb(tint, 0.92); ctx.lineWidth = 1.7;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.strokeStyle = rgb(tint, 0.28); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(x, y, r * 1.7, 0, 7); ctx.stroke();
    ctx.restore();
  }

  /* ------------------------------------------------------------- chrome */

  // the same falloff plate, flipped, used under the header strips
  function headPlate(ctx, x0, y0, w, h, flipX) {
    const p = readingPlate();
    ctx.save();
    ctx.translate(x0 + (flipX ? w : 0), y0 + h);
    ctx.scale(flipX ? -1 : 1, -1);
    ctx.drawImage(p, 0, 0, w, h);
    ctx.restore();
  }

  // everything in the header except the clock, which ticks on its own
  function titleBlock(ctx, title, sub) {
    const x = 96, y = 92;
    // two stacked plates: a wide soft one, then a tighter one directly under
    // the text, so the header stays readable over a lit neighbourhood without
    // any blur work (blur is the single most expensive text effect there is)
    headPlate(ctx, 0, 0, 1120, sub ? 290 : 230, false);
    headPlate(ctx, 0, 0, 760, sub ? 240 : 195, false);
    ctx.save();
    ctx.font = "600 15px " + SANS;
    ctx.fillStyle = rgb(GOLD, 0.95);
    tracked(ctx, "GLASSBOX · MOBILITY", x, y, 3.6);
    ctx.font = "500 64px " + SANS;
    ctx.fillStyle = rgb(INK, 0.98);
    ctx.fillText(title, x, y + 64);
    if (sub) {
      ctx.font = "400 18px " + SANS;
      ctx.fillStyle = rgb(INK, 0.72);
      ctx.fillText(sub, x, y + 100);
    }
    ctx.restore();
  }

  // the clock ticks every simulated minute, so it is drawn straight to the
  // frame rather than invalidating the whole cached header band
  function clockChip(ctx, title, clockText) {
    const x = 96, y = 92;
    ctx.font = "500 64px " + SANS;
    const tw = textW(ctx, title);
    const cw = 126, ch = 40, cx = x + tw + 24, cy = y + 64 - 31;
    roundRect(ctx, cx, cy, cw, ch, 20);
    ctx.fillStyle = "rgba(255,255,255,0.09)"; ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.24)"; ctx.lineWidth = 1; ctx.stroke();
    ctx.font = "600 21px " + SANS;
    ctx.fillStyle = rgb(INK, 0.96);
    const cwid = trackedWidth(ctx, clockText, 1.6);
    tracked(ctx, clockText, cx + (cw - cwid) / 2, cy + 28, 1.6);
  }

  function statStrip(ctx, items, rightEdge) {
    let x = rightEdge === undefined ? W - 96 : rightEdge;
    // the plate must run off the right edge of the frame, or its darkest
    // column leaves a visible vertical seam where it stops
    headPlate(ctx, Math.max(0, x - 1020), 0, W - Math.max(0, x - 1020), 200, true);
    for (const it of items.slice().reverse()) {
      ctx.font = "500 42px " + SANS;
      const nw = textW(ctx, it.n);
      ctx.font = "600 12px " + SANS;
      const lw = trackedWidth(ctx, it.l, 2.4);
      const bw = Math.max(nw, lw);
      x -= bw;
      ctx.font = "500 42px " + SANS;
      ctx.fillStyle = rgb(it.c || INK, 0.97);
      ctx.fillText(it.n, x + (bw - nw) / 2, 118);
      ctx.font = "600 12px " + SANS;
      ctx.fillStyle = rgb(it.c ? mix(it.c, INK, 0.4) : INK, 0.56);
      tracked(ctx, it.l, x + (bw - lw) / 2, 142, 2.4);
      x -= 52;
    }
  }

  // one cached 2-D falloff plate: dark at bottom-left, clear at top and right.
  // Built offscreen with destination-out so there is no seam where it ends.
  let _plate = null;
  function readingPlate() {
    if (_plate) return _plate;
    const PW = 1200, PH = 320;
    _plate = nc(PW, PH);
    const g = _plate.getContext("2d");
    const lg = g.createLinearGradient(0, 0, 0, PH);
    lg.addColorStop(0, "rgba(5,8,16,0)");
    lg.addColorStop(0.34, "rgba(5,8,16,0.46)");
    lg.addColorStop(1, "rgba(5,8,16,0.74)");
    g.fillStyle = lg; g.fillRect(0, 0, PW, PH);
    g.globalCompositeOperation = "destination-out";
    const hg = g.createLinearGradient(0, 0, PW, 0);
    hg.addColorStop(0, "rgba(0,0,0,0)");
    hg.addColorStop(0.62, "rgba(0,0,0,0)");
    hg.addColorStop(1, "rgba(0,0,0,1)");
    g.fillStyle = hg; g.fillRect(0, 0, PW, PH);
    return _plate;
  }

  function captionBlock(ctx, lines, maxW) {
    const x = 96, mw = maxW || 780;
    const laid = [];
    for (const ln of lines) {
      ctx.font = (ln.w || "400") + " " + (ln.s || 22) + "px " + SANS;
      for (const t of wrapText(ctx, ln.t, mw)) laid.push({ t: t, s: ln.s, w: ln.w, a: ln.a, c: ln.c });
    }
    if (!laid.length) return;
    let total = 0;
    for (const l of laid) total += (l.s || 22) * 1.42;
    let y = H - 96 - total + (laid[laid.length - 1].s || 22);
    // a soft reading plate so the text survives over a bright neighbourhood
    const top = y - (laid[0].s || 22) - 46;
    ctx.drawImage(readingPlate(), 0, top, x + mw + 210, H - top);
    ctx.drawImage(readingPlate(), 0, top + 24, x + mw + 40, H - top - 24);
    ctx.fillStyle = rgb(GOLD, 0.9);
    ctx.fillRect(x, y - (laid[0].s || 22) - 22, 44, 2);
    for (const ln of laid) {
      ctx.font = (ln.w || "400") + " " + (ln.s || 22) + "px " + SANS;
      ctx.fillStyle = rgb(ln.c || INK, ln.a === undefined ? 0.92 : Math.min(1, ln.a + 0.06));
      ctx.fillText(ln.t, x, y);
      y += (ln.s || 22) * 1.42;
    }
  }

  function provenanceFooter(ctx) {
    ctx.font = "400 12.5px " + SANS;
    ctx.fillStyle = rgb(INK, 0.46);
    ctx.fillText(PROVENANCE, 96, H - 40);
  }

  /* ================================================= the traveler dossier */

  function monoColor(world, pid) {
    let h = 0;
    for (let i = 0; i < pid.length; i++) h = (h * 31 + pid.charCodeAt(i)) >>> 0;
    const base = hexToRgb(MONO_HUES[h % MONO_HUES.length]);
    const p = world.prof[pid];
    const tone = { "inner city": -0.16, "suburb": 0.0, "small town": 0.14 }[p ? p.area_type : ""] || 0;
    return tone < 0 ? mix(base, [23, 15, 12], -tone) : mix(base, [240, 226, 206], tone);
  }

  // 2–3 verbatim sentences from the real card. Non-adjacent joins are marked.
  function cardExcerpt(world, pid, budget) {
    const text = world.card[pid];
    if (!text) return { parts: [], joined: "" };
    const sents = (text.match(/[^.!?]+[.!?]+/g) || [text]).map((s) => s.trim());
    const lim = budget || 360;
    const picked = [];
    let total = 0, lastIdx = -2, gapBefore = [];
    for (let i = 1; i < sents.length && picked.length < 3; i++) {
      const s = sents[i];
      if (s.length > lim) continue;
      if (total + s.length > lim) { if (picked.length) break; else continue; }
      gapBefore.push(picked.length > 0 && i !== lastIdx + 1);
      picked.push(s);
      total += s.length + 1;
      lastIdx = i;
    }
    if (!picked.length) { picked.push(sents[0]); gapBefore.push(false); }
    let joined = "";
    picked.forEach(function (s, i) {
      if (i) joined += gapBefore[i] ? " […] " : " ";
      joined += s;
    });
    return { parts: picked, gaps: gapBefore, joined: joined };
  }

  // 4–6 one-line entries, every one computed from this traveler's own minutes
  function dataDiary(world, pid, day) {
    const t = world.trav[pid], days = world.days;
    const must = [], nice = [];
    const row = (k, s, hot) => ({ k: k, s: s, hot: !!hot });

    const d1 = t.days[days[0]];
    if (d1) {
      must.push(row("DAY " + d1.day,
        "Leaves " + hhmm(d1.dep) + ", route " + (d1.route === "A" ? "A" : "ring") + ". " +
        d1.travel.toFixed(1) + " min" +
        (d1.delay >= 0.05 ? " — " + d1.delay.toFixed(1) + " of them stopped at the bridge." : ", no queue.")));
    }
    if (t.firstDayOnB) {
      const r = t.days[t.firstDayOnB];
      nice.push(row("DAY " + t.firstDayOnB,
        "First morning on the ring road: out " + hhmm(r.dep) + ", " +
        r.travel.toFixed(0) + " min flat, no bridge."));
    }
    // largest single-morning move in this traveler's own record, with ties
    let big = null, ties = 0;
    for (let i = 1; i < t.list.length; i++) {
      const s = t.list[i].dep - t.list[i - 1].dep;
      if (!big || Math.abs(s) > Math.abs(big.s) + 1e-6) { big = { s: s, day: t.list[i].day }; ties = 1; }
      else if (Math.abs(Math.abs(s) - Math.abs(big.s)) < 1e-6) ties++;
    }
    if (big && Math.abs(big.s) >= 1 && big.day !== day && big.day !== days[0] &&
        big.day !== t.firstDayOnB) {
      nice.push(row("DAY " + big.day,
        Math.abs(Math.round(big.s)) + " min " + (big.s < 0 ? "earlier" : "later") +
        " — the largest single step" +
        (ties > 1 ? ", repeated on " + ties + " mornings." : " of the " + t.list.length + ".")));
    }
    if (t.worst && t.worst.delay >= 1 && t.worst.day !== days[0] && t.worst.day !== day) {
      nice.push(row("DAY " + t.worst.day,
        "Longest wait of the " + t.list.length + ": " + t.worst.delay.toFixed(1) +
        " min stopped at the bridge."));
    }
    const cur = t.days[day];
    if (cur) {
      const prev = t.days[day - 1];
      let tail = "";
      if (prev) {
        const shift = cur.dep - prev.dep;
        tail = Math.abs(shift) < 0.5
          ? " Same departure as yesterday."
          : " " + Math.abs(Math.round(shift)) + " min " + (shift < 0 ? "earlier" : "later") + " than yesterday.";
        if (cur.route !== prev.route) {
          tail += " Switched to " + (cur.route === "B" ? "the ring road" : "route A") + ".";
        }
      }
      must.push(row("DAY " + day,
        "Today: out at " + hhmm(cur.dep) + ", in at " + hhmm(cur.arrive) + "." + tail, true));
    }
    const last = t.last;
    if (last && last.day !== day) {
      nice.push(row("DAY " + last.day,
        "Leaves " + hhmm(last.dep) + " — " +
        (Math.abs(t.drift) < 0.5 ? "the same minute as day " + days[0] + "." :
          durHM(t.drift) + (t.drift < 0 ? " earlier" : " later") + " than day " + days[0] + ".")));
    }
    must.push(row(t.list.length + " DAYS",
      t.switches + (t.switches === 1 ? " route switch · " : " route switches · ") +
      (t.daysOnB ? t.daysOnB + " of " + t.list.length + " mornings on the ring road."
                 : "never on the ring road.")));

    const out = must.slice(0, 1)
      .concat(nice.slice(0, Math.max(0, 6 - must.length)))
      .concat(must.slice(1));
    return out;
  }

  const DOSSIER_W = 620, DOSSIER_H = 940;

  function renderDossier(world, pid, day) {
    const key = pid + "|" + day;
    if (world._dossier[key]) return world._dossier[key];
    const c = nc(DOSSIER_W, DOSSIER_H), ctx = c.getContext("2d");
    const p = world.prof[pid], t = world.trav[pid];
    const name = world.name[pid] || pid;
    const tint = monoColor(world, pid);
    const PAD = 34;

    ctx.fillStyle = rgb(PAPER); ctx.fillRect(0, 0, DOSSIER_W, DOSSIER_H);
    // paper tooth
    let seed = hash(pid);
    for (let i = 0; i < 2600; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const x = (seed >>> 9) % DOSSIER_W;
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const y = (seed >>> 9) % DOSSIER_H;
      ctx.fillStyle = "rgba(120,102,80," + (0.012 + ((seed >>> 20) & 15) / 15 * 0.03) + ")";
      ctx.fillRect(x, y, 1.4, 1.4);
    }

    let y = PAD + 22;
    ctx.font = "600 12px " + SANS;
    ctx.fillStyle = rgb(PAPER_MUTE, 0.95);
    tracked(ctx, "TRAVELER DOSSIER", PAD, y, 3.2);
    ctx.font = "400 12px " + SANS;
    ctx.fillStyle = rgb(PAPER_MUTE, 0.75);
    const pw = ctx.measureText(pid).width;
    ctx.fillText(pid, DOSSIER_W - PAD - pw, y);
    y += 14;
    ctx.strokeStyle = rgb(PAPER_INK, 0.28); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD, y + 0.5); ctx.lineTo(DOSSIER_W - PAD, y + 0.5); ctx.stroke();

    // monogram chip
    y += 26;
    const MS = 106;
    ctx.fillStyle = rgb(tint); ctx.fillRect(PAD, y, MS, MS);
    ctx.font = "400 74px " + SERIF;
    ctx.fillStyle = "rgba(255,252,246,0.97)";
    const ini = name[0].toUpperCase();
    ctx.fillText(ini, PAD + MS / 2 - ctx.measureText(ini).width / 2, y + MS - 26);
    ctx.font = "600 9.5px " + SANS;
    ctx.fillStyle = rgb(PAPER_MUTE, 0.9);
    const mw = trackedWidth(ctx, "MONOGRAM", 2.2);
    tracked(ctx, "MONOGRAM", PAD + MS / 2 - mw / 2, y + MS + 15, 2.2);

    // name + profile
    const tx = PAD + MS + 26;
    ctx.font = "400 46px " + SERIF;
    ctx.fillStyle = rgb(PAPER_INK, 0.95);
    ctx.fillText(name, tx, y + 44);
    ctx.font = "400 14px " + SANS;
    ctx.fillStyle = rgb(PAPER_MUTE, 1);
    ctx.fillText(p.age_band + "  ·  " + p.occupation_type, tx, y + 70);
    ctx.fillText(p.household + "  ·  " + p.area_type, tx, y + 90);

    // route ribbon: one square per recorded morning
    y += MS + 34;
    ctx.font = "600 10px " + SANS;
    ctx.fillStyle = rgb(PAPER_MUTE, 0.95);
    tracked(ctx, "THE ROAD TAKEN, MORNING BY MORNING", PAD, y, 2.2);
    const legY = y;
    const legend = function (lx, col, txt) {
      ctx.fillStyle = rgb(col); ctx.fillRect(lx, legY - 8, 9, 9);
      ctx.font = "600 9.5px " + SANS;
      ctx.fillStyle = rgb(PAPER_MUTE, 0.95);
      tracked(ctx, txt, lx + 14, legY, 1.8);
    };
    ctx.font = "600 9.5px " + SANS;
    const wRing = trackedWidth(ctx, "RING ROAD", 1.8), wA = trackedWidth(ctx, "ROUTE A", 1.8);
    legend(DOSSIER_W - PAD - wRing - 14, RIBBON_B, "RING ROAD");
    legend(DOSSIER_W - PAD - wRing - 14 - wA - 32, RIBBON_A, "ROUTE A");

    y += 14;
    const n = t.list.length;
    const gap = 4, sq = Math.floor((DOSSIER_W - PAD * 2 - gap * (n - 1)) / n);
    t.list.forEach(function (r, i) {
      const x = PAD + i * (sq + gap);
      ctx.fillStyle = rgb(r.route === "B" ? RIBBON_B : RIBBON_A, r.day === day ? 1 : 0.42);
      ctx.fillRect(x, y, sq, sq);
      if (r.day === day) {
        ctx.fillStyle = rgb(PAPER_INK, 0.85);
        ctx.fillRect(x, y + sq + 3, sq, 2);
        ctx.font = "600 8.5px " + SANS;
        const lw = trackedWidth(ctx, "TODAY", 1.6);
        ctx.fillStyle = rgb(PAPER_INK, 0.8);
        tracked(ctx, "TODAY", Math.max(PAD, Math.min(DOSSIER_W - PAD - lw, x + sq / 2 - lw / 2)), y + sq + 17, 1.6);
      }
    });
    y += sq + 30;

    // three computed stat cells
    ctx.strokeStyle = rgb(PAPER_INK, 0.18);
    ctx.beginPath(); ctx.moveTo(PAD, y + 0.5); ctx.lineTo(DOSSIER_W - PAD, y + 0.5); ctx.stroke();
    y += 22;
    const cells = [
      { l: "ROUTE SWITCHES", v: String(t.switches) },
      { l: t.drift <= 0 ? "EARLIER BY DAY " + t.last.day : "LATER BY DAY " + t.last.day, v: durHM(t.drift) },
      { l: "RING ROAD", v: t.daysOnB ? t.daysOnB + "/" + n : "never" },
    ];
    const cw = (DOSSIER_W - PAD * 2) / 3;
    cells.forEach(function (c2, i) {
      const cx = PAD + i * cw;
      if (i) {
        ctx.strokeStyle = rgb(PAPER_INK, 0.14);
        ctx.beginPath(); ctx.moveTo(cx - 12, y - 12); ctx.lineTo(cx - 12, y + 40); ctx.stroke();
      }
      ctx.font = "600 9.5px " + SANS;
      ctx.fillStyle = rgb(PAPER_MUTE, 0.95);
      tracked(ctx, c2.l, cx, y, 1.9);
      ctx.font = "400 26px " + SERIF;
      ctx.fillStyle = rgb(PAPER_INK, 0.92);
      ctx.fillText(c2.v, cx, y + 32);
    });
    y += 60;
    ctx.strokeStyle = rgb(PAPER_INK, 0.18);
    ctx.beginPath(); ctx.moveTo(PAD, y + 0.5); ctx.lineTo(DOSSIER_W - PAD, y + 0.5); ctx.stroke();

    // verbatim card excerpt
    y += 34;
    const ex = cardExcerpt(world, pid, 330);
    ctx.font = "400 46px " + SERIF;
    ctx.fillStyle = rgb(PAPER_MUTE, 0.45);
    ctx.fillText("“", PAD - 4, y + 8);
    ctx.font = "400 18.5px " + SERIF;
    const quoteX = PAD + 26, quoteW = DOSSIER_W - PAD * 2 - 26;
    const lines = wrapText(ctx, ex.joined, quoteW);
    for (const ln of lines) {
      ctx.font = "400 18.5px " + SERIF;
      // render the […] marks in a lighter ink so joins are visible, not hidden
      if (ln.indexOf("[…]") >= 0) {
        const seg = ln.split("[…]");
        let cx = quoteX;
        seg.forEach(function (s, i) {
          if (i) {
            ctx.fillStyle = rgb(PAPER_MUTE, 0.7);
            ctx.fillText("[…]", cx, y); cx += ctx.measureText("[…]").width;
          }
          ctx.fillStyle = rgb(PAPER_INK, 0.9);
          ctx.fillText(s, cx, y); cx += ctx.measureText(s).width;
        });
      } else {
        ctx.fillStyle = rgb(PAPER_INK, 0.9);
        ctx.fillText(ln, quoteX, y);
      }
      y += 27;
    }
    y += 4;
    ctx.font = "italic 400 12px " + SERIF;
    ctx.fillStyle = rgb(PAPER_MUTE, 1);
    const joined = ex.gaps && ex.gaps.some(Boolean);
    ctx.fillText("verbatim from the study's biography card for " + pid +
      (joined ? " · non-adjacent sentences joined with […]" : " · consecutive sentences, unedited"),
      quoteX, y);

    // data diary
    y += 32;
    ctx.font = "600 10.5px " + SANS;
    ctx.fillStyle = rgb(PAPER_INK, 0.8);
    tracked(ctx, "DATA DIARY", PAD, y, 2.6);
    ctx.font = "600 9.5px " + SANS;
    ctx.fillStyle = rgb(PAPER_MUTE, 0.95);
    const rw = trackedWidth(ctx, "COMPUTED FROM THE TRAJECTORY LOG", 1.8);
    tracked(ctx, "COMPUTED FROM THE TRAJECTORY LOG", DOSSIER_W - PAD - rw, y, 1.8);
    y += 10;
    ctx.strokeStyle = rgb(PAPER_INK, 0.18);
    ctx.beginPath(); ctx.moveTo(PAD, y + 0.5); ctx.lineTo(DOSSIER_W - PAD, y + 0.5); ctx.stroke();
    y += 8;

    const diary = dataDiary(world, pid, day);
    for (const row of diary) {
      const rowH = 30;
      if (row.hot) {
        ctx.fillStyle = "rgba(200,170,120,0.26)";
        ctx.fillRect(PAD - 8, y, DOSSIER_W - PAD * 2 + 16, rowH);
        ctx.fillStyle = rgb(PAPER_INK, 0.8);
        ctx.fillRect(PAD - 8, y, 3, rowH);
      }
      ctx.font = (row.hot ? "700 " : "600 ") + "9.5px " + SANS;
      ctx.fillStyle = rgb(PAPER_INK, row.hot ? 0.9 : 0.6);
      tracked(ctx, row.k, PAD, y + 19, 1.7);
      ctx.font = "400 13.5px " + SANS;
      ctx.fillStyle = rgb(PAPER_INK, row.hot ? 0.95 : 0.82);
      const avail = DOSSIER_W - PAD - (PAD + 92);
      let s = row.s;
      while (ctx.measureText(s).width > avail && s.length > 8) s = s.slice(0, -2);
      if (s !== row.s) s = s.replace(/\s+\S*$/, "") + "…";
      ctx.fillText(s, PAD + 92, y + 19);
      y += rowH;
    }

    world._dossier[key] = { canvas: c, height: Math.min(DOSSIER_H, y + PAD) };
    return world._dossier[key];
  }

  function drawDossier(ctx, world, pid, day, x, y) {
    const d = renderDossier(world, pid, day);
    ctx.save();
    ctx.filter = "blur(26px)";
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(x + 6, y + 14, DOSSIER_W, d.height);
    ctx.restore();
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, DOSSIER_W, d.height); ctx.clip();
    ctx.drawImage(d.canvas, x, y);          // 1:1, no resample
    ctx.restore();
    ctx.strokeStyle = "rgba(0,0,0,0.35)"; ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, DOSSIER_W - 1, d.height - 1);
    return { x: x, y: y, w: DOSSIER_W, h: d.height };
  }

  /* ============================================================== render */

  /* The chrome is a screen-space overlay whose content changes far less often
   * than the frame does — a clock label ticks once a simulated minute, a stat
   * only when its count moves. Cache each band and redraw it on change; text
   * is the single most expensive thing in a 2D context. */
  const BAND_TOP_H = 262, BAND_BOT_Y = 690;
  function band(world, key, sig, y0, h, draw) {
    if (!world._bands) world._bands = Object.create(null);
    let e = world._bands[key];
    if (!e) e = world._bands[key] = { canvas: nc(W, h), sig: null };
    if (e.sig !== sig) {
      const g = e.canvas.getContext("2d");
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.globalAlpha = 1;
      g.globalCompositeOperation = "source-over";
      g.clearRect(0, 0, W, h);
      g.save(); g.translate(0, -y0); draw(g); g.restore();
      e.sig = sig;
    }
    return e.canvas;
  }

  function focusLayer(world, f) {
    const sig = [f.x | 0, f.y | 0, f.r | 0, f.a === undefined ? 0.44 : f.a].join(",");
    if (!world._focus) world._focus = { canvas: nc(W, H), sig: null };
    if (world._focus.sig !== sig) {
      const g = world._focus.canvas.getContext("2d");
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, W, H);
      const rg = g.createRadialGradient(f.x, f.y, f.r * 0.55, f.x, f.y, f.r * 2.1);
      rg.addColorStop(0, "rgba(6,9,18,0)");
      rg.addColorStop(1, "rgba(6,9,18," + (f.a === undefined ? 0.44 : f.a) + ")");
      g.fillStyle = rg; g.fillRect(0, 0, W, H);
      world._focus.sig = sig;
    }
    return world._focus.canvas;
  }

  function camPlace(cam) {
    if (!cam || !cam.zoom || cam.zoom === 1) return { z: 1, tx: 0, ty: 0 };
    const z = cam.zoom;
    const sx = cam.sx === undefined ? W / 2 : cam.sx;
    const sy = cam.sy === undefined ? H / 2 : cam.sy;
    return {
      z: z,
      tx: clamp(sx - cam.fx * z, W - W * z, 0),
      ty: clamp(sy - cam.fy * z, H - H * z, 0),
    };
  }

  /**
   * Draw one frame.
   *   ctx        2d context, any size, 16:9 assumed
   *   world      from Town.buildWorld(data)  (Town.buildStatic first)
   *   day        1..20
   *   clockMin   minutes after midnight
   *   opts       { title, sub, stats, caption, chips, focus, cam, inspect,
   *                share, chrome, hero }
   */
  function render(ctx, world, day, clockMin, opts) {
    opts = opts || {};
    if (!world._static) buildStatic(world);
    const canvas = ctx.canvas;
    const sw = canvas.width, sh = canvas.height;
    const k = sw / W;
    const L = lightOf(clockMin);
    const P = palette(L);
    const sp = world._sprites;
    const st = stateAt(world, day, clockMin);
    const ds = world.dayStats[day];

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, sw, sh);

    // with the dossier open, make sure its subject is not hiding behind it
    let cam = opts.cam;
    if (opts.inspect && world.trav[opts.inspect]) {
      const sub = locate(world, day, clockMin, opts.inspect);
      const probe = camPlace(cam);
      if (sub && sub.x * probe.z + probe.tx > W - DOSSIER_W - 150) {
        cam = { fx: sub.x, fy: sub.y, zoom: cam && cam.zoom ? cam.zoom : 1.10, sx: 600, sy: 500 };
      }
    }
    const cp = camPlace(cam);
    ctx.setTransform(k * cp.z, 0, 0, k * cp.z, k * cp.tx, k * cp.ty);
    // world -> design-space screen, used to keep tags out of the chrome
    const toScreen = (x, y) => ({ x: x * cp.z + cp.tx, y: y * cp.z + cp.ty });

    /* ---- static ground, cross-faded between night and morning ----------
     * The palette is linear in L, so blending the full-night and full-morning
     * passes reproduces the intermediate hour. The blend is cached at 1/48 of
     * a step, which is far below the eye's threshold on a dawn ramp but means
     * a scrub or a playback only pays for it every few frames. */
    ctx.globalAlpha = 1;
    const q = Math.round(L * 48) / 48;
    if (q <= 0) {
      ctx.drawImage(world._staticNight, 0, 0);
    } else if (q >= 1) {
      ctx.drawImage(world._staticDay, 0, 0);
    } else {
      if (world._baseQ !== q) {
        if (!world._baseMix) world._baseMix = nc(W, H);
        const bg = world._baseMix.getContext("2d");
        bg.setTransform(1, 0, 0, 1, 0, 0);
        bg.globalAlpha = 1;
        bg.drawImage(world._staticNight, 0, 0);
        bg.globalAlpha = q;
        bg.drawImage(world._staticDay, 0, 0);
        bg.globalAlpha = 1;
        world._baseQ = q;
      }
      ctx.drawImage(world._baseMix, 0, 0);
    }

    /* ---- lit homes ------------------------------------------------------ */
    const litSet = st.lit;
    ctx.globalCompositeOperation = "source-over";
    const litPlots = [];
    ctx.fillStyle = rgb(P.amberCore, 0.97);
    for (const pid of litSet) {
      const plot = world.plots[pid];
      if (!plot) continue;
      const dim = postDim(plot.x, plot.y);
      litPlots.push({ plot: plot, dim: dim });
      const ww = plot.ww, wh = plot.wh;
      ctx.globalAlpha = dim;
      for (const w of plot.wins) ctx.fillRect(w[0] - ww / 2, w[1] - wh / 2, ww, wh);
    }
    ctx.globalAlpha = 1;
    /* ---- district windows: lit at the recorded arrival ------------------ */
    const arrivedCells = [];
    for (const rec of st.arrived) {
      const c = world.district[rec.pid];
      if (c) arrivedCells.push(c);
    }
    ctx.fillStyle = rgb([255, 208, 132], 1);
    for (const c of arrivedCells) ctx.fillRect(c.x - c.w / 2, c.y - c.h / 2, c.w, c.h);
    ctx.fillStyle = rgb(P.amberCore, 1);
    for (const c of arrivedCells) ctx.fillRect(c.x - c.w / 2 + 0.6, c.y - c.h / 2 + 0.6, c.w - 1.2, c.h - 1.2);

    /* ---- cars ----------------------------------------------------------- */
    const cars = frameCars(world, st);
    world._cars = cars;
    for (const c of cars) { c.dim = postDim(c.x, c.y); drawCar(ctx, sp, P, c); }

    /* ---- additive light pass (pre-rendered sprites only) ---------------- */
    ctx.globalCompositeOperation = "lighter";
    const nightPool = P.night > 0.22;
    for (const lp of litPlots) {
      const plot = lp.plot, d = lp.dim;
      blitGlow(ctx, sp.lamp, plot.x, plot.y - plot.g.hh * 0.28,
        (11 + 9 * plot.s) * (1 + 1.35 * P.night), (0.52 + 0.40 * P.night) * d);
      if (nightPool) {
        blitGlow(ctx, sp.amber, plot.x, plot.y + plot.g.hd * 0.9,
          plot.g.w * (1.15 + 1.7 * P.night), 0.24 * P.night * d);
      }
    }
    for (const c of arrivedCells) {
      blitGlow(ctx, sp.deskLamp, c.x, c.y, 10 + 9 * P.night,
        (0.80 + 0.20 * P.night) * postDim(c.x, c.y));
    }
    if (st.arrived.length) {
      blitGlow(ctx, sp.amber, PLAZA.x, PLAZA.y - 60, 320,
        (0.05 + 0.10 * (st.arrived.length / world.pids.length)) * postDim(PLAZA.x, PLAZA.y - 60));
    }
    if (st.queued.length > 2) {
      const tail = ROUTE_A.at(Math.max(0.02, PINCH_FRAC - (st.queued.length >> 1) * QSLOT * 0.5));
      const mx = (tail.x + PINCH.x) / 2, my = (tail.y + PINCH.y) / 2;
      blitGlow(ctx, sp.hot, mx, my, 40 + st.queued.length * 1.6,
        Math.min(0.20, 0.02 + st.queued.length * 0.0022) * postDim(mx, my));
    }
    for (const c of cars) drawCarGlow(ctx, sp, P, c);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";

    /* ---- world-space annotations ---------------------------------------- */
    TAG_BOXES = [];
    const showTags = opts.tags === undefined ? (opts.chrome !== false && !opts.share) : opts.tags;
    if (showTags) {
      const reserved = [[60, 40, 940, 240], [900, 50, W, 175], [0, 800, 1140, H]];
      if (opts.inspect) reserved.push([W - DOSSIER_W - 80, 50, W, H]);
      const ok = function (wx, wy, bw, bh) {
        const p = toScreen(wx, wy);
        if (p.x - bw / 2 < 40 || p.x + bw / 2 > W - 40) return false;
        if (p.y - bh < 40 || p.y + 24 > H - 30) return false;
        for (const r of reserved) {
          if (p.x + bw / 2 > r[0] && p.x - bw / 2 < r[2] && p.y + 26 > r[1] && p.y - bh < r[3]) return false;
        }
        return true;
      };
      worldAnnotations(ctx, world, st, ok);
    }

    // hero rings / chips live in world space so they track the subject
    if (opts.chips && opts.chips.length) {
      // keep chips out of the screen-space chrome by mapping those boxes back
      const toWorld = (sx, sy) => ({ x: (sx - cp.tx) / cp.z, y: (sy - cp.ty) / cp.z });
      const screenBoxes = [[60, 30, 960, 250], [880, 40, W, 190], [40, 810, 1160, H]];
      if (opts.inspect) screenBoxes.push([W - DOSSIER_W - 90, 40, W, H]);
      const obstacles = screenBoxes.map(function (b) {
        const a = toWorld(b[0], b[1]), c2 = toWorld(b[2], b[3]);
        return { x0: a.x, y0: a.y, x1: c2.x, y1: c2.y };
      }).concat(TAG_BOXES);
      const rightLimit = opts.inspect ? W - DOSSIER_W - 110 : W - 120;
      const v0 = toWorld(120, 200), v1 = toWorld(rightLimit, H - 260);
      const chip = makeChipPlacer(obstacles);
      for (const c of opts.chips) {
        const loc = locate(world, day, clockMin, c.pid);
        if (!loc) continue;
        if (c.ring) heroRing(ctx, loc.x, loc.y, loc.s, c.tint || GOLD);
        chip(ctx, {
          x: loc.x + (c.dx === undefined ? 44 : c.dx),
          y: loc.y + (c.dy === undefined ? -120 : c.dy),
          side: c.side || "r", ax: loc.x, ay: loc.y,
          name: c.label || ((world.name[c.pid] || c.pid) + " · " + c.pid),
          tint: c.tint || GOLD, lines: c.lines,
          minX: v0.x, maxX: v1.x, minY: v0.y, maxY: v1.y,
          pivot: toWorld(W * 0.60, 0).x,
        });
      }
    }

    /* ---- screen-space chrome -------------------------------------------- */
    ctx.setTransform(k, 0, 0, k, 0, 0);

    if (opts.focus) ctx.drawImage(focusLayer(world, opts.focus), 0, 0);

    if (opts.chrome !== false) {
      const title = opts.title || ("Day " + day);
      const clockText = opts.clockText || hhmm(clockMin);
      const stats = opts.stats || defaultStats(world, st, ds);
      const cap = opts.caption === undefined ? defaultCaption(world, st, ds) : opts.caption;
      const rightEdge = opts.inspect ? 1180 : W - 96;
      const capW = opts.inspect ? 660 : 780;

      const sigTop = title + "|" + (opts.sub || "") + "|" + rightEdge + "|" +
        stats.map((s) => s.n + "~" + s.l + "~" + (s.c ? s.c.join(",") : "")).join(";");
      ctx.drawImage(band(world, "top", sigTop, 0, BAND_TOP_H, function (g) {
        titleBlock(g, title, opts.sub);
        statStrip(g, stats, rightEdge);
      }), 0, 0);
      clockChip(ctx, title, clockText);

      const sigBot = capW + "|" + cap.map(
        (l) => (l.s || 22) + "~" + (l.w || "") + "~" + (l.a === undefined ? "" : l.a) + "~" + l.t).join(";");
      ctx.drawImage(band(world, "bot", sigBot, BAND_BOT_Y, H - BAND_BOT_Y, function (g) {
        captionBlock(g, cap, capW);
        provenanceFooter(g);
      }), 0, BAND_BOT_Y);
    } else {
      provenanceFooter(ctx);
    }

    if (opts.inspect && world.trav[opts.inspect]) {
      drawDossier(ctx, world, opts.inspect, day, W - DOSSIER_W - 56, 70);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    world._lastFrame = { day: day, clock: clockMin, cam: cam || null, k: k, st: st };
    return st;
  }

  // the always-on caption: three sentences, every number read off this frame
  const isAre = (n) => n === 1 ? " is " : " are ";
  function defaultCaption(world, st, ds) {
    const n = world.pids.length;
    const onRoad = st.driving.length + st.queued.length;
    const first = world.dayStats[world.days[0]];
    let l1;
    if (!onRoad && !st.arrived.length) {
      const next = st.home.length ? Math.min.apply(null, st.home.map((r) => r.dep)) : null;
      l1 = "Nobody has left yet this morning" +
        (next === null ? "." : " — the first recorded departure is at " + hhmm(next) + ".");
    } else {
      l1 = "This minute: " +
        (st.arrived.length ? st.arrived.length + " of the " + n + " have arrived, " : "nobody has arrived yet, ") +
        (onRoad ? onRoad + isAre(onRoad) + "on the road" : "nobody is on the road") +
        (st.queued.length
          ? " and " + st.queued.length + isAre(st.queued.length) + "stopped at the bridge."
          : ", nobody is queueing.");
    }
    const l2 = "Across the whole morning " +
      (ds.nB === 0 ? "nobody takes the ring road" : ds.nB + " take the ring road") +
      " and the longest wait at the bridge is " + ds.maxDelayA.toFixed(1) + " min" +
      (ds.day === world.days[0] ? "." : " (day " + world.days[0] + ": " +
        (first.nB === 0 ? "nobody" : first.nB) + ", " + first.maxDelayA.toFixed(1) + " min).");
    const l3 = "Windows light 20 minutes before each recorded departure and go out on arrival: " +
      st.lit.length + " lit now. Click a house or a car for that traveler's dossier.";
    return [{ t: l1, s: 22 }, { t: l2, s: 16.5, a: 0.62 }, { t: l3, s: 15, a: 0.48 }];
  }

  function defaultStats(world, st, ds) {
    return [
      { n: String(st.driving.filter((d) => d.route === "A").length), l: "DRIVING ROUTE A" },
      { n: String(st.queued.length), l: "WAITING AT THE BRIDGE", c: HOT },
      { n: String(st.driving.filter((d) => d.route === "B").length), l: "ON THE RING ROAD", c: CY },
      { n: String(st.arrived.length), l: "ARRIVED" },
    ];
  }

  /* ============================================================ hit test */

  function hitTest(world, px, py) {
    // px,py in design space (1920x1080), already un-cammed by the caller
    let best = null;
    for (const c of world._cars) {
      const d = Math.hypot(c.x - px, c.y - py);
      const r = Math.max(9, 11 * c.s);
      if (d < r && (!best || d < best.d)) best = { kind: "car", pid: c.pid, d: d };
    }
    if (best) return best;
    for (const plot of world.plotList) {
      const dx = plot.x - px, dy = (plot.y - plot.g.hh * 0.4) - py;
      const d = Math.hypot(dx, dy * 1.4);
      const r = Math.max(8, plot.g.w * 0.62);
      if (d < r && (!best || d < best.d)) best = { kind: "house", pid: plot.pid, d: d };
    }
    if (best) return best;
    for (const pid of world.pids) {
      const c = world.district[pid];
      if (Math.abs(c.x - px) < 3.4 && Math.abs(c.y - py) < 4) return { kind: "desk", pid: pid, d: 0 };
    }
    return null;
  }

  // undo the camera so a click in canvas pixels lands in design space
  function screenToDesign(world, cx, cy, canvasW) {
    const f = world._lastFrame;
    const k = canvasW / W;
    let x = cx / k, y = cy / k;
    const cp = camPlace(f && f.cam);
    return { x: (x - cp.tx) / cp.z, y: (y - cp.ty) / cp.z };
  }

  /* =========================================================== the story */

  function ordinal(n) {
    const s = ["th", "st", "nd", "rd"], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }

  /* Five beats. Every number in every caption is read out of the world that
   * was just built from the three published files. The travelers featured are
   * the ones whose own logs carry the beat — see the comment on each. */
  function beats(world) {
    const days = world.days;
    const D1 = days[0], DN = days[days.length - 1];
    const s1 = world.dayStats[D1], sN = world.dayStats[DN];
    const out = [];

    /* ---- 1. the day-1 jam, at the recorded peak minute ----------------- */
    {
      const peak = s1.peakQueue;
      const clock = peak.minute;
      const st = stateAt(world, D1, clock);
      // the traveler at the head of the queue at this exact minute
      const head = st.queued.length ? st.queued[0].rec : null;
      const chips = [];
      if (head) {
        chips.push({
          pid: head.pid, tint: HOT, ring: true, dy: -150,
          lines: [
            "at the head of the queue — reached the bridge at " + hhmm(head.tPinch),
            "waits " + head.delay.toFixed(1) + " min here before crossing",
          ],
        });
      }
      // Hans t0122 is featured only where his own log carries it: he is inside
      // this queue at this minute, with the joint-4th longest wait of day 1.
      const hans = world.trav["t0122"] && world.trav["t0122"].days[D1];
      if (hans && clock >= hans.tPinch && clock < hans.tRelease) {
        const rk = s1.delayRank.rank["t0122"], tie = s1.delayRank.tie[rk];
        chips.push({
          pid: "t0122", tint: [255, 176, 130], dy: 128,
          lines: [
            "left " + hhmm(hans.dep) + " · stopped at the bridge since " + hhmm(hans.tPinch),
            hans.delay.toFixed(1) + " min of waiting — the " + ordinal(rk) + " longest of day " + D1 +
              (tie > 1 ? " (shared with " + (tie - 1) + " other)" : ""),
          ],
        });
      }
      out.push({
        id: "jam", day: D1, clock: clock,
        title: "Day " + D1, sub: "Everyone takes route A. The bridge cannot take them.",
        cam: { fx: PINCH.x - 170, fy: PINCH.y + 40, zoom: 1.20 },
        focus: { x: PINCH.x - 130, y: PINCH.y + 20, r: 330, a: 0.44 },
        chips: chips,
        stats: function (st2) {
          return [
            { n: String(st2.driving.filter((d) => d.route === "A").length), l: "DRIVING ROUTE A" },
            { n: String(st2.queued.length), l: "STOPPED AT THE BRIDGE", c: HOT },
            { n: String(s1.nB), l: "ON THE RING ROAD", c: CY },
            { n: String(st2.arrived.length), l: "ARRIVED" },
          ];
        },
        caption: function (st2) {
          return [
            { t: "The worst minute of day " + D1 + ": " + st2.queued.length +
                " of the " + world.pids.length + " are stopped at the bridge at once. " +
                "The longest wait anyone records today is " + s1.maxDelayA.toFixed(1) + " minutes, " +
                "and nobody has tried the ring road yet.", s: 22 },
            { t: "A house is lit from 20 minutes before its recorded departure until that traveler arrives. " +
                st2.lit.length + " are lit right now; " + st2.arrived.length + " are already at a desk.", s: 16.5, a: 0.60 },
          ];
        },
      });
    }

    /* ---- 2. day 2, the ring road is found ------------------------------ */
    if (days.length > 1) {
      const D2 = days[1], s2 = world.dayStats[D2];
      const first3 = s2.firstOntoB.slice(0, 3);
      // put the clock where all three of them are visibly out on the ring
      const deps = first3.map((p) => world.trav[p].days[D2].dep);
      const clock = Math.round(Math.max.apply(null, deps) + 7);
      const tints = [CY, [120, 224, 240], [176, 240, 250]];
      const chips = first3.map(function (pid, i) {
        const r = world.trav[pid].days[D2];
        return {
          pid: pid, tint: tints[i], ring: i === 0,
          dy: i === 0 ? -140 : (i === 1 ? 120 : -190),
          lines: [
            "left " + hhmm(r.dep) + " · " + r.travel.toFixed(0) + " min on the ring, no bridge",
            i === 0 ? "the first of the " + s2.nB + " onto it this morning"
                    : "the " + ordinal(i + 1) + " out of the gate today",
          ],
        };
      });
      out.push({
        id: "discovery", day: D2, clock: clock,
        title: "Day " + D2, sub: "Overnight, some of them found the long way round.",
        cam: { fx: ROUTE_B.at(0.42).x, fy: ROUTE_B.at(0.42).y - 40, zoom: 1.16 },
        focus: { x: ROUTE_B.at(0.42).x, y: ROUTE_B.at(0.42).y, r: 420, a: 0.34 },
        chips: chips,
        stats: function (st2) {
          return [
            { n: String(s2.nB), l: "TAKE THE RING ROAD TODAY", c: CY },
            { n: s2.maxDelayA.toFixed(1), l: "WORST WAIT, MINUTES", c: HOT },
            { n: s1.maxDelayA.toFixed(1), l: "WORST WAIT ON DAY " + D1 },
          ];
        },
        caption: function () {
          const l = first3.map((p) => (world.name[p] || p) + " (" + p + ") " +
            hhmm(world.trav[p].days[D2].dep)).join(", ");
          return [
            { t: s2.nB + " of the " + world.pids.length + " take the ring road on day " + D2 +
                " — none did on day " + D1 + ". The worst wait at the bridge drops from " +
                s1.maxDelayA.toFixed(1) + " minutes to " + s2.maxDelayA.toFixed(1) + ".", s: 22 },
            { t: "First three onto it, by their own recorded departure: " + l + ".", s: 16.5, a: 0.60 },
          ];
        },
      });
    }

    /* ---- 3. the switcher who goes back --------------------------------- */
    {
      // Hans t0122 if his log carries it; otherwise the busiest switcher.
      let pid = "t0122";
      if (!world.trav[pid] || world.trav[pid].switches < 2) {
        pid = world.pids.slice().sort((a, b) => world.trav[b].switches - world.trav[a].switches)[0];
      }
      const t = world.trav[pid];
      // the first morning he is back on route A after his first ring-road run
      let backDay = null;
      for (const r of t.list) {
        if (t.firstDayOnB && r.day > t.firstDayOnB && r.route === "A") { backDay = r.day; break; }
      }
      const day = backDay || t.list[Math.min(4, t.list.length - 1)].day;
      const rec = t.days[day];
      const clock = Math.round(rec.dep + rec.span * 0.55);
      const ds3 = world.dayStats[day];
      out.push({
        id: "switcher", day: day, clock: clock,
        title: "Day " + day, sub: "The ring road is not a one-way door.",
        cam: { fx: ROUTE_A.at(0.42).x, fy: ROUTE_A.at(0.42).y, zoom: 1.20 },
        focus: { x: ROUTE_A.at(0.42).x, y: ROUTE_A.at(0.42).y, r: 340, a: 0.40 },
        chips: [{
          pid: pid, tint: [255, 176, 130], ring: true, dy: -150,
          lines: [
            "back on route A · left " + hhmm(rec.dep) + " · " + rec.travel.toFixed(1) + " min today",
            t.switches + " switches in " + t.list.length + " mornings · " + t.daysOnB + " of them on the ring road",
          ],
        }],
        stats: function (st2) {
          return [
            { n: String(ds3.nB), l: "ON THE RING ROAD TODAY", c: CY },
            { n: ds3.maxDelayA.toFixed(1), l: "WORST WAIT, MINUTES", c: HOT },
            { n: String(t.switches), l: (world.name[pid] || pid).toUpperCase() + "'S SWITCHES" },
          ];
        },
        caption: function () {
          const route = t.list.map((r) => r.route === "B" ? "B" : "A").join("·");
          return [
            { t: (world.name[pid] || pid) + " (" + pid + ") tried the ring road on day " + t.firstDayOnB +
                ", stayed on it for " + (day - t.firstDayOnB) + " mornings, and is back on route A today. " +
                "Over the " + t.list.length + " recorded mornings he switches " + t.switches + " times.", s: 22 },
            { t: "His recorded route, morning by morning:  " + route +
                "   (A = route A, B = the ring road)", s: 15, a: 0.55 },
          ];
        },
      });
    }

    /* ---- 4. the ratchet: one lit window --------------------------------- */
    {
      // the traveler whose departure moves earliest of the 300
      let pid = world.pids[0];
      for (const p of world.pids) if (world.trav[p].drift < world.trav[pid].drift) pid = p;
      const t = world.trav[pid];
      // the mid-study morning where he is alone on the road
      let pick = null;
      for (const r of t.list) {
        const mid = Math.floor(r.dep + r.span * 0.5);
        const st2 = stateAt(world, r.day, mid);
        if (st2.lit.length === 1 && st2.lit[0] === pid) { pick = { day: r.day, clock: mid, rec: r }; }
        if (pick && r.day >= Math.round(world.days.length * 0.6)) break;
      }
      if (!pick) {
        const r = t.list[Math.floor(t.list.length * 0.6)];
        pick = { day: r.day, clock: Math.floor(r.dep + r.span * 0.5), rec: r };
      }
      const rec = pick.rec;
      const loc = locate(world, pick.day, pick.clock, pid);
      const plot = world.plots[pid];
      // frame the lit house AND the car — the beat is that they are the only two
      const fx = plot ? (plot.x + loc.x) / 2 : loc.x;
      const fy = plot ? (plot.y + loc.y) / 2 : loc.y;
      const spread = plot ? Math.hypot(plot.x - loc.x, plot.y - loc.y) : 200;
      const zoom = clamp(1500 / (spread + 420), 1.0, 1.30);
      const ex = cardExcerpt(world, pid, 210);
      out.push({
        id: "ratchet", day: pick.day, clock: pick.clock,
        title: "Day " + pick.day, sub: "One lit window. One car. Nobody else has left.",
        cam: { fx: fx, fy: fy - 40, zoom: zoom },
        focus: { x: loc.x, y: loc.y, r: 300, a: 0.42 },
        chips: [{
          pid: pid, tint: [255, 214, 158], ring: true, dy: -150,
          lines: [
            "left " + hhmm(rec.dep) + " · route " + (rec.route === "A" ? "A" : "ring") + " · " +
              rec.travel.toFixed(1) + " min, nobody to queue behind",
            "day " + world.days[0] + " he left " + hhmm(t.first.dep) + " · day " + t.last.day +
              " he leaves " + hhmm(t.last.dep),
          ],
        }],
        stats: function (st2) {
          return [
            { n: String(st2.driving.length + st2.queued.length), l: "ON THE ROAD" },
            { n: String(st2.lit.length), l: st2.lit.length === 1 ? "LIT WINDOW" : "LIT WINDOWS" },
            { n: String(st2.home.length), l: "STILL AT HOME" },
          ];
        },
        caption: function (st2) {
          const nextDep = st2.home.length
            ? Math.min.apply(null, st2.home.map((r) => r.dep)) - pick.clock : null;
          const mins = rec.travel.toFixed(0);
          const art = (mins[0] === "8" || mins === "11" || mins === "18") ? "an " : "a ";
          return [
            { t: (world.name[pid] || pid) + " (" + pid + ") left at " + hhmm(rec.dep) + " and is " +
                Math.round(pick.clock - rec.dep) + " minutes into " + art + mins + "-minute drive." +
                (nextDep !== null ? " The next of the " + world.pids.length + " does not leave for another " +
                  Math.round(nextDep) + " minutes." : ""), s: 22 },
            { t: "He has moved his departure " + durHM(t.drift) + " earlier since day " + world.days[0] + ", and " +
                (t.daysOnB ? "takes the ring road on " + t.daysOnB + " of " + t.list.length + " mornings"
                           : "has never once tried the ring road") + ".", s: 16.5, a: 0.60 },
            { t: "“" + (ex.parts[0] || "") + "”  — verbatim from his biography card", s: 15, a: 0.52 },
          ];
        },
        inspect: pid,
      });
    }

    /* ---- 5. day 20, the spread ------------------------------------------ */
    {
      const clock = 495;
      const widen = sN.depSD / s1.depSD;
      const spots = sN.firstOntoB.slice(0, 0); // filled below from who is en route
      out.push({
        id: "spread", day: DN, clock: clock,
        title: "Day " + DN, sub: "Same clock as day " + D1 + ". Nineteen mornings of learning later.",
        cam: null,
        chips: null,
        chipsFrom: function (st2) {
          const onB = st2.driving.filter((d) => d.route === "B")
            .sort((a, b) => a.rec.dep - b.rec.dep).slice(0, 3);
          return onB.map(function (d, i) {
            const t = world.trav[d.rec.pid];
            return {
              pid: d.rec.pid, tint: CY, dy: i === 1 ? 120 : -120 - i * 40,
              lines: [
                "ring road · left " + hhmm(d.rec.dep) + " · " + d.rec.travel.toFixed(0) + " min, no queue ever",
                "first tried it on day " + t.firstDayOnB + " · " + t.daysOnB + " of " + t.list.length + " mornings since",
              ],
            };
          });
        },
        stats: function (st2) {
          return [
            { n: String(sN.nB), l: "ON THE RING ROAD", c: CY },
            { n: sN.maxDelayA.toFixed(1), l: "WORST WAIT, MINUTES", c: HOT },
            { n: widen.toFixed(1) + "×", l: "WIDER DEPARTURE SPREAD" },
            { n: String(st2.arrived.length), l: "ALREADY ARRIVED" },
          ];
        },
        caption: function (st2) {
          return [
            { t: sN.nB + " travelers take the ring road on day " + DN + " and the worst wait at the bridge is " +
                sN.maxDelayA.toFixed(1) + " minutes, down from " + s1.maxDelayA.toFixed(1) + " on day " + D1 + ".", s: 22 },
            { t: "Departures are spread " + widen.toFixed(1) + " times wider than on day " + D1 +
                " (" + s1.depSD.toFixed(1) + " → " + sN.depSD.toFixed(1) + " minutes of spread), so at the same clock " +
                st2.arrived.length + " people are already at work and their houses have gone dark.", s: 16.5, a: 0.60 },
          ];
        },
      });
    }

    return out;
  }

  function beatOptions(world, beat) {
    const st = stateAt(world, beat.day, beat.clock);
    return {
      title: beat.title, sub: beat.sub,
      clockText: hhmm(beat.clock),
      cam: beat.cam, focus: beat.focus,
      chips: beat.chipsFrom ? beat.chipsFrom(st) : beat.chips,
      stats: beat.stats ? beat.stats(st) : undefined,
      caption: beat.caption ? beat.caption(st) : undefined,
    };
  }

  /* ============================================================== export */

  return {
    W: W, H: H, PROVENANCE: PROVENANCE,
    SANS: SANS, SERIF: SERIF,
    setCanvasFactory: setCanvasFactory,
    buildWorld: buildWorld,
    buildStatic: buildStatic,
    buildStaticSteps: buildStaticSteps,
    render: render,
    stateAt: stateAt,
    locate: locate,
    beats: beats,
    beatOptions: beatOptions,
    hitTest: hitTest,
    screenToDesign: screenToDesign,
    renderDossier: renderDossier,
    cardExcerpt: cardExcerpt,
    dataDiary: dataDiary,
    monoColor: monoColor,
    hhmm: hhmm, durHM: durHM, lightOf: lightOf, palette: palette,
    ROUTE_A: ROUTE_A, ROUTE_B: ROUTE_B, PINCH: PINCH, PLAZA: PLAZA,
    colors: { INK: INK, GOLD: GOLD, CY: CY, HOT: HOT },
  };
});
