#!/usr/bin/env node
/* make_bundle.mjs — regenerate mobility/app/data-bundle.js
 *
 * Why this exists: the viewer must also work when index.html is opened by
 * double-clicking it (file://), where fetch() is blocked by the browser. The
 * bundle is a plain <script> fallback holding the three published data files.
 *
 * What it does: READS the three files in mobility/app/data/ and writes their
 * exact bytes out as JavaScript string literals. It never edits them, and it
 * never adds, drops or reformats a single record — the loader parses the
 * bundle strings with the same code path it uses for a fetch() response, so
 * fetched and bundled data are identical by construction.
 *
 *   node mobility/app/make_bundle.mjs
 */
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(HERE, "data");
const OUT = path.join(HERE, "data-bundle.js");

const FILES = [
  ["trajectories", "trajectories.jsonl"],
  ["profiles", "profiles.json"],
  ["cards", "cards.jsonl"],
];

const parts = [];
const manifest = [];

for (const [key, file] of FILES) {
  const abs = path.join(DATA, file);
  const buf = fs.readFileSync(abs);                       // read only, never write
  const text = buf.toString("utf8");
  const sha = crypto.createHash("sha256").update(buf).digest("hex");
  manifest.push({ key, file, bytes: buf.length, sha256: sha });
  // JSON.stringify of the raw text gives a JS string literal of the exact bytes
  parts.push("  " + JSON.stringify(key) + ": " + JSON.stringify(text) + ",");
}

const header = [
  "/* data-bundle.js — GENERATED, do not edit by hand.",
  " * Run: node mobility/app/make_bundle.mjs",
  " *",
  " * Verbatim copies of the three published files in mobility/app/data/, held",
  " * as JS strings so the viewer also works from file:// where fetch() fails.",
  " * The viewer prefers fetch() and only falls back to this.",
  " *",
  ...manifest.map((m) =>
    " *   " + m.file.padEnd(22) + m.bytes + " bytes   sha256 " + m.sha256),
  " */",
].join("\n");

const body = [
  header,
  "window.__GBM_BUNDLE__ = {",
  '  generated: "' + new Date().toISOString().slice(0, 10) + '",',
  "  manifest: " + JSON.stringify(manifest) + ",",
  ...parts,
  "};",
  "",
].join("\n");

fs.writeFileSync(OUT, body);
console.log("wrote " + OUT + "  (" + body.length + " bytes)");
for (const m of manifest) console.log("  " + m.file + "  " + m.bytes + " B  " + m.sha256.slice(0, 16));
