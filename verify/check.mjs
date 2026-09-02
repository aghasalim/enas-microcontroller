// Recompute search-log aggregates in JavaScript and check them against README.md.
//
// Covers candidate counts, training minutes, best fitness/accuracy, seed
// comparison, and budget figures.

import { readFileSync } from "fs";
import { join } from "path";

const root = process.argv[2] || ".";

// Minimal CSV parser: split on commas respecting quoted fields.
function parseCSV(text) {
  const lines = text.trim().split("\n");
  const header = splitRow(lines[0]);
  return lines.slice(1).map((line) => {
    const vals = splitRow(line);
    const obj = {};
    header.forEach((h, i) => (obj[h] = vals[i]));
    return obj;
  });
}
function splitRow(line) {
  const out = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQ && line[i + 1] === '"') { cur += '"'; i++; continue; }
      inQ = !inQ; continue;
    }
    if (c === "," && !inQ) { out.push(cur); cur = ""; continue; }
    cur += c;
  }
  out.push(cur);
  return out;
}

const csv = readFileSync(join(root, "results", "search_log.csv"), "utf8").replace(/\r/g, "");
const rows = parseCSV(csv);
const readme = readFileSync(join(root, "README.md"), "utf8");

for (const r of rows) {
  r.params = parseInt(r.params, 10);
  r.macs = parseInt(r.macs, 10);
  r.peak_act = parseInt(r.peak_act, 10);
  r.acc = parseFloat(r.acc);
  r.fitness = r.fitness === "-inf" ? -Infinity : parseFloat(r.fitness);
  r.deployable = parseInt(r.deployable, 10);
  r.train_s = parseFloat(r.train_s);
}

const ok = rows.filter((r) => r.deployable === 1);
const seed = rows[0];
const best = ok.reduce((a, b) => (b.fitness > a.fitness ? b : a));

const rejected = rows.filter((r) => r.deployable === 0).length;

// Duplicated genomes.
const genomeCounts = {};
for (const r of rows) genomeCounts[r.genome] = (genomeCounts[r.genome] || 0) + 1;
const dupCount = Object.values(genomeCounts).filter((c) => c > 1).length;

const totalTrainMin = Math.round(rows.reduce((s, r) => s + r.train_s, 0) / 60);

// Dup training minutes.
const seenGenomes = {};
let dupTrainS = 0;
for (const r of rows) {
  if (r.genome in seenGenomes) dupTrainS += r.train_s;
  else seenGenomes[r.genome] = true;
}
const dupTrainMin = Math.round(dupTrainS / 60);

// Insert operator.
const insertRows = rows.filter((r) => r.mutation.startsWith("insert"));
const insertTrainable = insertRows.reduce((s, r) => s + r.deployable, 0);

const gapPoints = 100 * (best.acc - seed.acc);
const paramDiff = best.params - seed.params;
const macPct = 100 * (1 - best.macs / seed.macs);

const bestKB = (best.params + best.peak_act) / 1024;
const worstKB = Math.max(...ok.map((r) => (r.params + r.peak_act) / 1024));
const headroom = 250 / worstKB;

const moreAcc = ok.filter((r) => r.acc > best.acc);
const accRank = moreAcc.length + 1;

const want = [
  `${rows.length} candidates`,
  `${rows.length - 1} child slots`,
  `${rejected} rejected`,
  `${dupCount} evaluated twice`,
  `${totalTrainMin} minutes of training`,
  `${dupTrainMin} minutes of duplicated`,
  `${ok.length} trained candidates`,
  `drawn ${insertRows.length} times and produced ${insertTrainable} trainable`,
  `**${best.acc.toFixed(4)}**`,
  `**${best.fitness.toFixed(4)}**`,
  `${gapPoints.toFixed(1)} accuracy points`,
  `${paramDiff} more`,
  `${macPct.toFixed(1)}% fewer`,
  `generation ${best.gen}, candidate ${best.cand}`,
  `${bestKB.toFixed(1)} KB`,
  `${worstKB.toFixed(1)} KB`,
  `${headroom.toFixed(1)}x`,
  `${accRank}th by accuracy`,
];

let failures = 0;
for (const x of want) {
  const hit = readme.includes(x);
  console.log(`  ${hit ? "ok  " : "FAIL"} ${x}`);
  if (!hit) failures++;
}
if (failures > 0) {
  console.log(`${failures} of ${want.length} figures are not in README.md as written`);
  process.exit(1);
}
console.log(`JavaScript reproduces all ${want.length} figures from the search log`);
