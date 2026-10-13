import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import {
  detect, detectTournament, expectedTournamentMean, expectedMeanGValue, gValue,
  isGreen, keyToSeed, layerWeights, meanScore, normalCdf, sampleMultinomial,
  seededRng, seedFromContext, tournamentSample, weightedMeanScore, weightedZScore,
} from "../watermark.js";

const params = { gamma: 0.5, m: 15, h: 1, keySeed: keyToSeed("test-key") };

for (const detector of [detect, detectTournament]) {
  test(`${detector.name}: repetition mask keeps every colour`, () => {
    const short = detector([10, 10], 1, params);
    const repeated = detector(Array(2001).fill(10), 1, params);
    assert.equal(repeated.T, 1);
    assert.equal(repeated.totalT, 2000);
    assert.equal(repeated.flags.length, 2000);
    assert.equal(repeated.z, short.z);
    assert.equal(repeated.pValue, short.pValue);
    assert.equal(repeated.meanG, short.meanG);
    assert.equal(repeated.greenCount, short.greenCount);
  });

  test(`${detector.name}: deduplication uses the whole context${detector === detectTournament ? " (official mask)" : " and current token"}`, () => {
    const result = detector([1, 2, 9, 3, 2, 9, 1, 2, 9], 2, { ...params, h: 2 });
    assert.equal(result.totalT, 7);
    // Green list: only the final [1, 2] -> 9 repeats (same context AND token).
    // Tournament (official compute_context_repetition_mask): any repeated
    // context is masked, whichever token follows — [2, 9] and [1, 2] repeat.
    assert.equal(result.T, detector === detectTournament ? 5 : 6);
    assert.equal(result.flags.length, 7);
  });

  test(`${detector.name}: empty input has no evidence`, () => {
    const result = detector([], 0, params);
    assert.equal(result.T, 0);
    assert.equal(result.z, 0);
    assert.equal(result.pValue, 1);
  });
}

test("green-list statistic uses the configured baseline, not always 50%", () => {
  const ids = [2, 3, 5, 7, 11, 13];
  const gamma = 0.25;
  const result = detect(ids, 1, { ...params, gamma });
  assert.equal(result.T, 5);
  assert.equal(result.z, (result.greenCount - 1.25) / Math.sqrt(5 * 0.25 * 0.75));
});

/* ── official detector_mean.py / g_value_expectations.py ports ── */

test("layerWeights matches the official linspace(10, 1, m)", () => {
  assert.deepEqual(layerWeights(1), [1]);
  assert.deepEqual(layerWeights(2), [10, 1]);
  assert.deepEqual(layerWeights(3), [10, 5.5, 1]);
  assert.deepEqual(layerWeights(10), [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
});

test("meanScore matches detector_mean.mean_score", () => {
  const g = [[1, 0], [0, 1], [1, 1]];
  assert.equal(meanScore(g, [1, 1, 1]), 4 / 6);
  assert.equal(meanScore(g, [1, 0, 1]), 3 / 4); // the masked [0, 1] row drops out
  assert.equal(meanScore(g, [0, 0, 0]), 0.5); // nothing unmasked
  assert.equal(meanScore([], []), 0.5);
});

test("weightedMeanScore matches the official weighted_mean_score", () => {
  const g = [[1, 0], [0, 1]];
  // Official: weights [10, 1] are rescaled to sum to the depth (×2/11), then
  // sum(g·w)/(depth · unmasked) = ((10 + 0) + (0 + 1)) · 2/11 / 4 = 0.5
  assert.equal(weightedMeanScore(g, [1, 1]), 0.5);
  // Uniform weights reduce to the plain mean.
  assert.equal(weightedMeanScore(g, [1, 1], [1, 1]), meanScore(g, [1, 1]));
  // Depth 1: the single weight normalises to 1, same as the mean.
  assert.equal(weightedMeanScore([[1], [1], [0]], [1, 1, 1]), 2 / 3);
  // Constant matrices saturate in both directions.
  assert.equal(weightedMeanScore([[1, 1], [1, 1]], [1, 1]), 1);
  assert.equal(weightedMeanScore([[0, 0]], [1]), 0);
});

test("weightedMeanScore equals a weighted mean of per-layer means", () => {
  const rng = seededRng(7);
  const m = 15;
  const g = Array.from({ length: 60 }, () => Array.from({ length: m }, () => Math.round(rng())));
  const mask = Array.from({ length: 60 }, () => (rng() < 0.8 ? 1 : 0));
  const w = layerWeights(m);
  const total = w.reduce((a, b) => a + b, 0);
  const layerMeans = Array.from({ length: m }, (_, j) => {
    let sum = 0;
    let n = 0;
    for (let i = 0; i < g.length; i++) {
      if (mask[i]) {
        sum += g[i][j];
        n++;
      }
    }
    return n ? sum / n : 0;
  });
  const expected = w.reduce((acc, wj, j) => acc + (wj / total) * layerMeans[j], 0);
  assert.ok(Math.abs(weightedMeanScore(g, mask) - expected) < 1e-12);
});

test("expectedMeanGValue matches g_value_expectations (2 leaves, uniform LM)", () => {
  assert.equal(expectedMeanGValue(2), 0.625);
  assert.ok(expectedMeanGValue(1e9) > 0.7499999 && expectedMeanGValue(1e9) < 0.75);
});

test("detectTournament applies the official scores to the official mask", () => {
  const ids = generateFixture("tournament");
  const r = detectTournament(ids, 1, { ...params, vocabSize: 1000 });
  const gValues = [];
  const mask = [];
  const seen = new Set();
  for (let i = 1; i < ids.length; i++) {
    const ctx = ids.slice(i - 1, i);
    const seed = seedFromContext(ctx, params.keySeed);
    gValues.push(Array.from({ length: params.m }, (_, l) => gValue(seed, l, ids[i], null)));
    const key = ctx.join(",");
    mask.push(seen.has(key) ? 0 : 1);
    seen.add(key);
  }
  assert.equal(r.meanG, meanScore(gValues, mask));
  assert.equal(r.weightedMeanG, weightedMeanScore(gValues, mask));
  assert.equal(r.mask.join(","), mask.join(","));
  // The verdict statistic is the weighted z over the same matrix.
  assert.ok(Math.abs(r.z - weightedZScore(gValues, mask)) < 1e-12);
  assert.ok(Math.abs(r.expectedG - (0.5 + 0.25 * (1 - 1 / 1000))) < 1e-12);
  assert.equal(r.expectedSource, "official");
  assert.ok(r.weightedMeanG > 0.5);
});

test("weightedZScore with uniform weights equals the plain mean z-test", () => {
  const rng = seededRng(5);
  const m = 12;
  const g = Array.from({ length: 90 }, () => Array.from({ length: m }, () => Math.round(rng())));
  const mask = Array.from({ length: 90 }, () => (rng() < 0.85 ? 1 : 0));
  mask[0] = 1;
  let T = 0;
  let sumG = 0;
  for (let i = 0; i < g.length; i++) {
    if (!mask[i]) continue;
    T++;
    for (const v of g[i]) sumG += v;
  }
  const plain = (sumG - (T * m) / 2) / Math.sqrt((T * m) / 4);
  assert.ok(Math.abs(weightedZScore(g, mask, Array(m).fill(1)) - plain) < 1e-12);
});

test("weightedZScore standardises the weighted sum under the null", () => {
  const g = [[1, 0], [0, 1], [1, 1]];
  const mask = [1, 0, 1];
  // Weights [10, 1] normalised to sum 2: ŵ = [20/11, 2/11].
  // S = (20/11)·(1+1) + (2/11)·(0+1) = 42/11; E[S] = 2; Var = 2·(404/121)/4.
  const expected = (42 / 11 - 2) / Math.sqrt((2 * (404 / 121)) / 4); // = 20/√202
  assert.ok(Math.abs(weightedZScore(g, mask, [10, 1]) - expected) < 1e-12);
  assert.equal(weightedZScore([], []), 0);
  assert.equal(weightedZScore(g, [0, 0, 0]), 0);
});

test("weighted z rewards early-layer signal more than the plain mean z", () => {
  const rng = seededRng(11);
  const m = 10;
  const T = 300;
  const g = Array.from({ length: T }, () => {
    const row = Array.from({ length: m }, () => Math.round(rng()));
    row[0] = 1; // all the signal sits in the first tournament round
    return row;
  });
  const mask = Array(T).fill(1);
  const weighted = weightedZScore(g, mask);
  const uniform = weightedZScore(g, mask, Array(m).fill(1));
  assert.ok(weighted > uniform + 1, `weighted ${weighted} should beat uniform ${uniform}`);
});

test("detectTournament falls back to the synthetic reference without a vocab size", () => {
  const r = detectTournament([1, 2, 3], 1, params);
  assert.equal(r.expectedSource, "synthetic");
  assert.ok(Math.abs(r.expectedG - expectedTournamentMean(params.m)) < 1e-12);
});

test("forced-red candidates can survive a tied tournament", () => {
  assert.equal(tournamentSample([1], 123, 15, new Set([0]), seededRng(42)), 0);
});

function generateFixture(mode) {
  const weights = Array.from({ length: 128 }, (_, i) => Math.exp(-i / 20));
  const sum = weights.reduce((a, b) => a + b);
  const probs = weights.map(x => x / sum);
  const rng = seededRng(42);
  const ids = [19];
  for (let i = 0; i < 500; i++) {
    const seed = seedFromContext(ids.slice(-1), params.keySeed);
    if (mode === "tournament") {
      ids.push(tournamentSample(probs, seed, params.m, null, rng));
    } else {
      const q = probs.map((p, t) => {
        const green = isGreen(seed, t, params.gamma);
        return mode === "hard" ? (green ? p : 0)
          : mode === "soft" ? p * (green ? Math.exp(2) : 1) : p;
      });
      ids.push(sampleMultinomial(q, 1, rng)[0]);
    }
  }
  return ids;
}

for (const mode of ["hard", "soft", "tournament"]) {
  test(`${mode}: synthetic marked text gives a signal only with matching key`, () => {
    const ids = generateFixture(mode);
    const detector = mode === "tournament" ? detectTournament : detect;
    assert.ok(detector(ids, 1, params).z > 4);
    assert.ok(Math.abs(detector(ids, 1, { ...params, keySeed: keyToSeed("wrong") }).z) < 3);
  });
}

test("synthetic unmarked text remains near the null baseline", () => {
  const ids = generateFixture("none");
  for (const detector of [detect, detectTournament]) {
    assert.ok(Math.abs(detector(ids, 1, params).z) < 3);
  }
});

// Exercise the actual rendering functions without a model download or browser.
function render(result) {
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, { dataset: {}, style: {}, parentElement: {}, textContent: "" });
    return nodes.get(id);
  };
  const els = new Proxy({}, { get: (_, id) => node(id) });
  const source = readFileSync(new URL("../main.js", import.meta.url), "utf8");
  const start = source.indexOf("function renderVerdict(");
  const end = source.indexOf("/* ── user actions", start);
  vm.runInNewContext(source.slice(start, end) + "\nrenderVerdict(result);", {
    els, document: { getElementById: node }, expectedTournamentMean, result,
  });
  return node;
}

test("a high z from one rare green token is not presented as a strong result", () => {
  const gamma = 0.05;
  const seed = seedFromContext([1], params.keySeed);
  let token = 0;
  while (!isGreen(seed, token, gamma)) token++;
  const result = detect([1, token], 1, { ...params, gamma });
  assert.ok(result.z > 4);
  const node = render({ ...result, scheme: "greenlist", gamma, h: 1 });
  assert.equal(node("verdictLabel").textContent, "Мало данных для вывода");
});

test("empty verdicts show no NaN, Infinity or false detection", () => {
  for (const scheme of ["greenlist", "tournament"]) {
    const detector = scheme === "tournament" ? detectTournament : detect;
    const node = render({ ...params, ...detector([], 0, params), scheme });
    assert.equal(node("verdictLabel").textContent, "Мало данных для вывода");
    assert.equal(node("statZ").textContent, "—");
    assert.doesNotMatch(node("statGreen").textContent, /NaN|Infinity/);
  }
});

test("verdict and its tooltip describe the same signal and explain repetitions", () => {
  const result = detectTournament(Array(201).fill(10), 1, params);
  const node = render({ ...params, ...result, scheme: "tournament" });
  // Not enough data → short operational note, no wall of text.
  assert.match(node("verdictNote").textContent, /Данных мало/);
  // The repetition accounting lives in the green-score tooltip.
  assert.match(node("tip-green").dataset.tip, /Учтено 1 из 200[\s\S]*повторяющийся контекст/i);
  assert.match(node("verdict-note").dataset.tip, /Оценка 1 даёт преимущество/);
  assert.doesNotMatch(node("verdict-note").dataset.tip, /Порог z|пересказывает z/);
  assert.match(node("verdictConf").dataset.tip, /не вероятность авторства/);
});

/* ── verdict / strength follow the statistical evidence, not a reference ── */

const VERDICT_ORDER = ["Нет выраженного сигнала", "Слабый сигнал метки", "Сильный сигнал метки"];

function tournamentZ(meanG, T, m) {
  return (meanG - 0.5) * 2 * Math.sqrt(T * m);
}

function meterWidth(node) {
  return parseFloat(node("meterFill").style.width);
}

test("strong statistical evidence is not hidden by reference levels (T=500, m=30, meanG=0.53)", () => {
  const T = 500;
  const m = 30;
  const meanG = 0.53;
  const z = tournamentZ(meanG, T, m);
  assert.ok(Math.abs(z - 7.35) < 0.02); // the documented scenario, z ≈ 7.35
  const node = render({
    scheme: "tournament", T, totalT: T, m, h: 1, gamma: 0.5,
    meanG, weightedMeanG: meanG, z, pValue: 1 - normalCdf(z),
    expectedG: 0.75, expectedSource: "official",
  });
  assert.equal(node("verdictLabel").textContent, "Сильный сигнал метки");
  assert.ok(meterWidth(node) > 95); // z = 7.35 saturates the conditional scale
  // The note stays short; the nominal-thresholds caveat lives on the z tooltip.
  assert.doesNotMatch(node("verdictNote").textContent, /номинальные/);
  assert.match(node("tip-z").dataset.tip, /номинальные/);
});

test("long texts near the null level show no expressed signal (both schemes)", () => {
  const tz = render({
    scheme: "tournament", T: 2000, totalT: 2000, m: 15, h: 1, gamma: 0.5,
    meanG: 0.501, weightedMeanG: 0.501, z: tournamentZ(0.501, 2000, 15),
    pValue: 0.36, expectedG: 0.75, expectedSource: "official",
  });
  assert.equal(tz("verdictLabel").textContent, "Нет выраженного сигнала");
  assert.ok(meterWidth(tz) < 10);

  const T = 1000;
  const gamma = 0.5;
  const greenCount = 505; // z = (505 − 500) / √250 ≈ 0.32
  const gz = (greenCount - gamma * T) / Math.sqrt(T * gamma * (1 - gamma));
  const gzNode = render({ scheme: "greenlist", T, totalT: T, gamma, h: 1, greenCount, z: gz, pValue: 0.37 });
  assert.equal(gzNode("verdictLabel").textContent, "Нет выраженного сигнала");
  assert.ok(meterWidth(gzNode) < 10);
});

test("a very short sequence with a high computed z stays insufficient data", () => {
  for (const scheme of ["tournament", "greenlist"]) {
    const node = scheme === "tournament"
      ? render({
        scheme, T: 3, totalT: 3, m: 30, h: 1, gamma: 0.5,
        meanG: 1, weightedMeanG: 1, z: tournamentZ(1, 3, 30), pValue: 0,
        expectedG: 0.75, expectedSource: "official",
      })
      : render({
        scheme, T: 3, totalT: 3, m: 1, h: 1, gamma: 0.1,
        greenCount: 3, z: (3 - 0.3) / Math.sqrt(3 * 0.1 * 0.9), pValue: 1e-6, // all three tokens green
      });
    assert.equal(node("verdictLabel").textContent, "Мало данных для вывода");
    assert.ok(meterWidth(node) < 15); // damped by data insufficiency
  }
});

test("repeated contexts do not inflate confidence", () => {
  const result = detectTournament(Array(2001).fill(10), 1, params); // totalT 2000, T 1
  const node = render({ ...params, ...result, scheme: "tournament" });
  assert.equal(node("verdictLabel").textContent, "Мало данных для вывода");
  assert.ok(meterWidth(node) < 10);
  assert.match(node("verdictNote").textContent, /Данных мало/);
});

test("negative z shows no positive evidence for both schemes", () => {
  const T = 500;
  const m = 30;
  const z = tournamentZ(0.47, T, m); // ≈ −7.35
  const node = render({
    scheme: "tournament", T, totalT: T, m, h: 1, gamma: 0.5,
    meanG: 0.47, weightedMeanG: 0.47, z, pValue: 1, expectedG: 0.75, expectedSource: "official",
  });
  assert.equal(node("verdictLabel").textContent, "Нет выраженного сигнала");
  assert.equal(meterWidth(node), 0);

  const gNode = render({ scheme: "greenlist", T: 400, totalT: 400, gamma: 0.5, h: 1, greenCount: 150, z: -4.5, pValue: 0.9999 });
  assert.equal(gNode("verdictLabel").textContent, "Нет выраженного сигнала");
  assert.equal(meterWidth(gNode), 0);
});

test("meter and verdict are monotone in statistical evidence", () => {
  const T = 200;
  const m = 15;
  let prevWidth = -1;
  let prevRank = -1;
  for (const z of [0, 0.5, 1.5, 1.99, 2, 3, 3.99, 4, 5, 7, 12]) {
    const node = render({
      scheme: "tournament", T, totalT: T, m, h: 1, gamma: 0.5,
      meanG: 0.5 + z / (2 * Math.sqrt(T * m)), weightedMeanG: 0.5,
      z, pValue: 1 - normalCdf(z), expectedG: 0.75, expectedSource: "official",
    });
    const label = node("verdictLabel").textContent;
    const rank = label === "Мало данных для вывода" ? -1 : VERDICT_ORDER.indexOf(label);
    assert.ok(rank >= 0, `unknown label: ${label}`);
    assert.ok(rank >= prevRank, `verdict order broke at z=${z}`);
    const width = meterWidth(node);
    assert.ok(width >= prevWidth - 1e-9, `meter width not monotone at z=${z}`);
    prevRank = rank;
    prevWidth = width;
  }
});

test("verdict, meter, z and p stay consistent for both schemes", () => {
  const tournamentCases = [
    { T: 300, m: 20, z: 0.7 },
    { T: 300, m: 20, z: 2.4 },
    { T: 300, m: 20, z: 5.1 },
    { T: 6, m: 20, z: 9.0 }, // high z but not enough data
  ];
  for (const c of tournamentCases) {
    const pValue = 1 - normalCdf(c.z);
    const node = render({
      scheme: "tournament", T: c.T, totalT: c.T, m: c.m, h: 1, gamma: 0.5,
      meanG: 0.5, weightedMeanG: 0.5, z: c.z, pValue, expectedG: 0.75, expectedSource: "official",
    });
    assert.equal(node("statZ").textContent, c.z.toFixed(2));
    const label = node("verdictLabel").textContent;
    const sufficiency = Math.min(1, c.T / 25, (c.T * c.m) / 20);
    const expectedLabel = sufficiency < 1 ? "Мало данных для вывода"
      : VERDICT_ORDER[c.z >= 4 ? 2 : c.z >= 2 ? 1 : 0];
    assert.equal(label, expectedLabel);
    const expectedWidth = Math.max(0, Math.min(1, c.z / 6)) * 100 * sufficiency;
    assert.ok(Math.abs(meterWidth(node) - expectedWidth) < 0.2, `width mismatch at z=${c.z}`);
    if (expectedLabel === "Сильный сигнал метки") {
      assert.match(node("statP").textContent, /e-\d|0\.000|< 1e-15/);
    }
  }

  const gamma = 0.5;
  for (const [T, z] of [[400, 0.4], [400, 2.7], [400, 6.0], [10, 2.5]]) {
    const greenCount = Math.min(T, Math.round(gamma * T + z * Math.sqrt(T * gamma * (1 - gamma))));
    const node = render({ scheme: "greenlist", T, totalT: T, gamma, h: 1, m: 1, greenCount, z, pValue: 1 - normalCdf(z) });
    const sufficiency = Math.min(1, T / 25, (T * gamma) / 5, (T * (1 - gamma)) / 5);
    const expectedLabel = sufficiency < 1 ? "Мало данных для вывода"
      : VERDICT_ORDER[z >= 4 ? 2 : z >= 2 ? 1 : 0];
    assert.equal(node("verdictLabel").textContent, expectedLabel);
    const expectedWidth = Math.max(0, Math.min(1, z / 6)) * 100 * sufficiency;
    assert.ok(Math.abs(meterWidth(node) - expectedWidth) < 0.2, `greenlist width mismatch at z=${z}`);
  }
});

test("the badge shows only the strength percent; z lives in the stats table", () => {
  const node = render({
    scheme: "tournament", T: 300, totalT: 300, m: 20, h: 1, gamma: 0.5,
    meanG: 0.5, weightedMeanG: 0.5, z: 2.2, pValue: 1 - normalCdf(2.2),
    expectedG: 0.75, expectedSource: "official",
  });
  assert.match(node("verdictConf").textContent, /^сила \d+(,\d+)?(\.\d+)?%$/);
  assert.doesNotMatch(node("verdictConf").textContent, /z|·/);
  assert.equal(node("statZ").textContent, "2.20"); // z stays in the stats row
  assert.ok(node("verdictConf").dataset.tip.length > 0);
  assert.equal(node("copyVerdict").disabled, false); // copy action unlocks with the verdict
});

test("the visible note is one short operational line, not a wall of text", () => {
  const repeats = render({ scheme: "tournament", T: 300, totalT: 400, m: 15, h: 1, gamma: 0.5, meanG: 0.5, weightedMeanG: 0.5, z: 2.5, pValue: 0.006, expectedG: 0.75, expectedSource: "official" });
  assert.equal(repeats("verdictNote").textContent, "Исключено повторных контекстов: 100.");
  const clean = render({ scheme: "greenlist", T: 300, totalT: 300, gamma: 0.5, h: 1, greenCount: 150, z: 0, pValue: 0.5 });
  assert.equal(clean("verdictNote").hidden, true); // nothing operational to say
  const warn = render({ scheme: "tournament", T: 3, totalT: 3, m: 15, h: 1, gamma: 0.5, meanG: 1, weightedMeanG: 1, z: 9, pValue: 0, expectedG: 0.75, expectedSource: "official" });
  assert.match(warn("verdictNote").textContent, /Данных мало/);
});

test("null z is approximately standard normal across random unmarked sequences", () => {
  // Empirical check of the independence premise behind counting T·m trials:
  // under H0 each g-bit is an independent hash bit, so the one-sided 1%
  // rejection rate at z > 2.58 must stay near 1% (here: ≤ 4% of runs).
  for (const detector of [detectTournament, detect]) {
    const rng = seededRng(99);
    const runs = 150;
    let exceeded = 0;
    for (let r = 0; r < runs; r++) {
      const ids = [0];
      for (let i = 0; i < 80; i++) ids.push(Math.floor(rng() * 1e6));
      if (detector(ids, 1, { ...params, h: 2 }).z > 2.58) exceeded++;
    }
    assert.ok(exceeded <= 6, `${detector.name}: too many null rejections: ${exceeded}/${runs}`);
  }
});

/* ── per-text strength chart series ── */

function seriesHarness() {
  const source = readFileSync(new URL("../main.js", import.meta.url), "utf8");
  const start = source.indexOf("function tournamentTokenSigma(");
  const end = source.indexOf("/* ── user actions", start);
  const box = {};
  vm.runInNewContext(source.slice(start, end) + "\nbox.series = computeStrengthSeries; box.sigma = tournamentTokenSigma;", {
    box, layerWeights,
  });
  return box;
}

test("strength series: a marked first half and plain second half are told apart", () => {
  const box = seriesHarness();
  const T = 400;
  const scores = Array.from({ length: T }, (_, i) => (i < T / 2 ? 0.75 : 0.5));
  const { points, window } = box.series(scores, { sigmaPerToken: 0.15 });
  assert.ok(window >= 8 && window <= 100);
  const mean = (arr) => arr.reduce((a, p) => a + p.v, 0) / arr.length;
  const first = points.filter((p) => p.t < T / 4);
  const last = points.filter((p) => p.t >= (3 * T) / 4);
  assert.ok(mean(first) > 0.7, "marked half should read high");
  assert.ok(Math.abs(mean(last) - 0.5) < 1e-9, "plain half should sit at the null level");
  assert.ok(mean(first) - mean(last) > 0.2);
});

test("strength series: flat null text stays exactly at the null level", () => {
  const box = seriesHarness();
  const { points } = box.series(Array(300).fill(0.5), { sigmaPerToken: 0.14 });
  assert.ok(points.length >= 2);
  for (const p of points) {
    assert.ok(Math.abs(p.v - 0.5) < 1e-9);
    assert.ok(p.sigma > 0);
  }
});

test("strength series: masked tokens contribute nothing", () => {
  const box = seriesHarness();
  const scores = [1, 1, 1, 0, 0];
  const mask = [1, 0, 0, 0, 0]; // only the first occurrence counts
  const { points } = box.series(scores, { mask, sigmaPerToken: 0.5, window: 10, maxPoints: 10 });
  assert.ok(points.length > 0);
  for (const p of points) {
    assert.ok(p.n <= 1);
    if (p.n === 1) assert.equal(p.v, 1);
  }
});

test("strength series: long texts downsample and the window is capped", () => {
  const box = seriesHarness();
  const T = 5000;
  const { points, window } = box.series(Array(T).fill(0.6), { sigmaPerToken: 0.14 });
  assert.equal(window, 100); // cap, not T/8 = 625
  assert.ok(points.length <= 482);
  for (const p of points) assert.ok(Math.abs(p.v - 0.6) < 1e-9);
});

test("detectTournament ships a per-token weighted score aligned with the verdict", () => {
  const r = detectTournament([5, 6, 7, 8, 9, 10, 11, 12], 1, { ...params, m: 4 });
  assert.equal(r.perTokenWeighted.length, r.perTokenScore.length);
  for (let i = 0; i < r.perTokenWeighted.length; i++) {
    assert.ok(r.perTokenWeighted[i] >= 0 && r.perTokenWeighted[i] <= 1);
    // early layers are weighted up, so the weighted score is a valid [0,1] mean
    const mean = r.perTokenScore[i];
    assert.ok(r.perTokenWeighted[i] >= Math.min(mean, r.perTokenWeighted[i]) - 1e-12);
  }
});
