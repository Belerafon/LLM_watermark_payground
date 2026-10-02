import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import {
  detect, detectTournament, expectedTournamentMean, isGreen,
  keyToSeed, sampleMultinomial, seededRng, seedFromContext, tournamentSample,
} from "../watermark.js";

const params = { gamma: 0.5, m: 15, h: 1, keySeed: keyToSeed("test-key") };

for (const detector of [detect, detectTournament]) {
  test(`${detector.name}: repetition adds no statistical evidence, keeps every colour`, () => {
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

  test(`${detector.name}: deduplication uses the whole context and current token`, () => {
    const result = detector([1, 2, 9, 3, 2, 9, 1, 2, 9], 2, { ...params, h: 2 });
    assert.equal(result.totalT, 7);
    assert.equal(result.T, 6); // Only the final [1, 2] -> 9 repeats.
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
  assert.match(node("verdictNote").textContent, /Повторных сочетаний исключено: 199/);
  assert.match(node("verdict-note").dataset.tip, /Оценка 1 даёт преимущество/);
  assert.doesNotMatch(node("verdict-note").dataset.tip, /Порог z|пересказывает z/);
  assert.match(node("verdictConf").dataset.tip, /не вероятность авторства/);
});
