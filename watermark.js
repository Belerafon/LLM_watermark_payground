/**
 * Pure watermarking math for the green/red-list scheme from
 * "A Watermark for Large Language Models" (Kirchenbauer et al., 2023).
 *
 * No dependencies — usable from the web worker and from Node for testing.
 *
 * Instead of materialising a permuted green list of size γ·|V| each step
 * (expensive for 50k–150k vocabularies), a token is "green" iff a keyed hash
 * of (context seed, token id) falls below γ. This yields a pseudo-random
 * γ-fraction partition of the vocabulary per context, exactly reproducible
 * by the detector.
 */

const TOKEN_KEY = 0x9e3779b9;

/** 32-bit avalanche mixer (murmur3-style finalizer variant). */
function mix32(x) {
  x |= 0;
  x = Math.imul(x ^ (x >>> 16), 0x21f0aaad);
  x = Math.imul(x ^ (x >>> 15), 0x735a2d97);
  return (x ^ (x >>> 15)) >>> 0;
}

/**
 * Hash the user's secret watermarking key string into a 32-bit value.
 * @param {string} key
 * @returns {number} unsigned 32-bit key seed
 */
export function keyToSeed(key) {
  let s = 0x811c9dc5;
  const str = String(key ?? "");
  for (let i = 0; i < str.length; i++) s = mix32(s ^ str.charCodeAt(i));
  return s >>> 0;
}

/**
 * Fold the secret key and the last h context token ids into a 32-bit PRNG seed.
 * @param {ArrayLike<number|bigint>} contextIds
 * @param {number} keySeed from keyToSeed
 * @returns {number} unsigned 32-bit seed
 */
export function seedFromContext(contextIds, keySeed = 0) {
  let s = mix32(0x85ebca6b ^ keySeed);
  for (let i = 0; i < contextIds.length; i++) {
    s = mix32(s ^ mix32((Number(contextIds[i]) + 1) | 0));
  }
  return s >>> 0;
}

/**
 * Is `tokenId` in the green list for this seed?
 * @param {number} seed from seedFromContext
 * @param {number} tokenId
 * @param {number} gamma green-list fraction in (0, 1)
 * @param {Set<number>|null} forcedRed token ids that are always red
 */
export function isGreen(seed, tokenId, gamma, forcedRed = null) {
  if (forcedRed !== null && forcedRed.has(tokenId)) return false;
  const u = mix32(seed ^ mix32((tokenId ^ TOKEN_KEY) | 0)) / 4294967296;
  return u < gamma;
}

/** Standard normal CDF Φ(z) (Abramowitz & Stegun 26.2.17, ~1e-7 accuracy). */
export function normalCdf(z) {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989422804014327 * Math.exp((-z * z) / 2);
  const p =
    d *
    t *
    (0.31938153 +
      t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return z > 0 ? 1 - p : p;
}

/**
 * Run the watermark detector over a token sequence.
 *
 * For each position i >= startIndex, the green list is re-derived from the
 * preceding h tokens (the prompt's trailing tokens seed the first generated
 * position). The one-proportion z-test from the paper:
 *   z = (|s|_G − γT) / sqrt(T·γ·(1−γ))
 *
 * @param {number[]} ids full token sequence (prompt + generation)
 * @param {number} startIndex index of the first generated token
 * @param {{gamma: number, h: number, forcedRed?: Set<number>|null, keySeed?: number}} params
 */
export function detect(ids, startIndex, { gamma, h, forcedRed = null, keySeed = 0 }) {
  const flags = [];
  const seen = new Set();
  let T = 0;
  let greenCount = 0;
  for (let i = startIndex; i < ids.length; i++) {
    const ctx = ids.slice(Math.max(0, i - h), i);
    const seed = seedFromContext(ctx, keySeed);
    const green = isGreen(seed, ids[i], gamma, forcedRed);
    flags.push(green);
    // A repeated context/token pair reproduces the same hash, not new evidence.
    const pair = `${ctx.join(",")}:${ids[i]}`;
    if (seen.has(pair)) continue;
    seen.add(pair);
    T++;
    if (green) greenCount++;
  }
  const z = T > 0 ? (greenCount - gamma * T) / Math.sqrt(T * gamma * (1 - gamma)) : 0;
  const pValue = T > 0 ? 1 - normalCdf(z) : 1;
  return { flags, greenCount, T, totalT: flags.length, z, pValue };
}

/* ────────────────────────────────────────────────────────────────────────────
 * Tournament sampling (SynthID-Text, Dathathri et al., 2024)
 *
 * Each layer ℓ of the tournament has its own keyed pseudo-random function
 * g_ℓ(context, token) ∈ {0, 1}. At every generation step 2^m candidate tokens
 * are drawn i.i.d. from the model's distribution and played off in m knockout
 * rounds: in round ℓ the candidate with g_ℓ = 1 beats the one with g_ℓ = 0
 * (ties are broken by keeping the first). The survivor is emitted.
 *
 * Under no watermark every g-value is a fair coin; watermarked text has a
 * g-value mean pushed above 0.5, which the detector measures.
 * ──────────────────────────────────────────────────────────────────────────── */

const LAYER_KEY = 0x7f4a7c15;

/**
 * g-value of `tokenId` for tournament layer `layer` under this context seed.
 * Tokens on the forced red list always get g = 0 (they lose to 1, but can tie 0).
 * @returns {0|1}
 */
export function gValue(seed, layer, tokenId, forcedRed = null) {
  if (forcedRed !== null && forcedRed.has(tokenId)) return 0;
  const hv = mix32(seed ^ mix32(Math.imul(layer + 1, LAYER_KEY) | 0) ^ mix32((tokenId ^ TOKEN_KEY) | 0));
  return hv >>> 31; // top bit
}

/**
 * Deterministic uniform [0, 1) generator (mulberry32) for reproducible sampling.
 * @param {number} seed 32-bit integer
 * @returns {() => number}
 */
export function seededRng(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Draw `n` i.i.d. token ids from a probability vector (inverse-CDF sampling).
 * @param {ArrayLike<number>} probs normalised probabilities over the vocab
 * @param {number} n
 * @param {() => number} rng uniform [0, 1) source
 * @returns {number[]}
 */
export function sampleMultinomial(probs, n, rng = Math.random) {
  const V = probs.length;
  const cdf = new Float64Array(V);
  let acc = 0;
  for (let i = 0; i < V; i++) {
    acc += probs[i];
    cdf[i] = acc;
  }
  const total = acc; // ≈ 1, but guard against rounding
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    const u = rng() * total;
    let lo = 0, hi = V - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cdf[mid] > u) hi = mid;
      else lo = mid + 1;
    }
    out[k] = lo;
  }
  return out;
}

/**
 * One step of tournament sampling: the winner of an m-layer knockout among
 * 2^m candidates drawn i.i.d. from `probs`.
 *
 * The literal bracket is intractable for large m (the paper uses m = 30), so
 * the exact distribution of the winner is computed in closed form: one layer
 * of a match between two i.i.d. draws from q gives the winner distribution
 *   q'(x) = q(x) · (g(x) + P_q(g = 0)).
 * Applying this per layer and sampling the final distribution once is
 * identical in distribution to playing the 2^m-candidate tournament.
 *
 * @param {ArrayLike<number>} probs normalised probabilities over the vocab
 * @param {number} seed from seedFromContext
 * @param {number} m number of tournament layers (1–30)
 * @param {Set<number>|null} forcedRed
 * @param {() => number} rng
 */
export function tournamentSample(probs, seed, m, forcedRed = null, rng = Math.random) {
  const V = probs.length;
  const q = Float64Array.from(probs);
  for (let layer = 0; layer < m; layer++) {
    let p0 = 0;
    for (let t = 0; t < V; t++) {
      if (q[t] > 0 && !gValue(seed, layer, t, forcedRed)) p0 += q[t];
    }
    let sum = 0;
    for (let t = 0; t < V; t++) {
      if (q[t] > 0) {
        q[t] *= gValue(seed, layer, t, forcedRed) + p0;
        sum += q[t];
      }
    }
    for (let t = 0; t < V; t++) q[t] /= sum; // guard float drift
  }
  return sampleMultinomial(q, 1, rng)[0];
}

/**
 * Tournament-watermark detector: the mean g-value over all generated tokens and
 * all m layers. Under H0 each g ~ Bernoulli(0.5), so with N = T·m values
 *   z = (Σg − N/2) / sqrt(N/4).
 *
 * @param {number[]} ids full token sequence (prompt + generation)
 * @param {number} startIndex index of the first generated token
 * @param {{m: number, h: number, forcedRed?: Set<number>|null, keySeed?: number}} params
 */
/**
 * UI reference mean g for a tournament with a fixed artificial distribution.
 * This is neither a ceiling nor a calibration for the loaded language model.
 *
 * One Bernoulli layer has expectation (3−C)/4 (Dathathri / Omidi). C is the
 * chance two candidates are the same token. A peaked next-token distribution
 * has C ≈ 0.72, so one layer tops out near 0.57, not 0.75. Later layers
 * average in near-fair coins, so the mean slides toward 0.5 as m grows.
 * At m = 15 this particular artificial distribution yields a reference near 0.54.
 * A real model may produce a smaller or larger mean depending on its predictions.
 *
 * Deterministic: a fixed geometric distribution and fixed g-bits, so the scale
 * does not flicker, and a new m recomputes it.
 */
const TOURNAMENT_CEILING = new Map();
const CEILING_V = 40;
const CEILING_TRIALS = 64;
const CEILING_RATIO = 0.16;

function ceilingBit(trial, layer, rank) {
  const x = mix32((trial + 1) ^ mix32(Math.imul(layer + 1, LAYER_KEY)) ^ mix32(Math.imul(rank + 1, TOKEN_KEY)));
  return x >>> 31;
}

export function expectedTournamentMean(m) {
  const layers = Math.max(1, Math.min(30, Math.round(m) || 1));
  const cached = TOURNAMENT_CEILING.get(layers);
  if (cached !== undefined) return cached;

  const p = new Float64Array(CEILING_V);
  let sum = 0;
  for (let i = 0; i < CEILING_V; i++) {
    p[i] = CEILING_RATIO ** i;
    sum += p[i];
  }
  for (let i = 0; i < CEILING_V; i++) p[i] /= sum;

  let acc = 0;
  for (let trial = 0; trial < CEILING_TRIALS; trial++) {
    const q = Float64Array.from(p);
    const gs = [];
    for (let layer = 0; layer < layers; layer++) {
      const g = new Uint8Array(CEILING_V);
      let p0 = 0;
      for (let t = 0; t < CEILING_V; t++) {
        g[t] = ceilingBit(trial, layer, t);
        if (!g[t]) p0 += q[t];
      }
      gs.push(g);
      let s = 0;
      for (let t = 0; t < CEILING_V; t++) {
        q[t] *= g[t] + p0;
        s += q[t];
      }
      if (s > 0) for (let t = 0; t < CEILING_V; t++) q[t] /= s;
    }
    let mean = 0;
    for (const g of gs) {
      let mass = 0;
      for (let t = 0; t < CEILING_V; t++) if (g[t]) mass += q[t];
      mean += mass;
    }
    acc += mean / layers;
  }
  const value = acc / CEILING_TRIALS;
  TOURNAMENT_CEILING.set(layers, value);
  return value;
}

export function detectTournament(ids, startIndex, { m, h, forcedRed = null, keySeed = 0 }) {
  const perTokenScore = [];
  const seen = new Set();
  let T = 0;
  let sumG = 0;
  for (let i = startIndex; i < ids.length; i++) {
    const ctx = ids.slice(Math.max(0, i - h), i);
    const seed = seedFromContext(ctx, keySeed);
    let s = 0;
    for (let layer = 0; layer < m; layer++) s += gValue(seed, layer, ids[i], forcedRed);
    perTokenScore.push(s / m);
    const pair = `${ctx.join(",")}:${ids[i]}`;
    if (seen.has(pair)) continue;
    seen.add(pair);
    T++;
    sumG += s;
  }
  const N = T * m;
  const meanG = N > 0 ? sumG / N : 0.5;
  const z = N > 0 ? (sumG - N / 2) / Math.sqrt(N / 4) : 0;
  const pValue = N > 0 ? 1 - normalCdf(z) : 1;
  const flags = perTokenScore.map((s) => s >= 0.5);
  const expectedG = expectedTournamentMean(m);
  return { flags, perTokenScore, meanG, T, totalT: perTokenScore.length, m, z, pValue, expectedG };
}
