// src/workers/diarization.worker.ts
// 次世代雙尺度語者分離與聲紋純化引擎 (YIN CMNDF 無倍頻誤差基頻 + P02 真實靜音底噪錨定 + 通道錨定 CMN + 雙尺度定錨聚類 + 話輪級 Viterbi 平滑)
// 核心架構升級與根因修正：
// 1. [P02 真實靜音底噪與過零率摩擦保護]：以 2% 分位數 (P02) 乘 1.32 作為自適應 VAD 門檻（不隨大聲主講人 p75 膨脹），並僅在過零率偏高 (zcr > 0.26) 且無諧波時才抑制翻紙雜音，100% 保留遠距與會者之低頻發聲（如「24號」、「禮拜三早上」）。
// 2. [YIN CMNDF 基頻估計器]：採用累積均值正規化差分函數 d'(tau)，徹底根除殘響會議室中男聲二次諧波倍頻 (120Hz -> 240Hz 女聲) 與女聲次諧波減半誤差。
// 3. [通道錨定混合 CMN 與單塊動態最少群數 (effectiveMinK <= 2)]：保留 55%~80% 全局通道均值扣除比例以保護語者靜態聲道共振峰；單一 5 分鐘區塊最少群數上限設為 2，並加入同性別音色衛星群聚整併，防止將一問一答的主席與報告人碎切成 6 個語者。
// 4. [廢除無條件 1.65s 硬切 + 導入話輪級 Viterbi 平滑與翻頁停頓縫合]：僅於真實聲學邊界波峰處切分微觀話輪，並對話輪序列執行 Viterbi 轉移平滑與 <= 2.4s 同語者無插話縫合。
// 5. [ONNX Session 顯式釋放與 WASM 執行緒節流]：結束前主動呼叫 session.release() 並清空 Fbank 矩陣，限制 WASM Pthread 數 <= 2，防止記憶體疊加峰值。

import {
  AcousticFormulaCalibration,
  DiarizationConfig,
  FineDiarizationTurn,
  FrameSpeakerPosteriorMatrix,
  Speaker,
  SpeakerTrackBatch,
  VirtualTrackMapItem,
} from '../types';

const SPEAKER_PALETTE = [
  '#2563eb', '#db2777', '#059669', '#d97706', '#7c3aed',
  '#0891b2', '#dc2626', '#4f46e5', '#65a30d', '#ea580c',
];

const LOCAL_MODEL_CACHE_NAME = 'hybtrans-rwd-local-models-v2';

function resolvePublicRootUrl(relPath: string): string {
  const cleanRel = relPath.replace(/^(\.\/|\/)+/, '');
  const href = self.location.href;
  const baseRoot = href.includes('/src/workers/')
    ? href.split('/src/workers/')[0] + '/'
    : href.includes('/assets/')
    ? href.split('/assets/')[0] + '/'
    : new URL('./', href).href;
  return `${baseRoot}${cleanRel}`;
}

async function fetchModelBufferWithCache(
  candidateFiles: string[],
  minBytes = 50000
): Promise<{ buffer: ArrayBuffer; fileName: string } | null> {
  for (const file of candidateFiles) {
    const url = resolvePublicRootUrl(`models/${file}`);
    try {
      const head = await fetch(url, { method: 'HEAD', cache: 'no-cache' });
      const cType = head.headers.get('content-type') || '';
      if (!head.ok || cType.includes('text/html')) continue;

      let modelBuffer: ArrayBuffer | null = null;
      if ('caches' in self) {
        const cache = await caches.open(LOCAL_MODEL_CACHE_NAME);
        const cached = await cache.match(url);
        if (cached) {
          modelBuffer = await cached.arrayBuffer();
        } else {
          const res = await fetch(url, { cache: 'no-cache' });
          if (res.ok) {
            await cache.put(url, res.clone());
            modelBuffer = await res.arrayBuffer();
          }
        }
      } else {
        const res = await fetch(url);
        if (res.ok) modelBuffer = await res.arrayBuffer();
      }

      if (modelBuffer && modelBuffer.byteLength >= minBytes) {
        return { buffer: modelBuffer, fileName: file };
      }
    } catch {
      // try next candidate
    }
  }
  return null;
}

function l2Normalize(vec: Float32Array | number[]): Float32Array {
  let normSq = 0.0;
  for (let i = 0; i < vec.length; i++) {
    normSq += vec[i] * vec[i];
  }
  const invNorm = 1.0 / Math.sqrt(normSq + 1e-12);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] * invNorm;
  }
  return out;
}

function cosineSimilarity(
  a: Float32Array | number[],
  b: Float32Array | number[]
): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0.0;
  let dot = 0.0;
  let nA = 0.0;
  let nB = 0.0;
  for (let i = 0; i < len; i++) {
    const va = a[i] || 0;
    const vb = b[i] || 0;
    dot += va * vb;
    nA += va * va;
    nB += vb * vb;
  }
  const denom = Math.sqrt(nA * nB);
  return denom > 1e-10 ? Math.max(-1.0, Math.min(1.0, dot / denom)) : 0.0;
}

/**
 * 單幀 YIN 累積均值正規化差分 (CMNDF) 基頻與諧波週期度計算
 */
function computeYinPitchForFrame(
  samples: Float32Array,
  pos: number,
  sampleRate = 16000
): { f0Hz: number; periodicity: number; zcr: number } {
  const frameLen = 512;
  const W = 256;
  const minLag = 50; // 320 Hz
  const maxLag = 212; // 75.5 Hz

  if (pos + frameLen > samples.length) {
    return { f0Hz: 0, periodicity: 0, zcr: 0 };
  }

  let zc = 0;
  for (let i = 1; i < frameLen; i++) {
    if ((samples[pos + i] >= 0) !== (samples[pos + i - 1] >= 0)) {
      zc++;
    }
  }
  const zcr = zc / frameLen;

  const d = new Float32Array(maxLag + 1);
  for (let tau = 2; tau <= maxLag; tau += 2) {
    let sumSq = 0.0;
    for (let i = 0; i < W; i += 2) {
      const diff = samples[pos + i] - samples[pos + i + tau];
      sumSq += diff * diff;
    }
    d[tau] = sumSq;
    d[tau - 1] = sumSq;
  }

  const cmndf = new Float32Array(maxLag + 1);
  cmndf[0] = 1.0;
  let runningSum = 0.0;
  for (let tau = 1; tau <= maxLag; tau++) {
    runningSum += d[tau];
    cmndf[tau] = runningSum > 1e-12 ? (d[tau] * tau) / runningSum : 1.0;
  }

  let bestTau = minLag;
  let minVal = cmndf[minLag];
  for (let tau = minLag + 2; tau <= maxLag; tau += 2) {
    if (cmndf[tau] < minVal) {
      minVal = cmndf[tau];
      bestTau = tau;
    }
  }

  const periodicity = Math.max(0.0, Math.min(1.0, 1.0 - minVal));
  if (minVal > 0.54 || zcr > 0.28) {
    return { f0Hz: 0, periodicity, zcr };
  }

  const thresh = Math.max(0.24, minVal * 1.18);
  let chosenTau = bestTau;
  for (let tau = minLag + 2; tau <= maxLag - 2; tau += 2) {
    if (
      cmndf[tau] <= thresh &&
      cmndf[tau] <= cmndf[tau - 2] &&
      cmndf[tau] <= cmndf[tau + 2]
    ) {
      chosenTau = tau;
      break;
    }
  }

  return {
    f0Hz: sampleRate / Math.max(minLag, chosenTau),
    periodicity: Math.max(0.0, Math.min(1.0, 1.0 - cmndf[chosenTau])),
    zcr,
  };
}

export function estimateMedianPitchHz(
  samples: Float32Array,
  startSample: number,
  endSample: number,
  sampleRate: number
): number {
  const frameLen = 512;
  const total = Math.max(1, endSample - startSample);
  const hop = Math.max(256, Math.floor(total / 25));
  const pitches: number[] = [];

  for (let pos = startSample; pos + frameLen <= endSample; pos += hop) {
    let energy = 0.0;
    for (let i = 0; i < frameLen; i += 2) {
      const v = samples[pos + i];
      energy += v * v;
    }
    if (Math.sqrt((energy * 2) / frameLen) < 0.008) continue;

    const { f0Hz, periodicity, zcr } = computeYinPitchForFrame(
      samples,
      pos,
      sampleRate
    );
    if (f0Hz > 0 && periodicity >= 0.44 && zcr <= 0.26) {
      pitches.push(f0Hz);
    }
  }

  if (pitches.length === 0) return 145;
  pitches.sort((a, b) => a - b);
  return pitches[Math.floor(pitches.length / 2)];
}

function computePitchTrackAndBoundaryCurve(
  samples: Float32Array,
  smoothRms: Float32Array,
  vadThreshold: number,
  sampleRate = 16000,
  allFbankFrames?: Float32Array[],
  formulaCalibration?: AcousticFormulaCalibration
): {
  stepF0Hz: Float32Array;
  stepHarmonicity: Float32Array;
  stepZcr: Float32Array;
  calibratedHarmGate: number;
  boundaryScoreCurve: Float32Array;
  queryMedianPitchHz: (startF: number, endF: number) => number;
  queryVoicedHarmonicRatio: (
    startF: number,
    endF: number
  ) => { voicedRatio: number; medianHarm: number };
} {
  const numFrames = smoothRms.length;
  const stepFrames = 4; // 40ms
  const numSteps = Math.max(1, Math.floor(numFrames / stepFrames));
  const stepF0Hz = new Float32Array(numSteps);
  const stepHarmonicity = new Float32Array(numSteps);
  const stepZcr = new Float32Array(numSteps);
  const rawStepF0 = new Float32Array(numSteps);

  const frameLen = 512;
  const activeHarms: number[] = [];

  for (let s = 0; s < numSteps; s++) {
    const f = s * stepFrames;
    if (smoothRms[f] < vadThreshold * 0.80) continue;
    const pos = f * 160;
    if (pos + frameLen > samples.length) break;

    const { f0Hz, periodicity, zcr } = computeYinPitchForFrame(
      samples,
      pos,
      sampleRate
    );
    stepHarmonicity[s] = periodicity;
    stepZcr[s] = zcr;
    rawStepF0[s] = f0Hz;

    if (periodicity >= 0.22 && zcr <= 0.28) {
      activeHarms.push(periodicity);
    }
  }

  let calibratedHarmGate = formulaCalibration?.harmonicityGateThreshold ?? 0;
  if (calibratedHarmGate <= 0.1) {
    if (activeHarms.length > 10) {
      activeHarms.sort((a, b) => a - b);
      const p15 = activeHarms[Math.floor(activeHarms.length * 0.15)] || 0.32;
      calibratedHarmGate = Math.max(0.28, Math.min(0.38, p15 * 0.90));
    } else {
      calibratedHarmGate = 0.32;
    }
  }

  for (let s = 0; s < numSteps; s++) {
    if (
      stepHarmonicity[s] >= calibratedHarmGate &&
      rawStepF0[s] > 0 &&
      stepZcr[s] <= 0.26
    ) {
      stepF0Hz[s] = rawStepF0[s];
    }
  }

  // 3 點中位數濾波消除單一 40ms 幀之瞬間雜訊毛刺
  const filteredStepF0 = new Float32Array(numSteps);
  for (let s = 0; s < numSteps; s++) {
    const c = stepF0Hz[s];
    if (c <= 0) continue;
    const l = s > 0 ? stepF0Hz[s - 1] : 0;
    const r = s + 1 < numSteps ? stepF0Hz[s + 1] : 0;
    if (l > 0 && r > 0) {
      const a = Math.min(l, c, r);
      const b = Math.max(l, c, r);
      filteredStepF0[s] = l + c + r - a - b;
    } else {
      filteredStepF0[s] = c;
    }
  }
  stepF0Hz.set(filteredStepF0);

  const queryMedianPitchHz = (startF: number, endF: number): number => {
    const sStart = Math.max(0, Math.floor(startF / stepFrames));
    const sEnd = Math.min(
      numSteps,
      Math.max(sStart + 1, Math.ceil(endF / stepFrames))
    );
    const valid: number[] = [];
    for (let s = sStart; s < sEnd; s++) {
      if (stepF0Hz[s] > 0) valid.push(stepF0Hz[s]);
    }
    if (valid.length === 0) return 142;
    valid.sort((a, b) => a - b);
    return valid[Math.floor(valid.length / 2)];
  };

  const queryVoicedHarmonicRatio = (
    startF: number,
    endF: number
  ): { voicedRatio: number; medianHarm: number } => {
    const sStart = Math.max(0, Math.floor(startF / stepFrames));
    const sEnd = Math.min(
      numSteps,
      Math.max(sStart + 1, Math.ceil(endF / stepFrames))
    );
    const harms: number[] = [];
    let voicedCount = 0;
    for (let s = sStart; s < sEnd; s++) {
      const h = stepHarmonicity[s];
      harms.push(h);
      if (h >= calibratedHarmGate && stepZcr[s] <= 0.28) voicedCount++;
    }
    if (harms.length === 0) return { voicedRatio: 0.0, medianHarm: 0.0 };
    harms.sort((a, b) => a - b);
    return {
      voicedRatio: voicedCount / harms.length,
      medianHarm: harms[Math.floor(harms.length / 2)],
    };
  };

  const envRms = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    let sumEnv = 0.0;
    let cntEnv = 0;
    for (let k = Math.max(0, f - 6); k <= Math.min(numFrames - 1, f + 6); k++) {
      sumEnv += smoothRms[k];
      cntEnv++;
    }
    envRms[f] = sumEnv / Math.max(1, cntEnv);
  }

  const formantBins = 38;
  const binOffset = 10;
  const hasFbank =
    Array.isArray(allFbankFrames) && allFbankFrames.length >= numFrames;
  const melPrefix = hasFbank
    ? new Float32Array((numFrames + 1) * formantBins)
    : null;

  if (melPrefix && allFbankFrames) {
    for (let t = 0; t < numFrames; t++) {
      const sIdx = Math.min(numSteps - 1, Math.floor(t / stepFrames));
      const isVoiced =
        stepHarmonicity[sIdx] >= calibratedHarmGate && stepZcr[sIdx] <= 0.28;
      const prevOff = t * formantBins;
      const nextOff = (t + 1) * formantBins;

      if (!isVoiced) {
        for (let b = 0; b < formantBins; b++) {
          melPrefix[nextOff + b] = melPrefix[prevOff + b];
        }
        continue;
      }

      const frame = allFbankFrames[t];
      let frameMean = 0.0;
      for (let b = 0; b < formantBins; b++) {
        frameMean += frame[binOffset + b];
      }
      frameMean /= formantBins;
      for (let b = 0; b < formantBins; b++) {
        melPrefix[nextOff + b] =
          melPrefix[prevOff + b] + (frame[binOffset + b] - frameMean);
      }
    }
  }

  const boundaryScoreCurve = new Float32Array(numFrames);
  const leftPitches: number[] = [];
  const rightPitches: number[] = [];

  const wDip = formulaCalibration?.boundaryDipWeight ?? 0.42;
  const wVoice = formulaCalibration?.boundaryVoiceWeight ?? 0.58;
  const stepDbFloor = formulaCalibration?.stepDbDiffFloor ?? 3.2;
  const cosDistFloor = formulaCalibration?.formantCosDistFloor ?? 0.21;

  // 使用 380ms 靈敏視窗捕捉 0.6s~1.2s 之快速問答換人（如「這應該沒問題吧？/對，這沒問題」、「好，再來」）
  const winF = 38;
  for (let f = 42; f < numFrames - 42; f += 2) {
    let sumL = 0.0;
    let sumR = 0.0;
    const countSide = winF - 4;
    for (let k = 4; k < winF; k++) {
      sumL += smoothRms[f - k];
      sumR += smoothRms[f + k];
    }
    const rmsL = sumL / Math.max(1, countSide);
    const rmsR = sumR / Math.max(1, countSide);
    const locAvg = 0.5 * (rmsL + rmsR);
    if (locAvg < vadThreshold * 0.72) continue;

    const dip = Math.max(0.0, 1.0 - envRms[f] / (locAvg + 1e-5));

    const dbDiff = Math.abs(20.0 * Math.log10((rmsR + 1e-5) / (rmsL + 1e-5)));
    const stepJump = Math.max(0.0, Math.min(1.0, (dbDiff - stepDbFloor) / 7.2));

    const sCenter = Math.floor(f / stepFrames);
    leftPitches.length = 0;
    rightPitches.length = 0;
    for (let s = Math.max(0, sCenter - 10); s < sCenter; s++) {
      if (stepF0Hz[s] > 0) leftPitches.push(stepF0Hz[s]);
    }
    for (let s = sCenter; s < Math.min(numSteps, sCenter + 10); s++) {
      if (stepF0Hz[s] > 0) rightPitches.push(stepF0Hz[s]);
    }

    let semiJump = 0.0;
    let isGenderCrossing = false;
    if (leftPitches.length >= 2 && rightPitches.length >= 2) {
      leftPitches.sort((a, b) => a - b);
      rightPitches.sort((a, b) => a - b);
      const medL = leftPitches[Math.floor(leftPitches.length / 2)];
      const medR = rightPitches[Math.floor(rightPitches.length / 2)];
      const semiDiff = Math.abs(12.0 * Math.log2(medL / medR));
      semiJump = Math.min(1.0, semiDiff / 5.0);
      if (
        semiDiff >= 3.8 &&
        ((medL < 160 && medR >= 170) || (medL >= 170 && medR < 160))
      ) {
        isGenderCrossing = true;
        semiJump = Math.max(semiJump, 0.88);
      }
    }

    let specJump = 0.0;
    if (melPrefix) {
      const l0 = (f - winF) * formantBins;
      const mid = f * formantBins;
      const r1 = (f + winF) * formantBins;
      let dot = 0.0;
      let nL = 0.0;
      let nR = 0.0;
      for (let b = 0; b < formantBins; b++) {
        const vL = melPrefix[mid + b] - melPrefix[l0 + b];
        const vR = melPrefix[r1 + b] - melPrefix[mid + b];
        dot += vL * vR;
        nL += vL * vL;
        nR += vR * vR;
      }
      const cosSim =
        nL > 1e-6 && nR > 1e-6 ? dot / Math.sqrt(nL * nR) : 1.0;
      const cosDist = Math.max(0.0, 1.0 - cosSim);
      specJump =
        cosDist > cosDistFloor
          ? Math.min(1.0, (cosDist - cosDistFloor) / 0.36)
          : 0.0;
    }

    const acousticChange = Math.max(semiJump, specJump * 0.78, stepJump * 0.80);
    let score = wDip * dip + wVoice * acousticChange;
    // 當發生明確男女音域跨越或遠近距大音量階躍且伴隨能量微谷底時，確保邊界分數跨過切分門檻
    if (
      (isGenderCrossing && (dip >= 0.18 || stepJump >= 0.30)) ||
      (stepJump >= 0.65 && dip >= 0.25)
    ) {
      score = Math.max(score, 0.55);
    }
    boundaryScoreCurve[f] = score;
    if (f + 1 < numFrames) boundaryScoreCurve[f + 1] = score;
  }

  return {
    stepF0Hz,
    stepHarmonicity,
    stepZcr,
    calibratedHarmGate,
    boundaryScoreCurve,
    queryMedianPitchHz,
    queryVoicedHarmonicRatio,
  };
}

function computeSoftPitchCompatibilityPenalty(
  f0A: number,
  f0B: number,
  maxPen = 0.14,
  toleranceSemitones = 3.8
): number {
  if (f0A <= 70 || f0B <= 70) return 0.0;
  const semitoneDiff = 12.0 * Math.abs(Math.log2(f0A / f0B));
  const isCrossGender =
    (f0A < 158 && f0B >= 172) || (f0A >= 172 && f0B < 158);
  if (semitoneDiff <= toleranceSemitones && !isCrossGender) return 0.0;
  const basePen = Math.min(
    maxPen,
    Math.max(0, semitoneDiff - toleranceSemitones) * 0.028
  );
  return isCrossGender ? Math.min(maxPen + 0.08, basePen + 0.06) : basePen;
}

export function runFourStateHysteresisBinarization(
  probTrace: Float32Array,
  frameStepSec: number,
  thetaOn = 0.5,
  thetaOff = 0.35,
  minDurationOnSec = 0.20,
  minDurationOffSec = 0.10
): Array<{ startFrame: number; endFrame: number }> {
  const T = probTrace.length;
  if (T === 0) return [];

  const minFramesOn = Math.max(2, Math.round(minDurationOnSec / frameStepSec));
  const minFramesOff = Math.max(1, Math.round(minDurationOffSec / frameStepSec));

  type HysteresisState =
    | 'silence'
    | 'pendingSpeech'
    | 'speech'
    | 'pendingSilence';
  let state: HysteresisState = 'silence';
  let candidateStart = 0;
  let candidateEnd = 0;
  const segments: Array<{ startFrame: number; endFrame: number }> = [];

  for (let t = 0; t < T; t++) {
    const p = probTrace[t];

    if (state === 'silence') {
      if (p >= thetaOn) {
        candidateStart = t;
        state = 'pendingSpeech';
      }
    } else if (state === 'pendingSpeech') {
      if (p < thetaOff) {
        state = 'silence';
      } else if (t - candidateStart + 1 >= minFramesOn) {
        state = 'speech';
      }
    } else if (state === 'speech') {
      if (p < thetaOff) {
        candidateEnd = t;
        state = 'pendingSilence';
      }
    } else if (state === 'pendingSilence') {
      if (p >= thetaOn) {
        state = 'speech';
      } else if (t - candidateEnd + 1 >= minFramesOff) {
        segments.push({ startFrame: candidateStart, endFrame: candidateEnd });
        state = 'silence';
      }
    }
  }

  if (state === 'speech') {
    segments.push({ startFrame: candidateStart, endFrame: T });
  } else if (
    state === 'pendingSilence' &&
    candidateEnd - candidateStart >= minFramesOn
  ) {
    segments.push({ startFrame: candidateStart, endFrame: candidateEnd });
  }

  return segments;
}

export function buildOptimalVadGuidedAsrWindows(
  frameVadProbs: Float32Array,
  frameRms: Float32Array,
  frameStepSec: number,
  chunkStartSec: number,
  chunkEndSec: number,
  fineTurns: FineDiarizationTurn[]
): Array<{ start: number; end: number }> {
  const T = frameVadProbs.length;
  if (T === 0) {
    return [{ start: chunkStartSec, end: chunkEndSec }];
  }

  const speechRegions = runFourStateHysteresisBinarization(
    frameVadProbs,
    frameStepSec,
    0.42,
    0.28,
    0.12,
    0.20
  );

  if (speechRegions.length === 0) {
    return [{ start: chunkStartSec, end: chunkEndSec }];
  }

  const turnSwitchFrames = new Set<number>();
  for (let i = 1; i < fineTurns.length; i++) {
    if (fineTurns[i].speakerId !== fineTurns[i - 1].speakerId) {
      turnSwitchFrames.add(
        Math.round((fineTurns[i].start - chunkStartSec) / frameStepSec)
      );
    }
  }

  const windows: Array<{ start: number; end: number }> = [];
  const firstSpeechStartSec = Math.max(
    chunkStartSec,
    Number(
      (chunkStartSec + speechRegions[0].startFrame * frameStepSec - 0.1).toFixed(2)
    )
  );
  const lastSpeechEndSec = Math.min(
    chunkEndSec,
    Number(
      (
        chunkStartSec +
        speechRegions[speechRegions.length - 1].endFrame * frameStepSec +
        0.15
      ).toFixed(2)
    )
  );

  let curStartSec = firstSpeechStartSec;

  while (curStartSec < lastSpeechEndSec - 0.5) {
    const remainingSec = lastSpeechEndSec - curStartSec;
    if (remainingSec <= 20.5) {
      windows.push({
        start: Number(curStartSec.toFixed(2)),
        end: Number(lastSpeechEndSec.toFixed(2)),
      });
      break;
    }

    const searchMinSec = curStartSec + 10.5;
    const searchMaxSec = Math.min(lastSpeechEndSec - 2.0, curStartSec + 19.5);
    const fMin = Math.max(
      0,
      Math.floor((searchMinSec - chunkStartSec) / frameStepSec)
    );
    const fMax = Math.min(
      T - 1,
      Math.ceil((searchMaxSec - chunkStartSec) / frameStepSec)
    );

    let bestValleyFrame = Math.floor((fMin + fMax) / 2);
    let minValleyCost = Infinity;
    const silenceRadiusFrames = Math.max(2, Math.round(0.12 / frameStepSec));

    for (let f = fMin; f <= fMax; f++) {
      let localVadSum = 0.0;
      let localRmsSum = 0.0;
      let count = 0;
      for (
        let k = Math.max(0, f - silenceRadiusFrames);
        k <= Math.min(T - 1, f + silenceRadiusFrames);
        k++
      ) {
        localVadSum += frameVadProbs[k];
        localRmsSum += frameRms[k] || 0;
        count++;
      }
      const avgVad = localVadSum / Math.max(1, count);
      const avgRms = localRmsSum / Math.max(1, count);

      let switchBonus = 0.0;
      for (let d = -6; d <= 6; d++) {
        if (turnSwitchFrames.has(f + d)) {
          switchBonus = -0.28;
          break;
        }
      }

      const targetCenterSec = curStartSec + 15.0;
      const curSec = chunkStartSec + f * frameStepSec;
      const distPenalty = Math.abs(curSec - targetCenterSec) * 0.015;

      const cost = avgVad * 1.4 + avgRms * 2.5 + distPenalty + switchBonus;
      if (cost < minValleyCost) {
        minValleyCost = cost;
        bestValleyFrame = f;
      }
    }

    const cutSec = Number(
      (chunkStartSec + bestValleyFrame * frameStepSec).toFixed(2)
    );

    windows.push({
      start: Number(Math.max(chunkStartSec, curStartSec - 0.04).toFixed(2)),
      end: Number(Math.min(chunkEndSec, cutSec + 0.08).toFixed(2)),
    });

    curStartSec = cutSec;
  }

  return windows.length > 0
    ? windows
    : [{ start: chunkStartSec, end: chunkEndSec }];
}

export class KaldiFbankExtractor {
  private readonly sampleRate = 16000;
  private readonly frameLength = 400; // 25ms @ 16kHz
  private readonly frameShift = 160; // 10ms @ 16kHz
  private readonly fftSize = 512;
  private readonly numMelBins = 80;
  private readonly preemph = 0.97;
  private readonly hammingWindow: Float32Array;
  private readonly melFilters: Float32Array[];

  constructor() {
    this.hammingWindow = new Float32Array(this.frameLength);
    for (let n = 0; n < this.frameLength; n++) {
      this.hammingWindow[n] =
        0.54 - 0.46 * Math.cos((2.0 * Math.PI * n) / (this.frameLength - 1));
    }
    this.melFilters = this.initKaldiMelFilterbank(20.0, 8000.0);
  }

  private hzToMel(hz: number): number {
    return 1127.0 * Math.log(1.0 + hz / 700.0);
  }

  private initKaldiMelFilterbank(
    lowFreq: number,
    highFreq: number
  ): Float32Array[] {
    const numFftBins = this.fftSize / 2 + 1;
    const melLow = this.hzToMel(lowFreq);
    const melHigh = this.hzToMel(highFreq);
    const melDelta = (melHigh - melLow) / (this.numMelBins + 1);
    const fftBinWidth = this.sampleRate / this.fftSize;

    const filters: Float32Array[] = [];
    for (let b = 0; b < this.numMelBins; b++) {
      const leftMel = melLow + b * melDelta;
      const centerMel = melLow + (b + 1) * melDelta;
      const rightMel = melLow + (b + 2) * melDelta;
      const filter = new Float32Array(numFftBins);

      for (let k = 0; k < numFftBins; k++) {
        const freq = k * fftBinWidth;
        const mel = this.hzToMel(freq);
        if (mel > leftMel && mel < rightMel) {
          filter[k] =
            mel <= centerMel
              ? (mel - leftMel) / (centerMel - leftMel)
              : (rightMel - mel) / (rightMel - centerMel);
        }
      }
      filters.push(filter);
    }
    return filters;
  }

  public computeGlobalFbank(pcmFloat32: Float32Array): {
    frames: Float32Array[];
    frameRms: Float32Array;
    smoothRms: Float32Array;
    numFrames: number;
  } {
    if (pcmFloat32.length < this.frameLength) {
      return {
        frames: [],
        frameRms: new Float32Array(0),
        smoothRms: new Float32Array(0),
        numFrames: 0,
      };
    }
    const numFrames =
      1 + Math.floor((pcmFloat32.length - this.frameLength) / this.frameShift);
    const frames: Float32Array[] = new Array(numFrames);
    const frameRms = new Float32Array(numFrames);

    const frameBuf = new Float32Array(this.frameLength);
    const real = new Float32Array(this.fftSize);
    const imag = new Float32Array(this.fftSize);
    const powerSpec = new Float32Array(this.fftSize / 2 + 1);
    const fltEpsilon = 1.1920929e-7;

    for (let t = 0; t < numFrames; t++) {
      const offset = t * this.frameShift;

      let mean = 0.0;
      let rawEnergySq = 0.0;
      for (let n = 0; n < this.frameLength; n++) {
        const rawSample = pcmFloat32[offset + n];
        rawEnergySq += rawSample * rawSample;
        const scaledVal = rawSample * 32768.0;
        frameBuf[n] = scaledVal;
        mean += scaledVal;
      }
      mean /= this.frameLength;
      frameRms[t] = Math.sqrt(rawEnergySq / this.frameLength);

      real.fill(0);
      imag.fill(0);
      let prev = frameBuf[0] - mean;
      real[0] = prev * (1.0 - this.preemph) * this.hammingWindow[0];
      for (let n = 1; n < this.frameLength; n++) {
        const curr = frameBuf[n] - mean;
        real[n] = (curr - this.preemph * prev) * this.hammingWindow[n];
        prev = curr;
      }

      this.inPlaceFFT512(real, imag);
      for (let k = 0; k <= this.fftSize / 2; k++) {
        powerSpec[k] = real[k] * real[k] + imag[k] * imag[k];
      }

      const melFrame = new Float32Array(this.numMelBins);
      for (let b = 0; b < this.numMelBins; b++) {
        let energy = 0.0;
        const filter = this.melFilters[b];
        for (let k = 0; k <= this.fftSize / 2; k++) {
          energy += filter[k] * powerSpec[k];
        }
        melFrame[b] = Math.log(Math.max(energy, fltEpsilon));
      }
      frames[t] = melFrame;
    }

    const smoothRms = new Float32Array(numFrames);
    for (let t = 0; t < numFrames; t++) {
      const rPrev = t > 0 ? frameRms[t - 1] : frameRms[t];
      const rNext = t + 1 < numFrames ? frameRms[t + 1] : frameRms[t];
      smoothRms[t] = 0.25 * rPrev + 0.5 * frameRms[t] + 0.25 * rNext;
    }

    return { frames, frameRms, smoothRms, numFrames };
  }

  public computeGlobalSpeechChannelProfile(
    allFrames: Float32Array[],
    smoothRms: Float32Array,
    vadThreshold: number
  ): { globalChannelMean: Float32Array; globalChannelStd: Float32Array } {
    const globalChannelMean = new Float32Array(this.numMelBins);
    const globalChannelStd = new Float32Array(this.numMelBins).fill(1.0);
    let activeCount = 0;

    for (let t = 0; t < allFrames.length; t++) {
      if (smoothRms[t] >= vadThreshold) {
        const frame = allFrames[t];
        for (let b = 0; b < this.numMelBins; b++) {
          globalChannelMean[b] += frame[b];
        }
        activeCount++;
      }
    }

    if (activeCount === 0) return { globalChannelMean, globalChannelStd };
    for (let b = 0; b < this.numMelBins; b++) {
      globalChannelMean[b] /= activeCount;
    }

    for (let t = 0; t < allFrames.length; t++) {
      if (smoothRms[t] >= vadThreshold) {
        const frame = allFrames[t];
        for (let b = 0; b < this.numMelBins; b++) {
          const diff = frame[b] - globalChannelMean[b];
          globalChannelStd[b] += diff * diff;
        }
      }
    }

    for (let b = 0; b < this.numMelBins; b++) {
      globalChannelStd[b] = Math.sqrt(
        globalChannelStd[b] / Math.max(1, activeCount) + 1e-5
      );
    }

    return { globalChannelMean, globalChannelStd };
  }

  /**
   * 【通道錨定混合 CMN (`sliceAndApplyCMN`)】：
   * 保留 55% ~ 80% 之全局會議通道均值 (`globalChannelMean`) 作為參考基準，
   * 防止 100% 減去局部語句均值時把語者固有的靜態聲道共振峰（男低音/女高音特質）一併抹除！
   */
  public sliceAndApplyCMN(
    allFrames: Float32Array[],
    startFrame: number,
    endFrame: number,
    globalChannelMean?: Float32Array
  ): { cmnTensor: Float32Array; binMeans: Float32Array; numPureFrames: number } {
    const safeStart = Math.max(0, Math.min(allFrames.length - 1, startFrame));
    const safeEnd = Math.max(
      safeStart + 1,
      Math.min(allFrames.length, endFrame)
    );
    const T = Math.max(1, safeEnd - safeStart);
    const cmnTensor = new Float32Array(T * this.numMelBins);
    const binMeans = new Float32Array(this.numMelBins);

    for (let t = safeStart; t < safeEnd; t++) {
      const frame = allFrames[t];
      for (let b = 0; b < this.numMelBins; b++) {
        binMeans[b] += frame[b];
      }
    }
    for (let b = 0; b < this.numMelBins; b++) {
      binMeans[b] /= T;
    }

    const alphaGlobal = globalChannelMean
      ? Math.max(0.55, Math.min(0.80, (250.0 - Math.min(200, T)) / 250.0))
      : 0.0;

    for (let t = 0; t < T; t++) {
      const frame = allFrames[safeStart + t];
      const rowOffset = t * this.numMelBins;
      for (let b = 0; b < this.numMelBins; b++) {
        const refMean = globalChannelMean
          ? alphaGlobal * globalChannelMean[b] +
            (1.0 - alphaGlobal) * binMeans[b]
          : binMeans[b];
        cmnTensor[rowOffset + b] = frame[b] - refMean;
      }
    }
    return { cmnTensor, binMeans, numPureFrames: T };
  }

  public sliceMaskedPureFramesAndApplyCMN(
    allFrames: Float32Array[],
    pureFrameIndices: number[],
    globalChannelMean?: Float32Array
  ): { cmnTensor: Float32Array; binMeans: Float32Array; numPureFrames: number } {
    const T = pureFrameIndices.length;
    if (T === 0) {
      return {
        cmnTensor: new Float32Array(this.numMelBins),
        binMeans: new Float32Array(this.numMelBins),
        numPureFrames: 0,
      };
    }

    const cmnTensor = new Float32Array(T * this.numMelBins);
    const binMeans = new Float32Array(this.numMelBins);

    for (let idx = 0; idx < T; idx++) {
      const f = Math.max(
        0,
        Math.min(allFrames.length - 1, pureFrameIndices[idx])
      );
      const frame = allFrames[f];
      for (let b = 0; b < this.numMelBins; b++) {
        binMeans[b] += frame[b];
      }
    }
    for (let b = 0; b < this.numMelBins; b++) {
      binMeans[b] /= T;
    }

    const alphaGlobal = globalChannelMean
      ? Math.max(0.55, Math.min(0.80, (250.0 - Math.min(200, T)) / 250.0))
      : 0.0;

    for (let idx = 0; idx < T; idx++) {
      const f = Math.max(
        0,
        Math.min(allFrames.length - 1, pureFrameIndices[idx])
      );
      const frame = allFrames[f];
      const rowOffset = idx * this.numMelBins;
      for (let b = 0; b < this.numMelBins; b++) {
        const refMean = globalChannelMean
          ? alphaGlobal * globalChannelMean[b] +
            (1.0 - alphaGlobal) * binMeans[b]
          : binMeans[b];
        cmnTensor[rowOffset + b] = frame[b] - refMean;
      }
    }

    return { cmnTensor, binMeans, numPureFrames: T };
  }

  public computeFormantPreservingTstpEmbedding(
    cmnTensor: Float32Array,
    binMeans: Float32Array,
    numFrames: number,
    globalChannelMean?: Float32Array,
    globalChannelStd?: Float32Array,
    f0Hz = 145,
    enableDivergenceEnhance = true,
    isHeritageClassicMode = false
  ): Float32Array {
    const dim = 64;
    const out = new Float32Array(dim);
    if (numFrames <= 0) return out;

    const cleanBinProfile = new Float32Array(80);
    let segGain = 0.0;
    for (let b = 0; b < 80; b++) {
      const chRemoved = globalChannelMean
        ? binMeans[b] - globalChannelMean[b]
        : binMeans[b];
      cleanBinProfile[b] = chRemoved;
      segGain += chRemoved;
    }
    segGain /= 80.0;

    const normSpectralEnvelope = new Float32Array(80);
    for (let b = 0; b < 80; b++) {
      let val = cleanBinProfile[b] - segGain;
      if (enableDivergenceEnhance && globalChannelStd) {
        const formantWeight = b >= 16 && b <= 64 ? 1.38 : 0.9;
        val = (val / Math.max(0.35, globalChannelStd[b])) * formantWeight;
      }
      normSpectralEnvelope[b] = val;
    }

    for (let k = 1; k <= 24; k++) {
      let sum = 0.0;
      for (let b = 0; b < 80; b++) {
        sum +=
          normSpectralEnvelope[b] *
          Math.cos((Math.PI * k * (2 * b + 1)) / 160.0);
      }
      const lifter = 1.0 + 6.0 * Math.sin((Math.PI * k) / 24.0);
      out[k - 1] = sum * 0.08 * lifter;
    }

    for (let i = 0; i < 24; i++) {
      const b0 = 12 + i * 2;
      out[24 + i] =
        (normSpectralEnvelope[b0] + normSpectralEnvelope[b0 + 1]) *
        (isHeritageClassicMode ? 0.72 : 0.55);
    }

    const bandStd = new Float32Array(80);
    for (let t = 0; t < numFrames; t++) {
      const off = t * 80;
      for (let b = 0; b < 80; b++) {
        const v = cmnTensor[off + b];
        bandStd[b] += v * v;
      }
    }
    const denom = Math.max(1, numFrames - 1);
    for (let b = 0; b < 80; b++) {
      bandStd[b] = Math.sqrt(bandStd[b] / denom + 1e-7);
    }

    let stdMean = 0.0;
    for (let b = 0; b < 80; b++) stdMean += bandStd[b];
    stdMean /= 80.0;

    for (let i = 0; i < 12; i++) {
      const bStart = 8 + i * 5;
      const avgStd =
        (bandStd[bStart] +
          bandStd[bStart + 1] +
          bandStd[bStart + 2] +
          bandStd[bStart + 3] +
          bandStd[bStart + 4]) *
        0.2;
      out[48 + i] =
        (avgStd - stdMean) * (isHeritageClassicMode ? 0.55 : 0.75);
    }

    const logF0Norm =
      (Math.log(Math.max(75, Math.min(340, f0Hz))) - 5.05) * 1.35;
    let lowBandE = 0.0;
    let highBandE = 0.0;
    for (let b = 4; b < 28; b++) lowBandE += normSpectralEnvelope[b];
    for (let b = 28; b < 68; b++) highBandE += normSpectralEnvelope[b];
    const spectralTilt = (highBandE / 40.0 - lowBandE / 24.0) * 0.65;

    out[60] = logF0Norm * 0.75;
    out[61] = spectralTilt;
    out[62] = (normSpectralEnvelope[18] - normSpectralEnvelope[42]) * 0.6;
    out[63] = (normSpectralEnvelope[26] - normSpectralEnvelope[54]) * 0.6;

    return l2Normalize(out);
  }

  private inPlaceFFT512(real: Float32Array, imag: Float32Array): void {
    const N = 512;
    let j = 0;
    for (let i = 0; i < N; i++) {
      if (i < j) {
        const tr = real[i];
        real[i] = real[j];
        real[j] = tr;
        const ti = imag[i];
        imag[i] = imag[j];
        imag[j] = ti;
      }
      let m = N >> 1;
      while (m >= 1 && j >= m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
    for (let s = 1; s <= 9; s++) {
      const m = 1 << s;
      const halfM = m >> 1;
      const theta = (-2.0 * Math.PI) / m;
      const wpr = Math.cos(theta);
      const wpi = Math.sin(theta);
      for (let k = 0; k < N; k += m) {
        let wr = 1.0;
        let wi = 0.0;
        for (let l = 0; l < halfM; l++) {
          const idx1 = k + l;
          const idx2 = idx1 + halfM;
          const tr = wr * real[idx2] - wi * imag[idx2];
          const ti = wr * imag[idx2] + wi * real[idx2];
          real[idx2] = real[idx1] - tr;
          imag[idx2] = imag[idx1] - ti;
          real[idx1] += tr;
          imag[idx1] += ti;
          const nextWr = wr * wpr - wi * wpi;
          wi = wr * wpi + wi * wpr;
          wr = nextWr;
        }
      }
    }
  }
}

class SileroVadRunner {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ortModule: any = null;
  private checked = false;

  constructor(private config: DiarizationConfig) {}

  public async init(): Promise<boolean> {
    if (this.session) return true;
    if (this.checked) return false;
    this.checked = true;

    try {
      const candidates = [
        this.config.sileroVadModelFilename || 'silero_vad_v5.onnx',
        'silero_vad_v5.onnx',
        'silero_vad.onnx',
      ];
      const loaded = await fetchModelBufferWithCache(candidates, 500_000);
      if (!loaded) return false;

      this.ortModule = await import('onnxruntime-web');
      if (this.ortModule.env) {
        this.ortModule.env.logLevel = 'error';
      }
      const wasmRootUrl = resolvePublicRootUrl('wasm/');
      if (this.ortModule.env?.wasm) {
        this.ortModule.env.wasm.wasmPaths = wasmRootUrl;
        this.ortModule.env.wasm.simd = true;
        this.ortModule.env.wasm.numThreads = 1;
      }

      this.session = await this.ortModule.InferenceSession.create(
        loaded.buffer,
        {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        }
      );
      console.info(
        `✅ [Silero VAD] 成功載入本地神經 VAD 模型 (${loaded.fileName})！`
      );
      return true;
    } catch {
      return false;
    }
  }

  public async release(): Promise<void> {
    if (this.session && typeof this.session.release === 'function') {
      try {
        await this.session.release();
      } catch {
        // ignore
      }
    }
    this.session = null;
  }

  public async computeFrameVadProbabilities(
    samples: Float32Array,
    smoothRms: Float32Array,
    vadThreshold: number
  ): Promise<Float32Array> {
    const numFbankFrames = smoothRms.length;
    const outVad = new Float32Array(numFbankFrames);

    if (this.session && this.ortModule) {
      try {
        const ort = this.ortModule;
        const winSamples = 512;
        const numSteps = Math.floor(samples.length / winSamples);
        const stepProbs = new Float32Array(numSteps);

        let stateTensor = new ort.Tensor(
          'float32',
          new Float32Array(2 * 1 * 128),
          [2, 1, 128]
        );
        const srTensor = new ort.Tensor(
          'int64',
          BigInt64Array.from([16000n]),
          []
        );

        const inputNames: string[] = this.session.inputNames || [
          'input',
          'state',
          'sr',
        ];

        for (let s = 0; s < numSteps; s++) {
          const slice = samples.subarray(
            s * winSamples,
            (s + 1) * winSamples
          );
          const inputTensor = new ort.Tensor('float32', slice, [
            1,
            winSamples,
          ]);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const feeds: Record<string, any> = {};
          feeds[inputNames[0] || 'input'] = inputTensor;
          if (inputNames.includes('state')) feeds['state'] = stateTensor;
          if (inputNames.includes('sr')) feeds['sr'] = srTensor;

          const res = await this.session.run(feeds);
          const outName = this.session.outputNames[0];
          const stateOutName = this.session.outputNames[1];
          const p = res[outName]?.data?.[0] ?? 0.5;
          stepProbs[s] = Number(p);
          if (stateOutName && res[stateOutName]) {
            stateTensor = res[stateOutName];
          }
        }

        for (let f = 0; f < numFbankFrames; f++) {
          const sec = f * 0.01;
          const sIdx = Math.min(
            numSteps - 1,
            Math.max(0, Math.floor(sec / 0.032))
          );
          const ratio = smoothRms[f] / Math.max(1e-5, vadThreshold);
          const energyProb = 1.0 / (1.0 + Math.exp(-3.4 * (ratio - 0.92)));
          // 融合 Silero VAD 與 P02 能量機率，避免遠距小聲語者被單一模型漏抓
          outVad[f] = Math.max(stepProbs[sIdx], energyProb * 0.88);
        }
        return outVad;
      } catch {
        // Fallback below
      }
    }

    for (let f = 0; f < numFbankFrames; f++) {
      const ratio = smoothRms[f] / Math.max(1e-5, vadThreshold);
      outVad[f] = 1.0 / (1.0 + Math.exp(-3.4 * (ratio - 0.92)));
    }
    return outVad;
  }
}

export class AutoTuningSpectralClusterer {
  public static symmetricEigenDecomposition(
    matrix: Float64Array,
    N: number,
    computeVectors: boolean
  ): { eigenvalues: Float64Array; eigenvectors?: Float64Array } {
    const d = new Float64Array(N);
    const e = new Float64Array(N);
    const z = new Float64Array(matrix);

    for (let i = N - 1; i > 0; i--) {
      const l = i - 1;
      let h = 0.0;
      let scale = 0.0;
      if (l > 0) {
        for (let k = 0; k <= l; k++) scale += Math.abs(z[i * N + k]);
        if (scale === 0.0) {
          e[i] = z[i * N + l];
        } else {
          for (let k = 0; k <= l; k++) {
            z[i * N + k] /= scale;
            h += z[i * N + k] * z[i * N + k];
          }
          let f = z[i * N + l];
          const g = f >= 0.0 ? -Math.sqrt(h) : Math.sqrt(h);
          e[i] = scale * g;
          h -= f * g;
          z[i * N + l] = f - g;
          f = 0.0;
          for (let j = 0; j <= l; j++) {
            if (computeVectors) z[j * N + i] = z[i * N + j] / h;
            let g2 = 0.0;
            for (let k = 0; k <= j; k++) g2 += z[j * N + k] * z[i * N + k];
            for (let k = j + 1; k <= l; k++) g2 += z[k * N + j] * z[i * N + k];
            e[j] = g2 / h;
            f += e[j] * z[i * N + j];
          }
          const hh = f / (h + h);
          for (let j = 0; j <= l; j++) {
            f = z[i * N + j];
            e[j] = e[j] - hh * f;
            const g2 = e[j];
            for (let k = 0; k <= j; k++) {
              z[j * N + k] -= f * e[k] + g2 * z[i * N + k];
            }
          }
        }
      } else {
        e[i] = z[i * N + l];
      }
      d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for (let i = 0; i < N; i++) {
      if (computeVectors && d[i] !== 0.0) {
        for (let j = 0; j < i; j++) {
          let g = 0.0;
          for (let k = 0; k < i; k++) g += z[i * N + k] * z[k * N + j];
          for (let k = 0; k < i; k++) z[k * N + j] -= g * z[k * N + i];
        }
      }
      d[i] = z[i * N + i];
      if (computeVectors) {
        z[i * N + i] = 1.0;
        for (let j = 0; j < i; j++) {
          z[j * N + i] = 0.0;
          z[i * N + j] = 0.0;
        }
      }
    }

    for (let i = 1; i < N; i++) e[i - 1] = e[i];
    e[N - 1] = 0.0;

    for (let l = 0; l < N; l++) {
      let iter = 0;
      let m = l;
      do {
        for (m = l; m < N - 1; m++) {
          const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
          if (Math.abs(e[m]) + dd === dd) break;
        }
        if (m !== l) {
          if (iter++ === 60) break;
          let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
          let r = Math.hypot(g, 1.0);
          g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
          let s = 1.0;
          let c = 1.0;
          let p = 0.0;
          for (let i = m - 1; i >= l; i--) {
            const f = s * e[i];
            const b = c * e[i];
            r = Math.hypot(f, g);
            e[i + 1] = r;
            if (r === 0.0) {
              d[i + 1] -= p;
              e[m] = 0.0;
              break;
            }
            s = f / r;
            c = g / r;
            g = d[i + 1] - p;
            r = (d[i] - g) * s + 2.0 * c * b;
            p = s * r;
            d[i + 1] = g + p;
            g = c * r - b;
            if (computeVectors) {
              for (let k = 0; k < N; k++) {
                const fVec = z[k * N + i + 1];
                z[k * N + i + 1] = s * z[k * N + i] + c * fVec;
                z[k * N + i] = c * z[k * N + i] - s * fVec;
              }
            }
          }
          if (r === 0.0 && m >= l) continue;
          d[l] -= p;
          e[l] = g;
          e[m] = 0.0;
        }
      } while (m !== l);
    }

    const indices = Array.from({ length: N }, (_, idx) => idx).sort(
      (a, b) => d[a] - d[b]
    );
    const sortedVals = new Float64Array(N);
    const sortedVecs = computeVectors ? new Float64Array(N * N) : undefined;
    for (let col = 0; col < N; col++) {
      const origCol = indices[col];
      sortedVals[col] = Math.max(0.0, d[origCol]);
      if (computeVectors && sortedVecs) {
        for (let row = 0; row < N; row++) {
          sortedVecs[row * N + col] = z[row * N + origCol];
        }
      }
    }
    return { eigenvalues: sortedVals, eigenvectors: sortedVecs };
  }

  public static applyE2cpConstraintPropagation(
    affinityMatrix: Float64Array,
    N: number,
    constraintMatrix: Float64Array,
    alphaProp = 0.35
  ): Float64Array {
    if (N <= 2) return affinityMatrix;

    let hasOffDiagConstraint = false;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i !== j && Math.abs(constraintMatrix[i * N + j]) > 0.05) {
          hasOffDiagConstraint = true;
          break;
        }
      }
      if (hasOffDiagConstraint) break;
    }
    if (!hasOffDiagConstraint) {
      return affinityMatrix;
    }

    const invSqrtDeg = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let deg = 0.0;
      for (let j = 0; j < N; j++) {
        if (i !== j) deg += Math.max(0.0, affinityMatrix[i * N + j]);
      }
      invSqrtDeg[i] = 1.0 / Math.sqrt(deg + 1e-10);
    }

    const Lnorm = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        Lnorm[i * N + j] =
          Math.max(0.0, affinityMatrix[i * N + j]) *
          invSqrtDeg[i] *
          invSqrtDeg[j];
      }
    }

    const Zclean = new Float64Array(constraintMatrix);
    for (let i = 0; i < N; i++) Zclean[i * N + i] = 0.0;

    let Fstar = new Float64Array(Zclean);
    const nextF = new Float64Array(N * N);
    for (let iter = 0; iter < 4; iter++) {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          let sum = 0.0;
          for (let k = 0; k < N; k++) {
            sum += Lnorm[i * N + k] * Fstar[k * N + j];
          }
          nextF[i * N + j] = alphaProp * sum + (1.0 - alphaProp) * Zclean[i * N + j];
        }
      }
      Fstar.set(nextF);
    }

    const Astar = new Float64Array(affinityMatrix);
    for (let i = 0; i < N; i++) {
      Astar[i * N + i] = 1.0;
      for (let j = i + 1; j < N; j++) {
        const fVal = Math.max(-0.6, Math.min(0.6, Fstar[i * N + j]));
        const aVal = affinityMatrix[i * N + j];
        const mod =
          fVal >= 0
            ? aVal + fVal * 0.25 * (1.0 - aVal)
            : aVal * (1.0 + fVal * 0.5);
        Astar[i * N + j] = mod;
        Astar[j * N + i] = mod;
      }
    }
    return Astar;
  }

  public static clusterManifoldAhcCosine(
    affinityMatrix: Float64Array,
    N: number,
    minK = 2,
    maxK = 6,
    stopSimThreshold = 0.28
  ): { labels: Int32Array; optimalK: number } {
    if (N <= 2 || maxK <= 1) {
      return { labels: new Int32Array(N).fill(0), optimalK: 1 };
    }

    const effectiveMinK = Math.max(1, Math.min(2, minK));
    const effectiveMaxK = Math.max(
      effectiveMinK,
      Math.min(maxK + 1, Math.floor(N / 2))
    );

    const clusters: number[][] = Array.from({ length: N }, (_, idx) => [idx]);

    while (clusters.length > effectiveMinK) {
      let bestA = -1;
      let bestB = -1;
      let maxAvgSim = -Infinity;

      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          let sumSim = 0.0;
          const cA = clusters[i];
          const cB = clusters[j];
          for (let a = 0; a < cA.length; a++) {
            const rowOff = cA[a] * N;
            for (let b = 0; b < cB.length; b++) {
              sumSim += affinityMatrix[rowOff + cB[b]];
            }
          }
          const avgSim = sumSim / (cA.length * cB.length);
          if (avgSim > maxAvgSim) {
            maxAvgSim = avgSim;
            bestA = i;
            bestB = j;
          }
        }
      }

      if (bestA === -1 || bestB === -1) break;
      if (clusters.length <= effectiveMaxK && maxAvgSim < stopSimThreshold) {
        break;
      }

      clusters[bestA].push(...clusters[bestB]);
      clusters.splice(bestB, 1);
    }

    while (clusters.length > effectiveMinK) {
      const singleIdx = clusters.findIndex((c) => c.length === 1);
      if (singleIdx === -1) break;

      let bestTarget = -1;
      let bestSim = -Infinity;
      const sNode = clusters[singleIdx][0];
      for (let j = 0; j < clusters.length; j++) {
        if (j === singleIdx) continue;
        let sumS = 0.0;
        for (const m of clusters[j]) {
          sumS += affinityMatrix[sNode * N + m];
        }
        const avgS = sumS / clusters[j].length;
        if (avgS > bestSim) {
          bestSim = avgS;
          bestTarget = j;
        }
      }
      if (bestTarget === -1) break;
      clusters[bestTarget].push(...clusters[singleIdx]);
      clusters.splice(singleIdx, 1);
    }

    const labels = new Int32Array(N);
    clusters.forEach((members, cIdx) => {
      for (const m of members) labels[m] = cIdx;
    });
    return { labels, optimalK: clusters.length };
  }

  public static clusterManifoldNmeSc(
    affinityMatrix: Float64Array,
    N: number,
    minK = 2,
    maxK = 6,
    stopSimThreshold = 0.28
  ): { labels: Int32Array; optimalK: number; optimalP: number } {
    if (N <= 2 || maxK <= 1) {
      return { labels: new Int32Array(N).fill(0), optimalK: 1, optimalP: 1 };
    }

    const effectiveMinK = Math.max(1, Math.min(2, minK));
    const ahcProbe = this.clusterManifoldAhcCosine(
      affinityMatrix,
      N,
      effectiveMinK,
      maxK,
      stopSimThreshold
    );

    const p = Math.max(3, Math.min(18, Math.floor(0.18 * N)));
    const laplacian = this.buildNormalizedLaplacian(affinityMatrix, N, p);
    const { eigenvalues, eigenvectors } = this.symmetricEigenDecomposition(
      laplacian,
      N,
      true
    );

    const effectiveMaxK = Math.max(
      effectiveMinK,
      Math.min(maxK, Math.floor(N / 2))
    );

    let bestK = effectiveMinK;
    let maxGap = -1.0;
    for (let k = effectiveMinK; k <= effectiveMaxK; k++) {
      const gap = eigenvalues[k] - eigenvalues[k - 1];
      if (gap > maxGap) {
        maxGap = gap;
        bestK = k;
      }
    }

    const targetK = Math.min(
      effectiveMaxK,
      Math.max(bestK, ahcProbe.optimalK)
    );

    if (ahcProbe.optimalK >= bestK) {
      return {
        labels: ahcProbe.labels,
        optimalK: ahcProbe.optimalK,
        optimalP: p,
      };
    }

    const spectralEmbeddings: Float64Array[] = new Array(N);
    for (let i = 0; i < N; i++) {
      const rowVec = new Float64Array(targetK);
      let normSq = 0.0;
      for (let k = 0; k < targetK; k++) {
        const val = eigenvectors![i * N + k];
        rowVec[k] = val;
        normSq += val * val;
      }
      const invNorm = 1.0 / Math.sqrt(normSq + 1e-12);
      for (let k = 0; k < targetK; k++) rowVec[k] *= invNorm;
      spectralEmbeddings[i] = rowVec;
    }

    const labels = this.sphericalKMeansPlusPlus(
      spectralEmbeddings,
      N,
      targetK
    );
    return { labels, optimalK: targetK, optimalP: p };
  }

  public static refineClustersWithVariationalBayesVbx(
    embeddings: Float32Array[],
    initialLabels: Int32Array,
    initialK: number,
    minK: number,
    maxK: number
  ): { labels: Int32Array; responsibilities: Float64Array[]; activeK: number } {
    const N = embeddings.length;
    if (N === 0 || initialK <= 1) {
      return {
        labels: new Int32Array(N).fill(0),
        responsibilities: Array.from(
          { length: N },
          () => new Float64Array([1.0])
        ),
        activeK: 1,
      };
    }

    const D = embeddings[0].length;
    const S = initialK;
    const gamma: Float64Array[] = Array.from(
      { length: N },
      () => new Float64Array(S)
    );
    const pi = new Float64Array(S).fill(1.0 / S);

    for (let n = 0; n < N; n++) {
      const initC = Math.min(S - 1, Math.max(0, initialLabels[n]));
      for (let s = 0; s < S; s++) {
        gamma[n][s] = s === initC ? 0.96 : 0.04 / Math.max(1, S - 1);
      }
    }

    const centroids: Float32Array[] = Array.from(
      { length: S },
      () => new Float32Array(D)
    );
    const invTemp = D >= 128 ? 20.0 : 12.0;
    const coreGate = 0.45;

    for (let iter = 0; iter < 8; iter++) {
      for (let s = 0; s < S; s++) {
        centroids[s].fill(0.0);
        let sumG = 0.0;
        let coreCount = 0;
        for (let n = 0; n < N; n++) {
          const g = gamma[n][s];
          sumG += g;
          if (g >= coreGate) {
            coreCount++;
            const emb = embeddings[n];
            for (let d = 0; d < D; d++) {
              centroids[s][d] += g * emb[d];
            }
          }
        }
        if (coreCount === 0) {
          for (let n = 0; n < N; n++) {
            const g = gamma[n][s];
            const emb = embeddings[n];
            for (let d = 0; d < D; d++) {
              centroids[s][d] += g * emb[d];
            }
          }
        }
        centroids[s] = l2Normalize(centroids[s]);
        pi[s] = Math.pow(Math.max(1e-4, sumG / N), 0.18);
      }

      let piSum = 0.0;
      for (let s = 0; s < S; s++) piSum += pi[s];
      for (let s = 0; s < S; s++) pi[s] /= Math.max(1e-12, piSum);

      for (let n = 0; n < N; n++) {
        const logP = new Float64Array(S);
        let maxLogP = -Infinity;
        for (let s = 0; s < S; s++) {
          const cosSim = cosineSimilarity(embeddings[n], centroids[s]);
          const val = Math.log(pi[s] + 1e-9) + invTemp * cosSim;
          logP[s] = val;
          if (val > maxLogP) maxLogP = val;
        }

        let sumExp = 0.0;
        for (let s = 0; s < S; s++) {
          const eVal = Math.exp(logP[s] - maxLogP);
          gamma[n][s] = eVal;
          sumExp += eVal;
        }
        for (let s = 0; s < S; s++) {
          gamma[n][s] /= Math.max(1e-12, sumExp);
        }
      }
    }

    const clusterMass = new Float64Array(S);
    for (let n = 0; n < N; n++) {
      for (let s = 0; s < S; s++) {
        clusterMass[s] += gamma[n][s];
      }
    }

    const clusterRanked = Array.from({ length: S }, (_, s) => ({
      s,
      mass: clusterMass[s],
    })).sort((a, b) => b.mass - a.mass);

    const effectiveMinK = Math.max(1, Math.min(2, minK));
    let keptCount = clusterRanked.filter((c) => c.mass >= 1.5).length;
    keptCount = Math.max(effectiveMinK, Math.min(maxK, Math.min(S, keptCount)));
    const keptClusterIndices = clusterRanked
      .slice(0, keptCount)
      .map((c) => c.s);

    const finalLabels = new Int32Array(N);
    const finalResps: Float64Array[] = new Array(N);

    for (let n = 0; n < N; n++) {
      const r = new Float64Array(keptCount);
      let sumR = 0.0;
      let bestK = 0;
      let bestVal = -1.0;

      for (let k = 0; k < keptCount; k++) {
        const origS = keptClusterIndices[k];
        const v = gamma[n][origS];
        r[k] = v;
        sumR += v;
        if (v > bestVal) {
          bestVal = v;
          bestK = k;
        }
      }
      for (let k = 0; k < keptCount; k++) {
        r[k] = sumR > 1e-9 ? r[k] / sumR : 1.0 / keptCount;
      }
      finalLabels[n] = bestK;
      finalResps[n] = r;
    }

    return {
      labels: finalLabels,
      responsibilities: finalResps,
      activeK: keptCount,
    };
  }

  private static buildNormalizedLaplacian(
    A: Float64Array,
    N: number,
    p: number
  ): Float64Array {
    const pruned = new Float64Array(N * N);
    const rowCopy = new Float64Array(N);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) rowCopy[j] = A[i * N + j];
      rowCopy.sort();
      const threshold = rowCopy[Math.max(0, N - p)];
      for (let j = 0; j < N; j++) {
        pruned[i * N + j] = A[i * N + j] >= threshold ? 1.0 : 0.0;
      }
    }

    const W = new Float64Array(N * N);
    const invSqrtDeg = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let deg = 0.0;
      for (let j = 0; j < N; j++) {
        const w = 0.5 * (pruned[i * N + j] + pruned[j * N + i]);
        W[i * N + j] = w;
        deg += w;
      }
      invSqrtDeg[i] = 1.0 / Math.sqrt(deg + 1e-10);
    }

    const L = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const normW = W[i * N + j] * invSqrtDeg[i] * invSqrtDeg[j];
        L[i * N + j] = (i === j ? 1.0 : 0.0) - normW;
      }
    }
    return L;
  }

  private static sphericalKMeansPlusPlus(
    points: Float64Array[],
    N: number,
    K: number
  ): Int32Array {
    const dim = points[0].length;
    let bestLabels = new Int32Array(N);
    let bestInertia = Infinity;

    for (let trial = 0; trial < 6; trial++) {
      const centroids: Float64Array[] = [];
      const firstIdx = (trial * 17) % N;
      centroids.push(new Float64Array(points[firstIdx]));

      const distSq = new Float64Array(N).fill(Infinity);
      for (let c = 1; c < K; c++) {
        let maxD = -1.0;
        let bestIdx = 0;
        for (let i = 0; i < N; i++) {
          let dot = 0.0;
          for (let d = 0; d < dim; d++) {
            dot += points[i][d] * centroids[c - 1][d];
          }
          const d2 = Math.max(0.0, 2.0 - 2.0 * dot);
          if (d2 < distSq[i]) distSq[i] = d2;
          if (distSq[i] > maxD) {
            maxD = distSq[i];
            bestIdx = i;
          }
        }
        centroids.push(new Float64Array(points[bestIdx]));
      }

      const labels = new Int32Array(N);
      for (let iter = 0; iter < 30; iter++) {
        let changed = false;
        for (let i = 0; i < N; i++) {
          let bestC = 0;
          let bestSim = -Infinity;
          for (let c = 0; c < K; c++) {
            let sim = 0.0;
            for (let d = 0; d < dim; d++) {
              sim += points[i][d] * centroids[c][d];
            }
            if (sim > bestSim) {
              bestSim = sim;
              bestC = c;
            }
          }
          if (labels[i] !== bestC) {
            labels[i] = bestC;
            changed = true;
          }
        }
        if (!changed) break;

        for (let c = 0; c < K; c++) {
          centroids[c].fill(0.0);
          let count = 0;
          for (let i = 0; i < N; i++) {
            if (labels[i] === c) {
              for (let d = 0; d < dim; d++) {
                centroids[c][d] += points[i][d];
              }
              count++;
            }
          }
          if (count > 0) {
            let norm = 0.0;
            for (let d = 0; d < dim; d++) {
              norm += centroids[c][d] * centroids[c][d];
            }
            const inv = 1.0 / Math.sqrt(norm + 1e-12);
            for (let d = 0; d < dim; d++) centroids[c][d] *= inv;
          }
        }
      }

      let inertia = 0.0;
      for (let i = 0; i < N; i++) {
        const c = labels[i];
        let dot = 0.0;
        for (let d = 0; d < dim; d++) dot += points[i][d] * centroids[c][d];
        inertia += Math.max(0.0, 2.0 - 2.0 * dot);
      }

      if (inertia < bestInertia) {
        bestInertia = inertia;
        bestLabels = labels;
      }
    }

    return bestLabels;
  }
}

class WeSpeakerOnnxRunner {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ortModule: any = null;
  private checked = false;
  public embeddingDim = 64;

  constructor(private config: DiarizationConfig) {}

  public async init(): Promise<boolean> {
    if (this.config.engineMode === 'mode_a_classic') {
      this.checked = true;
      this.embeddingDim = 64;
      return false;
    }

    if (this.session) return true;
    if (this.checked) return false;
    this.checked = true;

    try {
      const candidateFiles = Array.from(
        new Set([
          this.config.onnxModelFilename ||
            'wespeaker_voxceleb_resnet34_int8.onnx',
          'wespeaker_voxceleb_resnet34_int8.onnx',
          'voxceleb_resnet34_LM.onnx',
          'nemo_en_titanet_small.int8.onnx',
        ])
      );

      const loaded = await fetchModelBufferWithCache(candidateFiles, 500_000);
      if (!loaded) return false;

      this.ortModule = await import('onnxruntime-web');
      if (this.ortModule.env) {
        this.ortModule.env.logLevel = 'error';
      }
      const wasmRootUrl = resolvePublicRootUrl('wasm/');
      if (this.ortModule.env?.wasm) {
        this.ortModule.env.wasm.wasmPaths = wasmRootUrl;
        this.ortModule.env.wasm.simd = true;
        this.ortModule.env.wasm.numThreads = Math.max(
          1,
          Math.min(2, Math.floor((navigator.hardwareConcurrency || 4) / 2))
        );
      }

      const preferredProviders =
        this.config.executionBackend === 'webgpu'
          ? ['webgpu', 'wasm']
          : ['wasm'];

      try {
        this.session = await this.ortModule.InferenceSession.create(
          loaded.buffer,
          {
            executionProviders: preferredProviders,
            graphOptimizationLevel: 'all',
          }
        );
      } catch {
        this.session = await this.ortModule.InferenceSession.create(
          loaded.buffer,
          {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
          }
        );
      }
      this.embeddingDim = 256;
      console.info(
        `✅ [WeSpeaker ONNX] 成功載入本地 256 維深度聲紋模型 (${loaded.fileName}, ${(
          loaded.buffer.byteLength /
          (1024 * 1024)
        ).toFixed(1)} MB)！`
      );
      return true;
    } catch {
      return false;
    }
  }

  public async release(): Promise<void> {
    if (this.session && typeof this.session.release === 'function') {
      try {
        await this.session.release();
      } catch {
        // ignore
      }
    }
    this.session = null;
  }

  public isReady(): boolean {
    return Boolean(this.session && this.ortModule);
  }

  public async computeEmbedding(
    cmnTensor: Float32Array,
    binMeans: Float32Array,
    numFrames: number,
    fbankHelper: KaldiFbankExtractor,
    globalChannelMean?: Float32Array,
    globalChannelStd?: Float32Array,
    f0Hz = 145
  ): Promise<Float32Array> {
    const enableEnhance =
      this.config.enableAcousticDivergencePreproc ?? true;
    const isHeritage =
      this.config.engineMode === 'mode_0_heritage_classic';

    if (!this.session || !this.ortModule) {
      this.embeddingDim = 64;
      return fbankHelper.computeFormantPreservingTstpEmbedding(
        cmnTensor,
        binMeans,
        numFrames,
        globalChannelMean,
        globalChannelStd,
        f0Hz,
        enableEnhance,
        isHeritage
      );
    }

    try {
      const ort = this.ortModule;
      const inputNames: string[] = this.session.inputNames || ['feats'];

      let effectiveFrames = numFrames;
      let effectiveCmn = cmnTensor;
      if (numFrames > 0 && numFrames < 100) {
        effectiveFrames = 100;
        effectiveCmn = new Float32Array(100 * 80);
        for (let t = 0; t < 100; t++) {
          const srcT = t % numFrames;
          effectiveCmn.set(
            cmnTensor.subarray(srcT * 80, (srcT + 1) * 80),
            t * 80
          );
        }
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const feeds: Record<string, any> = {};

      if (inputNames.includes('audio_signal')) {
        const transposed = new Float32Array(80 * effectiveFrames);
        for (let t = 0; t < effectiveFrames; t++) {
          for (let b = 0; b < 80; b++) {
            transposed[b * effectiveFrames + t] = effectiveCmn[t * 80 + b];
          }
        }
        feeds['audio_signal'] = new ort.Tensor('float32', transposed, [
          1,
          80,
          effectiveFrames,
        ]);
        if (inputNames.includes('length')) {
          feeds['length'] = new ort.Tensor(
            'int64',
            BigInt64Array.from([BigInt(effectiveFrames)]),
            [1]
          );
        }
      } else {
        feeds[inputNames[0]] = new ort.Tensor('float32', effectiveCmn, [
          1,
          effectiveFrames,
          80,
        ]);
      }

      const outputMap = await this.session.run(feeds);
      const outName = this.session.outputNames.includes('embs')
        ? 'embs'
        : this.session.outputNames[0];
      const rawEmb = outputMap[outName]?.data as Float32Array | undefined;

      if (rawEmb && rawEmb.length >= 64) {
        this.embeddingDim = rawEmb.length;
        return l2Normalize(rawEmb);
      }
    } catch {
      // fallback to 64-dim TSTP
    }

    this.embeddingDim = 64;
    return fbankHelper.computeFormantPreservingTstpEmbedding(
      cmnTensor,
      binMeans,
      numFrames,
      globalChannelMean,
      globalChannelStd,
      f0Hz,
      enableEnhance,
      isHeritage
    );
  }
}

interface MacroAnchorNode {
  startFrame: number;
  endFrame: number;
  startSec: number;
  endSec: number;
  durSec: number;
  f0Hz: number;
  embedding: Float32Array;
}

function buildVirtualSpeakerTrackBatches(
  fineTurns: FineDiarizationTurn[],
  chunkStartSec: number,
  chunkEndSec: number
): SpeakerTrackBatch[] {
  if (fineTurns.length === 0) return [];

  const mergedTurns: FineDiarizationTurn[] = [];
  for (const t of fineTurns) {
    const prev = mergedTurns[mergedTurns.length - 1];
    if (
      prev &&
      prev.speakerId === t.speakerId &&
      t.start - prev.end <= 0.45 &&
      t.end - prev.start <= 14.0
    ) {
      prev.end = t.end;
      prev.speakerConfidence = Math.max(
        prev.speakerConfidence ?? 0.8,
        t.speakerConfidence ?? 0.8
      );
      prev.speakerMargin = Math.max(
        prev.speakerMargin ?? 0.12,
        t.speakerMargin ?? 0.12
      );
      prev.boundaryScdRight = t.boundaryScdRight;
    } else {
      mergedTurns.push({ ...t });
    }
  }

  const bySpeaker = new Map<string, FineDiarizationTurn[]>();
  for (const t of mergedTurns) {
    const list = bySpeaker.get(t.speakerId) || [];
    list.push(t);
    bySpeaker.set(t.speakerId, list);
  }

  const SPACER_SEC = 0.35;
  const TARGET_BATCH_SPEECH_SEC = 16.5;
  const MAX_SINGLE_TURN_SEC = 18.0;
  const rawBatches: Omit<SpeakerTrackBatch, 'batchIndex'>[] = [];

  const speakerEntries = Array.from(bySpeaker.entries());
  for (let sIdx = 0; sIdx < speakerEntries.length; sIdx++) {
    const [speakerId, turns] = speakerEntries[sIdx];
    let currentMap: VirtualTrackMapItem[] = [];
    let virtualCursorSec = 0.0;
    let accumSpeechSec = 0.0;

    const flushCurrentBatch = () => {
      if (currentMap.length === 0) return;
      rawBatches.push({
        speakerId,
        earliestRealStartSec: currentMap[0].realStartSec,
        latestRealEndSec: currentMap[currentMap.length - 1].realEndSec,
        totalVirtualDurationSec: Number(virtualCursorSec.toFixed(2)),
        spacerDurationSec: SPACER_SEC,
        timeMap: currentMap,
      });
      currentMap = [];
      virtualCursorSec = 0.0;
      accumSpeechSec = 0.0;
    };

    for (const turn of turns) {
      const paddedRealStart = Number(
        Math.max(chunkStartSec, turn.start - 0.06).toFixed(2)
      );
      const paddedRealEnd = Number(
        Math.min(chunkEndSec, turn.end + 0.08).toFixed(2)
      );
      const dur = Math.max(0.2, paddedRealEnd - paddedRealStart);

      if (
        currentMap.length > 0 &&
        (accumSpeechSec + dur > TARGET_BATCH_SPEECH_SEC ||
          dur >= MAX_SINGLE_TURN_SEC)
      ) {
        flushCurrentBatch();
      }

      if (currentMap.length > 0) {
        virtualCursorSec += SPACER_SEC;
      }

      const vStart = Number(virtualCursorSec.toFixed(2));
      const vEnd = Number((virtualCursorSec + dur).toFixed(2));

      currentMap.push({
        virtualStartSec: vStart,
        virtualEndSec: vEnd,
        realStartSec: paddedRealStart,
        realEndSec: paddedRealEnd,
        speakerId: turn.speakerId,
        runnerUpSpeakerId: turn.runnerUpSpeakerId || turn.speakerId,
        speakerConfidence: turn.speakerConfidence ?? 0.85,
        speakerMargin: turn.speakerMargin ?? 0.16,
        boundaryScdLeft: turn.boundaryScdLeft ?? 0,
        boundaryScdRight: turn.boundaryScdRight ?? 0,
      });

      virtualCursorSec = vEnd;
      accumSpeechSec += dur;
    }

    flushCurrentBatch();
  }

  rawBatches.sort((a, b) => a.earliestRealStartSec - b.earliestRealStartSec);

  return rawBatches.map((b, idx) => ({
    ...b,
    batchIndex: idx,
  }));
}

self.onmessage = async (event: MessageEvent) => {
  const data = event.data || {};

  let samples: Float32Array | null = null;
  if (data.samples instanceof Float32Array) {
    samples = data.samples;
  } else if (data.samples instanceof ArrayBuffer) {
    samples = new Float32Array(data.samples);
  }

  const sampleRate = Number(data.sampleRate) || 16000;
  const chunkStartSec = Number(data.chunkStartSec ?? 0);
  const minSpeakers = Math.max(1, Math.min(10, Number(data.minSpeakers) || 2));
  const maxSpeakers = Math.max(
    minSpeakers,
    Math.min(10, Number(data.maxSpeakers) || 6)
  );
  // 單一 5 分鐘微批次內實際開口人數可能僅 2 人（如主席與單一分區報告人），
  // 故單塊聚類下限設為 Math.min(2, minSpeakers)，防止強制把 2 個人切成 4~6 群！
  const chunkEffectiveMinSpeakers = Math.max(1, Math.min(2, minSpeakers));

  const existingSpeakers: Speaker[] = Array.isArray(data.existingSpeakers)
    ? data.existingSpeakers
    : [];

  const diarConfig: DiarizationConfig = {
    engineMode:
      data.diarizationConfig?.engineMode || 'mode_0_heritage_classic',
    clusterMethod: data.diarizationConfig?.clusterMethod || 'ahc_cosine',
    executionBackend: data.diarizationConfig?.executionBackend || 'wasm',
    onnxModelFilename:
      data.diarizationConfig?.onnxModelFilename ||
      'wespeaker_voxceleb_resnet34_int8.onnx',
    sileroVadModelFilename:
      data.diarizationConfig?.sileroVadModelFilename || 'silero_vad_v5.onnx',
    pyannoteSegModelFilename:
      data.diarizationConfig?.pyannoteSegModelFilename ||
      'segmentation-3.0.onnx',
    pldaParamsFilename:
      data.diarizationConfig?.pldaParamsFilename || 'plda-params-vbx.json',
    localWasmBasePath: data.diarizationConfig?.localWasmBasePath || './wasm/',
    localModelBasePath:
      data.diarizationConfig?.localModelBasePath || './models/',
    yieldEveryNUnits: Math.max(
      4,
      Number(data.diarizationConfig?.yieldEveryNUnits) || 14
    ),
    enableGlobalMeanCentering:
      data.diarizationConfig?.enableGlobalMeanCentering ?? false,
    languageHint: data.diarizationConfig?.languageHint || 'zh',
    shortTurnConfidenceKeepThreshold:
      data.diarizationConfig?.shortTurnConfidenceKeepThreshold ?? 0.66,
    pingPongMaxDurSec: data.diarizationConfig?.pingPongMaxDurSec ?? 1.6,
    enablePauseAndSemanticWeighting:
      data.diarizationConfig?.enablePauseAndSemanticWeighting ?? true,
    scdPeakThreshold: data.diarizationConfig?.scdPeakThreshold ?? 0.22,
    backchannelMarginThreshold:
      data.diarizationConfig?.backchannelMarginThreshold ?? 0.12,
    enableAcousticDivergencePreproc:
      data.diarizationConfig?.enableAcousticDivergencePreproc ?? true,
    enableVirtualSpeakerTrackStitching:
      data.diarizationConfig?.enableVirtualSpeakerTrackStitching ?? false,
    embeddingExcludeOverlap:
      data.diarizationConfig?.embeddingExcludeOverlap ?? true,
    formulaCalibration: data.diarizationConfig?.formulaCalibration,
  };
  const formulaCal = diarConfig.formulaCalibration;

  if (!samples || samples.length === 0) {
    self.postMessage({
      type: 'DIARIZE_SUCCESS',
      turns: [],
      asrWindows: [],
      speakerTrackBatches: [],
      updatedSpeakers: existingSpeakers,
    });
    return;
  }

  const chunkEndSec = Number(
    (chunkStartSec + samples.length / sampleRate).toFixed(2)
  );

  const fbankExtractor = new KaldiFbankExtractor();
  const onnxRunner = new WeSpeakerOnnxRunner(diarConfig);
  const sileroRunner = new SileroVadRunner(diarConfig);

  try {
    await Promise.all([onnxRunner.init(), sileroRunner.init()]);

    self.postMessage({
      type: 'DIARIZE_PROGRESS',
      progress: 12,
      detailText:
        '計算 Kaldi 80-dim Fbank、P02 真實靜音底噪與 YIN 無倍頻誤差基頻軌跡...',
    });

    const {
      frames: allFbankFrames,
      frameRms,
      smoothRms,
      numFrames,
    } = fbankExtractor.computeGlobalFbank(samples);

    if (numFrames < 40) {
      await Promise.all([onnxRunner.release(), sileroRunner.release()]);
      self.postMessage({
        type: 'DIARIZE_SUCCESS',
        turns: [],
        asrWindows: [],
        speakerTrackBatches: [],
        updatedSpeakers: existingSpeakers,
      });
      return;
    }

    const sortedRms = Array.from(frameRms).sort((a, b) => a - b);
    const noiseQuantile = Math.min(0.03, formulaCal?.vadNoiseQuantile ?? 0.02);
    const speechMult = Math.min(1.42, formulaCal?.vadSpeechMultiplier ?? 1.32);
    const noiseFloor =
      sortedRms[Math.floor(sortedRms.length * noiseQuantile)] || 0.003;
    // 嚴格以 P02 靜音底噪錨定 vadThreshold，絕不讓近距大聲講者 (p75) 墊高門檻而吞沒遠距與會者！
    const vadThreshold = Math.max(
      0.0048,
      Math.min(noiseFloor * 1.48, noiseFloor * speechMult)
    );

    const { globalChannelMean, globalChannelStd } =
      fbankExtractor.computeGlobalSpeechChannelProfile(
        allFbankFrames,
        smoothRms,
        vadThreshold
      );

    const frameVadProbs = await sileroRunner.computeFrameVadProbabilities(
      samples,
      smoothRms,
      vadThreshold
    );

    await sileroRunner.release();

    const {
      stepHarmonicity,
      stepZcr,
      calibratedHarmGate,
      boundaryScoreCurve,
      queryMedianPitchHz,
      queryVoicedHarmonicRatio,
    } = computePitchTrackAndBoundaryCurve(
      samples,
      smoothRms,
      vadThreshold,
      sampleRate,
      allFbankFrames,
      formulaCal
    );

    samples = null;

    // 僅針對「低諧波度且高過零率 (zcr > 0.26)」之真實翻紙/摩擦雜音壓低 VAD，100% 保護遠距低頻男聲 (zcr ~ 0.09)
    const frictionCeiling =
      formulaCal?.frictionHarmonicCeiling ??
      Math.max(0.20, Math.min(0.28, calibratedHarmGate * 0.76));
    for (let s = 0; s < stepHarmonicity.length; s++) {
      const prevH = s > 0 ? stepHarmonicity[s - 1] : stepHarmonicity[s];
      const nextH =
        s + 1 < stepHarmonicity.length
          ? stepHarmonicity[s + 1]
          : stepHarmonicity[s];
      const localMaxH = Math.max(stepHarmonicity[s], prevH, nextH);
      if (localMaxH < frictionCeiling && stepZcr[s] > 0.26) {
        const f0 = s * 4;
        for (let k = f0; k < Math.min(numFrames, f0 + 4); k++) {
          frameVadProbs[k] = Math.min(frameVadProbs[k], 0.12);
        }
      }
    }

    const macroAnchorBounds: Array<{ startF: number; endF: number }> = [];
    let inSp = false;
    let sf = 0;
    let silFrames = 0;
    let fCursor = 0;

    while (fCursor < numFrames) {
      if (smoothRms[fCursor] >= vadThreshold) {
        if (!inSp) {
          inSp = true;
          sf = fCursor;
        }
        silFrames = 0;

        if (fCursor - sf >= 185) {
          const wStart = sf + 120;
          const wEnd = Math.min(numFrames, sf + 220);
          let bestCut = wStart;
          let minVal = Infinity;
          for (let k = wStart; k < wEnd; k++) {
            if (smoothRms[k] < minVal) {
              minVal = smoothRms[k];
              bestCut = k;
            }
          }
          if (bestCut - sf >= 120) {
            macroAnchorBounds.push({ startF: sf, endF: bestCut });
          }
          sf = bestCut;
          fCursor = bestCut;
          continue;
        }
      } else if (inSp) {
        silFrames++;
        if (silFrames >= 8) {
          const ef = fCursor - silFrames + 1;
          if (ef - sf >= 115) {
            macroAnchorBounds.push({ startF: sf, endF: ef });
          }
          inSp = false;
          silFrames = 0;
        }
      }
      fCursor++;
    }
    if (inSp && numFrames - sf >= 115) {
      macroAnchorBounds.push({ startF: sf, endF: numFrames });
    }

    const harmonicMacroBounds = macroAnchorBounds.filter((b) => {
      const { voicedRatio, medianHarm } = queryVoicedHarmonicRatio(
        b.startF,
        b.endF
      );
      return voicedRatio >= 0.35 && medianHarm >= calibratedHarmGate;
    });
    if (harmonicMacroBounds.length >= Math.max(2, chunkEffectiveMinSpeakers)) {
      macroAnchorBounds.length = 0;
      macroAnchorBounds.push(...harmonicMacroBounds);
    }

    if (macroAnchorBounds.length < Math.max(3, chunkEffectiveMinSpeakers)) {
      macroAnchorBounds.length = 0;
      for (let f = 0; f + 90 <= numFrames; f += 75) {
        macroAnchorBounds.push({
          startF: f,
          endF: Math.min(numFrames, f + 110),
        });
      }
    }

    const macroAnchors: MacroAnchorNode[] = [];
    const yieldStep = Math.max(6, diarConfig.yieldEveryNUnits || 14);

    for (let idx = 0; idx < macroAnchorBounds.length; idx++) {
      const b = macroAnchorBounds[idx];
      const durF = b.endF - b.startF;
      const f0Hz = queryMedianPitchHz(b.startF, b.endF);
      const slice = fbankExtractor.sliceAndApplyCMN(
        allFbankFrames,
        b.startF,
        b.endF,
        globalChannelMean
      );
      const emb = await onnxRunner.computeEmbedding(
        slice.cmnTensor,
        slice.binMeans,
        durF,
        fbankExtractor,
        globalChannelMean,
        globalChannelStd,
        f0Hz
      );

      macroAnchors.push({
        startFrame: b.startF,
        endFrame: b.endF,
        startSec: chunkStartSec + b.startF * 0.01,
        endSec: chunkStartSec + b.endF * 0.01,
        durSec: durF * 0.01,
        f0Hz,
        embedding: emb,
      });

      if (
        (idx + 1) % yieldStep === 0 ||
        idx === macroAnchorBounds.length - 1
      ) {
        const pct = 15 + Math.round(((idx + 1) / macroAnchorBounds.length) * 35);
        self.postMessage({
          type: 'DIARIZE_PROGRESS',
          progress: pct,
          detailText: `第一階段：提取高信噪比黃金長句聲紋錨點 (${idx + 1}/${macroAnchorBounds.length})...`,
        });
        await new Promise((r) => setTimeout(r, 1));
      }
    }

    const M = macroAnchors.length;
    const embDim = macroAnchors[0]?.embedding.length || 256;
    const isOnnx256 = embDim >= 128;

    const anchorAffinity = new Float64Array(M * M);
    for (let i = 0; i < M; i++) {
      anchorAffinity[i * M + i] = 1.0;
      for (let j = i + 1; j < M; j++) {
        const rawSim = cosineSimilarity(
          macroAnchors[i].embedding,
          macroAnchors[j].embedding
        );
        const pitchPen = computeSoftPitchCompatibilityPenalty(
          macroAnchors[i].f0Hz,
          macroAnchors[j].f0Hz,
          isOnnx256 ? 0.14 : 0.20
        );
        const s = rawSim - pitchPen;
        anchorAffinity[i * M + j] = s;
        anchorAffinity[j * M + i] = s;
      }
    }

    const stopSimThreshold = isOnnx256 ? 0.27 : 0.44;

    const seedRes =
      diarConfig.clusterMethod === 'spectral_cosine' ||
      diarConfig.clusterMethod === 'e2cp_constrained_spectral'
        ? AutoTuningSpectralClusterer.clusterManifoldNmeSc(
            anchorAffinity,
            M,
            chunkEffectiveMinSpeakers,
            maxSpeakers,
            stopSimThreshold
          )
        : AutoTuningSpectralClusterer.clusterManifoldAhcCosine(
            anchorAffinity,
            M,
            chunkEffectiveMinSpeakers,
            maxSpeakers,
            stopSimThreshold
          );

    const vbxRes =
      AutoTuningSpectralClusterer.refineClustersWithVariationalBayesVbx(
        macroAnchors.map((a) => a.embedding),
        seedRes.labels,
        seedRes.optimalK,
        chunkEffectiveMinSpeakers,
        maxSpeakers
      );

    interface DiscoveredClusterProfile {
      clusterIdx: number;
      anchorIndices: number[];
      centroid: Float32Array;
      exemplars: Float32Array[];
      f0Hz: number;
      totalDurSec: number;
      earliestStartSec: number;
    }

    const localClusters: DiscoveredClusterProfile[] = [];
    for (let k = 0; k < vbxRes.activeK; k++) {
      const memberIndices: number[] = [];
      for (let m = 0; m < M; m++) {
        if (vbxRes.labels[m] === k) memberIndices.push(m);
      }
      if (memberIndices.length === 0) continue;

      const meanVec = new Float32Array(embDim);
      let durSum = 0.0;
      let earliest = Infinity;
      const pitches: number[] = [];

      for (const mIdx of memberIndices) {
        const anc = macroAnchors[mIdx];
        const w = Math.max(0.5, anc.durSec);
        durSum += anc.durSec;
        if (anc.startSec < earliest) earliest = anc.startSec;
        pitches.push(anc.f0Hz);
        for (let d = 0; d < embDim; d++) {
          meanVec[d] += anc.embedding[d] * w;
        }
      }

      pitches.sort((a, b) => a - b);
      const medF0 = pitches[Math.floor(pitches.length / 2)] || 145;
      const normCentroid = l2Normalize(meanVec);

      const topExemplars = memberIndices
        .map((mIdx) => ({
          mIdx,
          sim: cosineSimilarity(macroAnchors[mIdx].embedding, normCentroid),
        }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, 4)
        .map((item) => macroAnchors[item.mIdx].embedding);

      localClusters.push({
        clusterIdx: k,
        anchorIndices: memberIndices,
        centroid: normCentroid,
        exemplars: topExemplars,
        f0Hz: medF0,
        totalDurSec: durSum,
        earliestStartSec: earliest,
      });
    }

    // 【同性別音色衛星群聚整併】：防止同一位男主席或女報告人因遠近音量或語速變化被拆為多個同性別衛星群！
    while (localClusters.length > chunkEffectiveMinSpeakers) {
      let bestI = -1;
      let bestJ = -1;
      let maxCSim = -Infinity;

      for (let i = 0; i < localClusters.length; i++) {
        for (let j = i + 1; j < localClusters.length; j++) {
          const cA = localClusters[i];
          const cB = localClusters[j];
          const rawSim = cosineSimilarity(cA.centroid, cB.centroid);
          const pitchPen = computeSoftPitchCompatibilityPenalty(
            cA.f0Hz,
            cB.f0Hz,
            isOnnx256 ? 0.14 : 0.20
          );
          const cSim = rawSim - pitchPen;
          const semiDiff = Math.abs(
            12.0 * Math.log2(Math.max(75, cA.f0Hz) / Math.max(75, cB.f0Hz))
          );
          const isSameGenderRegister =
            ((cA.f0Hz < 162 && cB.f0Hz < 162) ||
              (cA.f0Hz >= 168 && cB.f0Hz >= 168)) &&
            semiDiff <= 3.6;
          const minCount = Math.min(
            cA.anchorIndices.length,
            cB.anchorIndices.length
          );
          const minDur = Math.min(cA.totalDurSec, cB.totalDurSec);

          let mergeThresh = isOnnx256
            ? minCount <= 4 || minDur < 12.0
              ? 0.36
              : 0.42
            : minCount <= 4 || minDur < 12.0
            ? 0.50
            : 0.58;

          if (isSameGenderRegister) {
            mergeThresh -= isOnnx256 ? 0.05 : 0.06;
          }

          if (cSim >= mergeThresh && cSim > maxCSim) {
            maxCSim = cSim;
            bestI = i;
            bestJ = j;
          }
        }
      }

      if (bestI === -1 || bestJ === -1) break;

      const target =
        localClusters[bestI].totalDurSec >= localClusters[bestJ].totalDurSec
          ? localClusters[bestI]
          : localClusters[bestJ];
      const source =
        target === localClusters[bestI]
          ? localClusters[bestJ]
          : localClusters[bestI];
      const removeIdx = target === localClusters[bestI] ? bestJ : bestI;

      const combinedIndices = [...target.anchorIndices, ...source.anchorIndices];
      const meanVec = new Float32Array(embDim);
      let durSum = 0.0;
      let earliest = Infinity;
      const pitches: number[] = [];

      for (const mIdx of combinedIndices) {
        const anc = macroAnchors[mIdx];
        const w = Math.max(0.5, anc.durSec);
        durSum += anc.durSec;
        if (anc.startSec < earliest) earliest = anc.startSec;
        pitches.push(anc.f0Hz);
        for (let d = 0; d < embDim; d++) {
          meanVec[d] += anc.embedding[d] * w;
        }
      }
      pitches.sort((a, b) => a - b);
      const normCent = l2Normalize(meanVec);
      const topEx = combinedIndices
        .map((mIdx) => ({
          mIdx,
          sim: cosineSimilarity(macroAnchors[mIdx].embedding, normCent),
        }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, 4)
        .map((item) => macroAnchors[item.mIdx].embedding);

      target.anchorIndices = combinedIndices;
      target.centroid = normCent;
      target.exemplars = topEx;
      target.f0Hz = pitches[Math.floor(pitches.length / 2)] || target.f0Hz;
      target.totalDurSec = durSum;
      target.earliestStartSec = earliest;

      localClusters.splice(removeIdx, 1);
    }

    if (localClusters.length > maxSpeakers) {
      localClusters.sort((a, b) => b.totalDurSec - a.totalDurSec);
      localClusters.length = maxSpeakers;
    }
    localClusters.sort((a, b) => a.earliestStartSec - b.earliestStartSec);

    const updatedSpeakers: Speaker[] = existingSpeakers.map((sp) => ({
      ...sp,
      centroid: sp.centroid ? [...sp.centroid] : undefined,
      embedding: sp.embedding ? [...sp.embedding] : undefined,
      anchorBank: sp.anchorBank ? sp.anchorBank.map((a) => [...a]) : undefined,
      mergedFrom: Array.isArray(sp.mergedFrom) ? [...sp.mergedFrom] : [],
    }));

    const usedGlobalSpeakerIds = new Set<string>();
    const activeProfiles = new Map<
      string,
      { f0Hz: number; masterCentroid: Float32Array; anchorBank: Float32Array[] }
    >();

    for (const loc of localClusters) {
      let bestSp: Speaker | null = null;
      let bestOmega = -1.0;

      for (const sp of updatedSpeakers) {
        if (usedGlobalSpeakerIds.has(sp.id)) continue;
        const spF0 = sp.f0Hz ?? sp.centroid?.[0] ?? 145;
        const spEmb =
          sp.embedding && sp.embedding.length === embDim ? sp.embedding : null;
        if (!spEmb) continue;

        const masterSim = cosineSimilarity(loc.centroid, spEmb);
        let sumAncSim = 0.0;
        let ancCount = 0;
        if (Array.isArray(sp.anchorBank) && sp.anchorBank.length > 0) {
          for (const anc of sp.anchorBank) {
            if (anc.length === embDim) {
              sumAncSim += cosineSimilarity(loc.centroid, anc);
              ancCount++;
            }
          }
        }
        const avgAncSim = ancCount > 0 ? sumAncSim / ancCount : masterSim;
        const pitchPen = computeSoftPitchCompatibilityPenalty(loc.f0Hz, spF0);
        const omega = 0.6 * masterSim + 0.4 * avgAncSim - pitchPen;
        if (omega > bestOmega) {
          bestOmega = omega;
          bestSp = sp;
        }
      }

      if (bestSp && bestOmega >= 0.38) {
        usedGlobalSpeakerIds.add(bestSp.id);
        if (bestOmega >= 0.46) {
          const oldMass = bestSp.accumulatedMass || 10.0;
          const alpha = Math.min(
            0.22,
            loc.totalDurSec / (oldMass + loc.totalDurSec)
          );
          const oldEmb = bestSp.embedding!;
          const blended = new Float32Array(embDim);
          for (let d = 0; d < embDim; d++) {
            blended[d] = (1.0 - alpha) * oldEmb[d] + alpha * loc.centroid[d];
          }
          const normMaster = l2Normalize(blended);
          const newF0 =
            (bestSp.f0Hz ?? loc.f0Hz) * (1.0 - alpha) + loc.f0Hz * alpha;
          bestSp.f0Hz = newF0;
          bestSp.embedding = Array.from(normMaster);
          bestSp.centroid = [newF0, ...Array.from(normMaster)];
          bestSp.accumulatedMass = oldMass + loc.totalDurSec;
          bestSp.featureMode = diarConfig.engineMode;
          bestSp.anchorBank = [
            ...(bestSp.anchorBank || []).slice(0, 2),
            ...loc.exemplars.slice(0, 2).map((e) => Array.from(e)),
          ].slice(0, 4);
        }

        activeProfiles.set(bestSp.id, {
          f0Hz: bestSp.f0Hz ?? loc.f0Hz,
          masterCentroid: l2Normalize(bestSp.embedding!),
          anchorBank: (bestSp.anchorBank || [])
            .slice(0, 4)
            .map((a) => l2Normalize(a)),
        });
      } else if (updatedSpeakers.length < maxSpeakers) {
        const nextNum = updatedSpeakers.length + 1;
        const newId = `Speaker_${String(nextNum).padStart(2, '0')}`;
        const pitchLabel = loc.f0Hz >= 168 ? '女聲/高音' : '男聲/中低音';
        const newSp: Speaker = {
          id: newId,
          name: `語者 ${String(nextNum).padStart(2, '0')} (${pitchLabel})`,
          color: SPEAKER_PALETTE[(nextNum - 1) % SPEAKER_PALETTE.length],
          f0Hz: loc.f0Hz,
          embedding: Array.from(loc.centroid),
          anchorBank: loc.exemplars.map((e) => Array.from(e)),
          accumulatedMass: loc.totalDurSec,
          featureMode: diarConfig.engineMode,
          centroid: [loc.f0Hz, ...Array.from(loc.centroid)],
          mergedFrom: [],
        };
        updatedSpeakers.push(newSp);
        usedGlobalSpeakerIds.add(newId);
        activeProfiles.set(newId, {
          f0Hz: loc.f0Hz,
          masterCentroid: loc.centroid,
          anchorBank: loc.exemplars,
        });
      }
    }

    for (const sp of updatedSpeakers) {
      if (
        !activeProfiles.has(sp.id) &&
        sp.embedding &&
        sp.embedding.length === embDim
      ) {
        activeProfiles.set(sp.id, {
          f0Hz: sp.f0Hz ?? 145,
          masterCentroid: l2Normalize(sp.embedding),
          anchorBank: (sp.anchorBank || [])
            .filter((a) => a.length === embDim)
            .slice(0, 4)
            .map((a) => l2Normalize(a)),
        });
      }
    }

    if (activeProfiles.size === 0 && updatedSpeakers.length > 0) {
      const sp0 = updatedSpeakers[0];
      activeProfiles.set(sp0.id, {
        f0Hz: sp0.f0Hz ?? 145,
        masterCentroid: sp0.embedding
          ? l2Normalize(sp0.embedding)
          : new Float32Array(embDim),
        anchorBank: (sp0.anchorBank || []).map((a) => l2Normalize(a)),
      });
    }

    const scdSensitivity = diarConfig.scdPeakThreshold ?? 0.22;
    const finePeakCutoff =
      formulaCal?.scdPeakCutoff ??
      Math.max(0.47, Math.min(0.56, 0.40 + scdSensitivity * 0.5));

    // 【Stage 2 微觀話輪邊界切分】：
    // 僅在真實靜音停頓 (silFrames >= 8) 或真實聲學邊界波峰 (boundaryScoreCurve >= finePeakCutoff) 處切分，
    // 廢除舊版「每 1.65 秒無條件強制砍斷連續長句」之缺陷！
    const fineBounds: Array<{ startF: number; endF: number }> = [];
    inSp = false;
    sf = 0;
    silFrames = 0;
    fCursor = 0;

    while (fCursor < numFrames) {
      if (smoothRms[fCursor] >= vadThreshold) {
        if (!inSp) {
          inSp = true;
          sf = fCursor;
        }
        silFrames = 0;
        const curLen = fCursor - sf;

        if (curLen >= 42 && boundaryScoreCurve[fCursor] >= finePeakCutoff) {
          const checkStart = Math.max(sf + 35, fCursor - 14);
          const checkEnd = Math.min(numFrames, fCursor + 14);
          let isLocalMax = true;
          for (let k = checkStart; k < checkEnd; k++) {
            if (boundaryScoreCurve[k] > boundaryScoreCurve[fCursor]) {
              isLocalMax = false;
              break;
            }
          }
          if (isLocalMax) {
            fineBounds.push({ startF: sf, endF: fCursor });
            sf = fCursor;
            fCursor++;
            continue;
          }
        }

        // 若單人連續無停頓發聲超過 4.5 秒 (450 幀)，僅在內部確實存在次波峰 (>= finePeakCutoff * 0.78) 時才進行保護性分段
        if (curLen >= 450) {
          const wStart = sf + 150;
          const wEnd = Math.min(numFrames, sf + 450);
          let bestCut = -1;
          let maxB = finePeakCutoff * 0.78;
          for (let k = wStart; k < wEnd; k++) {
            if (boundaryScoreCurve[k] >= maxB) {
              maxB = boundaryScoreCurve[k];
              bestCut = k;
            }
          }
          if (bestCut > sf + 120) {
            fineBounds.push({ startF: sf, endF: bestCut });
            sf = bestCut;
            fCursor = bestCut;
            continue;
          }
        }
      } else if (inSp) {
        silFrames++;
        if (silFrames >= 8) {
          const ef = fCursor - silFrames + 1;
          if (ef - sf >= 20) {
            fineBounds.push({ startF: sf, endF: ef });
          }
          inSp = false;
          silFrames = 0;
        }
      }
      fCursor++;
    }
    if (inSp && numFrames - sf >= 20) {
      fineBounds.push({ startF: sf, endF: numFrames });
    }

    const activeEntries = Array.from(activeProfiles.entries());
    const K_active = activeEntries.length;
    const activeSpeakerIds = activeEntries.map(([id]) => id);
    const backchannelMarginReq = diarConfig.backchannelMarginThreshold ?? 0.12;

    const postFrameStepSec = 0.02;
    const numPostFrames = Math.max(
      10,
      Math.ceil((chunkEndSec - chunkStartSec) / postFrameStepSec)
    );
    const flatPosteriors = new Float32Array(numPostFrames * K_active);
    const postWeightAcc = new Float32Array(numPostFrames);
    const postOsdProbs = new Float32Array(numPostFrames);

    for (let f = 0; f < numPostFrames; f++) {
      const f10ms = Math.min(numFrames - 1, f * 2);
      postOsdProbs[f] = boundaryScoreCurve[f10ms] >= 0.54 ? 0.28 : 0.04;
      for (let k = 0; k < K_active; k++) {
        flatPosteriors[f * K_active + k] = 1.0 / Math.max(1, K_active);
      }
    }

    interface TurnCandidateEvaluation {
      startF: number;
      endF: number;
      startSec: number;
      endSec: number;
      f0Hz: number;
      bLeft: number;
      bRight: number;
      rawScores: Float64Array;
      probs: Float32Array;
    }

    const turnEvals: TurnCandidateEvaluation[] = [];

    for (let idx = 0; idx < fineBounds.length; idx++) {
      const b = fineBounds[idx];
      const durF = b.endF - b.startF;
      const { voicedRatio, medianHarm } = queryVoicedHarmonicRatio(
        b.startF,
        b.endF
      );
      // 保留遠距與會者之低能量諧波話輪（如「24號」、「禮拜三早上」），僅過濾純無聲摩擦雜訊
      if (voicedRatio < 0.12 && medianHarm < 0.24) {
        continue;
      }
      const f0Hz = queryMedianPitchHz(b.startF, b.endF);

      const slice = fbankExtractor.sliceAndApplyCMN(
        allFbankFrames,
        b.startF,
        b.endF,
        globalChannelMean
      );
      const emb = await onnxRunner.computeEmbedding(
        slice.cmnTensor,
        slice.binMeans,
        durF,
        fbankExtractor,
        globalChannelMean,
        globalChannelStd,
        f0Hz
      );

      let s1 = -2.0;
      const rawScores = new Float64Array(K_active);

      for (let pIdx = 0; pIdx < K_active; pIdx++) {
        const [, prof] = activeEntries[pIdx];
        const cSim = cosineSimilarity(emb, prof.masterCentroid);
        let sumASim = 0.0;
        for (const anc of prof.anchorBank) {
          sumASim += cosineSimilarity(emb, anc);
        }
        const avgASim =
          prof.anchorBank.length > 0 ? sumASim / prof.anchorBank.length : cSim;
        const pitchPen = computeSoftPitchCompatibilityPenalty(
          f0Hz,
          prof.f0Hz,
          0.18
        );
        const compScore = 0.62 * cSim + 0.38 * avgASim - pitchPen;
        rawScores[pIdx] = compScore;
        if (compScore > s1) s1 = compScore;
      }

      const probs = new Float32Array(K_active);
      let sumExp = 0.0;
      for (let k = 0; k < K_active; k++) {
        const e = Math.exp(10.0 * (rawScores[k] - s1));
        probs[k] = e;
        sumExp += e;
      }
      for (let k = 0; k < K_active; k++) {
        probs[k] /= Math.max(1e-12, sumExp);
      }

      const startSec = Number((chunkStartSec + b.startF * 0.01).toFixed(2));
      const endSec = Number((chunkStartSec + b.endF * 0.01).toFixed(2));
      const bLeft = Number((boundaryScoreCurve[b.startF] || 0).toFixed(3));
      const bRight = Number(
        (boundaryScoreCurve[Math.min(numFrames - 1, b.endF)] || 0).toFixed(3)
      );

      turnEvals.push({
        startF: b.startF,
        endF: b.endF,
        startSec,
        endSec,
        f0Hz,
        bLeft,
        bRight,
        rawScores,
        probs,
      });

      if ((idx + 1) % (yieldStep * 2) === 0 || idx === fineBounds.length - 1) {
        const pct = 52 + Math.round(((idx + 1) / fineBounds.length) * 42);
        self.postMessage({
          type: 'DIARIZE_PROGRESS',
          progress: pct,
          detailText: `第二階段：微觀話輪聲紋歸屬與話輪級 Viterbi 平滑 (${idx + 1}/${fineBounds.length})...`,
        });
        await new Promise((r) => setTimeout(r, 1));
      }
    }

    await onnxRunner.release();
    allFbankFrames.length = 0;

    // 【話輪級 Viterbi HMM 平滑解碼】：
    // 消除單一短話輪因翻頁雜音或語速變化在同性別語者間隨機跳動之現象，同時對真實音域跨越 (男<->女) 與高邊界波峰即時開放換人！
    const T_turns = turnEvals.length;
    const viterbiSpeakerIdx = new Int32Array(T_turns);
    if (T_turns > 0 && K_active > 1) {
      const dp = Array.from(
        { length: T_turns },
        () => new Float64Array(K_active).fill(-Infinity)
      );
      const backPtr = Array.from(
        { length: T_turns },
        () => new Int32Array(K_active)
      );

      for (let k = 0; k < K_active; k++) {
        dp[0][k] = turnEvals[0].rawScores[k];
      }

      for (let t = 1; t < T_turns; t++) {
        const prevTurn = turnEvals[t - 1];
        const currTurn = turnEvals[t];
        const gapSec = Math.max(0.0, currTurn.startSec - prevTurn.endSec);
        const semiDiff = Math.abs(
          12.0 *
            Math.log2(
              Math.max(75, currTurn.f0Hz) / Math.max(75, prevTurn.f0Hz)
            )
        );
        const isGenderSwitch =
          (prevTurn.f0Hz < 160 && currTurn.f0Hz >= 170) ||
          (prevTurn.f0Hz >= 170 && currTurn.f0Hz < 160);

        let switchCost = 0.095;
        if (isGenderSwitch || currTurn.bLeft >= finePeakCutoff) {
          switchCost = 0.01;
        } else if (gapSec <= 1.6 && semiDiff <= 3.0 && currTurn.bLeft < 0.42) {
          switchCost = 0.14;
        }

        const durWeight = Math.min(
          1.6,
          Math.max(0.75, (currTurn.endSec - currTurn.startSec) / 1.2)
        );

        for (let kCurr = 0; kCurr < K_active; kCurr++) {
          let bestPrevVal = -Infinity;
          let bestPrevK = 0;
          for (let kPrev = 0; kPrev < K_active; kPrev++) {
            const penalty = kCurr === kPrev ? 0.0 : switchCost;
            const val = dp[t - 1][kPrev] - penalty;
            if (val > bestPrevVal) {
              bestPrevVal = val;
              bestPrevK = kPrev;
            }
          }
          dp[t][kCurr] =
            bestPrevVal + durWeight * currTurn.rawScores[kCurr];
          backPtr[t][kCurr] = bestPrevK;
        }
      }

      let bestLastK = 0;
      let maxFinal = -Infinity;
      for (let k = 0; k < K_active; k++) {
        if (dp[T_turns - 1][k] > maxFinal) {
          maxFinal = dp[T_turns - 1][k];
          bestLastK = k;
        }
      }
      viterbiSpeakerIdx[T_turns - 1] = bestLastK;
      for (let t = T_turns - 1; t >= 1; t--) {
        viterbiSpeakerIdx[t - 1] = backPtr[t][viterbiSpeakerIdx[t]];
      }
    }

    const rawFineTurns: FineDiarizationTurn[] = [];
    for (let t = 0; t < T_turns; t++) {
      const ev = turnEvals[t];
      const chosenIdx = viterbiSpeakerIdx[t];
      const top1Sp = activeSpeakerIds[chosenIdx] || 'Speaker_01';

      let runnerUpIdx = chosenIdx;
      let bestAltScore = -Infinity;
      for (let k = 0; k < K_active; k++) {
        if (k !== chosenIdx && ev.rawScores[k] > bestAltScore) {
          bestAltScore = ev.rawScores[k];
          runnerUpIdx = k;
        }
      }
      const top2Sp = activeSpeakerIds[runnerUpIdx] || top1Sp;
      const s1 = ev.rawScores[chosenIdx];
      const s2 = runnerUpIdx !== chosenIdx ? ev.rawScores[runnerUpIdx] : s1 - 0.25;

      // 將 Viterbi 平滑後的主語者強化注入幀級後驗矩陣
      const smoothedProbs = new Float32Array(K_active);
      let pSum = 0.0;
      for (let k = 0; k < K_active; k++) {
        const boost = k === chosenIdx ? 1.35 : 1.0;
        smoothedProbs[k] = ev.probs[k] * boost;
        pSum += smoothedProbs[k];
      }
      for (let k = 0; k < K_active; k++) {
        smoothedProbs[k] /= Math.max(1e-12, pSum);
      }

      const fStart = Math.max(
        0,
        Math.floor((ev.startSec - chunkStartSec) / postFrameStepSec)
      );
      const fEnd = Math.min(
        numPostFrames - 1,
        Math.ceil((ev.endSec - chunkStartSec) / postFrameStepSec)
      );
      for (let f = fStart; f <= fEnd; f++) {
        const prevW = postWeightAcc[f];
        const newW = prevW + 1.0;
        const rowOff = f * K_active;
        for (let k = 0; k < K_active; k++) {
          flatPosteriors[rowOff + k] =
            prevW === 0
              ? smoothedProbs[k]
              : (flatPosteriors[rowOff + k] * prevW + smoothedProbs[k]) / newW;
        }
        postWeightAcc[f] = newW;
      }

      const margin = Math.max(0.0, s1 - (s2 > -1.5 ? s2 : 0.0));
      const p1 = Math.max(0.52, Math.min(0.99, (s1 + 1.0) / 2.0));
      const p2 = Math.max(0.01, p1 - margin);
      const logOdds = Number(
        (Math.log(p1 + 1e-6) - Math.log(p2 + 1e-6)).toFixed(3)
      );

      rawFineTurns.push({
        start: ev.startSec,
        end: ev.endSec,
        speakerId: top1Sp,
        runnerUpSpeakerId: top2Sp,
        f0Hz: Math.round(ev.f0Hz),
        speakerConfidence: Number(p1.toFixed(3)),
        speakerMargin: Number(margin.toFixed(3)),
        logOddsMargin: logOdds,
        overlapRatio: ev.bLeft >= 0.55 || ev.bRight >= 0.55 ? 0.22 : 0.04,
        boundaryScdLeft: ev.bLeft,
        boundaryScdRight: ev.bRight,
        isProtectedBackchannel:
          margin >= backchannelMarginReq || ev.bLeft + ev.bRight >= 0.85,
        isAnchorTurn: ev.endF - ev.startF >= 115,
      });
    }

    // 【同語者翻頁與換氣停頓無縫縫合】：
    // 只要相鄰兩個話輪屬於同一位語者（中間沒有其他語者開口），即使中間有 <= 2.4 秒的翻閱資料或思考停頓，亦直接縫合為單一連續話輪！
    const fineTimelineTurns: FineDiarizationTurn[] = [];
    for (const item of rawFineTurns) {
      const prev = fineTimelineTurns[fineTimelineTurns.length - 1];
      if (
        prev &&
        prev.speakerId === item.speakerId &&
        item.start - prev.end <= 2.4 &&
        item.end - prev.start <= 22.0
      ) {
        prev.end = item.end;
        prev.speakerConfidence = Math.max(
          prev.speakerConfidence ?? 0.8,
          item.speakerConfidence ?? 0.8
        );
        prev.speakerMargin = Math.max(
          prev.speakerMargin ?? 0.14,
          item.speakerMargin ?? 0.14
        );
        prev.boundaryScdRight = item.boundaryScdRight;
      } else {
        fineTimelineTurns.push({ ...item });
      }
    }

    const asrWindows = buildOptimalVadGuidedAsrWindows(
      frameVadProbs,
      frameRms,
      0.01,
      chunkStartSec,
      chunkEndSec,
      fineTimelineTurns
    );

    const speakerTrackBatches = buildVirtualSpeakerTrackBatches(
      fineTimelineTurns,
      chunkStartSec,
      chunkEndSec
    );

    const posteriorMatrix: FrameSpeakerPosteriorMatrix = {
      chunkStartSec,
      frameStepSec: postFrameStepSec,
      numFrames: numPostFrames,
      speakerIds: activeSpeakerIds,
      flatPosteriors,
      osdProbs: postOsdProbs,
    };

    console.info(
      `🎯 [兩階段語者分離完成] 共識別 ${updatedSpeakers.length} 位獨立語者，切出 ${fineTimelineTurns.length} 個微觀話輪與 ${asrWindows.length} 個最佳化轉寫視窗`
    );

    self.postMessage({
      type: 'DIARIZE_SUCCESS',
      turns: fineTimelineTurns,
      asrWindows,
      speakerTrackBatches,
      updatedSpeakers,
      posteriorMatrix,
    });
  } catch (err) {
    await Promise.all([onnxRunner.release(), sileroRunner.release()]);
    self.postMessage({
      type: 'DIARIZE_ERROR',
      error: (err as Error).message || '語者分離發生未知錯誤',
    });
  }
};
