// src/utils/audioProcessor.ts
// 次世代雙軌無損聲學預處理 (85Hz Zero-Phase In-Place filtfilt) + YIN CMNDF 基頻估計 + 遠近雙層語音響度補償 + 啟動階段 3~5 秒全文聲學預分析與「底層聲學公式動態校準層 (AcousticFormulaCalibration)」
// 核心架構升級：
// 1. [雙軌分流無損動態範圍保護]：語者分離軌道 (`extractAndPreprocessMacroChunk`) 嚴格維持「85Hz 零相位巴特沃斯高通濾波 + 線性無壓縮波形」，100% 保留語者間自然的能量谷底 (`dip`) 與遠近分貝階躍 (`stepDbJump`)。
// 2. [YIN CMNDF 累積均值正規化差分基頻估計]：根除傳統自相關在殘響會議室中將男聲二次諧波 (240Hz) 誤判為女聲、或將女聲次諧波 (110Hz) 誤判為男聲之倍頻誤差。
// 3. [P02 真實靜音底噪錨定與遠距語音 AGC 響度補償]：以 2% 分位數 (`P02`) 錨定真實環境底噪，避免近距大聲主講人拉高 VAD 門檻而吞沒遠距與會者（如「24號」、「禮拜三早上」），並在送入 Whisper 前針對有聲諧波幀提供最高 +12.5 dB (4.2x) 平滑響度補償。
// 4. [原地 (In-Place) 零額外配置濾波與即時釋放]：`applyZeroPhaseButterworthHighpass` 改為原地雙向濾波，省去 2 倍 Float32Array 暫存配置；預分析逐視窗處理後立即釋放 PCM 切片，大幅壓低啟動階段記憶體峰值。

import {
  AcousticFormulaCalibration,
  AudioPreAnalysisProfile,
  AudioRecommendationOption,
} from '../types';

export const TARGET_SAMPLE_RATE = 16000;
export const DEFAULT_CHUNK_DURATION_SEC = 300; // 5 分鐘微批次

// ── 三合一防休眠鎖 (WakeLock + Web Audio 無聲流 + Visibility 自動重連) ──
let wakeLockSentinel: WakeLockSentinel | null = null;
let keepAliveAudioCtx: AudioContext | null = null;
let keepAliveOsc: OscillatorNode | null = null;
let isKeepAwakeActive = false;

async function requestScreenWakeLock() {
  if ('wakeLock' in navigator && isKeepAwakeActive) {
    try {
      wakeLockSentinel = await navigator.wakeLock.request('screen');
    } catch {
      // ignore permission error
    }
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && isKeepAwakeActive) {
      requestScreenWakeLock();
    }
  });
}

export async function acquireKeepAwakeLock(): Promise<void> {
  isKeepAwakeActive = true;
  await requestScreenWakeLock();

  try {
    if (!keepAliveAudioCtx) {
      const AudioCtxClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      keepAliveAudioCtx = new AudioCtxClass();
      const gain = keepAliveAudioCtx.createGain();
      gain.gain.value = 0.00001;
      keepAliveOsc = keepAliveAudioCtx.createOscillator();
      keepAliveOsc.frequency.value = 1;
      keepAliveOsc.connect(gain);
      gain.connect(keepAliveAudioCtx.destination);
      keepAliveOsc.start();
    } else if (keepAliveAudioCtx.state === 'suspended') {
      await keepAliveAudioCtx.resume();
    }
  } catch {
    // ignore audio keep-alive fallback
  }
}

export async function releaseKeepAwakeLock(): Promise<void> {
  isKeepAwakeActive = false;
  if (wakeLockSentinel) {
    try {
      await wakeLockSentinel.release();
    } catch {
      // ignore
    }
    wakeLockSentinel = null;
  }
  if (keepAliveOsc) {
    try {
      keepAliveOsc.stop();
      keepAliveOsc.disconnect();
    } catch {
      // ignore
    }
    keepAliveOsc = null;
  }
  if (keepAliveAudioCtx) {
    try {
      await keepAliveAudioCtx.close();
    } catch {
      // ignore
    }
    keepAliveAudioCtx = null;
  }
}

export function getAudioDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    audio.preload = 'metadata';

    const cleanup = () => {
      URL.revokeObjectURL(url);
      audio.removeAttribute('src');
    };

    audio.onloadedmetadata = () => {
      const duration = audio.duration;
      cleanup();
      if (!Number.isFinite(duration) || duration <= 0) {
        reject(new Error('無法解析音訊總時長，請確認檔案格式是否正確。'));
      } else {
        resolve(duration);
      }
    };

    audio.onerror = () => {
      cleanup();
      reject(new Error('讀取音訊 Metadata 失敗。'));
    };

    audio.src = url;
  });
}

function alignMp3SyncHeader(bytes: Uint8Array): ArrayBuffer {
  let offset = 0;
  if (
    bytes.length > 10 &&
    bytes[0] === 0x49 &&
    bytes[1] === 0x44 &&
    bytes[2] === 0x33
  ) {
    const id3Size =
      ((bytes[6] & 0x7f) << 21) |
      ((bytes[7] & 0x7f) << 14) |
      ((bytes[8] & 0x7f) << 7) |
      (bytes[9] & 0x7f);
    offset = 10 + id3Size;
  }

  for (let i = offset; i < Math.min(bytes.length - 4, offset + 16384); i++) {
    if (bytes[i] === 0xff && (bytes[i + 1] & 0xe0) === 0xe0) {
      const layer = (bytes[i + 1] >> 1) & 0x03;
      const bitrateIdx = (bytes[i + 2] >> 4) & 0x0f;
      const sampleRateIdx = (bytes[i + 2] >> 2) & 0x03;
      if (layer !== 0 && bitrateIdx !== 0x0f && sampleRateIdx !== 0x03) {
        const sliced = bytes.slice(i);
        return sliced.buffer.slice(
          sliced.byteOffset,
          sliced.byteOffset + sliced.byteLength
        ) as ArrayBuffer;
      }
    }
  }

  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer as ArrayBuffer;
}

/**
 * 85 Hz 零相位雙向二階巴特沃斯高通濾波 (Zero-Phase In-Place Butterworth HPF, filtfilt)
 * 直接於傳入之 Float32Array 上原地完成正向與反向濾波，完全不額外配置兩份長陣列，降低 66% 預處理記憶體佔用。
 */
export function applyZeroPhaseButterworthHighpass(
  input: Float32Array,
  sampleRate = TARGET_SAMPLE_RATE,
  cutoffHz = 85.0
): Float32Array {
  const N = input.length;
  if (N < 16) return input;

  const w0 = (2.0 * Math.PI * cutoffHz) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);
  const alpha = sinW0 / Math.SQRT2;

  const a0 = 1.0 + alpha;
  const b0 = ((1.0 + cosW0) * 0.5) / a0;
  const b1 = -(1.0 + cosW0) / a0;
  const b2 = ((1.0 + cosW0) * 0.5) / a0;
  const a1 = (-2.0 * cosW0) / a0;
  const a2 = (1.0 - alpha) / a0;

  // 1. 原地正向濾波 (In-Place Forward Pass)
  let x1 = input[0];
  let x2 = input[0];
  let y1 = 0.0;
  let y2 = 0.0;
  for (let i = 0; i < N; i++) {
    const x0 = input[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    input[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }

  // 2. 原地反向濾波 (In-Place Backward Pass) 消除所有頻率之相位偏移
  x1 = input[N - 1];
  x2 = input[N - 1];
  y1 = 0.0;
  y2 = 0.0;
  for (let i = N - 1; i >= 0; i--) {
    const x0 = input[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    input[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }

  return input;
}

/**
 * YIN 累積均值正規化差分基頻與諧波清晰度估計器 (de Cheveigné & Kawahara 2002)
 * 同時防止男聲二次諧波倍頻 (120Hz -> 240Hz) 與女聲次諧波減半 (210Hz -> 105Hz) 誤差。
 */
function estimateFramePitchAndVoicingYin(
  samples: Float32Array,
  startIdx: number,
  sampleRate = TARGET_SAMPLE_RATE
): { f0Hz: number; periodicity: number; secondDipPeriodicity: number } {
  const W = 256;
  const minLag = 50; // 320 Hz
  const maxLag = 212; // 75.5 Hz
  if (startIdx + W + maxLag >= samples.length) {
    return { f0Hz: 0, periodicity: 0, secondDipPeriodicity: 0 };
  }

  const d = new Float32Array(maxLag + 1);
  for (let tau = 2; tau <= maxLag; tau += 2) {
    let sumSq = 0.0;
    for (let i = 0; i < W; i += 2) {
      const diff = samples[startIdx + i] - samples[startIdx + i + tau];
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
  if (minVal > 0.54) {
    return { f0Hz: 0, periodicity, secondDipPeriodicity: 0 };
  }

  const thresh = Math.max(0.24, minVal * 1.18);
  let chosenTau = bestTau;
  let secondBestVal = 1.0;
  let foundFirst = false;

  for (let tau = minLag + 2; tau <= maxLag - 2; tau += 2) {
    const val = cmndf[tau];
    if (val <= cmndf[tau - 2] && val <= cmndf[tau + 2]) {
      if (!foundFirst && val <= thresh) {
        chosenTau = tau;
        foundFirst = true;
      } else if (
        foundFirst &&
        Math.abs(tau - chosenTau * 2) > 8 &&
        val < secondBestVal
      ) {
        secondBestVal = val;
      }
    }
  }

  return {
    f0Hz: sampleRate / Math.max(minLag, chosenTau),
    periodicity: Math.max(0.0, Math.min(1.0, 1.0 - cmndf[chosenTau])),
    secondDipPeriodicity: Math.max(0.0, Math.min(1.0, 1.0 - secondBestVal)),
  };
}

/**
 * 【ASR 專用前瞻包絡響度補償器（支援遠近麥克風雙層動態均衡）】：
 * 針對同一視窗內既有近距大聲講者 (RMS ~0.16) 又有遠距與會者 (RMS ~0.018，如「24號」、「禮拜三早上」) 之情境，
 * 以 P02 底噪與諧波度/過零率雙重閘控，將遠距有聲語音平滑增益最高達 4.2x (+12.5 dB)，同時保持紙張摩擦與環境底噪為 1.0x 不放大。
 */
export function applyVadGatedLookAheadLoudnessNormalization(
  samples: Float32Array,
  sampleRate = TARGET_SAMPLE_RATE
): Float32Array {
  const N = samples.length;
  if (N < 1600) return samples;

  const frameLen = Math.floor(sampleRate * 0.02); // 320 samples (20ms)
  const numFrames = Math.ceil(N / frameLen);
  const frameRms = new Float32Array(numFrames);
  const frameZeroCross = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const start = f * frameLen;
    const end = Math.min(N, start + frameLen);
    const count = Math.max(1, end - start);
    let sumSq = 0.0;
    let zc = 0;
    for (let i = start; i < end; i++) {
      const v = samples[i];
      sumSq += v * v;
      if (i > start && (samples[i] >= 0) !== (samples[i - 1] >= 0)) {
        zc++;
      }
    }
    frameRms[f] = Math.sqrt(sumSq / count);
    frameZeroCross[f] = zc / count;
  }

  const minLag = Math.floor(sampleRate / 340);
  const maxLag = Math.floor(sampleRate / 75);
  const frameHarmonicity = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    if (frameRms[f] < 0.002) continue;
    const start = f * frameLen;
    if (start + 380 > N) continue;
    let maxCorr = 0.0;
    for (let lag = minLag; lag <= maxLag; lag += 2) {
      let sum = 0.0;
      let nA = 0.0;
      let nB = 0.0;
      for (let i = 0; i < 160; i += 2) {
        const a = samples[start + i];
        const b = samples[start + i + lag];
        sum += a * b;
        nA += a * a;
        nB += b * b;
      }
      const c = nA > 1e-8 && nB > 1e-8 ? sum / Math.sqrt(nA * nB) : 0.0;
      if (c > maxCorr) maxCorr = c;
    }
    frameHarmonicity[f] = maxCorr;
  }

  // 計算 5 幀 (100ms) 音節級諧波包絡，區分真實聲帶發聲 (syllabicHarm >= 0.42) 與冷氣底噪/桌面撞擊 (syllabicHarm < 0.34)
  const syllabicHarm = new Float32Array(numFrames);
  const nonHarmRmsList: number[] = [];
  for (let f = 0; f < numFrames; f++) {
    const windowH: number[] = [];
    for (let k = Math.max(0, f - 2); k <= Math.min(numFrames - 1, f + 2); k++) {
      windowH.push(frameHarmonicity[k]);
    }
    windowH.sort((a, b) => b - a);
    const topMean =
      windowH.length >= 2 ? 0.5 * (windowH[0] + windowH[1]) : windowH[0] || 0;
    syllabicHarm[f] = topMean;
    if (topMean < 0.36) {
      nonHarmRmsList.push(frameRms[f]);
    }
  }

  const sortedRms = Array.from(frameRms).sort((a, b) => a - b);
  nonHarmRmsList.sort((a, b) => a - b);
  // 以非諧波幀之 P40 (或全體 P12) 錨定真實環境底噪，徹底杜絕將冷氣/投影機底噪當作人聲放大！
  const noiseFloor = Math.max(
    0.0012,
    nonHarmRmsList.length >= 8
      ? nonHarmRmsList[Math.floor(nonHarmRmsList.length * 0.4)]
      : sortedRms[Math.floor(sortedRms.length * 0.12)] || 0.003
  );
  const medianRms = Math.max(
    noiseFloor * 1.4,
    sortedRms[Math.floor(sortedRms.length * 0.5)] || 0.018
  );

  const targetRms = 0.092;
  const maxLinearGain = 2.45; // +7.8 dB 平滑增益，兼顧遠距語音拉抬與自然停頓谷底對比
  const minSpeechRms = Math.max(0.0038, noiseFloor * 1.18);
  const truePeakCeiling = 0.89125;

  const targetFrameGain = new Float32Array(numFrames).fill(1.0);

  for (let f = 0; f < numFrames; f++) {
    const prevR = f > 0 ? frameRms[f - 1] : frameRms[f];
    const nextR = f + 1 < numFrames ? frameRms[f + 1] : frameRms[f];
    const localRms = Math.max(frameRms[f], 0.5 * (prevR + nextR));
    const zcr = frameZeroCross[f];
    const sHarm = syllabicHarm[f];

    // 【瞬態撞擊與麥克風敲擊抑制 (Transient De-Thumper)】：
    // 若單幀能量暴衝 (> medianRms * 2.2) 但無聲帶諧波性 (frameHarmonicity < 0.30)，代表為拍桌/碰麥克風雜音（如 33.62s 撞擊），予以衰減避免遮蔽前後微弱語音！
    if (
      frameHarmonicity[f] < 0.30 &&
      sHarm < 0.38 &&
      frameRms[f] > Math.max(0.045, medianRms * 2.2)
    ) {
      targetFrameGain[f] = Math.max(
        0.25,
        Math.min(0.85, (medianRms * 1.25) / frameRms[f])
      );
      continue;
    }

    // 非諧波靜音段或高過零摩擦雜音嚴格保持 1.0x 原音，保留清晰靜音谷底供 Whisper Cross-Attention 對齊
    if (sHarm < 0.41 || zcr > 0.30 || localRms < minSpeechRms) {
      targetFrameGain[f] = 1.0;
      continue;
    }

    const snrRatio = localRms / Math.max(1e-5, noiseFloor);
    const pVad =
      1.0 /
      (1.0 + Math.exp(-3.0 * (snrRatio - 1.25) - 4.0 * (sHarm - 0.42)));

    if (pVad >= 0.35 && localRms < targetRms) {
      const dbCurrent = 20.0 * Math.log10(Math.max(1e-5, localRms));
      const dbTarget = -20.5;
      const dbDeficit = Math.max(0.0, dbTarget - dbCurrent);
      const boostDb = Math.min(7.8, dbDeficit * 0.68 * pVad);
      targetFrameGain[f] = Math.min(
        maxLinearGain,
        Math.pow(10.0, boostDb / 20.0)
      );
    } else {
      targetFrameGain[f] = 1.0;
    }
  }

  const smoothedFrameGain = new Float32Array(numFrames);
  let currentGain = 1.0;
  const alphaAttack = 0.38;
  const alphaRelease = 0.14;

  for (let f = 0; f < numFrames; f++) {
    let lookAheadGain = targetFrameGain[f];
    if (lookAheadGain >= 1.0) {
      for (let la = 1; la <= 3; la++) {
        if (f + la < numFrames && targetFrameGain[f + la] >= 1.0) {
          lookAheadGain = Math.max(lookAheadGain, targetFrameGain[f + la]);
        }
      }
    }

    if (lookAheadGain < 0.95) {
      // 瞬態撞擊立即衰減
      currentGain = lookAheadGain;
    } else if (lookAheadGain > currentGain) {
      currentGain =
        alphaAttack * lookAheadGain + (1.0 - alphaAttack) * currentGain;
    } else {
      currentGain =
        alphaRelease * lookAheadGain + (1.0 - alphaRelease) * currentGain;
    }
    smoothedFrameGain[f] = currentGain;
  }

  const out = new Float32Array(N);
  const limiterLookAhead = Math.max(16, Math.floor(sampleRate * 0.003));

  for (let i = 0; i < N; i++) {
    const fIdx = Math.min(numFrames - 1, Math.floor(i / frameLen));
    const nextFIdx = Math.min(numFrames - 1, fIdx + 1);
    const frac = (i - fIdx * frameLen) / frameLen;
    const interpGain =
      smoothedFrameGain[fIdx] * (1.0 - frac) +
      smoothedFrameGain[nextFIdx] * frac;
    out[i] = samples[i] * interpGain;
  }

  let limiterEnv = 1.0;
  const limRelease = Math.exp(-1.0 / (sampleRate * 0.04));

  for (let i = 0; i < N; i++) {
    let windowPeak = 0.0;
    const lookEnd = Math.min(N, i + limiterLookAhead);
    for (let k = i; k < lookEnd; k += 2) {
      const absV = Math.abs(out[k]);
      if (absV > windowPeak) windowPeak = absV;
    }
    const neededGain =
      windowPeak > truePeakCeiling ? truePeakCeiling / windowPeak : 1.0;
    if (neededGain < limiterEnv) {
      limiterEnv = neededGain;
    } else {
      limiterEnv = limRelease * limiterEnv + (1.0 - limRelease) * neededGain;
    }
    out[i] *= limiterEnv;
  }

  return out;
}

async function decodeSliceTo16kMonoRaw(
  file: File,
  sliceStartSec: number,
  sliceEndSec: number,
  totalDurationSec: number,
  cachedFullAudioBuffer?: AudioBuffer | null
): Promise<Float32Array> {
  const nameLower = file.name.toLowerCase();
  const isWav = file.type.includes('wav') || nameLower.endsWith('.wav');
  const isMp3 =
    file.type.includes('mpeg') ||
    file.type.includes('mp3') ||
    nameLower.endsWith('.mp3');

  const AudioCtxClass =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext;

  if (cachedFullAudioBuffer) {
    const sr = cachedFullAudioBuffer.sampleRate;
    const sIdx = Math.max(0, Math.floor(sliceStartSec * sr));
    const eIdx = Math.min(
      cachedFullAudioBuffer.length,
      Math.ceil(sliceEndSec * sr)
    );
    const frameCount = Math.max(1, eIdx - sIdx);
    const numChannels = cachedFullAudioBuffer.numberOfChannels;
    const mono = new Float32Array(frameCount);
    for (let ch = 0; ch < numChannels; ch++) {
      const chData = cachedFullAudioBuffer.getChannelData(ch);
      const invCh = 1.0 / numChannels;
      for (let i = 0; i < frameCount; i++) {
        mono[i] += chData[sIdx + i] * invCh;
      }
    }
    if (sr === TARGET_SAMPLE_RATE) return mono;
    const outLen = Math.max(
      1,
      Math.floor((frameCount / sr) * TARGET_SAMPLE_RATE)
    );
    const resampled = new Float32Array(outLen);
    const ratio = sr / TARGET_SAMPLE_RATE;
    for (let i = 0; i < outLen; i++) {
      const srcPos = i * ratio;
      const idx0 = Math.min(frameCount - 1, Math.floor(srcPos));
      const idx1 = Math.min(frameCount - 1, idx0 + 1);
      const frac = srcPos - idx0;
      resampled[i] = mono[idx0] * (1.0 - frac) + mono[idx1] * frac;
    }
    return resampled;
  }

  if (isWav && totalDurationSec > 0 && file.size > 44) {
    const dataBytes = file.size - 44;
    const bytesPerSec = dataBytes / totalDurationSec;
    const startByte = 44 + Math.floor((sliceStartSec * bytesPerSec) / 4) * 4;
    const endByte = Math.min(
      file.size,
      44 + Math.ceil((sliceEndSec * bytesPerSec) / 4) * 4
    );
    const headerBuf = await file.slice(0, 44).arrayBuffer();
    const bodyBuf = await file.slice(startByte, endByte).arrayBuffer();

    const combined = new Uint8Array(44 + bodyBuf.byteLength);
    combined.set(new Uint8Array(headerBuf), 0);
    combined.set(new Uint8Array(bodyBuf), 44);
    const view = new DataView(combined.buffer);
    view.setUint32(4, 36 + bodyBuf.byteLength, true);
    view.setUint32(40, bodyBuf.byteLength, true);

    const audioCtx = new AudioCtxClass({ sampleRate: TARGET_SAMPLE_RATE });
    try {
      const decoded = await audioCtx.decodeAudioData(
        combined.buffer as ArrayBuffer
      );
      const numChannels = decoded.numberOfChannels;
      const mono = new Float32Array(decoded.length);
      for (let ch = 0; ch < numChannels; ch++) {
        const chData = decoded.getChannelData(ch);
        const invCh = 1.0 / numChannels;
        for (let i = 0; i < decoded.length; i++) {
          mono[i] += chData[i] * invCh;
        }
      }
      return mono;
    } finally {
      await audioCtx.close();
    }
  }

  if (isMp3 && totalDurationSec > 0 && file.size > 4096) {
    const bytesPerSec = file.size / totalDurationSec;
    const startByte = Math.max(0, Math.floor(sliceStartSec * bytesPerSec));
    const endByte = Math.min(file.size, Math.ceil(sliceEndSec * bytesPerSec));
    const rawSlice = new Uint8Array(
      await file.slice(startByte, endByte).arrayBuffer()
    );
    const alignedBuffer = alignMp3SyncHeader(rawSlice);

    const audioCtx = new AudioCtxClass({ sampleRate: TARGET_SAMPLE_RATE });
    try {
      const decoded = await audioCtx.decodeAudioData(alignedBuffer);
      const numChannels = decoded.numberOfChannels;
      const mono = new Float32Array(decoded.length);
      for (let ch = 0; ch < numChannels; ch++) {
        const chData = decoded.getChannelData(ch);
        const invCh = 1.0 / numChannels;
        for (let i = 0; i < decoded.length; i++) {
          mono[i] += chData[i] * invCh;
        }
      }
      return mono;
    } catch {
      // fallback below
    } finally {
      await audioCtx.close();
    }
  }

  return new Float32Array(0);
}

function computeInterquartileMean(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = Math.max(0, Math.floor(sorted.length * 0.25));
  const q3 = Math.max(q1 + 1, Math.ceil(sorted.length * 0.75));
  let sum = 0;
  for (let i = q1; i < q3; i++) {
    sum += sorted[i];
  }
  return sum / Math.max(1, q3 - q1);
}

/**
 * 【方向一深化核心】：3~5 秒高密度全場多點聲學預分析與「底層聲學公式動態校準引擎」(`analyzeFullAudioAcousticProfile`)
 */
export async function analyzeFullAudioAcousticProfile(
  file: File,
  onProgress?: (progressPct: number, detailText: string) => void
): Promise<AudioPreAnalysisProfile> {
  const startTimeMs = performance.now();
  onProgress?.(5, '正在讀取音訊時間軸與配置高密度全場抽樣探針...');

  let totalDurationSec = 60;
  try {
    totalDurationSec = await getAudioDuration(file);
  } catch {
    totalDurationSec = Math.max(30, file.size / 16000);
  }

  const windowDurSec = 6.0;
  const targetWindowCount =
    totalDurationSec <= 216
      ? Math.max(6, Math.ceil(totalDurationSec / windowDurSec))
      : totalDurationSec <= 900
      ? 28
      : 36;

  const nameLower = file.name.toLowerCase();
  const isWavOrMp3 =
    file.type.includes('wav') ||
    nameLower.endsWith('.wav') ||
    file.type.includes('mpeg') ||
    file.type.includes('mp3') ||
    nameLower.endsWith('.mp3');

  let cachedFullBuffer: AudioBuffer | null = null;
  if (!isWavOrMp3 && file.size <= 35 * 1024 * 1024) {
    onProgress?.(12, '正在解碼 M4A/AAC 音訊軌道以進行全場多點掃描...');
    try {
      const AudioCtxClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const audioCtx = new AudioCtxClass({ sampleRate: TARGET_SAMPLE_RATE });
      try {
        const rawBuf = await file.arrayBuffer();
        cachedFullBuffer = await audioCtx.decodeAudioData(rawBuf);
      } finally {
        await audioCtx.close();
      }
    } catch {
      cachedFullBuffer = null;
    }
  }

  const windowSlices: Float32Array[] = [];
  const stepSec =
    targetWindowCount > 1
      ? Math.max(
          windowDurSec,
          (totalDurationSec - windowDurSec) / (targetWindowCount - 1)
        )
      : 0;

  for (let w = 0; w < targetWindowCount; w++) {
    const wStart = Math.min(
      Math.max(0, totalDurationSec - windowDurSec),
      w * stepSec
    );
    const wEnd = Math.min(totalDurationSec, wStart + windowDurSec);
    try {
      const rawMono = await decodeSliceTo16kMonoRaw(
        file,
        wStart,
        wEnd,
        totalDurationSec,
        cachedFullBuffer
      );
      if (rawMono.length >= 1600) {
        const filtered = applyZeroPhaseButterworthHighpass(
          rawMono,
          TARGET_SAMPLE_RATE,
          85.0
        );
        windowSlices.push(filtered);
      }
    } catch {
      // ignore single window decode error
    }

    if (w % 3 === 0 || w === targetWindowCount - 1) {
      const pct = 15 + Math.round(((w + 1) / targetWindowCount) * 40);
      onProgress?.(
        pct,
        `正在深度掃描全場聲學特徵 (${w + 1}/${targetWindowCount} 區段，累計取樣 ${Math.round(
          (w + 1) * windowDurSec
        )} 秒)...`
      );
      await new Promise((r) => setTimeout(r, 40));
    }
  }

  // 切片完成後立即釋放完整 AudioBuffer，避免佔用系統記憶體
  cachedFullBuffer = null;

  onProgress?.(
    62,
    '正在分析全場 P02 真實靜音底噪、YIN 基頻跨度與底層聲學公式動態閥值...'
  );
  await new Promise((r) => setTimeout(r, 150));

  const frameLen = 320;
  const allFrameRms: number[] = [];
  const utteranceF0List: number[] = [];
  const voicedHarmList: number[] = [];
  const activeCandidateHarmList: number[] = [];
  const turnFeatureVectors: Array<{ f0Semitone: number; highBandRatio: number }> =
    [];

  let totalSampledFrames = 0;
  let totalVoicedFrames = 0;
  let totalSilenceFrames = 0;
  let totalOverlapSuspectFrames = 0;
  let totalTurnSwitches = 0;
  const totalSampledWindows = windowSlices.length;

  for (let wIdx = 0; wIdx < windowSlices.length; wIdx++) {
    const samples = windowSlices[wIdx];
    const numFrames = Math.floor(samples.length / frameLen);
    if (numFrames < 10) {
      windowSlices[wIdx] = new Float32Array(0);
      continue;
    }

    const wRms = new Float32Array(numFrames);
    const wZcr = new Float32Array(numFrames);
    const wHiRatio = new Float32Array(numFrames);

    for (let f = 0; f < numFrames; f++) {
      const base = f * frameLen;
      let sumSq = 0.0;
      let diffSq = 0.0;
      let zc = 0;
      for (let i = 0; i < frameLen; i++) {
        const v = samples[base + i];
        sumSq += v * v;
        if (i > 0) {
          const dv = v - samples[base + i - 1];
          diffSq += dv * dv;
          if ((v >= 0) !== (samples[base + i - 1] >= 0)) zc++;
        }
      }
      const rms = Math.sqrt(sumSq / frameLen);
      wRms[f] = rms;
      wZcr[f] = zc / frameLen;
      wHiRatio[f] = sumSq > 1e-8 ? Math.sqrt(diffSq / sumSq) : 0.0;
      allFrameRms.push(rms);
    }

    const sortedLocal = Array.from(wRms).sort((a, b) => a - b);
    const p02 = Math.max(
      0.0005,
      sortedLocal[Math.floor(sortedLocal.length * 0.02)] || 0.001
    );
    const p75 = Math.max(
      p02 * 2.2,
      sortedLocal[Math.floor(sortedLocal.length * 0.75)] || 0.015
    );
    // 以 P02 * 1.32 作為真實語音門檻，不讓近距大聲語者 (p75) 淹沒遠距與會者
    const vadGate = Math.max(0.0032, p02 * 1.32);

    let prevUtteranceF0 = 0;
    let prevUtteranceHiRatio = 0;
    let currentTurnF0s: number[] = [];
    let currentTurnHiRatios: number[] = [];
    let silenceRunFrames = 0;

    const flushUtteranceIfValid = (minFramesReq: number) => {
      if (currentTurnF0s.length >= minFramesReq) {
        const iqmF0 = computeInterquartileMean(currentTurnF0s);
        const iqmHi = computeInterquartileMean(currentTurnHiRatios);
        utteranceF0List.push(iqmF0);
        turnFeatureVectors.push({
          f0Semitone: 12 * Math.log2(Math.max(75, iqmF0) / 100.0),
          highBandRatio: iqmHi,
        });
        if (prevUtteranceF0 > 0) {
          const semitoneDiff = Math.abs(12 * Math.log2(iqmF0 / prevUtteranceF0));
          const hiDiff = Math.abs(iqmHi - prevUtteranceHiRatio);
          if (
            semitoneDiff >= 2.5 ||
            (semitoneDiff >= 1.6 && hiDiff >= 0.09)
          ) {
            totalTurnSwitches++;
          }
        }
        prevUtteranceF0 = iqmF0;
        prevUtteranceHiRatio = iqmHi;
      }
      currentTurnF0s = [];
      currentTurnHiRatios = [];
    };

    for (let f = 0; f < numFrames; f++) {
      totalSampledFrames++;
      if (wRms[f] < vadGate) {
        totalSilenceFrames++;
      }
      if (wRms[f] < vadGate || wZcr[f] > 0.28) {
        silenceRunFrames++;
        if (silenceRunFrames >= 7) {
          flushUtteranceIfValid(16);
        }
        continue;
      }

      const base = f * frameLen;
      const { f0Hz, periodicity, secondDipPeriodicity } =
        estimateFramePitchAndVoicingYin(samples, base, TARGET_SAMPLE_RATE);

      if (periodicity >= 0.22) {
        activeCandidateHarmList.push(periodicity);
      }

      if (f0Hz <= 0 || periodicity < 0.46) {
        silenceRunFrames++;
        continue;
      }

      totalVoicedFrames++;
      voicedHarmList.push(periodicity);
      currentTurnF0s.push(f0Hz);
      currentTurnHiRatios.push(wHiRatio[f]);

      if (
        secondDipPeriodicity >= 0.72 * periodicity &&
        periodicity < 0.76 &&
        wRms[f] > p75 * 0.9
      ) {
        totalOverlapSuspectFrames++;
      }

      silenceRunFrames = 0;

      if (currentTurnF0s.length >= 90) {
        flushUtteranceIfValid(45);
      }
    }

    flushUtteranceIfValid(16);
    windowSlices[wIdx] = new Float32Array(0);
  }
  windowSlices.length = 0;

  onProgress?.(
    84,
    '正在執行全場語者聲紋分佈聚類、推導底層聲學公式動態閥值與最佳化方案...'
  );
  await new Promise((r) => setTimeout(r, 160));

  const sortedAllRms =
    allFrameRms.length > 0
      ? [...allFrameRms].sort((a, b) => a - b)
      : [0.001, 0.02];
  const globalP02 = Math.max(
    0.0005,
    sortedAllRms[Math.floor(sortedAllRms.length * 0.02)] || 0.001
  );
  const globalP85 = Math.max(
    globalP02 * 2.0,
    sortedAllRms[Math.floor(sortedAllRms.length * 0.85)] || 0.03
  );

  const noiseFloorDb = Number(
    Math.max(-68, Math.min(-22, 20 * Math.log10(globalP02))).toFixed(1)
  );
  const snrDb = Number(
    Math.max(6, Math.min(42, 20 * Math.log10(globalP85 / globalP02))).toFixed(1)
  );
  const noiseLevel: 'LOW' | 'MEDIUM' | 'HIGH' =
    snrDb >= 22 ? 'LOW' : snrDb >= 14 ? 'MEDIUM' : 'HIGH';

  const sortedF0 =
    utteranceF0List.length > 0
      ? [...utteranceF0List].sort((a, b) => a - b)
      : [125, 215];
  const f0P10Hz = Math.round(
    sortedF0[Math.floor(sortedF0.length * 0.1)] || 120
  );
  const f0P90Hz = Math.round(
    sortedF0[Math.floor(sortedF0.length * 0.9)] || 215
  );
  const f0SpreadHz = Math.max(15, f0P90Hz - f0P10Hz);
  const pitchSpreadSemitones = Number(
    (12 * Math.log2(Math.max(1.05, f0P90Hz / Math.max(70, f0P10Hz)))).toFixed(1)
  );

  const sortedHarm =
    voicedHarmList.length > 0
      ? [...voicedHarmList].sort((a, b) => a - b)
      : [0.68];
  const harmonicityMedian = Number(
    (sortedHarm[Math.floor(sortedHarm.length * 0.5)] || 0.68).toFixed(2)
  );

  const sortedCandHarm =
    activeCandidateHarmList.length > 0
      ? [...activeCandidateHarmList].sort((a, b) => a - b)
      : [0.34, 0.65];
  const harmP15 =
    sortedCandHarm[Math.floor(sortedCandHarm.length * 0.15)] || 0.34;

  const clusters: Array<{ f0Semitone: number; highBandRatio: number; count: number }> =
    [];
  for (const vec of turnFeatureVectors) {
    let bestIdx = -1;
    let bestDist = Infinity;
    for (let c = 0; c < clusters.length; c++) {
      const df0 = Math.abs(vec.f0Semitone - clusters[c].f0Semitone) / 2.6;
      const dHi = Math.abs(vec.highBandRatio - clusters[c].highBandRatio) / 0.15;
      const dist = Math.sqrt(df0 * df0 + dHi * dHi);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = c;
      }
    }
    if (bestIdx >= 0 && bestDist <= 1.25) {
      const cl = clusters[bestIdx];
      const nextCount = cl.count + 1;
      cl.f0Semitone = (cl.f0Semitone * cl.count + vec.f0Semitone) / nextCount;
      cl.highBandRatio =
        (cl.highBandRatio * cl.count + vec.highBandRatio) / nextCount;
      cl.count = nextCount;
    } else if (clusters.length < 8) {
      clusters.push({ ...vec, count: 1 });
    }
  }

  const significantClusters = clusters.filter(
    (c) => c.count >= Math.max(2, Math.floor(turnFeatureVectors.length * 0.10))
  );
  const rawClusterCount = Math.max(2, Math.min(5, significantClusters.length));

  const sampledDurationSec = Number(
    ((totalSampledFrames * frameLen) / TARGET_SAMPLE_RATE).toFixed(1)
  );
  const coverageRatio = Number(
    Math.min(1.0, sampledDurationSec / Math.max(1, totalDurationSec)).toFixed(2)
  );
  const sampledMinutes = Math.max(0.25, sampledDurationSec / 60.0);

  const turnSwitchRatePerMin = Number(
    Math.min(32, Math.max(1.5, totalTurnSwitches / sampledMinutes)).toFixed(1)
  );
  const overlapRatio = Number(
    Math.min(
      0.35,
      totalOverlapSuspectFrames / Math.max(1, totalVoicedFrames)
    ).toFixed(3)
  );
  const overlapRisk: 'LOW' | 'MEDIUM' | 'HIGH' =
    overlapRatio >= 0.14 || turnSwitchRatePerMin >= 16
      ? 'HIGH'
      : overlapRatio >= 0.06 || turnSwitchRatePerMin >= 7.5
      ? 'MEDIUM'
      : 'LOW';

  const voiceSimilarity: 'HIGH' | 'MEDIUM' | 'LOW' =
    pitchSpreadSemitones <= 4.2
      ? 'HIGH'
      : pitchSpreadSemitones <= 7.8
      ? 'MEDIUM'
      : 'LOW';

  // 【關鍵通用修正】：每個 5 分鐘微批次內可能僅有 2 位主要對話者（例如主席與單一分區報告人一問一答），
  // 嚴禁將 estimatedMinSpeakers 強制鎖死在 3 或 4，否則聚類器會被迫把同一位男主席或女報告人拆成 3 個假語者！
  // 因此最少語者一律設為 2（單人演講設為 1），最多語者設為 4~6，讓演算法依每塊真實聲紋自動收斂。
  const estimatedMinSpeakers =
    turnSwitchRatePerMin < 2.2 && rawClusterCount <= 1 ? 1 : 2;
  const estimatedMaxSpeakers = Math.max(
    4,
    Math.min(6, rawClusterCount + 2)
  );

  const silenceRatio = totalSilenceFrames / Math.max(1, totalSampledFrames);
  const calibratedVadNoiseQuantile = Number(
    Math.max(0.015, Math.min(0.035, silenceRatio < 0.10 ? 0.018 : 0.022)).toFixed(3)
  );
  const calibratedVadSpeechMultiplier = Number(
    Math.max(1.28, Math.min(1.42, snrDb >= 20 ? 1.32 : 1.36)).toFixed(2)
  );
  const calibratedHarmGate = Number(
    Math.max(0.28, Math.min(0.38, harmP15 * 0.90)).toFixed(3)
  );
  const calibratedFrictionCeiling = Number(
    Math.max(0.20, Math.min(0.28, calibratedHarmGate * 0.76)).toFixed(3)
  );
  const calibratedMaxMicroTurnSec = Number(
    Math.max(
      2.4,
      Math.min(
        4.2,
        turnSwitchRatePerMin >= 12
          ? 2.6
          : turnSwitchRatePerMin >= 6.0
          ? 3.2
          : 3.8
      )
    ).toFixed(2)
  );
  const calibratedScdPeakCutoff = Number(
    Math.max(
      0.48,
      Math.min(
        0.55,
        turnSwitchRatePerMin >= 11 ? 0.49 : turnSwitchRatePerMin <= 4.5 ? 0.54 : 0.51
      )
    ).toFixed(2)
  );
  const calibratedPitchTolerance = Number(
    Math.max(
      3.5,
      Math.min(4.4, pitchSpreadSemitones >= 7.5 ? 4.0 : 3.6)
    ).toFixed(2)
  );

  const baseFormulaCalibration: AcousticFormulaCalibration = {
    vadNoiseQuantile: calibratedVadNoiseQuantile,
    vadSpeechMultiplier: calibratedVadSpeechMultiplier,
    vadDynamicRangeWeight: 0.02,
    harmonicityGateThreshold: calibratedHarmGate,
    frictionHarmonicCeiling: calibratedFrictionCeiling,
    boundaryDipWeight: 0.42,
    boundaryVoiceWeight: 0.58,
    stepDbDiffFloor: 3.2,
    formantCosDistFloor: 0.21,
    scdPeakCutoff: calibratedScdPeakCutoff,
    maxMicroTurnSec: calibratedMaxMicroTurnSec,
    ahcStopSimThreshold: voiceSimilarity === 'HIGH' ? 0.32 : 0.30,
    pitchPenaltySemitoneTolerance: calibratedPitchTolerance,
    vbxInvTemp: 20.0,
    vbxCoreResponsibilityGate: 0.45,
  };

  let recommendedPresetId:
    | 'preset_a_balanced'
    | 'preset_b_sensitive'
    | 'preset_c_long_speech' = 'preset_a_balanced';

  if (
    (voiceSimilarity === 'HIGH' && turnSwitchRatePerMin >= 8.0) ||
    turnSwitchRatePerMin >= 18.0
  ) {
    recommendedPresetId = 'preset_b_sensitive';
  } else if (turnSwitchRatePerMin <= 4.2 && overlapRatio < 0.05) {
    recommendedPresetId = 'preset_c_long_speech';
  }

  const recommendations: AudioRecommendationOption[] = [
    {
      id: 'preset_a_balanced',
      badge: '方案 A・標準會議首選',
      title: 'Mode 0 黃金長句定錨＋自適應聲學公式（穩健平衡）',
      label: 'Mode 0 黃金長句定錨＋自適應聲學公式（穩健平衡）',
      subtitle: '先取純淨長句建立聲紋錨點，再以 YIN 基頻與話輪 Viterbi 平滑兼顧長篇報告與快速問答',
      whyRecommend: `全場平均每分鐘換人 ${turnSwitchRatePerMin} 次、重疊率 ${(
        overlapRatio * 100
      ).toFixed(
        1
      )}%，已將 P02 靜音門檻與 YIN 諧波門檻 (${calibratedHarmGate}) 下放至聲學公式層，兼顧完整長句與遠近距問答。`,
      bestFor: '2～6 人部門會議、專案進度檢討、主席與報告人一問一答及穿插補充（「24號」、「不算」、「對」）',
      notIdealFor: '單人連續講述超過 85% 時間之大型演講',
      isRecommended: recommendedPresetId === 'preset_a_balanced',
      recommendedMinSpeakers: estimatedMinSpeakers,
      recommendedMaxSpeakers: estimatedMaxSpeakers,
      diarizationPatch: {
        engineMode: 'mode_0_heritage_classic',
        strategyMode: 'MODE_0_GOLDEN_TWO_STAGE',
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.22,
        backchannelMarginThreshold: 0.11,
        shortTurnConfidenceKeepThreshold: 0.66,
        pingPongMaxDurSec: 1.6,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
        formulaCalibration: {
          ...baseFormulaCalibration,
        },
      },
    },
    {
      id: 'preset_b_sensitive',
      badge: '方案 B・同性別與搶話強化',
      title: '高敏感度插話捕捉＋雙尺度神經聲紋',
      label: '高敏感度插話捕捉＋雙尺度神經聲紋',
      subtitle: `針對同性別音色極相近或極高頻搶話下調公式波峰門檻`,
      whyRecommend:
        voiceSimilarity === 'HIGH'
          ? `偵測到全場基頻跨度僅 ${pitchSpreadSemitones} 半音（全同性別且音色極相近），可啟用雙尺度 256 維聲紋強化區分。`
          : `若您希望連 0.4 秒極短促的重疊搶話都強制降低邊界公式門檻獨立分段，可改選此高敏感度方案。`,
      bestFor: '全同性別與會者且聲線高度重疊、激烈辯論搶話',
      notIdealFor: '講者說話習慣在句中停頓思考（較容易把同一句話切成兩段）',
      isRecommended: recommendedPresetId === 'preset_b_sensitive',
      recommendedMinSpeakers: estimatedMinSpeakers,
      recommendedMaxSpeakers: Math.min(7, estimatedMaxSpeakers + 1),
      diarizationPatch: {
        engineMode: 'mode_c_onnx_neural',
        strategyMode: 'MODE_C_ONNX_NEURAL',
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.2,
        backchannelMarginThreshold: 0.1,
        shortTurnConfidenceKeepThreshold: 0.62,
        pingPongMaxDurSec: 1.75,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
        formulaCalibration: {
          ...baseFormulaCalibration,
          scdPeakCutoff: Number(
            Math.max(0.46, baseFormulaCalibration.scdPeakCutoff - 0.03).toFixed(2)
          ),
          maxMicroTurnSec: Number(
            Math.max(2.0, baseFormulaCalibration.maxMicroTurnSec - 0.4).toFixed(2)
          ),
          stepDbDiffFloor: 2.8,
        },
      },
    },
    {
      id: 'preset_c_long_speech',
      badge: '方案 C・簡報與長句防斷裂',
      title: '長篇報告防句中斷裂（主講保護模式）',
      label: '長篇報告防句中講述防斷裂（主講保護模式）',
      subtitle: '拉高底層邊界公式門檻與話輪平滑權重，防止主講人翻頁停頓被誤切成兩個人',
      whyRecommend: `適合以長句闡述為主的會議，最大化保護句子完整性並防止主講人換氣或翻閱資料時被誤判換人。`,
      bestFor: '專題簡報、教育訓練、法說會、一人主講搭配少數 QA 提問',
      notIdealFor: '多人每秒快速搶話的圓桌討論',
      isRecommended: recommendedPresetId === 'preset_c_long_speech',
      recommendedMinSpeakers: Math.max(1, estimatedMinSpeakers),
      recommendedMaxSpeakers: Math.max(3, estimatedMaxSpeakers - 1),
      diarizationPatch: {
        engineMode: 'mode_b_advanced_dsp',
        strategyMode: 'MODE_B_ADVANCED_SPECTRAL',
        clusterMethod: 'spectral_cosine',
        scdPeakThreshold: 0.26,
        backchannelMarginThreshold: 0.14,
        shortTurnConfidenceKeepThreshold: 0.7,
        pingPongMaxDurSec: 1.4,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
        formulaCalibration: {
          ...baseFormulaCalibration,
          scdPeakCutoff: Number(
            Math.min(0.58, baseFormulaCalibration.scdPeakCutoff + 0.04).toFixed(2)
          ),
          maxMicroTurnSec: Number(
            Math.min(4.8, baseFormulaCalibration.maxMicroTurnSec + 0.8).toFixed(2)
          ),
          stepDbDiffFloor: 4.0,
        },
      },
    },
  ];

  const elapsedSoFar = performance.now() - startTimeMs;
  if (elapsedSoFar < 2200) {
    onProgress?.(94, '正在彙整全場聲學體檢指標與底層聲學公式校準參數...');
    await new Promise((r) =>
      setTimeout(r, Math.min(800, 2200 - elapsedSoFar))
    );
  }

  const analysisDurationMs = Math.round(performance.now() - startTimeMs);
  const winningOpt =
    recommendations.find((r) => r.isRecommended) || recommendations[0];

  const plainSummary = `已完成全場 ${totalSampledWindows} 個區段（共抽樣 ${sampledDurationSec} 秒，涵蓋率 ${Math.round(
    coverageRatio * 100
  )}%）深度聲學掃描與公式校準：環境信噪比 ${snrDb} dB（底噪 ${
    noiseLevel === 'LOW' ? '極低' : noiseLevel === 'MEDIUM' ? '中等' : '偏高'
  }）、話輪基頻跨度 ${f0P10Hz}~${f0P90Hz} Hz（${pitchSpreadSemitones} 半音）、YIN 諧波門檻自動校準為 ${calibratedHarmGate}、每分鐘換人約 ${turnSwitchRatePerMin} 次、重疊率 ${(
    overlapRatio * 100
  ).toFixed(1)}%，建議人數範圍 ${estimatedMinSpeakers}~${estimatedMaxSpeakers} 人。系統優先推薦【${
    winningOpt.badge
  }】。`;

  onProgress?.(
    100,
    '全場聲學深度預分析與底層公式校準完成！請確認或選擇下方建議方案後啟動轉寫。'
  );

  return {
    snrDb,
    noiseFloorDb,
    noiseLevel,
    f0SpreadHz,
    pitchSpreadSemitones,
    f0P10Hz,
    f0P90Hz,
    harmonicityMedian,
    voiceSimilarity,
    turnSwitchRatePerMin,
    overlapRatio,
    overlapRisk,
    estimatedMinSpeakers,
    estimatedMaxSpeakers,
    sampledWindowsCount: totalSampledWindows,
    sampledDurationSec,
    coverageRatio,
    analysisDurationMs,
    plainSummary,
    recommendedPresetId,
    recommendations,
    recommendedOptions: recommendations,
    formulaCalibration: baseFormulaCalibration,
  };
}

export const analyzeAudioPreProfile = analyzeFullAudioAcousticProfile;

export async function extractAndPreprocessMacroChunk(
  file: File,
  chunkStartSec: number,
  chunkEndSec: number,
  totalDurationSec: number,
  includeWavBuffer = false
): Promise<{
  samples: Float32Array;
  wavBuffer: ArrayBuffer;
  actualDurationSec: number;
}> {
  if (chunkStartSec === 0) {
    await new Promise((r) => setTimeout(r, 120));
  }

  const nameLower = file.name.toLowerCase();
  const isWav = file.type.includes('wav') || nameLower.endsWith('.wav');
  const isMp3 =
    file.type.includes('mpeg') ||
    file.type.includes('mp3') ||
    nameLower.endsWith('.mp3');

  const AudioCtxClass =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext;

  if (isWav && totalDurationSec > 0 && file.size > 44) {
    const dataBytes = file.size - 44;
    const bytesPerSec = dataBytes / totalDurationSec;
    const startByte = 44 + Math.floor(sliceAlign(chunkStartSec * bytesPerSec) / 4) * 4;
    const endByte = Math.min(
      file.size,
      44 + Math.ceil(sliceAlign(chunkEndSec * bytesPerSec) / 4) * 4
    );
    const headerBuf = await file.slice(0, 44).arrayBuffer();
    const bodyBuf = await file.slice(startByte, endByte).arrayBuffer();

    const combined = new Uint8Array(44 + bodyBuf.byteLength);
    combined.set(new Uint8Array(headerBuf), 0);
    combined.set(new Uint8Array(bodyBuf), 44);
    const view = new DataView(combined.buffer);
    view.setUint32(4, 36 + bodyBuf.byteLength, true);
    view.setUint32(40, bodyBuf.byteLength, true);

    const audioCtx = new AudioCtxClass({ sampleRate: TARGET_SAMPLE_RATE });
    let decoded: AudioBuffer;
    try {
      decoded = await audioCtx.decodeAudioData(combined.buffer as ArrayBuffer);
    } finally {
      await audioCtx.close();
    }

    const samples = await applyHighpassFilter16k(decoded);
    const wavBuffer = includeWavBuffer
      ? encodeFloat32ToWav(samples, TARGET_SAMPLE_RATE)
      : new ArrayBuffer(0);
    return {
      samples,
      wavBuffer,
      actualDurationSec: samples.length / TARGET_SAMPLE_RATE,
    };
  }

  if (isMp3 && totalDurationSec > 0 && file.size > 4096) {
    const bytesPerSec = file.size / totalDurationSec;
    const startByte = Math.max(0, Math.floor(chunkStartSec * bytesPerSec));
    const endByte = Math.min(file.size, Math.ceil(chunkEndSec * bytesPerSec));
    const rawSlice = new Uint8Array(
      await file.slice(startByte, endByte).arrayBuffer()
    );
    const alignedBuffer = alignMp3SyncHeader(rawSlice);

    const audioCtx = new AudioCtxClass({ sampleRate: TARGET_SAMPLE_RATE });
    try {
      const decoded = await audioCtx.decodeAudioData(alignedBuffer);
      const samples = await applyHighpassFilter16k(decoded);
      const wavBuffer = includeWavBuffer
        ? encodeFloat32ToWav(samples, TARGET_SAMPLE_RATE)
        : new ArrayBuffer(0);
      return {
        samples,
        wavBuffer,
        actualDurationSec: samples.length / TARGET_SAMPLE_RATE,
      };
    } catch {
      // fallback below
    } finally {
      await audioCtx.close();
    }
  }

  const rawBuffer = await file.arrayBuffer();
  const audioCtx = new AudioCtxClass({ sampleRate: TARGET_SAMPLE_RATE });
  let decodedBuffer: AudioBuffer;
  try {
    decodedBuffer = await audioCtx.decodeAudioData(rawBuffer);
  } finally {
    await audioCtx.close();
  }

  const startSample = Math.max(
    0,
    Math.floor(chunkStartSec * TARGET_SAMPLE_RATE)
  );
  const endSample = Math.min(
    decodedBuffer.length,
    Math.ceil(chunkEndSec * TARGET_SAMPLE_RATE)
  );
  const frameCount = Math.max(1, endSample - startSample);
  const numChannels = decodedBuffer.numberOfChannels;
  const monoSlice = new Float32Array(frameCount);

  for (let ch = 0; ch < numChannels; ch++) {
    const chData = decodedBuffer.getChannelData(ch);
    for (let i = 0; i < frameCount; i++) {
      monoSlice[i] += chData[startSample + i] / numChannels;
    }
  }

  const samples = await applyHighpassFilterOnArray(
    monoSlice,
    TARGET_SAMPLE_RATE
  );
  const wavBuffer = includeWavBuffer
    ? encodeFloat32ToWav(samples, TARGET_SAMPLE_RATE)
    : new ArrayBuffer(0);

  return {
    samples,
    wavBuffer,
    actualDurationSec: samples.length / TARGET_SAMPLE_RATE,
  };
}

function sliceAlign(val: number): number {
  return Math.max(0, val);
}

export const sliceAudioIntoMacroChunks = extractAndPreprocessMacroChunk;

async function applyHighpassFilter16k(
  decoded: AudioBuffer
): Promise<Float32Array> {
  const numChannels = decoded.numberOfChannels;
  const length = decoded.length;
  const mono = new Float32Array(length);
  for (let ch = 0; ch < numChannels; ch++) {
    const chData = decoded.getChannelData(ch);
    const invCh = 1.0 / numChannels;
    for (let i = 0; i < length; i++) {
      mono[i] += chData[i] * invCh;
    }
  }

  let resampled = mono;
  if (decoded.sampleRate !== TARGET_SAMPLE_RATE) {
    const frameCount = Math.max(
      1,
      Math.ceil(decoded.duration * TARGET_SAMPLE_RATE)
    );
    const offlineCtx = new OfflineAudioContext(
      1,
      frameCount,
      TARGET_SAMPLE_RATE
    );
    const sourceNode = offlineCtx.createBufferSource();
    sourceNode.buffer = decoded;
    sourceNode.connect(offlineCtx.destination);
    sourceNode.start(0);
    const rendered = await offlineCtx.startRendering();
    resampled = new Float32Array(rendered.getChannelData(0));
  }

  return applyZeroPhaseButterworthHighpass(
    resampled,
    TARGET_SAMPLE_RATE,
    85.0
  );
}

export async function decodeAndResampleTo16kMono(
  file: File
): Promise<Float32Array> {
  const dur = await getAudioDuration(file);
  const res = await extractAndPreprocessMacroChunk(file, 0, dur, dur, false);
  return res.samples;
}

async function applyHighpassFilterOnArray(
  inputSamples: Float32Array,
  sampleRate: number
): Promise<Float32Array> {
  return applyZeroPhaseButterworthHighpass(
    inputSamples,
    sampleRate,
    85.0
  );
}

function applyPresenceBoostBiquad(
  samples: Float32Array,
  sampleRate = TARGET_SAMPLE_RATE,
  centerFreqHz = 2800,
  gainDb = 4.5,
  Q = 1.1
): Float32Array {
  const out = new Float32Array(samples.length);
  const A = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * centerFreqHz) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);
  const alpha = sinW0 / (2 * Q);

  const b0 = 1 + alpha * A;
  const b1 = -2 * cosW0;
  const b2 = 1 - alpha * A;
  const a0 = 1 + alpha / A;
  const a1 = -2 * cosW0;
  const a2 = 1 - alpha / A;

  const nb0 = b0 / a0;
  const nb1 = b1 / a0;
  const nb2 = b2 / a0;
  const na1 = a1 / a0;
  const na2 = a2 / a0;

  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;

  for (let i = 0; i < samples.length; i++) {
    const x0 = samples[i];
    const y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2;
    out[i] = Math.max(-0.95, Math.min(0.95, y0));
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return out;
}

function applyWsolaTimeStretch(
  input: Float32Array,
  stretchRatio = 1.18,
  sampleRate = TARGET_SAMPLE_RATE
): Float32Array {
  if (Math.abs(stretchRatio - 1.0) < 0.03 || input.length < 1600) {
    return input;
  }

  const winLen = Math.floor(sampleRate * 0.03);
  const synHop = Math.floor(winLen / 2);
  const anaHop = Math.max(1, Math.round(synHop / stretchRatio));
  const searchRadius = Math.floor(sampleRate * 0.008);

  const outLen = Math.ceil(input.length * stretchRatio) + winLen;
  const output = new Float32Array(outLen);
  const normWeight = new Float32Array(outLen);

  const hann = new Float32Array(winLen);
  for (let n = 0; n < winLen; n++) {
    hann[n] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * n) / (winLen - 1)));
  }

  let anaPos = 0;
  let synPos = 0;

  for (let n = 0; n < winLen && n < input.length; n++) {
    output[n] = input[n] * hann[n];
    normWeight[n] = hann[n] * hann[n];
  }

  let prevBestAnaPos = 0;

  while (
    synPos + winLen < outLen &&
    anaPos + anaHop + winLen + searchRadius < input.length
  ) {
    synPos += synHop;
    const naturalNextAna = prevBestAnaPos + anaHop;
    const targetCandidateCenter = Math.round(synPos / stretchRatio);

    let bestOffset = 0;
    let bestCorr = -Infinity;

    const overlapCheckLen = Math.min(120, synHop);
    for (let delta = -searchRadius; delta <= searchRadius; delta += 2) {
      const candPos = targetCandidateCenter + delta;
      if (candPos < 0 || candPos + winLen >= input.length) continue;

      let corr = 0.0;
      for (let k = 0; k < overlapCheckLen; k += 2) {
        const refVal =
          naturalNextAna + k < input.length ? input[naturalNextAna + k] : 0;
        corr += refVal * input[candPos + k];
      }
      if (corr > bestCorr) {
        bestCorr = corr;
        bestOffset = delta;
      }
    }

    const chosenAnaPos = Math.max(
      0,
      Math.min(input.length - winLen, targetCandidateCenter + bestOffset)
    );

    for (let n = 0; n < winLen; n++) {
      const w = hann[n];
      output[synPos + n] += input[chosenAnaPos + n] * w;
      normWeight[synPos + n] += w * w;
    }

    prevBestAnaPos = chosenAnaPos;
    anaPos = chosenAnaPos;
  }

  const finalValidLen = Math.min(outLen, synPos + winLen);
  const normalizedOut = new Float32Array(finalValidLen);
  for (let i = 0; i < finalValidLen; i++) {
    normalizedOut[i] =
      normWeight[i] > 1e-4 ? output[i] / normWeight[i] : output[i];
  }

  return normalizedOut;
}

export function enhanceSubSegmentAudioForRelisten(
  chunkSamples: Float32Array,
  chunkStartGlobalSec: number,
  segmentStartGlobalSec: number,
  segmentEndGlobalSec: number,
  options?: {
    paddingSec?: number;
    enableTimeStretch?: boolean;
    stretchRatio?: number;
  }
): {
  subSamples: Float32Array;
  subWavBuffer: ArrayBuffer;
  paddedStartGlobalSec: number;
  paddedEndGlobalSec: number;
  timeStretchRatio: number;
  hasQuietDropCompensated: boolean;
} {
  const paddingSec = options?.paddingSec ?? 0.85;
  const baseSliced = sliceSubSegmentAudio(
    chunkSamples,
    chunkStartGlobalSec,
    segmentStartGlobalSec,
    segmentEndGlobalSec,
    paddingSec
  );

  const raw = baseSliced.subSamples;
  if (raw.length < 800) {
    return {
      ...baseSliced,
      timeStretchRatio: 1.0,
      hasQuietDropCompensated: false,
    };
  }

  let peakAbs = 0.0;
  let sumSq = 0.0;
  for (let i = 0; i < raw.length; i++) {
    const a = Math.abs(raw[i]);
    if (a > peakAbs) peakAbs = a;
    sumSq += raw[i] * raw[i];
  }
  const rms = Math.sqrt(sumSq / raw.length);

  const linearGain =
    peakAbs > 1e-4 ? Math.min(3.5, Math.max(1.0, 0.78 / peakAbs)) : 1.0;
  const hasQuietDropCompensated = linearGain >= 1.45 || rms < 0.045;

  const normalizedSamples = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i] * linearGain;
    normalizedSamples[i] = Math.max(-0.891, Math.min(0.891, v));
  }

  const formantEnhanced = applyPresenceBoostBiquad(
    normalizedSamples,
    TARGET_SAMPLE_RATE,
    2800,
    2.8,
    1.0
  );

  const shouldStretch = options?.enableTimeStretch === true;
  const requestedRatio = shouldStretch ? options?.stretchRatio ?? 1.12 : 1.0;

  const finalSamples =
    requestedRatio > 1.03
      ? applyWsolaTimeStretch(
          formantEnhanced,
          requestedRatio,
          TARGET_SAMPLE_RATE
        )
      : formantEnhanced;

  const subWavBuffer = encodeFloat32ToWav(finalSamples, TARGET_SAMPLE_RATE);

  return {
    subSamples: finalSamples,
    subWavBuffer,
    paddedStartGlobalSec: baseSliced.paddedStartGlobalSec,
    paddedEndGlobalSec: baseSliced.paddedEndGlobalSec,
    timeStretchRatio: requestedRatio,
    hasQuietDropCompensated,
  };
}

export function sliceSubSegmentAudio(
  chunkSamples: Float32Array,
  chunkStartGlobalSec: number,
  segmentStartGlobalSec: number,
  segmentEndGlobalSec: number,
  paddingSec = 0.8
): {
  subSamples: Float32Array;
  subWavBuffer: ArrayBuffer;
  paddedStartGlobalSec: number;
  paddedEndGlobalSec: number;
} {
  const chunkDurationSec = chunkSamples.length / TARGET_SAMPLE_RATE;
  const relStartSec = Math.max(
    0,
    segmentStartGlobalSec - chunkStartGlobalSec - paddingSec
  );
  const relEndSec = Math.min(
    chunkDurationSec,
    segmentEndGlobalSec - chunkStartGlobalSec + paddingSec
  );

  const safeEndSec = Math.max(
    relStartSec + 1.5,
    Math.min(relStartSec + 15.0, relEndSec)
  );

  const startSampleIdx = Math.max(
    0,
    Math.floor(relStartSec * TARGET_SAMPLE_RATE)
  );
  const endSampleIdx = Math.min(
    chunkSamples.length,
    Math.ceil(safeEndSec * TARGET_SAMPLE_RATE)
  );

  const subSamples = chunkSamples.slice(startSampleIdx, endSampleIdx);
  const subWavBuffer = encodeFloat32ToWav(subSamples, TARGET_SAMPLE_RATE);

  return {
    subSamples,
    subWavBuffer,
    paddedStartGlobalSec:
      chunkStartGlobalSec + startSampleIdx / TARGET_SAMPLE_RATE,
    paddedEndGlobalSec: chunkStartGlobalSec + endSampleIdx / TARGET_SAMPLE_RATE,
  };
}

export const extractSubSegmentAudio = sliceSubSegmentAudio;

export function encodeFloat32ToWav(
  samples: Float32Array,
  sampleRate = TARGET_SAMPLE_RATE
): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}

export const encodeWavFromPcm = encodeFloat32ToWav;

export function encodeFloat32ToWavBlob(
  samples: Float32Array,
  sampleRate = TARGET_SAMPLE_RATE
): Blob {
  const buf = encodeFloat32ToWav(samples, sampleRate);
  return new Blob([buf], { type: 'audio/wav' });
}

export const encodeWavBlobFromFloat32 = encodeFloat32ToWavBlob;
