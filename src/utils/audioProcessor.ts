// src/utils/audioProcessor.ts
// 次世代雙軌無損聲學預處理 (85Hz Zero-Phase filtfilt) + 啟動階段 3~5 秒全文聲學預分析與「底層聲學公式動態校準層 (AcousticFormulaCalibration)」
// 核心架構升級：
// 1. [雙軌分流無損動態範圍保護]：語者分離軌道 (`extractAndPreprocessMacroChunk`) 嚴格維持「85Hz 零相位巴特沃斯高通濾波 + 線性無壓縮波形」，100% 保留語者間自然的能量谷底 (`dip`) 與遠近分貝階躍 (`stepDbJump`)，不再因時變壓縮增益抹平語者邊界；前瞻包絡響度正規化 (`applyVadGatedLookAheadLoudnessNormalization`) 專供 ASR 弱音視窗補償使用。
// 2. [預分析參數下放至底層聲學公式層]：`analyzeFullAudioAcousticProfile` 除推薦方案 A/B/C 外，更直接根據音檔實測之底噪分位數、有聲母音 P15 諧波度、音域跨度與換人頻率，動態推導出 `AcousticFormulaCalibration` 注入 `diarization.worker.ts` 底層公式。

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

/**
 * 修正 TypeScript 5.7 Uint8Array.buffer (ArrayBufferLike) 型別相容性
 */
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
 * 【研究報告 5.1 核心】：85 Hz 零相位雙向二階巴特沃斯高通濾波 (Zero-Phase Forward-Backward Butterworth HPF, filtfilt)
 * 先正向通過一次 85Hz 二階 Biquad 高通濾波器，再反轉陣列反向通過一次，
 * 達成 -24 dB/octave 陡峭濾除空調與桌面低頻轟鳴，且相位延遲嚴格為 0，完美保護爆破音瞬態與共振峰相位！
 */
export function applyZeroPhaseButterworthHighpass(
  input: Float32Array,
  sampleRate = TARGET_SAMPLE_RATE,
  cutoffHz = 85.0
): Float32Array {
  const N = input.length;
  if (N < 16) return new Float32Array(input);

  const w0 = (2.0 * Math.PI * cutoffHz) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);
  const alpha = sinW0 / Math.SQRT2; // Q = 1 / sqrt(2) = 0.7071 (Butterworth)

  const a0 = 1.0 + alpha;
  const b0 = ((1.0 + cosW0) * 0.5) / a0;
  const b1 = -(1.0 + cosW0) / a0;
  const b2 = ((1.0 + cosW0) * 0.5) / a0;
  const a1 = (-2.0 * cosW0) / a0;
  const a2 = (1.0 - alpha) / a0;

  const forward = new Float32Array(N);

  // 1. 正向濾波 (Forward Pass)
  let x1 = input[0];
  let x2 = input[0];
  let y1 = 0.0;
  let y2 = 0.0;
  for (let i = 0; i < N; i++) {
    const x0 = input[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    forward[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }

  // 2. 反向濾波 (Backward Pass) 消除所有頻率之相位偏移
  const out = new Float32Array(N);
  x1 = forward[N - 1];
  x2 = forward[N - 1];
  y1 = 0.0;
  y2 = 0.0;
  for (let i = N - 1; i >= 0; i--) {
    const x0 = forward[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    out[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }

  return out;
}

/**
 * 【ASR 專用前瞻包絡響度補償器】：VAD 閘控之零相位失真前瞻包絡響度正規化與 -1 dBFS 真實峰值限制器
 * 注意：此函式專供 ASR 弱音片段補償使用；語者分離 (`diarization.worker.ts`) 則使用純線性 85Hz 高通波形，
 * 以 100% 保留真實能量谷底 (`dip`) 與遠近語者分貝差 (`stepDbJump`)！
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

  const sortedRms = Array.from(frameRms).sort((a, b) => a - b);
  const noiseFloor = Math.max(
    0.0006,
    sortedRms[Math.floor(sortedRms.length * 0.03)] || 0.0015
  );

  const targetRms = 0.115; // 約 -18.8 dBFS
  const maxLinearGain = 2.45; // 最高 +7.8 dB 溫和補償
  const minSpeechRms = Math.max(0.003, noiseFloor * 1.55);
  const truePeakCeiling = 0.89125; // -1.0 dBFS

  const targetFrameGain = new Float32Array(numFrames).fill(1.0);
  const frameHarmonicity = new Float32Array(numFrames);
  const minLag = Math.floor(sampleRate / 340);
  const maxLag = Math.floor(sampleRate / 78);

  for (let f = 0; f < numFrames; f++) {
    if (frameRms[f] < minSpeechRms * 0.75) continue;
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

  for (let f = 0; f < numFrames; f++) {
    const prevR = f > 0 ? frameRms[f - 1] : frameRms[f];
    const nextR = f + 1 < numFrames ? frameRms[f + 1] : frameRms[f];
    const localRms = Math.max(frameRms[f], 0.5 * (prevR + nextR));
    const zcr = frameZeroCross[f];
    const prevH = f > 0 ? frameHarmonicity[f - 1] : frameHarmonicity[f];
    const nextH =
      f + 1 < numFrames ? frameHarmonicity[f + 1] : frameHarmonicity[f];
    const localHarm = Math.max(frameHarmonicity[f], prevH, nextH);

    const snrRatio = localRms / Math.max(1e-5, noiseFloor);
    const isAperiodicFriction =
      localHarm < 0.34 && (zcr > 0.44 || snrRatio < 2.0);
    const pVad = isAperiodicFriction
      ? 0.08
      : 1.0 / (1.0 + Math.exp(-2.0 * (snrRatio - 1.65)));

    // 偵測夾在有聲母音之間的弱子音過渡幀（如快速連讀之「禮物 l-w」、「委員 w-y」字首流音/滑音谷底）
    const isInterVowelWeakConsonant =
      frameHarmonicity[f] < 0.48 &&
      (prevH >= 0.48 || nextH >= 0.48) &&
      frameRms[f] >= minSpeechRms * 0.65 &&
      frameRms[f] < Math.max(prevR, nextR) * 0.72;
    const consonantClarityBonusDb = isInterVowelWeakConsonant ? 1.65 : 0.0;

    if (pVad < 0.3 || localRms < minSpeechRms) {
      targetFrameGain[f] = 1.0;
    } else if (localRms < targetRms || consonantClarityBonusDb > 0) {
      const dbCurrent = 20.0 * Math.log10(Math.max(1e-5, localRms));
      const dbTarget = -18.8;
      const dbDeficit = Math.max(0.0, dbTarget - dbCurrent);
      const boostDb = Math.min(
        8.2,
        dbDeficit * 0.5 * pVad + consonantClarityBonusDb
      );
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
  const alphaAttack = 0.5;
  const alphaRelease = 0.08;

  for (let f = 0; f < numFrames; f++) {
    let lookAheadGain = targetFrameGain[f];
    if (f + 1 < numFrames) {
      lookAheadGain = Math.max(lookAheadGain, targetFrameGain[f + 1]);
    }
    if (f + 2 < numFrames) {
      lookAheadGain = Math.max(lookAheadGain, targetFrameGain[f + 2]);
    }

    if (lookAheadGain > currentGain) {
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

  // 【ASR 音頻細節強化】：針對 ASR 音軌施加 2650Hz (+2.3dB, Q=1.05) 聲母與 F2/F3 第二/三共振峰清晰度補償，
  // 強化快速連讀中易衰減的流音/滑音 (/l/, /w/, /y/, /j/, /q/, /x/ 如「禮物」、「委員」、「工項」)，
  // 而語者分離音軌 (Track 1) 仍保持 100% 純線性不經過此濾波！
  return applyPresenceBoostBiquad(out, sampleRate, 2650, 2.3, 1.05);
}

/**
 * 抽取單一區間之 16kHz 單聲道波形（供全文預分析抽樣視窗與巨集區塊共用）
 */
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

/**
 * 計算陣列之四分位去極端值平均 (25% ~ 75% Interquartile Mean)，
 * 徹底濾除華語四聲聲調（陰平、陽平、上聲、去聲）造成的瞬時基頻飆高或壓低！
 */
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
 * 1. 話輪級去聲調基頻中位數 (Utterance-Level Interquartile Mean F0) 估算真實換人頻率與語者數。
 * 2. 將實測聲學分佈（P05/P75 能量分位數、有聲幀 P15 諧波清晰度、音域半音跨度、換人頻率）直接推導為底層 `AcousticFormulaCalibration`，
 *    讓 `diarization.worker.ts` 的 VAD、自相關 F0 諧波門檻、150ms 音節包絡邊界曲線與 Core-Gated VBx 公式自動適應當前音檔！
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

  const windowDurSec = 7.5;
  const targetWindowCount =
    totalDurationSec <= 180
      ? Math.max(4, Math.ceil(totalDurationSec / windowDurSec))
      : totalDurationSec <= 900
      ? 24
      : 32;

  const nameLower = file.name.toLowerCase();
  const isWavOrMp3 =
    file.type.includes('wav') ||
    nameLower.endsWith('.wav') ||
    file.type.includes('mpeg') ||
    file.type.includes('mp3') ||
    nameLower.endsWith('.mp3');

  let cachedFullBuffer: AudioBuffer | null = null;
  if (!isWavOrMp3 && file.size <= 45 * 1024 * 1024) {
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

  onProgress?.(
    62,
    '正在分析全場雙分位底噪、有聲母音 P15 諧波分佈與底層聲學公式動態閥值...'
  );
  await new Promise((r) => setTimeout(r, 150));

  const frameLen = 320; // 20ms @ 16kHz
  const minLag = Math.floor(TARGET_SAMPLE_RATE / 340); // ~47 samples (340Hz)
  const maxLag = Math.floor(TARGET_SAMPLE_RATE / 75); // ~213 samples (75Hz)

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

  for (let wIdx = 0; wIdx < windowSlices.length; wIdx++) {
    const samples = windowSlices[wIdx];
    const numFrames = Math.floor(samples.length / frameLen);
    if (numFrames < 10) continue;

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
    const p05 = Math.max(
      0.0006,
      sortedLocal[Math.floor(sortedLocal.length * 0.05)] || 0.001
    );
    const p75 = Math.max(
      p05 * 2.2,
      sortedLocal[Math.floor(sortedLocal.length * 0.75)] || 0.015
    );
    const vadGate = Math.max(0.0032, p05 * 1.7 + (p75 - p05) * 0.08);

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
            semitoneDiff >= 2.4 ||
            (semitoneDiff >= 1.5 && hiDiff >= 0.08)
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
      if (wRms[f] < vadGate || wZcr[f] > 0.45) {
        silenceRunFrames++;
        if (silenceRunFrames >= 7) {
          flushUtteranceIfValid(18);
        }
        continue;
      }

      const base = f * frameLen;
      if (base + maxLag + 160 >= samples.length) continue;

      const corrProfile = new Float32Array(maxLag + 1);
      let maxCorr = 0.0;
      for (let lag = minLag; lag <= maxLag; lag += 2) {
        let sum = 0.0;
        let nA = 0.0;
        let nB = 0.0;
        for (let i = 0; i < 160; i += 2) {
          const a = samples[base + i];
          const b = samples[base + i + lag];
          sum += a * b;
          nA += a * a;
          nB += b * b;
        }
        const c = nA > 1e-8 && nB > 1e-8 ? sum / Math.sqrt(nA * nB) : 0.0;
        corrProfile[lag] = c;
        if (c > maxCorr) maxCorr = c;
      }

      if (maxCorr >= 0.24) {
        activeCandidateHarmList.push(maxCorr);
      }

      if (maxCorr < 0.36) {
        silenceRunFrames++;
        continue;
      }

      let chosenLag = minLag;
      let secondPeakCorr = 0.0;
      const firstPeakThresh = 0.86 * maxCorr;
      let foundFirstPeak = false;

      for (let lag = minLag + 2; lag <= maxLag - 2; lag += 2) {
        const c = corrProfile[lag];
        if (c >= corrProfile[lag - 2] && c >= corrProfile[lag + 2]) {
          if (!foundFirstPeak && c >= firstPeakThresh) {
            chosenLag = lag;
            foundFirstPeak = true;
          } else if (
            foundFirstPeak &&
            Math.abs(lag - chosenLag * 2) > 8 &&
            c > secondPeakCorr
          ) {
            secondPeakCorr = c;
          }
        }
      }

      const f0Hz = TARGET_SAMPLE_RATE / Math.max(minLag, chosenLag);
      totalVoicedFrames++;
      voicedHarmList.push(maxCorr);
      currentTurnF0s.push(f0Hz);
      currentTurnHiRatios.push(wHiRatio[f]);

      if (
        secondPeakCorr >= 0.68 * maxCorr &&
        maxCorr < 0.74 &&
        wRms[f] > p75 * 0.9
      ) {
        totalOverlapSuspectFrames++;
      }

      silenceRunFrames = 0;

      if (currentTurnF0s.length >= 90) {
        flushUtteranceIfValid(45);
      }
    }

    flushUtteranceIfValid(18);
  }

  onProgress?.(
    84,
    '正在執行全場語者聲紋分佈聚類、推導底層聲學公式動態閥值與最佳化方案...'
  );
  await new Promise((r) => setTimeout(r, 180));

  const sortedAllRms =
    allFrameRms.length > 0
      ? [...allFrameRms].sort((a, b) => a - b)
      : [0.001, 0.02];
  const globalP05 = Math.max(
    0.0005,
    sortedAllRms[Math.floor(sortedAllRms.length * 0.05)] || 0.001
  );
  const globalP85 = Math.max(
    globalP05 * 2.0,
    sortedAllRms[Math.floor(sortedAllRms.length * 0.85)] || 0.03
  );

  const noiseFloorDb = Number(
    Math.max(-68, Math.min(-22, 20 * Math.log10(globalP05))).toFixed(1)
  );
  const snrDb = Number(
    Math.max(6, Math.min(42, 20 * Math.log10(globalP85 / globalP05))).toFixed(1)
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
      : [0.38, 0.65];
  const harmP15 =
    sortedCandHarm[Math.floor(sortedCandHarm.length * 0.15)] || 0.38;

  // 利用 2D (話輪去聲調 F0 半音 + 高頻共振峰比) 凝聚估計全場活躍語者叢集數
  const clusters: Array<{ f0Semitone: number; highBandRatio: number; count: number }> =
    [];
  for (const vec of turnFeatureVectors) {
    let bestIdx = -1;
    let bestDist = Infinity;
    for (let c = 0; c < clusters.length; c++) {
      const df0 = Math.abs(vec.f0Semitone - clusters[c].f0Semitone) / 2.2;
      const dHi = Math.abs(vec.highBandRatio - clusters[c].highBandRatio) / 0.12;
      const dist = Math.sqrt(df0 * df0 + dHi * dHi);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = c;
      }
    }
    if (bestIdx >= 0 && bestDist <= 1.15) {
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
    (c) => c.count >= Math.max(2, Math.floor(turnFeatureVectors.length * 0.08))
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

  const estimatedMinSpeakers =
    turnSwitchRatePerMin < 3.0 && rawClusterCount <= 1
      ? 1
      : Math.max(2, Math.min(4, rawClusterCount));
  const estimatedMaxSpeakers = Math.max(
    estimatedMinSpeakers + 2,
    Math.min(6, rawClusterCount + 2)
  );

  // =========================================================================
  // 【方向一深化】：根據全場實測分佈，動態計算底層聲學公式校準參數 (AcousticFormulaCalibration)
  // =========================================================================
  const silenceRatio = totalSilenceFrames / Math.max(1, totalSampledFrames);
  const calibratedVadNoiseQuantile = Number(
    Math.max(0.03, Math.min(0.08, silenceRatio < 0.12 ? 0.04 : 0.05)).toFixed(3)
  );
  const calibratedVadSpeechMultiplier = Number(
    Math.max(1.52, Math.min(1.88, snrDb >= 20 ? 1.65 : 1.78)).toFixed(2)
  );
  const calibratedHarmGate = Number(
    Math.max(0.34, Math.min(0.42, harmP15 * 0.95)).toFixed(3)
  );
  const calibratedFrictionCeiling = Number(
    Math.max(0.26, Math.min(0.35, calibratedHarmGate * 0.82)).toFixed(3)
  );
  const calibratedMaxMicroTurnSec = Number(
    Math.max(
      1.45,
      Math.min(
        1.85,
        turnSwitchRatePerMin >= 10
          ? 1.55
          : turnSwitchRatePerMin >= 5.5
          ? 1.65
          : 1.80
      )
    ).toFixed(2)
  );
  const calibratedScdPeakCutoff = Number(
    Math.max(
      0.50,
      Math.min(
        0.56,
        turnSwitchRatePerMin >= 11 ? 0.51 : turnSwitchRatePerMin <= 4.5 ? 0.55 : 0.53
      )
    ).toFixed(2)
  );
  const calibratedPitchTolerance = Number(
    Math.max(
      3.4,
      Math.min(4.2, pitchSpreadSemitones >= 8.0 ? 3.8 : 3.5)
    ).toFixed(2)
  );

  const baseFormulaCalibration: AcousticFormulaCalibration = {
    vadNoiseQuantile: calibratedVadNoiseQuantile,
    vadSpeechMultiplier: calibratedVadSpeechMultiplier,
    vadDynamicRangeWeight: 0.06,
    harmonicityGateThreshold: calibratedHarmGate,
    frictionHarmonicCeiling: calibratedFrictionCeiling,
    boundaryDipWeight: 0.45,
    boundaryVoiceWeight: 0.55,
    stepDbDiffFloor: 3.5,
    formantCosDistFloor: 0.22,
    scdPeakCutoff: calibratedScdPeakCutoff,
    maxMicroTurnSec: calibratedMaxMicroTurnSec,
    ahcStopSimThreshold: voiceSimilarity === 'HIGH' ? 0.30 : 0.28,
    pitchPenaltySemitoneTolerance: calibratedPitchTolerance,
    vbxInvTemp: 22.0,
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
      subtitle: '先取純淨長句建立聲紋錨點，再以自適應諧波與音節包絡公式精準切分短句與附和詞',
      whyRecommend: `全場平均每分鐘換人 ${turnSwitchRatePerMin} 次、重疊率 ${(
        overlapRatio * 100
      ).toFixed(
        1
      )}%，已將實測諧波門檻 (${calibratedHarmGate}) 與微觀視窗 (${calibratedMaxMicroTurnSec}s) 下放至聲學公式層，兼顧完整長句與快速問答。`,
      bestFor: '2～6 人部門會議、專案討論、男女與同性別混合且有來回接話（「不算」、「對」）',
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
            Math.max(0.48, baseFormulaCalibration.scdPeakCutoff - 0.03).toFixed(2)
          ),
          maxMicroTurnSec: Number(
            Math.max(1.38, baseFormulaCalibration.maxMicroTurnSec - 0.15).toFixed(2)
          ),
          stepDbDiffFloor: 3.0,
        },
      },
    },
    {
      id: 'preset_c_long_speech',
      badge: '方案 C・簡報與長句防斷裂',
      title: '長篇報告防句中斷裂（主講保護模式）',
      label: '長篇報告防句中斷裂（主講保護模式）',
      subtitle: '拉高底層邊界公式門檻與微觀視窗上限，防止主講人換氣停頓被誤切成兩個人',
      whyRecommend: `適合以長句闡述為主的會議，最大化保護句子完整性並防止主講人換氣時被誤判換人。`,
      bestFor: '專題簡報、教育訓練、法說會、一人主講搭配少數 QA 提問',
      notIdealFor: '多人每秒快速搶話的圓桌討論',
      isRecommended: recommendedPresetId === 'preset_c_long_speech',
      recommendedMinSpeakers: Math.max(1, estimatedMinSpeakers - 1),
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
            Math.min(2.05, baseFormulaCalibration.maxMicroTurnSec + 0.25).toFixed(2)
          ),
          stepDbDiffFloor: 4.2,
        },
      },
    },
  ];

  const elapsedSoFar = performance.now() - startTimeMs;
  if (elapsedSoFar < 2400) {
    onProgress?.(94, '正在彙整全場聲學體檢指標與底層聲學公式校準參數...');
    await new Promise((r) =>
      setTimeout(r, Math.min(1000, 2400 - elapsedSoFar))
    );
  }

  const analysisDurationMs = Math.round(performance.now() - startTimeMs);
  const winningOpt =
    recommendations.find((r) => r.isRecommended) || recommendations[0];

  const plainSummary = `已完成全場 ${
    windowSlices.length
  } 個區段（共抽樣 ${sampledDurationSec} 秒，涵蓋率 ${Math.round(
    coverageRatio * 100
  )}%）深度聲學掃描與公式校準：環境信噪比 ${snrDb} dB（底噪 ${
    noiseLevel === 'LOW' ? '極低' : noiseLevel === 'MEDIUM' ? '中等' : '偏高'
  }）、話輪基頻跨度 ${f0P10Hz}~${f0P90Hz} Hz（${pitchSpreadSemitones} 半音）、有聲母音諧波門檻自動校準為 ${calibratedHarmGate}、每分鐘換人約 ${turnSwitchRatePerMin} 次（微觀切分視窗校準為 ${calibratedMaxMicroTurnSec}s）、重疊率 ${(
    overlapRatio * 100
  ).toFixed(1)}%，預估約 ${estimatedMinSpeakers}~${estimatedMaxSpeakers} 位語者。系統優先推薦【${
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
    sampledWindowsCount: windowSlices.length,
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

/** 相容別名：供舊版或外部元件呼叫 */
export const analyzeAudioPreProfile = analyzeFullAudioAcousticProfile;

export async function extractAndPreprocessMacroChunk(
  file: File,
  chunkStartSec: number,
  chunkEndSec: number,
  totalDurationSec: number
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
    const startByte = 44 + Math.floor((chunkStartSec * bytesPerSec) / 4) * 4;
    const endByte = Math.min(
      file.size,
      44 + Math.ceil((chunkEndSec * bytesPerSec) / 4) * 4
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
    const wavBuffer = encodeFloat32ToWav(samples, TARGET_SAMPLE_RATE);
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
      const wavBuffer = encodeFloat32ToWav(samples, TARGET_SAMPLE_RATE);
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
  const wavBuffer = encodeFloat32ToWav(samples, TARGET_SAMPLE_RATE);

  return {
    samples,
    wavBuffer,
    actualDurationSec: samples.length / TARGET_SAMPLE_RATE,
  };
}

/** 相容匯出別名 */
export const sliceAudioIntoMacroChunks = extractAndPreprocessMacroChunk;

/**
 * 將 AudioBuffer 轉為 16kHz 單聲道後，僅執行「85Hz 零相位雙向巴特沃斯高通濾波 (`filtfilt`)」：
 * 嚴格保持原始線性動態包絡，不在此階段套用時變壓縮增益，確保 `diarization.worker.ts` 擁有 100% 真實的能量谷底 (`dip`) 與遠近語者分貝差！
 */
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
  const res = await extractAndPreprocessMacroChunk(file, 0, dur, dur);
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

/**
 * 雙二階峰值濾波器 (Biquad Peaking EQ)：針對 2.8kHz 唇齒音與 F2/F3 共振峰增強 +4.5dB，
 * 讓模糊的平翹舌、鼻音與氣音子音在重聽時變得清晰銳利。
 */
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

/**
 * 波形相似度重疊相加演算法 (Pitch-Preserving WSOLA Time-Stretching)
 */
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

/**
 * 【重聽專用無失真聲學正規化與清晰度強化器】
 */
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
    peakAbs > 1e-4 ? Math.min(2.6, Math.max(1.0, 0.78 / peakAbs)) : 1.0;
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