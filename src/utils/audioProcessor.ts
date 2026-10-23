// src/utils/audioProcessor.ts
// 次世代零相位雙向巴特沃斯高通濾波 (Zero-Phase filtfilt) + VAD 閘控前瞻包絡響度正規化 + -1 dBFS 真實峰值限制器

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
  const b0 = (1.0 + cosW0) * 0.5 / a0;
  const b1 = -(1.0 + cosW0) / a0;
  const b2 = (1.0 + cosW0) * 0.5 / a0;
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
 * 【研究報告 5.1 核心】：VAD 閘控之零相位失真前瞻包絡響度正規化與 -1 dBFS 真實峰值限制器
 * (VAD-Gated Look-Ahead Envelope Compression & True-Peak Limiter)
 * - 解決遠端與會者小聲插話 (-38 dBFS) 被 Whisper 當成噪音吞字，同時防止靜音區放大空調底噪引發幻覺跳針。
 * - 15ms 前瞻視窗 + 雙時間常數 (Attack 5ms / Release 120ms) + 3ms 前瞻 -1.0 dBFS (0.891) 峰值限制器。
 */
export function applyVadGatedLookAheadLoudnessNormalization(
  samples: Float32Array,
  sampleRate = TARGET_SAMPLE_RATE
): Float32Array {
  const N = samples.length;
  if (N < 1600) return samples;

  // 1. 以 20ms (320 samples) 為一幀計算局部 RMS 包絡與自適應底噪門檻
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

  // 估計第 18 百分位數作為會議室空調底噪 (Noise Floor)
  const sortedRms = Array.from(frameRms).sort((a, b) => a - b);
  const noiseFloor = Math.max(
    0.0008,
    sortedRms[Math.floor(sortedRms.length * 0.18)] || 0.002
  );

  // 目標語音響度 -18 dBFS (約 RMS = 0.1259)，最大向上補償增益 +12 dB (3.98 倍)
  const targetRms = 0.1259; // -18 dBFS
  const maxLinearGain = 3.981; // +12 dB
  const minSpeechRms = Math.max(0.004, noiseFloor * 2.1); // 約 -45 dBFS 以上且高於底噪 2.1 倍
  const truePeakCeiling = 0.89125; // -1.0 dBFS

  // 2. 計算每一幀之 VAD 語音機率估計 P_VAD(f) 與目標包絡增益 G_target(f)
  const targetFrameGain = new Float32Array(numFrames).fill(1.0);

  // 計算每幀之快速自相關諧波週期度 (Voiced Harmonicity)，區分真實人聲母音與翻紙/摩擦/碰撞雜音
  const frameHarmonicity = new Float32Array(numFrames);
  const minLag = Math.floor(sampleRate / 320); // 50 samples
  const maxLag = Math.floor(sampleRate / 80);  // 200 samples
  for (let f = 0; f < numFrames; f++) {
    if (frameRms[f] < minSpeechRms * 0.8) continue;
    const start = f * frameLen;
    if (start + 400 > N) continue;
    let maxCorr = 0.0;
    for (let lag = minLag; lag <= maxLag; lag += 6) {
      let sum = 0.0;
      let nA = 0.0;
      let nB = 0.0;
      for (let i = 0; i < 200; i += 4) {
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
    // 結合前後 1 幀 (共 60ms) 進行微觀平滑，避免單幀爆破音誤判
    const prevR = f > 0 ? frameRms[f - 1] : frameRms[f];
    const nextR = f + 1 < numFrames ? frameRms[f + 1] : frameRms[f];
    const localRms = Math.max(frameRms[f], 0.5 * (prevR + nextR));
    const zcr = frameZeroCross[f];
    const prevH = f > 0 ? frameHarmonicity[f - 1] : frameHarmonicity[f];
    const nextH = f + 1 < numFrames ? frameHarmonicity[f + 1] : frameHarmonicity[f];
    const localHarm = Math.max(frameHarmonicity[f], 0.5 * (prevH + nextH));

    // 計算軟性語音機率 P_VAD \in [0, 1]：若缺乏聲帶諧波週期性 (localHarm < 0.45，如紙張摩擦或移動雜音)，強制壓低 P_VAD 禁止放大雜音！
    const snrRatio = localRms / Math.max(1e-5, noiseFloor);
    const isAperiodicFriction = (zcr > 0.42 && snrRatio < 3.5) || localHarm < 0.45;
    const pVad = isAperiodicFriction
      ? 0.05
      : 1.0 / (1.0 + Math.exp(-2.2 * (snrRatio - 2.3)));

    if (pVad < 0.35 || localRms < minSpeechRms) {
      // 非語音靜音區：強制鎖定增益為 0 dB (1.0)，絕對不放大會議室底噪！
      targetFrameGain[f] = 1.0;
    } else if (localRms < targetRms) {
      // 小聲插話區間 (-45 dBFS ~ -18 dBFS)：依壓縮比 2.5:1 計算溫和補償增益
      const dbCurrent = 20.0 * Math.log10(Math.max(1e-5, localRms));
      const dbTarget = -18.0;
      const dbDeficit = Math.max(0.0, dbTarget - dbCurrent);
      // 壓縮補償：提振 (1 - 1/2.5) = 60% 的響度落差，且受 P_VAD 軟閘控與 +12dB 上限保護
      const boostDb = Math.min(12.0, dbDeficit * 0.6 * pVad);
      targetFrameGain[f] = Math.min(
        maxLinearGain,
        Math.pow(10.0, boostDb / 20.0)
      );
    } else {
      // 音量已充足 (> -18 dBFS)：維持 1.0 原音動態
      targetFrameGain[f] = 1.0;
    }
  }

  // 3. 15ms 前瞻視窗 (Look-Ahead 1 frame) + 非對稱攻擊/釋放平滑 (Attack 5ms / Release 120ms)
  const smoothedFrameGain = new Float32Array(numFrames);
  let currentGain = 1.0;
  const alphaAttack = 0.65; // 快速反應以保護字頭子音
  const alphaRelease = 0.14; // 緩慢釋放以保護句尾韻母不產生泵浦效應 (Pumping)

  for (let f = 0; f < numFrames; f++) {
    // 前瞻 1 幀 (20ms) 預先拉升字首增益
    const lookAheadGain =
      f + 1 < numFrames
        ? Math.max(targetFrameGain[f], targetFrameGain[f + 1])
        : targetFrameGain[f];

    if (lookAheadGain > currentGain) {
      currentGain =
        alphaAttack * lookAheadGain + (1.0 - alphaAttack) * currentGain;
    } else {
      currentGain =
        alphaRelease * lookAheadGain + (1.0 - alphaRelease) * currentGain;
    }
    smoothedFrameGain[f] = currentGain;
  }

  // 4. 逐樣本線性插值套用包絡增益 + 3ms (48 samples) 前瞻 -1.0 dBFS 真實峰值限制器 (True-Peak Limiter)
  const out = new Float32Array(N);
  const limiterLookAhead = Math.max(16, Math.floor(sampleRate * 0.003)); // 48 samples (3ms)

  for (let i = 0; i < N; i++) {
    const fIdx = Math.min(numFrames - 1, Math.floor(i / frameLen));
    const nextFIdx = Math.min(numFrames - 1, fIdx + 1);
    const frac = (i - fIdx * frameLen) / frameLen;
    const interpGain =
      smoothedFrameGain[fIdx] * (1.0 - frac) +
      smoothedFrameGain[nextFIdx] * frac;
    out[i] = samples[i] * interpGain;
  }

  // 3ms 前瞻防削波軟限幅 (True-Peak Limiter at -1.0 dBFS = 0.89125)
  let limiterEnv = 1.0;
  const limRelease = Math.exp(-1.0 / (sampleRate * 0.04)); // 40ms 恢復常數

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

/**
 * 將 AudioBuffer 轉為 16kHz 單聲道後，依序執行：
 * 1. 85Hz 零相位雙向巴特沃斯高通濾波 (`filtfilt`)
 * 2. VAD 閘控前瞻包絡響度正規化與 -1 dBFS 真實峰值限制器
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

  const zeroPhaseFiltered = applyZeroPhaseButterworthHighpass(
    resampled,
    TARGET_SAMPLE_RATE,
    85.0
  );
  return applyVadGatedLookAheadLoudnessNormalization(
    zeroPhaseFiltered,
    TARGET_SAMPLE_RATE
  );
}

async function applyHighpassFilterOnArray(
  inputSamples: Float32Array,
  sampleRate: number
): Promise<Float32Array> {
  const zeroPhaseFiltered = applyZeroPhaseButterworthHighpass(
    inputSamples,
    sampleRate,
    85.0
  );
  return applyVadGatedLookAheadLoudnessNormalization(
    zeroPhaseFiltered,
    sampleRate
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
    out[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return out;
}

/**
 * 波形相似度重疊相加演算法 (Pitch-Preserving WSOLA Time-Stretching)：
 * 在不改變人聲基頻 (不變調) 的前提下，將語速放慢 (例如 stretchRatio = 1.18 對應 0.85x 語速)，
 * 讓連音、搶話或太短促的漏字獲得更充裕的 Whisper 編碼幀數。
 */
function applyWsolaTimeStretch(
  input: Float32Array,
  stretchRatio = 1.18, // > 1.0 代表放慢展開
  sampleRate = TARGET_SAMPLE_RATE
): Float32Array {
  if (Math.abs(stretchRatio - 1.0) < 0.03 || input.length < 1600) {
    return input;
  }

  const winLen = Math.floor(sampleRate * 0.03); // 30ms 視窗 (480 samples)
  const synHop = Math.floor(winLen / 2); // 15ms 合成步長 (240 samples)
  const anaHop = Math.max(1, Math.round(synHop / stretchRatio)); // 分析步長
  const searchRadius = Math.floor(sampleRate * 0.008); // ±8ms 最佳相位對齊搜尋半徑

  const outLen = Math.ceil(input.length * stretchRatio) + winLen;
  const output = new Float32Array(outLen);
  const normWeight = new Float32Array(outLen);

  const hann = new Float32Array(winLen);
  for (let n = 0; n < winLen; n++) {
    hann[n] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * n) / (winLen - 1)));
  }

  let anaPos = 0;
  let synPos = 0;

  // 第一個視窗直接寫入
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

    // 在預期重疊區利用互相關 (Cross-Correlation) 尋找相位最連續的切點，徹底消除機械音
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
 * 預設停用會造成 Whisper 頻譜相位失真與幻覺的 WSOLA 時間拉伸 (enableTimeStretch = false)
 * 改用「整段線性峰值正規化 (Utterance-Level Linear Peak Normalization) + 溫和尾音補償 + 2.8kHz 咬字增強 (+2.8dB)」
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