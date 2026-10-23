// src/utils/audioProcessor.ts

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

async function applyHighpassFilter16k(
  decoded: AudioBuffer
): Promise<Float32Array> {
  const frameCount = Math.max(
    1,
    Math.ceil(decoded.duration * TARGET_SAMPLE_RATE)
  );
  const offlineCtx = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
  const sourceNode = offlineCtx.createBufferSource();
  sourceNode.buffer = decoded;

  const highpass = offlineCtx.createBiquadFilter();
  highpass.type = 'highpass';
  highpass.frequency.value = 85;
  highpass.Q.value = 0.707;

  sourceNode.connect(highpass);
  highpass.connect(offlineCtx.destination);
  sourceNode.start(0);

  const rendered = await offlineCtx.startRendering();
  return new Float32Array(rendered.getChannelData(0));
}

async function applyHighpassFilterOnArray(
  inputSamples: Float32Array,
  sampleRate: number
): Promise<Float32Array> {
  const frameCount = Math.max(1, inputSamples.length);
  const offlineCtx = new OfflineAudioContext(1, frameCount, sampleRate);
  const buf = offlineCtx.createBuffer(1, frameCount, sampleRate);
  buf.getChannelData(0).set(inputSamples);

  const sourceNode = offlineCtx.createBufferSource();
  sourceNode.buffer = buf;

  const highpass = offlineCtx.createBiquadFilter();
  highpass.type = 'highpass';
  highpass.frequency.value = 85;
  highpass.Q.value = 0.707;

  sourceNode.connect(highpass);
  highpass.connect(offlineCtx.destination);
  sourceNode.start(0);

  const rendered = await offlineCtx.startRendering();
  return new Float32Array(rendered.getChannelData(0));
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
 * 【Issue 2 第 7 步核心：重聽專用無失真聲學正規化與清晰度強化器】
 * 修正重點：
 * 1. 預設停用會造成 Whisper 頻譜相位失真與幻覺的 WSOLA 時間拉伸 (enableTimeStretch = false) 及 50ms 劇烈起伏 AGC。
 * 2. 改用「整段線性峰值正規化 (Utterance-Level Linear Peak Normalization) + 溫和尾音補償 + 2.8kHz 咬字增強 (+3.0dB)」，
 *    確保 Whisper 接收到 100% 相位連續、自然清晰的人聲波形。
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

  // ── 階段 1：整段線性音量正規化與柔和壓限（保持自然語音包絡，不產生呼吸泵浦雜訊） ──
  let peakAbs = 0.0;
  let sumSq = 0.0;
  for (let i = 0; i < raw.length; i++) {
    const a = Math.abs(raw[i]);
    if (a > peakAbs) peakAbs = a;
    sumSq += raw[i] * raw[i];
  }
  const rms = Math.sqrt(sumSq / raw.length);

  // 若整段偏小聲 (rms < 0.06)，給予一致的線性放大（最高 2.6 倍），維持字與字之間的自然相對動態
  const linearGain =
    peakAbs > 1e-4 ? Math.min(2.6, Math.max(1.0, 0.78 / peakAbs)) : 1.0;
  const hasQuietDropCompensated = linearGain >= 1.45 || rms < 0.045;

  const normalizedSamples = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i] * linearGain;
    normalizedSamples[i] = Math.max(-0.95, Math.min(0.95, v));
  }

  // ── 階段 2：溫和的 2.8kHz 唇齒音清晰度強化 (+2.8dB)，不扭曲母音共振峰 ──
  const formantEnhanced = applyPresenceBoostBiquad(
    normalizedSamples,
    TARGET_SAMPLE_RATE,
    2800,
    2.8,
    1.0
  );

  // 預設不啟用 WSOLA 變速，確保 Whisper 聲學特徵 100% 原汁原味
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