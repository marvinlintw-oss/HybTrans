// src/workers/localAsr.worker.ts
import { env, pipeline } from '@huggingface/transformers';
import * as OpenCC from 'opencc-js';
import {
  FineDiarizationTurn,
  LocalAsrModel,
  Segment,
  VirtualTrackMapItem,
  WordSpan,
} from '../types';

env.allowLocalModels = false;
env.useBrowserCache = true;

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

const MODEL_ID_MAP: Record<LocalAsrModel, string> = {
  'Whisper-Large-v3-Turbo': 'onnx-community/whisper-large-v3-turbo',
  'SenseVoice-Small': 'onnx-community/whisper-small',
  'Breeze-ASR': 'onnx-community/whisper-base',
};

const MULTILINGUAL_QUESTION_END_REGEX =
  /([？?]|怎麼辦|對不對|是不是|好不好|可以嗎|是嗎|對嗎|嗎[。]?|呢[。]?|か[。？?]?|でしょうか[。？?]?|ますか[。？?]?|ですか[。？?]?|じゃないですか[。？?]?|right\?|okay\?)$/i;

const MULTILINGUAL_BACKCHANNEL_REGEX =
  /^(不算|對[，。]?|對啊|對對對|沒錯|可以|可以啊|沒有[，。]?|沒有啊|是吧|好[，。]?|好的|了解|はい|ええ|うん|なるほど|そうですね|そうですか|確かに|了解です|承知しました|yeah|yes|yep|right|uh-huh|exactly|sure|okay|ok|i see|got it)[，。！？!?,.\s]*$/i;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let asrInstance: any = null;
let loadedModelKey: string | null = null;
let maxReportedProgress = 0;
const fileProgressMap = new Map<string, { loaded: number; total: number }>();

/**
 * 探測當前瀏覽器與顯卡是否支援 WebGPU 及 `shader-f16` (fp16) 擴充功能。
 * 避免在不支援 fp16 的顯卡上強行指定 `dtype: { encoder_model: 'fp16' }` 導致崩潰。
 */
async function probeWebGpuCapabilities(): Promise<{
  hasWebGpu: boolean;
  supportsFp16: boolean;
}> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as unknown as { gpu?: any }).gpu;
    if (!gpu || typeof gpu.requestAdapter !== 'function') {
      return { hasWebGpu: false, supportsFp16: false };
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return { hasWebGpu: false, supportsFp16: false };
    }
    const supportsFp16 = Boolean(
      adapter.features &&
        typeof adapter.features.has === 'function' &&
        adapter.features.has('shader-f16')
    );
    return { hasWebGpu: true, supportsFp16 };
  } catch {
    return { hasWebGpu: false, supportsFp16: false };
  }
}

async function ensureAsrPipeline(modelName: LocalAsrModel) {
  const modelId =
    MODEL_ID_MAP[modelName] || MODEL_ID_MAP['Whisper-Large-v3-Turbo'];
  if (asrInstance && loadedModelKey === modelId) return asrInstance;

  if (asrInstance && typeof asrInstance.dispose === 'function') {
    try {
      await asrInstance.dispose();
    } catch {
      // ignore
    }
  }

  maxReportedProgress = 5;
  fileProgressMap.clear();

  self.postMessage({
    type: 'ASR_DOWNLOAD_PROGRESS',
    active: true,
    modelName,
    fileName: '正在檢測 GPU 精度支援並準備載入語音模型...',
    progress: 5,
  });

  const progressCallback = (info: {
    status?: string;
    file?: string;
    loaded?: number;
    total?: number;
    progress?: number;
  }) => {
    const fileName = info.file || '';
    if (!fileName.endsWith('.onnx')) return;

    if (info.status === 'progress' || info.status === 'download') {
      if (
        typeof info.loaded === 'number' &&
        typeof info.total === 'number' &&
        info.total > 0
      ) {
        fileProgressMap.set(fileName, {
          loaded: info.loaded,
          total: info.total,
        });
        let sumLoaded = 0;
        let sumTotal = 0;
        const entries = Array.from(fileProgressMap.values());
        for (let i = 0; i < entries.length; i++) {
          sumLoaded += entries[i].loaded;
          sumTotal += entries[i].total;
        }
        const combinedPct =
          sumTotal > 0 ? Math.round((sumLoaded / sumTotal) * 100) : 20;
        maxReportedProgress = Math.max(
          maxReportedProgress,
          Math.min(99, combinedPct)
        );
      } else if (typeof info.progress === 'number') {
        maxReportedProgress = Math.max(
          maxReportedProgress,
          Math.min(99, Math.round(info.progress))
        );
      }

      const shortFile = fileName.split('/').pop() || fileName;
      self.postMessage({
        type: 'ASR_DOWNLOAD_PROGRESS',
        active: true,
        modelName,
        fileName: shortFile,
        progress: maxReportedProgress,
      });
    }
  };

  const { hasWebGpu, supportsFp16 } = await probeWebGpuCapabilities();

  try {
    if (hasWebGpu && supportsFp16) {
      asrInstance = await pipeline('automatic-speech-recognition', modelId, {
        device: 'webgpu',
        dtype: {
          encoder_model: 'fp16',
          decoder_model_merged: 'q4',
        },
        progress_callback: progressCallback,
      });
    } else if (hasWebGpu) {
      console.info(
        'ℹ️ [LocalASR] 偵測到顯卡不支援 shader-f16，自動採用 WebGPU fp32 + q4 相容模式'
      );
      asrInstance = await pipeline('automatic-speech-recognition', modelId, {
        device: 'webgpu',
        dtype: {
          encoder_model: 'fp32',
          decoder_model_merged: 'q4',
        },
        progress_callback: progressCallback,
      });
    } else {
      throw new Error('WebGPU unavailable, fallback to WASM');
    }
  } catch (firstErr) {
    console.warn(
      '⚠️ [LocalASR] WebGPU 初始化自動切換至相容精度模式 (fp32 / WASM):',
      firstErr
    );
    fileProgressMap.clear();
    try {
      if (hasWebGpu) {
        asrInstance = await pipeline('automatic-speech-recognition', modelId, {
          device: 'webgpu',
          dtype: {
            encoder_model: 'fp32',
            decoder_model_merged: 'q4',
          },
          progress_callback: progressCallback,
        });
      } else {
        asrInstance = await pipeline('automatic-speech-recognition', modelId, {
          device: 'wasm',
          dtype: 'q8',
          progress_callback: progressCallback,
        });
      }
    } catch {
      asrInstance = await pipeline('automatic-speech-recognition', modelId, {
        device: 'wasm',
        dtype: 'q8',
        progress_callback: progressCallback,
      });
    }
  }

  self.postMessage({
    type: 'ASR_DOWNLOAD_PROGRESS',
    active: false,
    modelName,
    fileName: '',
    progress: 100,
  });

  loadedModelKey = modelId;
  return asrInstance;
}

function restoreMultilingualPunctuation(rawInput: string): string {
  const hasKana = /[\u3040-\u30ff]/.test(rawInput);
  let text = hasKana ? rawInput.trim() : toTwp(rawInput.trim());
  if (!text) return '';

  text = text.replace(/([\u4e00-\u9fa5])\1{2,}/g, '$1');
  text = text.replace(/([\u4e00-\u9fa5]{2,6})\1{2,}/g, '$1');
  text = text.replace(
    /^(字幕|請不吝點讚|訂閱轉發|謝謝觀看|Amara\.org).*$/gi,
    ''
  );

  const cjkRatio =
    (text.match(/[\u3040-\u9fa5]/g)?.length || 0) / Math.max(1, text.length);

  if (cjkRatio >= 0.25) {
    text = text
      .replace(/,/g, '，')
      .replace(/\?/g, '？')
      .replace(/!/g, '！')
      .replace(/([\u3040-\u9fa5])\s+([\u3040-\u9fa5])/g, '$1，$2');
  }

  text = text.replace(
    /^(不是|沒有|對|好|那|另外|不過|其實|老實說|基本上|はい|ええ|いや|なるほど)(?=[，。？！]?[^\u3000-\u303f\uff00-\uffefa-zA-Z要])/,
    '$1，'
  );
  text = text.replace(
    /^(對對對|沒有沒有|不用不用)(?=[^\u3000-\u303f\uff00-\uffef])/,
    '$1，'
  );

  text = text.replace(
    /([\u4e00-\u9fa5A-Za-z0-9]{4,})(但是|可是|因為|所以|如果說|不然的話|否則|那如果|而且|然後|ですが|けれども|しかし|ただ)/g,
    '$1，$2'
  );

  text = text.replace(
    /(的時候|的話|來說|的部分|這件事)(?=[\u4e00-\u9fa5]{3,})/g,
    '$1，'
  );

  text = text.replace(
    /(怎麼辦|對不對|是不是|好不好|可以嗎|是嗎|對嗎|多少錢|為什麼|哪裡|でしょうか|ますか|ですか)[。，？?]*$/g,
    '$1？'
  );
  text = text.replace(/(嗎|呢)[。，？?]*$/g, '$1？');
  text = text.replace(
    /(不知道啊|太遠了|太久了|超亂的)[。，！!]*$/g,
    '$1！'
  );

  text = text
    .replace(/[？?]{2,}/g, '？')
    .replace(/[！!]{2,}/g, '！')
    .replace(/？[，。、]+/g, '？')
    .replace(/！[，。、]+/g, '！')
    .replace(/[，、]{2,}/g, '，')
    .replace(/，([。？！])/g, '$1');

  if (
    text.length >= 2 &&
    cjkRatio >= 0.25 &&
    !/[，。？！、；：.?!]$/.test(text)
  ) {
    text += '。';
  }
  return text;
}

function buildWordSpans(
  cleanText: string,
  segId: string,
  startSec: number,
  endSec: number,
  baseConfidence: number,
  acousticThreshold = 0.72
): WordSpan[] {
  const tokens =
    cleanText.match(
      /[\u4e00-\u9fa5]{1,2}|[\u3040-\u30ff]{1,3}|[a-zA-Z0-9']+\s*|[^\s\u4e00-\u9fa5\u3040-\u30ffa-zA-Z0-9]/g
    ) || [cleanText];
  const dur = Math.max(0.2, endSec - startSec);
  const step = dur / Math.max(1, tokens.length);

  return tokens.map((tok, idx) => {
    const isPunct = /^[^\u4e00-\u9fa5\u3040-\u30ffa-zA-Z0-9]+$/.test(tok);
    const conf = isPunct
      ? 0.99
      : Number(Math.min(0.98, Math.max(0.45, baseConfidence)).toFixed(2));
    return {
      id: `${segId}_w_${idx}`,
      text: tok,
      start: Number((startSec + idx * step).toFixed(2)),
      end: Number((startSec + (idx + 1) * step).toFixed(2)),
      confidence: conf,
      status: conf < acousticThreshold ? 'acoustic_uncertain' : 'normal',
    };
  });
}

/**
 * 【Piecewise Linear Time Unwarping：虛擬音軌時間軸無縫還原演算法】
 * 當送入 Whisper 的是「語者 A 的第 1 段 + 0.35s 靜音隔板 + 語者 A 的第 2 段...」：
 * 1. 將 Whisper 輸出之每個 Chunk / 子句根據其虛擬時間區間 `[vStart, vEnd]` 對準 `timeMap` 中重疊最大之真實發聲區段。
 * 2. 若單一 Whisper Chunk 橫跨了 `0.35s` 靜音隔板（即同時覆蓋兩個不連續的 `VirtualTrackMapItem`），
 *    自動在隔板位置將文字依標點或時間比例切開，分別映射回各自真實的會議時間區段 `[realStartSec, realEndSec]`！
 */
function unwarpVirtualTrackChunksToSegments(
  rawChunks: Array<{ text: string; timestamp?: [number, number | null] }>,
  timeMap: VirtualTrackMapItem[],
  speakerId: string,
  chunkIndex: number,
  batchIndex: number,
  acousticThreshold: number
): Segment[] {
  if (timeMap.length === 0 || rawChunks.length === 0) return [];

  const totalVirtualSec = timeMap[timeMap.length - 1].virtualEndSec;
  const unwarpedUnits: Array<{
    text: string;
    realStart: number;
    realEnd: number;
    speakerId: string;
    runnerUpSpeakerId: string;
    confidence: number;
    acousticMargin: number;
  }> = [];

  for (let i = 0; i < rawChunks.length; i++) {
    const ch = rawChunks[i];
    const cleanText = restoreMultilingualPunctuation(ch.text || '');
    if (!cleanText) continue;

    const vStart =
      typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
    const vEnd =
      typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
        ? ch.timestamp[1]
        : Math.min(totalVirtualSec, vStart + 3.5);

    const overlappingItems = timeMap.filter(
      (item) =>
        Math.min(vEnd, item.virtualEndSec) -
          Math.max(vStart, item.virtualStartSec) >
        0.08
    );

    if (overlappingItems.length === 0) {
      const vCenter = 0.5 * (vStart + vEnd);
      let nearest = timeMap[0];
      let minDist = Infinity;
      for (const item of timeMap) {
        const itemCenter = 0.5 * (item.virtualStartSec + item.virtualEndSec);
        const d = Math.abs(vCenter - itemCenter);
        if (d < minDist) {
          minDist = d;
          nearest = item;
        }
      }
      overlappingItems.push(nearest);
    }

    if (overlappingItems.length === 1) {
      const item = overlappingItems[0];
      const vSpan = Math.max(0.15, item.virtualEndSec - item.virtualStartSec);
      const rSpan = Math.max(0.15, item.realEndSec - item.realStartSec);

      const clampedVStart = Math.max(
        item.virtualStartSec,
        Math.min(item.virtualEndSec - 0.1, vStart)
      );
      const clampedVEnd = Math.max(
        clampedVStart + 0.15,
        Math.min(item.virtualEndSec, vEnd)
      );

      const relRatioStart = (clampedVStart - item.virtualStartSec) / vSpan;
      const relRatioEnd = (clampedVEnd - item.virtualStartSec) / vSpan;

      const realStart = Number(
        (item.realStartSec + relRatioStart * rSpan).toFixed(2)
      );
      const realEnd = Number(
        Math.max(
          realStart + 0.2,
          item.realStartSec + relRatioEnd * rSpan
        ).toFixed(2)
      );

      unwarpedUnits.push({
        text: cleanText,
        realStart,
        realEnd,
        speakerId: item.speakerId || speakerId,
        runnerUpSpeakerId:
          item.runnerUpSpeakerId || item.speakerId || speakerId,
        confidence: item.speakerConfidence,
        acousticMargin: item.speakerMargin,
      });
    } else {
      const clauses = cleanText
        .split(/(?<=[，。？！、；,.?!])/)
        .map((s) => s.trim())
        .filter(Boolean);

      if (clauses.length >= overlappingItems.length) {
        let cCursor = 0;
        for (let k = 0; k < overlappingItems.length; k++) {
          const item = overlappingItems[k];
          const takeCount =
            k === overlappingItems.length - 1
              ? clauses.length - cCursor
              : Math.max(
                  1,
                  Math.round(clauses.length / overlappingItems.length)
                );
          const subText = restoreMultilingualPunctuation(
            clauses.slice(cCursor, cCursor + takeCount).join('')
          );
          cCursor += takeCount;
          if (!subText) continue;

          unwarpedUnits.push({
            text: subText,
            realStart: item.realStartSec,
            realEnd: item.realEndSec,
            speakerId: item.speakerId || speakerId,
            runnerUpSpeakerId:
              item.runnerUpSpeakerId || item.speakerId || speakerId,
            confidence: item.speakerConfidence,
            acousticMargin: item.speakerMargin,
          });
        }
      } else {
        let bestItem = overlappingItems[0];
        let maxOv = -1;
        for (const item of overlappingItems) {
          const ov =
            Math.min(vEnd, item.virtualEndSec) -
            Math.max(vStart, item.virtualStartSec);
          if (ov > maxOv) {
            maxOv = ov;
            bestItem = item;
          }
        }
        unwarpedUnits.push({
          text: cleanText,
          realStart: bestItem.realStartSec,
          realEnd: bestItem.realEndSec,
          speakerId: bestItem.speakerId || speakerId,
          runnerUpSpeakerId:
            bestItem.runnerUpSpeakerId || bestItem.speakerId || speakerId,
          confidence: bestItem.speakerConfidence,
          acousticMargin: bestItem.speakerMargin,
        });
      }
    }
  }

  const mergedUnits: typeof unwarpedUnits = [];
  for (const u of unwarpedUnits) {
    const prev = mergedUnits[mergedUnits.length - 1];
    if (
      prev &&
      prev.speakerId === u.speakerId &&
      u.realStart >= prev.realEnd - 0.2 &&
      u.realStart - prev.realEnd <= 1.8 &&
      prev.text.length + u.text.length <= 240
    ) {
      const needsComma = !/[，。？！、；：,.?!]$/.test(prev.text);
      prev.text = (needsComma ? `${prev.text}，` : prev.text) + u.text;
      prev.realEnd = Math.max(prev.realEnd, u.realEnd);
      prev.acousticMargin = Math.max(prev.acousticMargin, u.acousticMargin);
    } else {
      mergedUnits.push({ ...u });
    }
  }

  return mergedUnits.map((m, idx) => {
    const segId = `seg_c${chunkIndex}_tb${batchIndex}_s${idx}`;
    return {
      id: segId,
      chunkIndex,
      speakerId: m.speakerId,
      runnerUpSpeakerId: m.runnerUpSpeakerId,
      acousticMargin: m.acousticMargin,
      start: m.realStart,
      end: m.realEnd,
      words: buildWordSpans(
        m.text,
        segId,
        m.realStart,
        m.realEnd,
        m.confidence,
        acousticThreshold
      ),
      isCorrected: false,
      relistenCount: 0,
    };
  });
}

interface AtomicTokenUnit {
  text: string;
  start: number;
  end: number;
  center: number;
  speakerId?: string;
  runnerUpSpeakerId?: string;
  confidence?: number;
  acousticMargin?: number;
  boundaryScdLeft?: number;
  boundaryScdRight?: number;
}

function decomposeChunksIntoAtomicUnits(
  rawChunks: Array<{ text: string; timestamp?: [number, number | null] }>,
  windowStartSec: number,
  windowEndSec: number,
  fineTurns: FineDiarizationTurn[]
): AtomicTokenUnit[] {
  const units: AtomicTokenUnit[] = [];

  const acousticSplitAnchors: Array<{ time: number; strength: number }> = [];
  const windowTurns = fineTurns.filter(
    (ft) => ft.end > windowStartSec && ft.start < windowEndSec
  );

  for (let i = 0; i < windowTurns.length; i++) {
    const cur = windowTurns[i];
    const prev = i > 0 ? windowTurns[i - 1] : null;
    const isSpeakerSwitch = prev && prev.speakerId !== cur.speakerId;
    const pauseGap = prev ? Math.max(0, cur.start - prev.end) : 0;
    const scdLeft = cur.boundaryScdLeft || 0;

    if (isSpeakerSwitch || pauseGap >= 0.12 || scdLeft >= 0.16) {
      const strength =
        (isSpeakerSwitch ? 0.5 : 0.0) +
        Math.min(0.4, pauseGap) +
        scdLeft * 0.8;
      acousticSplitAnchors.push({ time: cur.start, strength });
    }
    if ((cur.boundaryScdRight || 0) >= 0.18) {
      acousticSplitAnchors.push({
        time: cur.end,
        strength: (cur.boundaryScdRight || 0) * 0.8,
      });
    }
  }

  const speechMinStart =
    windowTurns.length > 0 ? windowTurns[0].start : windowStartSec;
  const speechMaxEnd =
    windowTurns.length > 0
      ? windowTurns[windowTurns.length - 1].end
      : windowEndSec;

  for (let i = 0; i < rawChunks.length; i++) {
    const ch = rawChunks[i];
    const rawText = (ch.text || '').trim();
    if (!rawText) continue;

    const relStart =
      typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
    const relEnd =
      typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
        ? ch.timestamp[1]
        : Math.min(windowEndSec - windowStartSec, relStart + 3.8);

    let gStart = Number((windowStartSec + relStart).toFixed(2));
    let gEnd = Number(
      Math.max(gStart + 0.35, windowStartSec + relEnd).toFixed(2)
    );

    if (i === 0 && Math.abs(gStart - speechMinStart) <= 0.6) {
      gStart = Number(Math.max(windowStartSec, speechMinStart).toFixed(2));
    }
    if (
      i === rawChunks.length - 1 &&
      gEnd > speechMaxEnd &&
      gEnd - speechMaxEnd <= 1.2
    ) {
      gEnd = Number(Math.max(gStart + 0.35, speechMaxEnd).toFixed(2));
    }

    const prePunct = restoreMultilingualPunctuation(rawText);
    if (!prePunct) continue;

    const marked = prePunct
      .replace(/([。？！?!])/g, '$1|')
      .replace(
        /([，、,])\s*(?=可是|但是|不是|不過|那如果|所以我|那我|不算|はい|ええ|なるほど|そうですね|しかし|ただ|yeah|right|but\s|however)/gi,
        '$1|'
      );

    const subStrings = marked
      .split('|')
      .flatMap((part) =>
        part.length >= 14 ? part.split(/(?<=[，、,])/) : [part]
      )
      .map((s) => s.trim())
      .filter((s) => s.length > 0 && !/^[，。？！、,.?!]+$/.test(s));

    if (subStrings.length === 0) continue;

    const totalChars = Math.max(
      1,
      subStrings.reduce((acc, s) => acc + s.length, 0)
    );
    const spanDur = Math.max(0.3, gEnd - gStart);

    const boundaries = new Float64Array(subStrings.length + 1);
    boundaries[0] = gStart;
    boundaries[subStrings.length] = gEnd;

    let cursor = 0;
    for (let k = 0; k < subStrings.length - 1; k++) {
      cursor += subStrings[k].length;
      const rawSplit = gStart + (cursor / totalChars) * spanDur;
      const prevBound = boundaries[k];
      const maxAllowed = gEnd - (subStrings.length - 1 - k) * 0.18;
      const minAllowed = prevBound + 0.18;

      const isPragmaticTurnBoundary =
        MULTILINGUAL_QUESTION_END_REGEX.test(subStrings[k]) ||
        MULTILINGUAL_BACKCHANNEL_REGEX.test(subStrings[k + 1]);
      const snapRadius = isPragmaticTurnBoundary ? 0.65 : 0.45;

      let snappedSplit = rawSplit;
      let bestScore = -1.0;

      for (const anchor of acousticSplitAnchors) {
        if (anchor.time < minAllowed || anchor.time > maxAllowed) continue;
        const dist = Math.abs(anchor.time - rawSplit);
        if (dist <= snapRadius) {
          const score = anchor.strength - dist * 0.6;
          if (score > bestScore) {
            bestScore = score;
            snappedSplit = anchor.time;
          }
        }
      }

      boundaries[k + 1] = Number(
        Math.max(minAllowed, Math.min(maxAllowed, snappedSplit)).toFixed(2)
      );
    }

    for (let k = 0; k < subStrings.length; k++) {
      const uStart = boundaries[k];
      const uEnd = Math.max(uStart + 0.18, boundaries[k + 1]);
      units.push({
        text: restoreMultilingualPunctuation(subStrings[k]),
        start: Number(uStart.toFixed(2)),
        end: Number(uEnd.toFixed(2)),
        center: Number((0.5 * (uStart + uEnd)).toFixed(2)),
      });
    }
  }

  return units;
}

function alignUnitsWithScdViterbiAndPragmatics(
  units: AtomicTokenUnit[],
  fineTurns: FineDiarizationTurn[],
  backchannelMarginThreshold = 0.12,
  pingPongMaxDurSec = 1.5,
  engineMode = 'mode_0_heritage_classic'
): AtomicTokenUnit[] {
  const L = units.length;
  if (L === 0 || fineTurns.length === 0) return units;

  const speakerSet = new Set<string>();
  for (const ft of fineTurns) {
    if (ft.speakerId) speakerSet.add(ft.speakerId);
    if (ft.runnerUpSpeakerId) speakerSet.add(ft.runnerUpSpeakerId);
  }
  const speakers = Array.from(speakerSet);
  const G = speakers.length;

  if (G === 0) return units;
  if (G === 1) {
    return units.map((u) => ({
      ...u,
      speakerId: speakers[0],
      runnerUpSpeakerId: speakers[0],
      confidence: 0.9,
      acousticMargin: 1.0,
    }));
  }

  const getLocalScd = (t: number, radius = 0.35): number => {
    let maxVal = 0.0;
    for (const ft of fineTurns) {
      if (Math.abs(ft.start - t) <= radius) {
        maxVal = Math.max(maxVal, ft.boundaryScdLeft || 0);
      }
      if (Math.abs(ft.end - t) <= radius) {
        maxVal = Math.max(maxVal, ft.boundaryScdRight || 0);
      }
    }
    return maxVal;
  };

  const unitSpeakerScores: Float64Array[] = new Array(L);

  for (let l = 0; l < L; l++) {
    const u = units[l];
    const scores = new Float64Array(G).fill(0.35);
    let weightSum = 0.0;

    for (const ft of fineTurns) {
      const ovStart = Math.max(u.start - 0.12, ft.start);
      const ovEnd = Math.min(u.end + 0.12, ft.end);
      if (ovEnd > ovStart) {
        const w = ovEnd - ovStart;
        weightSum += w;
        const conf = ft.speakerConfidence ?? 0.8;
        const margin = ft.speakerMargin ?? 0.12;
        const secondScore = Math.max(0.2, conf - margin);

        for (let g = 0; g < G; g++) {
          const spId = speakers[g];
          if (spId === ft.speakerId) {
            scores[g] += w * conf;
          } else if (spId === ft.runnerUpSpeakerId) {
            scores[g] += w * secondScore;
          } else {
            scores[g] += w * Math.max(0.1, secondScore - 0.12);
          }
        }
      }
    }

    if (weightSum > 0) {
      for (let g = 0; g < G; g++) {
        scores[g] = (scores[g] - 0.35) / weightSum;
      }
    } else {
      let nearest = fineTurns[0];
      let minGap = Infinity;
      for (const ft of fineTurns) {
        const gap = Math.min(
          Math.abs(u.center - ft.start),
          Math.abs(u.center - ft.end)
        );
        if (gap < minGap) {
          minGap = gap;
          nearest = ft;
        }
      }
      for (let g = 0; g < G; g++) {
        if (speakers[g] === nearest.speakerId) {
          scores[g] = nearest.speakerConfidence ?? 0.8;
        } else if (speakers[g] === nearest.runnerUpSpeakerId) {
          scores[g] = Math.max(
            0.25,
            (nearest.speakerConfidence ?? 0.8) - (nearest.speakerMargin ?? 0.12)
          );
        } else {
          scores[g] = 0.25;
        }
      }
    }

    unitSpeakerScores[l] = scores;
    u.boundaryScdLeft = getLocalScd(u.start, 0.28);
    u.boundaryScdRight = getLocalScd(u.end, 0.28);
  }

  const dp = new Float64Array(L * G).fill(Infinity);
  const backptr = new Int32Array(L * G).fill(0);

  const gamma = 13.5;
  const lambdaBase =
    engineMode === 'mode_0_heritage_classic'
      ? 2.1
      : engineMode === 'mode_a_classic'
      ? 4.2
      : 3.1;
  const alphaScd = 3.5;

  for (let l = 0; l < L; l++) {
    const u = units[l];
    const rawScores = unitSpeakerScores[l];

    let maxScaled = -Infinity;
    const scaled = new Float64Array(G);
    for (let g = 0; g < G; g++) {
      scaled[g] = gamma * rawScores[g];
      if (scaled[g] > maxScaled) maxScaled = scaled[g];
    }
    let sumExp = 0.0;
    for (let g = 0; g < G; g++) {
      sumExp += Math.exp(scaled[g] - maxScaled);
    }
    const logDenom = maxScaled + Math.log(sumExp);

    for (let g = 0; g < G; g++) {
      const emitCost = -(scaled[g] - logDenom);
      if (l === 0) {
        dp[g] = emitCost;
      } else {
        const prevU = units[l - 1];
        const pauseGap = Math.max(0.0, u.start - prevU.end);
        const boundaryScd = u.boundaryScdLeft || 0.0;

        let lingFactor = 1.0;
        if (/[。？！?!.]$/.test(prevU.text) || pauseGap >= 0.35) {
          lingFactor = 0.25;
        } else if (/[，、,;；]$/.test(prevU.text) || pauseGap >= 0.12) {
          lingFactor = 0.55;
        }

        const switchPenalty =
          lambdaBase * Math.exp(-alphaScd * boundaryScd) * lingFactor;

        let bestPrevCost = Infinity;
        let bestPrevG = 0;
        for (let prevG = 0; prevG < G; prevG++) {
          const transCost = prevG === g ? 0.0 : switchPenalty;
          const candCost = dp[(l - 1) * G + prevG] + transCost;
          if (candCost < bestPrevCost) {
            bestPrevCost = candCost;
            bestPrevG = prevG;
          }
        }
        dp[l * G + g] = emitCost + bestPrevCost;
        backptr[l * G + g] = bestPrevG;
      }
    }
  }

  const assignedIdx = new Int32Array(L);
  let bestLastG = 0;
  let minFinalCost = Infinity;
  for (let g = 0; g < G; g++) {
    if (dp[(L - 1) * G + g] < minFinalCost) {
      minFinalCost = dp[(L - 1) * G + g];
      bestLastG = g;
    }
  }
  assignedIdx[L - 1] = bestLastG;
  for (let l = L - 1; l > 0; l--) {
    assignedIdx[l - 1] = backptr[l * G + assignedIdx[l]];
  }

  const dualPeakMarginReq = Number(
    (backchannelMarginThreshold * 0.45).toFixed(3)
  );
  let idx = 0;
  while (idx < L) {
    let endIdx = idx;
    while (endIdx + 1 < L && assignedIdx[endIdx + 1] === assignedIdx[idx]) {
      endIdx++;
    }
    const turnDur = units[endIdx].end - units[idx].start;

    if (
      idx > 0 &&
      endIdx < L - 1 &&
      assignedIdx[idx - 1] === assignedIdx[endIdx + 1] &&
      turnDur < pingPongMaxDurSec
    ) {
      const spkA = assignedIdx[idx - 1];
      const spkB = assignedIdx[idx];

      let marginSum = 0.0;
      let combinedText = '';
      for (let k = idx; k <= endIdx; k++) {
        marginSum += unitSpeakerScores[k][spkB] - unitSpeakerScores[k][spkA];
        combinedText += units[k].text;
      }
      const avgMarginBA = marginSum / (endIdx - idx + 1);
      const dualScdPeakSum =
        (units[idx].boundaryScdLeft || 0) +
        (units[endIdx].boundaryScdRight || 0);
      const isLexicalBackchannel =
        MULTILINGUAL_BACKCHANNEL_REGEX.test(combinedText);

      const isTrueBackchannel =
        avgMarginBA >= backchannelMarginThreshold ||
        (avgMarginBA >= dualPeakMarginReq && dualScdPeakSum >= 0.5) ||
        isLexicalBackchannel;

      if (!isTrueBackchannel) {
        for (let k = idx; k <= endIdx; k++) {
          assignedIdx[k] = spkA;
        }
      }
    }
    idx = endIdx + 1;
  }

  for (let l = 0; l < L; l++) {
    const g1 = assignedIdx[l];
    const scores = unitSpeakerScores[l];
    const s1 = scores[g1];
    let g2 = g1;
    let s2 = -1.0;
    for (let g = 0; g < G; g++) {
      if (g !== g1 && scores[g] > s2) {
        s2 = scores[g];
        g2 = g;
      }
    }
    const margin = Math.max(0.0, s1 - (s2 >= 0 ? s2 : 0));
    units[l].speakerId = speakers[g1];
    units[l].runnerUpSpeakerId = speakers[g2];
    units[l].confidence = Number(Math.min(0.98, Math.max(0.55, s1)).toFixed(2));
    units[l].acousticMargin = Number(margin.toFixed(3));
  }

  for (let l = 1; l < L; l++) {
    const prev = units[l - 1];
    const curr = units[l];
    const currDur = curr.end - curr.start;
    const isLocked = (curr.acousticMargin || 0) >= 0.16 && currDur >= 1.2;

    const prevIsQuestion = MULTILINGUAL_QUESTION_END_REGEX.test(prev.text);
    const currIsShortAnswer = MULTILINGUAL_BACKCHANNEL_REGEX.test(curr.text);

    if (
      prevIsQuestion &&
      currIsShortAnswer &&
      curr.speakerId === prev.speakerId
    ) {
      if (curr.runnerUpSpeakerId && curr.runnerUpSpeakerId !== prev.speakerId) {
        curr.speakerId = curr.runnerUpSpeakerId;
      } else {
        const altSp = speakers.find((s) => s !== prev.speakerId);
        if (altSp) curr.speakerId = altSp;
      }
      continue;
    }

    const isRebuttalQuestion =
      /^(可是|不是[，]?|不過|しかし|ただ|but\s)/i.test(curr.text) &&
      MULTILINGUAL_QUESTION_END_REGEX.test(curr.text);
    if (
      !isLocked &&
      isRebuttalQuestion &&
      curr.speakerId === prev.speakerId &&
      curr.runnerUpSpeakerId &&
      curr.runnerUpSpeakerId !== prev.speakerId &&
      (curr.boundaryScdLeft || 0) >= 0.15
    ) {
      curr.speakerId = curr.runnerUpSpeakerId;
    }
  }

  return units;
}

function consolidateUnitsIntoSegments(
  units: AtomicTokenUnit[],
  chunkIndex: number,
  windowIndex: number,
  acousticThreshold: number
): Segment[] {
  const merged: AtomicTokenUnit[] = [];

  for (const item of units) {
    const prev = merged[merged.length - 1];
    if (
      prev &&
      prev.speakerId === item.speakerId &&
      item.start - prev.end <= 3.2 &&
      prev.text.length + item.text.length <= 260
    ) {
      const needsComma = !/[，。？！、；：,.?!]$/.test(prev.text);
      prev.text = (needsComma ? `${prev.text}，` : prev.text) + item.text;
      prev.end = item.end;
      prev.confidence = Math.max(
        prev.confidence || 0.85,
        item.confidence || 0.85
      );
      prev.acousticMargin = Math.max(
        prev.acousticMargin || 0.1,
        item.acousticMargin || 0.1
      );
    } else {
      merged.push({ ...item });
    }
  }

  return merged.map((m, idx) => {
    const segId = `seg_c${chunkIndex}_w${windowIndex}_s${idx}`;
    const spId = m.speakerId || 'Speaker_01';
    return {
      id: segId,
      chunkIndex,
      speakerId: spId,
      runnerUpSpeakerId: m.runnerUpSpeakerId || spId,
      acousticMargin: m.acousticMargin ?? 0.15,
      start: m.start,
      end: m.end,
      words: buildWordSpans(
        m.text,
        segId,
        m.start,
        m.end,
        m.confidence || 0.88,
        acousticThreshold
      ),
      isCorrected: false,
      relistenCount: 0,
    };
  });
}

self.onmessage = async (event: MessageEvent) => {
  const data = event.data;

  if (data.type === 'UNLOAD_MODEL' || data.type === 'DISPOSE') {
    if (asrInstance && typeof asrInstance.dispose === 'function') {
      try {
        await asrInstance.dispose();
      } catch {
        // ignore
      }
    }
    asrInstance = null;
    loadedModelKey = null;
    self.postMessage({ type: 'UNLOAD_DONE' });
    return;
  }

  /**
   * 【核心升級】語者專屬虛擬音軌拼接轉寫模式 (TRANSCRIBE_SPEAKER_TRACK_BATCH)
   */
  if (data.type === 'TRANSCRIBE_SPEAKER_TRACK_BATCH') {
    const {
      stitchedSamples,
      timeMap,
      speakerId,
      chunkIndex,
      batchIndex,
      modelName,
      languageHint = 'zh',
      acousticThreshold = 0.72,
    } = data as {
      stitchedSamples: Float32Array;
      timeMap: VirtualTrackMapItem[];
      speakerId: string;
      chunkIndex: number;
      batchIndex: number;
      modelName: LocalAsrModel;
      languageHint?: string;
      acousticThreshold?: number;
    };

    try {
      const transcriber = await ensureAsrPipeline(modelName);
      const audioInput =
        stitchedSamples instanceof Float32Array
          ? stitchedSamples
          : new Float32Array(stitchedSamples);

      if (audioInput.length < 1600 || !timeMap || timeMap.length === 0) {
        self.postMessage({
          type: 'SPEAKER_BATCH_TRANSCRIBED',
          segments: [],
          batchIndex,
        });
        return;
      }

      const resolvedWhisperLang =
        languageHint === 'en'
          ? 'english'
          : languageHint === 'ja'
          ? 'japanese'
          : 'chinese';

      const output = await transcriber(audioInput, {
        language: resolvedWhisperLang,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
      });

      const totalVSec = timeMap[timeMap.length - 1].virtualEndSec;
      const rawChunks: Array<{
        text: string;
        timestamp?: [number, number | null];
      }> =
        output && Array.isArray(output.chunks) && output.chunks.length > 0
          ? output.chunks
          : [
              {
                text: typeof output?.text === 'string' ? output.text : '',
                timestamp: [0, totalVSec],
              },
            ];

      const unwarpedSegments = unwarpVirtualTrackChunksToSegments(
        rawChunks,
        timeMap,
        speakerId,
        chunkIndex,
        batchIndex,
        acousticThreshold
      );

      self.postMessage({
        type: 'SPEAKER_BATCH_TRANSCRIBED',
        segments: unwarpedSegments,
        batchIndex,
      });
    } catch (err) {
      self.postMessage({
        type: 'ASR_ERROR',
        error: (err as Error).message || '語者虛擬音軌轉寫發生錯誤',
      });
    }
    return;
  }

  if (data.type === 'TRANSCRIBE_WINDOW_AND_ALIGN') {
    const {
      windowSamples,
      windowStartSec,
      windowEndSec,
      fineTurns,
      chunkIndex,
      windowIndex,
      modelName,
      languageHint = 'zh',
      acousticThreshold = 0.72,
      backchannelMarginThreshold = 0.12,
      pingPongMaxDurSec = 1.5,
      engineMode = 'mode_0_heritage_classic',
    } = data as {
      windowSamples: Float32Array;
      windowStartSec: number;
      windowEndSec: number;
      fineTurns: FineDiarizationTurn[];
      chunkIndex: number;
      windowIndex: number;
      modelName: LocalAsrModel;
      languageHint?: string;
      acousticThreshold?: number;
      asrBeamSize?: number;
      backchannelMarginThreshold?: number;
      pingPongMaxDurSec?: number;
      engineMode?: string;
    };

    try {
      const transcriber = await ensureAsrPipeline(modelName);
      const audioInput =
        windowSamples instanceof Float32Array
          ? windowSamples
          : new Float32Array(windowSamples);

      if (audioInput.length < 1600) {
        self.postMessage({
          type: 'WINDOW_TRANSCRIBED',
          segments: [],
          windowIndex,
        });
        return;
      }

      const resolvedWhisperLang =
        languageHint === 'en'
          ? 'english'
          : languageHint === 'ja'
          ? 'japanese'
          : 'chinese';

      const output = await transcriber(audioInput, {
        language: resolvedWhisperLang,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
      });

      const rawChunks: Array<{
        text: string;
        timestamp?: [number, number | null];
      }> =
        output && Array.isArray(output.chunks) && output.chunks.length > 0
          ? output.chunks
          : [
              {
                text: typeof output?.text === 'string' ? output.text : '',
                timestamp: [0, windowEndSec - windowStartSec],
              },
            ];

      const atomicUnits = decomposeChunksIntoAtomicUnits(
        rawChunks,
        windowStartSec,
        windowEndSec,
        fineTurns
      );
      const alignedUnits = alignUnitsWithScdViterbiAndPragmatics(
        atomicUnits,
        fineTurns,
        backchannelMarginThreshold,
        pingPongMaxDurSec,
        engineMode
      );
      const finalSegments = consolidateUnitsIntoSegments(
        alignedUnits,
        chunkIndex,
        windowIndex,
        acousticThreshold
      );

      self.postMessage({
        type: 'WINDOW_TRANSCRIBED',
        segments: finalSegments,
        windowIndex,
      });
    } catch (err) {
      self.postMessage({
        type: 'ASR_ERROR',
        error: (err as Error).message || '語音轉寫發生錯誤',
      });
    }
    return;
  }

  if (data.type === 'RELISTEN_SUB_SEGMENT') {
    const {
      subSamples,
      targetSegment,
      modelName,
      languageHint = 'zh',
    } = data as {
      subSamples: Float32Array;
      targetSegment: Segment;
      modelName: LocalAsrModel;
      languageHint?: string;
      promptHint?: string;
    };

    try {
      const transcriber = await ensureAsrPipeline(modelName);
      const audioInput =
        subSamples instanceof Float32Array
          ? subSamples
          : new Float32Array(subSamples);

      const resolvedWhisperLang =
        languageHint === 'en'
          ? 'english'
          : languageHint === 'ja'
          ? 'japanese'
          : 'chinese';

      // 針對已經過 enhanceSubSegmentAudioForRelisten (+12dB 弱音補償 + 2.8kHz 咬字增強 + 0.86x WSOLA 慢速展開) 的音訊，
      // 採用高解析度步長進行重聽解碼，抓出突然變小聲或太短促的漏字
      const output = await transcriber(audioInput, {
        language: resolvedWhisperLang,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 20,
        stride_length_s: 3,
      });

      const rawText = Array.isArray(output?.chunks)
        ? output.chunks
            .map((c: { text?: string }) => (c.text || '').trim())
            .filter(Boolean)
            .join('，')
        : typeof output?.text === 'string'
        ? output.text
        : '';

      const cleanText = restoreMultilingualPunctuation(rawText);
      // 對齊回原目標段落之真實會議時間區間 [targetSegment.start, targetSegment.end]
      const newWords = cleanText
        ? buildWordSpans(
            cleanText,
            `${targetSegment.id}_re`,
            targetSegment.start,
            targetSegment.end,
            0.95
          ).map((w) => ({
            ...w,
            status: 'auto_corrected' as const,
            reason: '經微觀弱音放大(+12dB)與慢速展開重聽補漏字校準',
          }))
        : null;

      self.postMessage({
        type: 'RELISTEN_SUCCESS',
        segmentId: targetSegment.id,
        newWords,
      });
    } catch (err) {
      self.postMessage({
        type: 'RELISTEN_ERROR',
        error: (err as Error).message,
      });
    }
  }
};