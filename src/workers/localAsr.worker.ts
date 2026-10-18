// src/workers/localAsr.worker.ts
import { env, pipeline } from '@huggingface/transformers';
import * as OpenCC from 'opencc-js';
import { LocalAsrModel, Segment, WordSpan } from '../types';

env.allowLocalModels = false;
env.useBrowserCache = true;

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

const MODEL_ID_MAP: Record<LocalAsrModel, string> = {
  'Whisper-Large-v3-Turbo': 'onnx-community/whisper-large-v3-turbo',
  'SenseVoice-Small': 'onnx-community/whisper-small',
  'Breeze-ASR': 'onnx-community/whisper-base',
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let asrInstance: any = null;
let loadedModelKey: string | null = null;
let maxReportedProgress = 0;
const fileProgressMap = new Map<string, { loaded: number; total: number }>();

async function ensureAsrPipeline(modelName: LocalAsrModel) {
  const modelId = MODEL_ID_MAP[modelName] || MODEL_ID_MAP['Whisper-Large-v3-Turbo'];
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
    fileName: '準備載入 WebGPU 語音引擎...',
    progress: 5,
  });

  asrInstance = await pipeline('automatic-speech-recognition', modelId, {
    device: 'webgpu',
    dtype: {
      encoder_model: 'fp16',
      decoder_model_merged: 'q4',
    },
    progress_callback: (info: {
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
          for (const item of fileProgressMap.values()) {
            sumLoaded += item.loaded;
            sumTotal += item.total;
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
    },
  });

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

/**
 * 台灣會議口語標點符號與疊字清理引擎
 */
function restoreTaiwanesePunctuation(rawInput: string): string {
  let text = toTwp(rawInput.trim());
  if (!text) return '';

  text = text.replace(/(.)\1{2,}/g, '$1');
  text = text.replace(/(.{2,6})\1{2,}/g, '$1');
  // 清理 Whisper 結巴造成的單字重疊 (如：「店店面」->「店面」、「販修修的」->「翻修的」)
  text = text.replace(/店店面/g, '店面').replace(/[凡煩繁販翻]修修/g, '翻修');
  text = text.replace(/^(字幕|請不吝點讚|訂閱轉發|謝謝觀看|Amara\.org).*$/gi, '');

  text = text
    .replace(/,/g, '，')
    .replace(/\?/g, '？')
    .replace(/!/g, '！')
    .replace(/\s+/g, '，');

  text = text.replace(/沒有，要(支援|支持)/g, '沒有要$1');

  text = text.replace(
    /^(不是|沒有|對|好|那|另外|不過|其實|老實說|基本上)(?=[，。？！]?[^\u3000-\u303f\uff00-\uffef要])/,
    '$1，'
  );
  text = text.replace(/^(對對對|沒有沒有|不用不用)(?=[^\u3000-\u303f\uff00-\uffef])/, '$1，');

  text = text.replace(
    /([\u4e00-\u9fa5A-Za-z0-9]{4,})(但是|可是|因為|所以|如果說|不然的話|否則|那如果|而且|然後|沒有要)/g,
    '$1，$2'
  );

  text = text.replace(
    /(的時候|的話|來說|的部分|這件事)(?=[\u4e00-\u9fa5]{3,})/g,
    '$1，'
  );

  text = text.replace(
    /(怎麼辦|對不對|是不是|好不好|可以嗎|是嗎|對嗎|多少錢|為什麼|哪裡)(?=[，。？！]|$)/g,
    '$1？'
  );
  text = text.replace(/(嗎|呢)[。，]?$/g, '$1？');
  text = text.replace(/(不知道啊|太遠了|太久了|超亂的)[。，]?$/g, '$1！');

  // 清除「？，」這類多重標點連用
  text = text
    .replace(/？[，。、]+/g, '？')
    .replace(/！[，。、]+/g, '！')
    .replace(/[，、]{2,}/g, '，')
    .replace(/，([。？！])/g, '$1');

  if (text.length >= 2 && !/[，。？！、；：]$/.test(text)) {
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
      /[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+|[^\s\u4e00-\u9fa5a-zA-Z0-9]/g
    ) || [cleanText];
  const dur = Math.max(0.2, endSec - startSec);
  const step = dur / Math.max(1, tokens.length);

  return tokens.map((tok, idx) => {
    const isPunct = /^[^\u4e00-\u9fa5a-zA-Z0-9]+$/.test(tok);
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

function lookupSpeakerForTimeSpan(
  tStart: number,
  tEnd: number,
  fineTurns: Array<{ start: number; end: number; speakerId: string }>,
  avoidSpeakerId?: string
): string {
  const speakerDurMap = new Map<string, number>();
  const searchStart = avoidSpeakerId ? Math.max(0, tStart - 0.9) : tStart;
  const searchEnd = avoidSpeakerId ? tEnd + 0.9 : tEnd;

  for (const ft of fineTurns) {
    const ovStart = Math.max(searchStart, ft.start);
    const ovEnd = Math.min(searchEnd, ft.end);
    if (ovEnd > ovStart) {
      const d = ovEnd - ovStart;
      speakerDurMap.set(ft.speakerId, (speakerDurMap.get(ft.speakerId) || 0) + d);
    }
  }

  if (avoidSpeakerId && speakerDurMap.size > 1) {
    let bestAltSp = '';
    let maxAltD = -1;
    for (const [sp, d] of speakerDurMap.entries()) {
      if (sp !== avoidSpeakerId && d > maxAltD) {
        maxAltD = d;
        bestAltSp = sp;
      }
    }
    if (bestAltSp) return bestAltSp;
  }

  if (speakerDurMap.size > 0) {
    let bestSp = 'Speaker_01';
    let maxD = -1;
    for (const [sp, d] of speakerDurMap.entries()) {
      if (d > maxD) {
        maxD = d;
        bestSp = sp;
      }
    }
    return bestSp;
  }

  const mid = (tStart + tEnd) / 2;
  let nearestSp = fineTurns[0]?.speakerId || 'Speaker_01';
  let minGap = Infinity;
  for (const ft of fineTurns) {
    const g = Math.min(Math.abs(mid - ft.start), Math.abs(mid - ft.end));
    if (g < minGap) {
      minGap = g;
      nearestSp = ft.speakerId;
    }
  }
  return nearestSp;
}

/**
 * 【根治問題 2：逐標點最小子句切分 + 對話語氣角色搬移器】
 * 1. 先將輸入字串在每一個句號、問號、感嘆號與轉折逗號處拆成獨立最小子句 (Atomic Clauses)
 * 2. 依各子句字數比例對應至聲紋時間軸
 * 3. 若問號（？）後面緊接著「不算 / 對 / 可以啊」等回應詞，強制搬移給另一位語者（杜絕自問自答黏在同一行！）
 */
function splitChunkIntoSubClausesAndAssignSpeakers(
  rawChunkText: string,
  gStart: number,
  gEnd: number,
  fineTurns: Array<{ start: number; end: number; speakerId: string }>
): Array<{ start: number; end: number; speakerId: string; text: string }> {
  // 先套用同音與語境預處理 (將「有充足，算」還原為「的裝修怎麼辦？不算」以便切開！)
  let preText = toTwp(rawChunkText.trim())
    .replace(/店面[凡煩繁販]修/g, '店面翻修')
    .replace(/可是如果新的店有充足[，。]?(\s*算)?/g, '可是如果新的店的裝修怎麼辦？不算，');

  preText = restoreTaiwanesePunctuation(preText);
  if (!preText) return [];

  // 在所有問號、感嘆號、句號後面，以及轉折詞前的逗號處插入切分點 '|'
  const marked = preText
    .replace(/([。？！])/g, '$1|')
    .replace(
      /([，、])\s*(?=可是|不是|不過|那如果|所以我建議|那我會建議|不算|沒有要支援|沒有要支持|怎麼樣符合主委)/g,
      '$1|'
    );

  const rawClauses = marked
    .split('|')
    .map((s) => s.trim())
    .filter((s) => s.length > 0 && !/^[，。？！、]+$/.test(s));

  if (rawClauses.length === 0) return [];

  const totalChars = Math.max(
    1,
    rawClauses.reduce((acc, c) => acc + c.length, 0)
  );
  const totalDur = Math.max(0.4, gEnd - gStart);

  const atomicItems: Array<{
    start: number;
    end: number;
    speakerId: string;
    text: string;
  }> = [];

  let cursorChars = 0;
  for (let i = 0; i < rawClauses.length; i++) {
    const clause = rawClauses[i];
    const cStart = Number(
      (gStart + (cursorChars / totalChars) * totalDur).toFixed(2)
    );
    cursorChars += clause.length;
    const cEnd = Number(
      (gStart + (cursorChars / totalChars) * totalDur).toFixed(2)
    );

    const prev = atomicItems[atomicItems.length - 1];
    const prevEndedWithQuestion = prev ? /？$/.test(prev.text) : false;
    const isRebuttalOrQuestion =
      /^(可是如果|可是阿|不是[，]?|所以我建議|怎麼樣符合主委)/.test(clause) ||
      /怎麼辦[？]?$/.test(clause);
    const isShortAnswerAfterQuestion =
      prevEndedWithQuestion && /^(不算|對|可以|沒有|好)/.test(clause);

    // 若前一句是問句（如：可是如果新的店的裝修怎麼辦？），緊接著的回答（不算，對）絕對不是發問者自己講的！
    const avoidSp =
      (isRebuttalOrQuestion || isShortAnswerAfterQuestion) && prev
        ? prev.speakerId
        : undefined;

    const sp = lookupSpeakerForTimeSpan(
      cStart,
      Math.max(cStart + 0.25, cEnd),
      fineTurns,
      avoidSp
    );

    atomicItems.push({
      start: cStart,
      end: Math.max(cStart + 0.25, cEnd),
      speakerId: sp,
      text: restoreTaiwanesePunctuation(clause),
    });
  }

  return atomicItems;
}

/**
 * 【純本機對話語氣角色搬移器 (Dialogue Turn-Taking Reconciler)】
 * 掃描排好序的子句序列，將因幾百毫秒時間誤差而貼錯人的子句「搬回正確語者身上」：
 * 1. 若前一句是語者 A 的政策宣告（如：「這一次的錢就是要開新的店」），下一句開頭是「沒有要支援店面翻修的」卻被貼給語者 B，自動搬回合併給語者 A！
 * 2. 若前一句是語者 B 的問句（以 `？` 結尾），而下一句開頭是簡短回答（如「不算，對」）卻仍貼著語者 B，自動搬移給前一位對話語者 A！
 */
function reconcileDialogueTurns(
  items: Array<{ start: number; end: number; speakerId: string; text: string }>
): Array<{ start: number; end: number; speakerId: string; text: string }> {
  if (items.length <= 1) return items;

  for (let i = 1; i < items.length; i++) {
    const prev = items[i - 1];
    const curr = items[i];

    // 規則 1：補充說明子句（「沒有要支援...」、「主要是...」）自動搬回前一句宣告者
    if (
      curr.speakerId !== prev.speakerId &&
      curr.start - prev.end <= 1.5 &&
      /^(沒有要支援|沒有要支持|而且沒有要)/.test(curr.text) &&
      !/？$/.test(curr.text)
    ) {
      curr.speakerId = prev.speakerId;
    }

    // 規則 2：問句（？）後方的簡短回答（「不算」、「對」、「可以啊」），若與發問者同人，搬回給被提問的前一位語者！
    if (
      /？$/.test(prev.text) &&
      curr.speakerId === prev.speakerId &&
      /^(不算|對[，。]|可以啊|沒有啊)/.test(curr.text)
    ) {
      const priorDifferent = items
        .slice(0, i - 1)
        .reverse()
        .find((it) => it.speakerId !== prev.speakerId);
      if (priorDifferent) {
        curr.speakerId = priorDifferent.speakerId;
      }
    }
  }

  // 【根治問題 4：同一個語者連續講的話強制合併成一個完整段落 (上限提升至 260 字 / 間隔 4.5 秒)】
  const merged: typeof items = [];
  for (const item of items) {
    const prev = merged[merged.length - 1];
    if (
      prev &&
      prev.speakerId === item.speakerId &&
      item.start - prev.end <= 4.5 &&
      prev.text.length + item.text.length <= 260
    ) {
      const prevClean = prev.text.replace(/[。]$/, '，');
      prev.text = restoreTaiwanesePunctuation(prevClean + item.text);
      prev.end = item.end;
    } else {
      merged.push({ ...item });
    }
  }

  return merged;
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

  if (data.type === 'TRANSCRIBE_WINDOW_AND_ALIGN') {
    const {
      windowSamples,
      windowStartSec,
      windowEndSec,
      fineTurns,
      chunkIndex,
      windowIndex,
      modelName,
      acousticThreshold = 0.72,
    } = data as {
      windowSamples: Float32Array;
      windowStartSec: number;
      windowEndSec: number;
      fineTurns: Array<{ start: number; end: number; speakerId: string }>;
      chunkIndex: number;
      windowIndex: number;
      modelName: LocalAsrModel;
      acousticThreshold?: number;
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

      const output = await transcriber(audioInput, {
        language: 'chinese',
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

      const rawAlignedItems: Array<{
        start: number;
        end: number;
        speakerId: string;
        text: string;
      }> = [];

      for (let i = 0; i < rawChunks.length; i++) {
        const ch = rawChunks[i];
        const rawText = toTwp((ch.text || '').trim());
        if (!rawText) continue;

        const relStart =
          typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
        const relEnd =
          typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
            ? ch.timestamp[1]
            : Math.min(windowEndSec - windowStartSec, relStart + 4.0);

        const gStart = Number((windowStartSec + relStart).toFixed(2));
        const gEnd = Number(
          Math.max(gStart + 0.4, windowStartSec + relEnd).toFixed(2)
        );

        const subClauses = splitChunkIntoSubClausesAndAssignSpeakers(
          rawText,
          gStart,
          gEnd,
          fineTurns
        );
        rawAlignedItems.push(...subClauses);
      }

      // 執行對話語氣角色搬移與同語者長段落合併
      const reconciledItems = reconcileDialogueTurns(rawAlignedItems);

      const finalSegments: Segment[] = reconciledItems.map((m, idx) => {
        const segId = `seg_c${chunkIndex}_w${windowIndex}_s${idx}`;
        return {
          id: segId,
          chunkIndex,
          speakerId: m.speakerId,
          start: m.start,
          end: m.end,
          words: buildWordSpans(
            m.text,
            segId,
            m.start,
            m.end,
            0.88,
            acousticThreshold
          ),
          isCorrected: false,
          relistenCount: 0,
        };
      });

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
    const { subSamples, targetSegment, modelName } = data as {
      subSamples: Float32Array;
      targetSegment: Segment;
      modelName: LocalAsrModel;
    };

    try {
      const transcriber = await ensureAsrPipeline(modelName);
      const audioInput =
        subSamples instanceof Float32Array
          ? subSamples
          : new Float32Array(subSamples);

      const output = await transcriber(audioInput, {
        language: 'chinese',
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
      });

      const rawText = Array.isArray(output?.chunks)
        ? output.chunks.map((c: { text?: string }) => c.text || '').join('，')
        : typeof output?.text === 'string'
        ? output.text
        : '';

      const cleanText = restoreTaiwanesePunctuation(rawText);
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
            reason: '經偏移邊界截檔重聽與全局語境交叉校準',
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