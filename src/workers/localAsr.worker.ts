// src/workers/localAsr.worker.ts
// 次世代本地 WebGPU Whisper 語音轉寫與聲學換人錨點耦合引擎
// 修正重點：
// 1. 嚴格保留 probeWebGpuCapabilities() 鐵律（不支援 shader-f16 時自動降級 encoder_model: 'fp32', decoder_model_merged: 'q4'）
// 2. 針對 Whisper 輸出之子句或字詞 Chunks，內建「語用轉折/搶話拆句 (Discourse Clause Splitting) + 聲學換人切點吸附 (Acoustic Switch-Point Snapping)」，
//    徹底解決 Whisper 將「不是，啊如果新的店的裝修怎麼辦？不算，不算。」或「...你這樣寫的時候如果剛好就只要限定開新店就好了」黏在同一 Chunk 時的時間錯位問題！

import { env, pipeline } from '@huggingface/transformers';
import * as OpenCC from 'opencc-js';
import {
  FineDiarizationTurn,
  FrameSpeakerPosteriorMatrix,
  LocalAsrModel,
  Segment,
  VirtualTrackMapItem,
  WordSpan,
} from '../types';
import {
  decodeWordLevelSpeakersViterbi,
  groupAlignedWordsIntoSegments,
  RawWordTimestampItem,
  synthesizePosteriorMatrixFromFineTurns,
} from '../utils/viterbiWordAligner';

env.allowLocalModels = false;
env.useBrowserCache = true;

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

const MODEL_ID_MAP: Record<LocalAsrModel, string> = {
  'Whisper-Large-v3-Turbo': 'onnx-community/whisper-large-v3-turbo',
  'SenseVoice-Small': 'onnx-community/whisper-small',
  'Breeze-ASR': 'onnx-community/whisper-base',
};

const DEFAULT_ALIGNMENT_HEADS: Record<LocalAsrModel, number[][]> = {
  'Whisper-Large-v3-Turbo': [
    [2, 4],
    [2, 11],
    [3, 3],
    [3, 6],
    [3, 11],
    [3, 14],
  ],
  'SenseVoice-Small': [
    [5, 3],
    [5, 9],
    [8, 0],
    [8, 4],
    [8, 7],
    [8, 8],
    [9, 0],
    [9, 7],
    [9, 9],
    [10, 5],
  ],
  'Breeze-ASR': [
    [3, 1],
    [4, 2],
    [4, 3],
    [4, 7],
    [5, 1],
    [5, 2],
    [5, 4],
    [5, 6],
  ],
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let asrInstance: any = null;
let loadedModelKey: string | null = null;
let maxReportedProgress = 0;
const fileProgressMap = new Map<string, { loaded: number; total: number }>();

/**
 * 探測當前瀏覽器與顯卡是否支援 WebGPU 及 `shader-f16` (fp16) 擴充功能。
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

function ensureAlignmentHeadsPatched(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transcriber: any,
  modelName: LocalAsrModel
): void {
  try {
    const genCfg = transcriber?.model?.generation_config;
    if (
      genCfg &&
      (!Array.isArray(genCfg.alignment_heads) ||
        genCfg.alignment_heads.length === 0)
    ) {
      genCfg.alignment_heads =
        DEFAULT_ALIGNMENT_HEADS[modelName] ||
        DEFAULT_ALIGNMENT_HEADS['Whisper-Large-v3-Turbo'];
    }
  } catch {
    // ignore
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function releaseResidualGpuTensors(outputObj: any): void {
  try {
    if (!outputObj || typeof outputObj !== 'object') return;
    for (const key of Object.keys(outputObj)) {
      const val = outputObj[key];
      if (val && typeof val.dispose === 'function') {
        val.dispose();
      }
    }
  } catch {
    // ignore
  }
}

async function ensureAsrPipeline(modelName: LocalAsrModel) {
  const modelId =
    MODEL_ID_MAP[modelName] || MODEL_ID_MAP['Whisper-Large-v3-Turbo'];
  if (asrInstance && loadedModelKey === modelId) {
    ensureAlignmentHeadsPatched(asrInstance, modelName);
    return asrInstance;
  }

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

  ensureAlignmentHeadsPatched(asrInstance, modelName);

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

function estimateRepetitionCompressionRatio(text: string): number {
  const clean = text.replace(/\s+/g, '').trim();
  if (clean.length < 8) return 1.0;

  // 偵測無標點或跨標點之重複跳針迴圈 (如「謝謝。謝謝。謝謝。謝謝。」)
  if (
    /([\u4e00-\u9fa5a-zA-Z0-9]{1,6})(?:[，。？！、；：,.?!\s]*\1){3,}/i.test(
      clean
    )
  ) {
    return 3.2;
  }

  const pureChars = clean.replace(/[，。？！、；：,.?!]/g, '');
  if (pureChars.length < 8) return 1.0;

  const bigrams = new Set<string>();
  for (let i = 0; i < pureChars.length - 1; i++) {
    bigrams.add(pureChars.slice(i, i + 2));
  }
  const uniqueRatio = bigrams.size / Math.max(1, pureChars.length - 1);
  return Number((1.0 / Math.max(0.15, uniqueRatio)).toFixed(2));
}

function restoreMultilingualPunctuation(rawInput: string): string {
  const hasKana = /[\u3040-\u30ff]/.test(rawInput);
  let text = hasKana ? rawInput.trim() : toTwp(rawInput.trim());
  if (!text) return '';

  // 1. 消除單字或詞組連續無標點重複、以及跨標點跳針重複 (如紙張摩擦造成之「常。謝謝。謝謝。謝謝。...」)
  text = text.replace(/([\u4e00-\u9fa5])\1{2,}/g, '$1');
  text = text.replace(/([\u4e00-\u9fa5]{2,6})\1{2,}/g, '$1');
  text = text.replace(
    /([\u4e00-\u9fa5]{1,6})(?:[，。？！、；：,.?!\s]+\1){2,}/g,
    '$1'
  );
  // 若句首因摩擦雜音殘留單一無意義孤字緊接句號與短詞 (如「常。謝謝。」)，自動剝除該雜訊孤字
  text = text.replace(/^[常嗯恩啊喔欸][。，、,.]\s*(?=謝謝|好|對|可以)/, '');
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

  // 依據通用華語/日語/英語句法連接詞補入子句逗號
  text = text.replace(
    /([\u4e00-\u9fa5A-Za-z0-9]{4,})(但是|可是|不過|因為|所以|如果說|不然的話|否則|那如果|而且|然後|ですが|けれども|しかし|ただ)/g,
    '$1，$2'
  );

  text = text.replace(
    /(的時候|的話|來說|的部分|這件事)(?=[\u4e00-\u9fa5]{3,})/g,
    '$1，'
  );

  text = text.replace(
    /(怎麼辦|對不對|是不是|好不好|可以嗎|是嗎|對嗎|多少錢|為什麼|哪裡|でしょうか|ますか|ですか)(?=[，。？?]|$)/g,
    '$1？'
  );
  text = text.replace(/(嗎|呢)[。，？?]*$/g, '$1？');

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

/**
 * 通用語法起手式字元正則（僅用於當聲學軌道偵測到句中無標點換人時，在換人時間點 ±3 字內對齊最自然的詞語邊界，避免把單一詞彙切開）
 */
/**
 * 僅保留真正能作為新子句主詞或轉折連詞之起手標記
 * 嚴格排除「他、她、好、對、算」等極易作為前句「動賓受詞 (寄給他)」或「程度補語 (都很好)」的單字！
 */
const UNIVERSAL_CLAUSE_STARTER_REGEX =
  /^(我們|你們|他們|這個|那個|這樣|那樣|但是|可是|不過|因為|所以|如果|要是|不是|沒有|不用|不然|否則|其實|只是|那邊|這邊|現在|後來|還是|或者|而且|然後|就|那)/;

/**
 * 檢查切點前後是否處於不可切斷之「強語法價鍵結構 (Syntactic Valence Bound Pair)」：
 * - 動賓/介賓結構：如「寄給｜他」、「提醒｜他們」、「告訴｜我」
 * - 狀中程度結構：如「都很｜好」、「非常｜好」、「不｜對」
 * - 詞幹+句末助詞：如「...｜的」、「...｜了」、「...｜嗎」
 */
function isSyntacticallyInseparableBoundary(
  leftStr: string,
  rightStr: string
): boolean {
  const lClean = leftStr.replace(/[，。？！、；：,.?!\s]+$/g, '');
  const rClean = rightStr.replace(/^[，。？！、；：,.?!\s]+/g, '');
  if (!lClean || !rClean) return false;
  if (/[，。？！、；：,.?!]$/.test(leftStr.trim())) return false;

  // 1. 動詞/介詞 + 代名詞受詞 (如：寄給+他、提醒+他們、交給+你)
  if (
    /[給寄送交找問請跟對向讓叫幫替被把為愛看聽說告訴通知聯絡協助感謝謝謝陪帶]$/.test(
      lClean
    ) &&
    /^(我|你|您|他|她|它|我們|你們|他們|她們|大家|誰|自己)/.test(rClean)
  ) {
    return true;
  }

  // 2. 程度/範圍/否定副詞 + 形容詞/助動詞補語 (如：都很+好、非常+好、太+遠、不+對)
  if (
    /(很|太|更|最|超|蠻|挺|都|還|不|沒|別|真|非常|相當|特別|十分|比較|稍微|有點|已經)$/.test(
      lClean
    ) &&
    /^(好|對|棒|行|可以|錯|大|小|多|少|高|低|快|慢|遠|近|久|難|易|重要|清楚|可能|確定|喜歡|想要|需要|應該|願意)/.test(
      rClean
    )
  ) {
    return true;
  }

  // 3. 句末助詞、時態助詞或複數詞尾 (的/得/地/了/嗎/呢/吧/啊/呀/喔/啦/過/著/們)
  if (
    /^(的|得|地|了|嗎|呢|吧|啊|呀|喔|哦|啦|囉|嘛|齁|吼|過|著|們)([，。？！、；：,.?!\s]|$)/.test(
      rClean
    )
  ) {
    return true;
  }

  // 4. 數詞/指示詞 + 量詞 (如：一+個、這+個、那+些)
  if (
    /[一二三四五六七八九十百千萬兩這那某每各幾]$/.test(lClean) &&
    /^(個|位|名|次|件|種|些|點|處|家|間|筆|項|套|場|人|天|年|月|週)/.test(
      rClean
    )
  ) {
    return true;
  }

  return false;
}

/**
 * 依據通用標點與語法邊界將文字拆解為基本子句 (Universal Discourse Clauses)
 * 完全不依賴任何特定會議詞彙，僅在「句末標點 (。？！)」、「句末或句尾逗號後之獨立短應答詞」與「長句逗號後之通用轉折連接詞」處切分。
 */
function splitTextIntoDiscourseClauses(cleanClause: string): string[] {
  let t = cleanClause;
  // 1. 在句號、問號、感嘆號後切開
  t = t.replace(/([。？！?!])/g, '$1|');
  // 2. 在句末終止符後緊接之通用獨立短應答詞 (如「？|不算，」、「。|對，」) 後切開
  t = t.replace(
    /([|。？！?!]\s*(?:不算|算|對|對啊|對對對|好|好的|可以|沒錯|沒有|不是|了解|はい|ええ|うん|なるほど|yeah|yes|no|right|sure|ok|okay)[，、,])\s*/gi,
    '$1|'
  );
  // 3. 當長子句 (>=4字) 以逗號結尾，且後方緊接句尾獨立短應答詞 (如「神秘如果我們再寄給他，好。」) 時，在逗號後切開！
  t = t.replace(
    /([\u4e00-\u9fa5]{4,}[，、,])\s*(?=(?:好|好的|好啊|對|對啊|對對對|可以|沒錯|算|不算|了解|ok|okay)[。？！?!|\s]*$)/gi,
    '$1|'
  );
  // 4. 在長子句 (>=5字) 逗號後若緊接通用轉折/條件連詞切開
  t = t.replace(
    /([\u4e00-\u9fa5]{5,}[，、,])\s*(?=不是[，]|可是|不過|但是|然而|那如果|不然|否則|はい|しかし|ただ|but\s|however)/gi,
    '$1|'
  );

  return t
    .split('|')
    .map((p) => p.trim())
    .filter((p) => p.length > 0 && !/^[，。？！、,.?!]+$/.test(p));
}

/**
 * 防止奇數字長中文子句在句尾留下「單字孤兒 Token」(例如 13 字「但我覺得他給團隊提醒都很好，」切成 2*6+1 導致最後一字「好，」獨立成單字 Token 而往下掉)
 * 若子句最末 Token 為單一漢字，自動與前一個雙字 Token 合併為穩固的 3 字收束詞組 (如「都很好，」、「再寄給他。」)！
 */
function tokenizeClausePreventingTailOrphan(cText: string): string[] {
  const rawTokens =
    cText.match(
      /[\u4e00-\u9fa5]{1,2}[，。？！、；：,.?!]*|[\u3040-\u30ff]{1,3}[，。？！、；：,.?!]*|[a-zA-Z0-9']+[，。？！、；：,.?!]*|[，。？！、；：,.?!]+/g
    ) || [cText];

  const cleaned: string[] = [];
  for (const t of rawTokens) {
    const st = t.trim();
    if (!st) continue;
    if (/^[，。？！、；：,.?!]+$/.test(st) && cleaned.length > 0) {
      if (!/[，。？！、；：,.?!]$/.test(cleaned[cleaned.length - 1])) {
        cleaned[cleaned.length - 1] += st[0];
      }
      continue;
    }
    cleaned.push(st);
  }

  if (cleaned.length >= 2) {
    const last = cleaned[cleaned.length - 1];
    const prev = cleaned[cleaned.length - 2];
    const lastPure = last.replace(/[，。？！、；：,.?!\s]/g, '');
    const prevPure = prev.replace(/[，。？！、；：,.?!\s]/g, '');
    if (
      lastPure.length === 1 &&
      /^[\u4e00-\u9fa5]$/.test(lastPure) &&
      !/[，。？！、；：,.?!]$/.test(prev) &&
      /^[\u4e00-\u9fa5]{1,2}$/.test(prevPure)
    ) {
      cleaned.splice(cleaned.length - 2, 2, prev + last);
    }
  }

  return cleaned;
}

/**
 * 【通用字元級聲學換人切點投影 (Character-Level Acoustic Switch Projection)】：
 * 當某位與會者在另一人講話中途無停頓打斷搶話，導致 Whisper 輸出的單一子句 [cStart, cEnd] 內部沒有標點、
 * 卻跨越了 `fineTurns` 偵測到的真實聲學換人時間點 `midClauseSwitches` 時：
 * 1. 依字元數比例將聲學換人時間 `st` 映射為子句內的字元位置 `rawIdx`。
 * 2. 在 `rawIdx ± 3` 字範圍內尋找逗號或通用語法起手詞邊界，並嚴格避開「動賓 (寄給他) / 狀中 (都很好)」強約束結構！
 */
function splitClauseAtAcousticSwitches(
  cText: string,
  cStart: number,
  cEnd: number,
  midClauseSwitches: number[]
): Array<{ text: string; start: number; end: number; isAcousticSplit: boolean }> {
  if (midClauseSwitches.length === 0 || cText.length < 8) {
    return [{ text: cText, start: cStart, end: cEnd, isAcousticSplit: false }];
  }
  const chars = Array.from(cText);
  const L = chars.length;
  const dur = Math.max(0.2, cEnd - cStart);

  const splitPoints: Array<{ charIdx: number; timeSec: number }> = [];
  for (const st of midClauseSwitches) {
    const rel = Math.max(0.18, Math.min(0.82, (st - cStart) / dur));
    const rawIdx = Math.round(rel * L);
    let bestIdx = -1;
    let bestScore = -5.0;

    for (
      let idx = Math.max(3, rawIdx - 3);
      idx <= Math.min(L - 3, rawIdx + 3);
      idx++
    ) {
      const leftPart = chars.slice(0, idx).join('');
      const restStr = chars.slice(idx).join('');
      if (isSyntacticallyInseparableBoundary(leftPart, restStr)) {
        continue;
      }

      const distPen = Math.abs(idx - rawIdx) * 0.4;
      const prevChar = chars[idx - 1];
      let score = -distPen;
      if (/[，、；：,:;\s]/.test(prevChar)) {
        score += 3.5;
      } else if (UNIVERSAL_CLAUSE_STARTER_REGEX.test(restStr)) {
        score += 2.2;
      }
      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
      }
    }
    // 僅在確實找到合法語法邊界 (逗號或明確起手連詞，且非動賓/狀中結構) 時才執行句中無標點切分，防止誤切同一人的連續句子！
    if (bestIdx >= 3 && bestScore >= 0.4) {
      splitPoints.push({ charIdx: bestIdx, timeSec: st });
    }
  }

  splitPoints.sort((a, b) => a.charIdx - b.charIdx);
  const result: Array<{
    text: string;
    start: number;
    end: number;
    isAcousticSplit: boolean;
  }> = [];
  let prevCharIdx = 0;
  let prevTime = cStart;

  for (const sp of splitPoints) {
    if (sp.charIdx - prevCharIdx < 2 || L - sp.charIdx < 2) continue;
    const subStr = chars
      .slice(prevCharIdx, sp.charIdx)
      .join('')
      .replace(/^[，、\s]+/, '')
      .trim();
    if (subStr) {
      result.push({
        text: subStr,
        start: prevTime,
        end: sp.timeSec,
        isAcousticSplit: prevCharIdx > 0,
      });
    }
    prevCharIdx = sp.charIdx;
    prevTime = sp.timeSec;
  }

  const tailStr = chars
    .slice(prevCharIdx)
    .join('')
    .replace(/^[，、\s]+/, '')
    .trim();
  if (tailStr) {
    result.push({
      text: tailStr,
      start: prevTime,
      end: cEnd,
      isAcousticSplit: prevCharIdx > 0,
    });
  }

  return result.length > 0
    ? result
    : [{ text: cText, start: cStart, end: cEnd, isAcousticSplit: false }];
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

async function runWhisperWithWordTimestampsAndFallback(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transcriber: any,
  audioInput: Float32Array,
  resolvedWhisperLang: string,
  chunkLengthSec = 30,
  strideLengthSec = 5
): Promise<{
  chunks: Array<{ text: string; timestamp?: [number, number | null] }>;
  isWordLevel: boolean;
}> {
  let output: {
    text?: string;
    chunks?: Array<{ text: string; timestamp?: [number, number | null] }>;
  } | null = null;
  let isWordLevel = false;

  // 先使用子句級時間戳記 (return_timestamps: true) 確保標點符號與完整語意結構不流失，
  // 若單一 chunk 過長或無標點則結合聲學換人切點對齊！
  try {
    output = await transcriber(audioInput, {
      language: resolvedWhisperLang,
      task: 'transcribe',
      return_timestamps: true,
      chunk_length_s: chunkLengthSec,
      stride_length_s: strideLengthSec,
    });
    isWordLevel = false;
  } catch {
    output = null;
  }

  if (!output || !Array.isArray(output.chunks) || output.chunks.length === 0) {
    try {
      output = await transcriber(audioInput, {
        language: resolvedWhisperLang,
        task: 'transcribe',
        return_timestamps: 'word',
        chunk_length_s: chunkLengthSec,
        stride_length_s: strideLengthSec,
      });
      if (output && Array.isArray(output.chunks) && output.chunks.length > 0) {
        isWordLevel = true;
      }
    } catch {
      // ignore
    }
  }

  const fullText =
    typeof output?.text === 'string'
      ? output.text
      : (output?.chunks || []).map((c) => c.text).join('');
  const compRatio = estimateRepetitionCompressionRatio(fullText);

  if (compRatio > 2.4) {
    console.warn(
      `⚠️ [LocalASR] 偵測到壓縮比異常 (${compRatio} > 2.4)，啟動防跳針退火重試...`
    );
    releaseResidualGpuTensors(output);
    try {
      const retryOut = await transcriber(audioInput, {
        language: resolvedWhisperLang,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: Math.min(20, chunkLengthSec),
        stride_length_s: 3,
        temperature: 0.2,
      });
      if (
        retryOut &&
        Array.isArray(retryOut.chunks) &&
        retryOut.chunks.length > 0
      ) {
        output = retryOut;
        isWordLevel = false;
      }
    } catch {
      // keep first output
    }
  }

  const rawChunks: Array<{
    text: string;
    timestamp?: [number, number | null];
  }> =
    output && Array.isArray(output.chunks) && output.chunks.length > 0
      ? output.chunks.map((c) => ({
          text: c.text || '',
          timestamp: c.timestamp
            ? [c.timestamp[0], c.timestamp[1]]
            : undefined,
        }))
      : [
          {
            text: typeof output?.text === 'string' ? output.text : '',
            timestamp: [0, audioInput.length / 16000],
          },
        ];

  releaseResidualGpuTensors(output);

  return {
    chunks: rawChunks,
    isWordLevel,
  };
}

/**
 * 【核心修正 7】：將 Whisper Chunks 拆解為獨立語用子句並精準吸附至 `fineTurns` 聲學換人切點！
 * 徹底解決當 Whisper 把多位語者的快速問答（如「...裝修怎麼辦？不算，不算。」或「...如果剛好就只要限定開新店就好了」）
 * 黏合在單一時間戳區間內時，純字數比例插值導致短句時間漂移與語者錯位的問題！
 */
function normalizeWhisperChunksToWordItems(
  rawChunks: Array<{ text: string; timestamp?: [number, number | null] }>,
  windowStartSec: number,
  windowEndSec: number,
  fineTurns: FineDiarizationTurn[],
  isWordLevel: boolean
): RawWordTimestampItem[] {
  const windowDur = Math.max(0.4, windowEndSec - windowStartSec);
  const windowTurns = fineTurns.filter(
    (ft) => ft.end > windowStartSec && ft.start < windowEndSec
  );
  const speechMinStart =
    windowTurns.length > 0 ? windowTurns[0].start : windowStartSec;
  const speechMaxEnd =
    windowTurns.length > 0
      ? windowTurns[windowTurns.length - 1].end
      : windowEndSec;

  // 收集視窗內所有真實聲學換人時間點 (Switch Anchors)
  const switchTimes: number[] = [];
  for (let i = 1; i < windowTurns.length; i++) {
    if (windowTurns[i].speakerId !== windowTurns[i - 1].speakerId) {
      switchTimes.push(windowTurns[i].start);
    }
  }

  const rawItems: RawWordTimestampItem[] = [];

  if (isWordLevel) {
    for (let i = 0; i < rawChunks.length; i++) {
      const ch = rawChunks[i];
      const rawTok = toTwp((ch.text || '').trim());
      if (!rawTok) continue;
      if (/^(字幕|請不吝點讚|訂閱轉發|謝謝觀看|Amara\.org)/i.test(rawTok)) {
        continue;
      }

      const relStart =
        typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
      const relEnd =
        typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
          ? ch.timestamp[1]
          : Math.min(windowDur, relStart + 0.24);

      const gStart = Number(
        Math.max(windowStartSec, windowStartSec + relStart).toFixed(3)
      );
      const gEnd = Number(
        Math.max(
          gStart + 0.04,
          Math.min(windowEndSec + 0.2, windowStartSec + relEnd)
        ).toFixed(3)
      );

      const subTokens =
        rawTok.match(
          /[\u4e00-\u9fa5]{1,2}[，。？！、；：,.?!]*|[\u3040-\u30ff]{1,3}[，。？！、；：,.?!]*|[a-zA-Z0-9']+[，。？！、；：,.?!]*|[，。？！、；：,.?!]+/g
        ) || [rawTok];

      const subSpan = Math.max(0.04, gEnd - gStart);
      const step = subSpan / Math.max(1, subTokens.length);

      for (let k = 0; k < subTokens.length; k++) {
        const st = subTokens[k].trim();
        if (!st) continue;
        if (/^[，。？！、；：,.?!]+$/.test(st) && rawItems.length > 0) {
          const prev = rawItems[rawItems.length - 1];
          if (!/[，。？！、；：,.?!]$/.test(prev.text)) {
            prev.text += st[0];
          }
          continue;
        }
        rawItems.push({
          text: st,
          start: Number((gStart + k * step).toFixed(3)),
          end: Number((gStart + (k + 1) * step).toFixed(3)),
          confidence: 0.9,
        });
      }
    }
    return rawItems;
  }

  // ── 主力路徑：子句級時間戳記 + 語用拆句 + 聲學換人切點對齊 (Switch-Point Boundary Snapping) ──
  for (let i = 0; i < rawChunks.length; i++) {
    const ch = rawChunks[i];
    const cleanClause = restoreMultilingualPunctuation(ch.text || '');
    if (!cleanClause) continue;

    const relStart =
      typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
    const relEnd =
      typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
        ? ch.timestamp[1]
        : Math.min(windowDur, relStart + 3.6);

    let gStart = Number((windowStartSec + relStart).toFixed(3));
    let gEnd = Number(
      Math.max(gStart + 0.25, windowStartSec + relEnd).toFixed(3)
    );

    if (i === 0 && Math.abs(gStart - speechMinStart) <= 0.6) {
      gStart = Number(Math.max(windowStartSec, speechMinStart).toFixed(3));
    }
    if (
      i === rawChunks.length - 1 &&
      gEnd > speechMaxEnd &&
      gEnd - speechMaxEnd <= 1.2
    ) {
      gEnd = Number(Math.max(gStart + 0.25, speechMaxEnd).toFixed(3));
    }

    // 修正 Whisper 子句右邊界延遲 (Right-Boundary Lag Snapping)：
    // 當 Whisper 單一子句結束時間 gEnd 比真實聲學換人時間點晚了 0.05s ~ 0.65s 時，將 gEnd 對齊回聲學換人點，防止句尾最後一字跨入下一位語者時段！
    for (const st of switchTimes) {
      if (
        st > gStart + 0.45 &&
        gEnd > st &&
        gEnd - st <= 0.65 &&
        (st - gStart) / Math.max(0.2, gEnd - gStart) >= 0.68
      ) {
        gEnd = Number(st.toFixed(3));
        break;
      }
    }

    const clauses = splitTextIntoDiscourseClauses(cleanClause);
    const numClauses = clauses.length;
    if (numClauses === 0) continue;

    // 找出落在當前 Chunk [gStart + 0.22, gEnd - 0.22] 內部的真實聲學換人點
    const internalSwitches = switchTimes.filter(
      (t) => t >= gStart + 0.22 && t <= gEnd - 0.22
    );
    const claimedSwitches = new Set<number>();

    const clauseBounds = new Float64Array(numClauses + 1);
    clauseBounds[0] = gStart;
    clauseBounds[numClauses] = gEnd;

    if (numClauses > 1 && internalSwitches.length === numClauses - 1) {
      // 若標點子句切分數剛好等於該區段內的聲學換人次數，直接 100% 對位至聲學換人切點
      for (let k = 0; k < internalSwitches.length; k++) {
        clauseBounds[k + 1] = internalSwitches[k];
        claimedSwitches.add(internalSwitches[k]);
      }
    } else if (numClauses > 1) {
      const totalChars = Math.max(
        1,
        clauses.reduce((acc, c) => acc + c.length, 0)
      );
      let charAcc = 0;
      for (let k = 0; k < numClauses - 1; k++) {
        charAcc += clauses[k].length;
        const rawSplit = gStart + (charAcc / totalChars) * (gEnd - gStart);
        const minAllowed = clauseBounds[k] + 0.18;
        const maxAllowed = gEnd - (numClauses - 1 - k) * 0.18;

        let snapped = rawSplit;
        let minDist = 0.85;
        let bestSw: number | null = null;
        for (const st of internalSwitches) {
          if (!claimedSwitches.has(st) && st >= minAllowed && st <= maxAllowed) {
            const d = Math.abs(st - rawSplit);
            if (d < minDist) {
              minDist = d;
              snapped = st;
              bestSw = st;
            }
          }
        }
        if (bestSw !== null) claimedSwitches.add(bestSw);
        clauseBounds[k + 1] = Math.max(
          minAllowed,
          Math.min(maxAllowed, snapped)
        );
      }
    }

    for (let cIdx = 0; cIdx < numClauses; cIdx++) {
      const cText = clauses[cIdx];
      const cStart = clauseBounds[cIdx];
      const cEnd = Math.max(cStart + 0.16, clauseBounds[cIdx + 1]);

      // 檢查該子句內部是否仍包含尚未被標點吸附之「句中無標點聲學換人點」
      const midClauseSwitches = internalSwitches.filter(
        (st) =>
          !claimedSwitches.has(st) && st >= cStart + 0.35 && st <= cEnd - 0.35
      );

      const acousticSubClauses = splitClauseAtAcousticSwitches(
        cText,
        cStart,
        cEnd,
        midClauseSwitches
      );

      for (let sIdx = 0; sIdx < acousticSubClauses.length; sIdx++) {
        const sub = acousticSubClauses[sIdx];
        const tokens = tokenizeClausePreventingTailOrphan(sub.text);

        const spanDur = Math.max(0.14, sub.end - sub.start);
        const step = spanDur / Math.max(1, tokens.length);

        for (let k = 0; k < tokens.length; k++) {
          const tok = tokens[k].trim();
          if (!tok) continue;
          if (/^[，。？！、；：,.?!]+$/.test(tok) && rawItems.length > 0) {
            const prev = rawItems[rawItems.length - 1];
            if (!/[，。？！、；：,.?!]$/.test(prev.text)) {
              prev.text += tok[0];
            }
            continue;
          }
          rawItems.push({
            text: tok,
            start: Number((sub.start + k * step).toFixed(3)),
            end: Number((sub.start + (k + 1) * step).toFixed(3)),
            confidence: 0.88,
            isAcousticTurnBoundary:
              k === 0 && (cIdx > 0 || sub.isAcousticSplit),
          });
        }
      }
    }
  }

  return rawItems;
}

/**
 * 虛擬音軌時間軸還原演算法 (僅在使用者手動開啟 Track-First 時使用，且嚴格限制跨段合併距離 <= 0.45s)
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
        Math.min(item.virtualEndSec - 0.06, vStart)
      );
      const clampedVEnd = Math.max(
        clampedVStart + 0.08,
        Math.min(item.virtualEndSec, vEnd)
      );

      const relRatioStart = (clampedVStart - item.virtualStartSec) / vSpan;
      const relRatioEnd = (clampedVEnd - item.virtualStartSec) / vSpan;

      const realStart = Number(
        (item.realStartSec + relRatioStart * rSpan).toFixed(2)
      );
      const realEnd = Number(
        Math.max(
          realStart + 0.12,
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
      const clauses = splitTextIntoDiscourseClauses(cleanText);

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

  // 嚴格禁止跨 >0.40s 間隙合併，防止把中間隔著其他語者發言的兩段話黏成同一段！
  const mergedUnits: typeof unwarpedUnits = [];
  for (const u of unwarpedUnits) {
    const prev = mergedUnits[mergedUnits.length - 1];
    if (
      prev &&
      prev.speakerId === u.speakerId &&
      u.realStart >= prev.realEnd - 0.1 &&
      u.realStart - prev.realEnd <= 0.4 &&
      prev.text.length + u.text.length <= 160
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
    const isLocked =
      m.acousticMargin >= 0.16 && m.realEnd - m.realStart >= 1.2;
    return {
      id: segId,
      chunkIndex,
      speakerId: m.speakerId,
      runnerUpSpeakerId: m.runnerUpSpeakerId,
      acousticMargin: m.acousticMargin,
      logOddsMargin: isLocked ? 2.1 : 1.1,
      overlapRatio: 0.05,
      slotTier: isLocked
        ? 'tier1_locked_anchor'
        : 'tier2_candidate_constrained',
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

      const { chunks: rawChunks } =
        await runWhisperWithWordTimestampsAndFallback(
          transcriber,
          audioInput,
          resolvedWhisperLang,
          30,
          5
        );

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
      posteriorMatrix,
      chunkIndex,
      windowIndex,
      modelName,
      languageHint = 'zh',
      acousticThreshold = 0.72,
      backchannelMarginThreshold = 0.12,
      pingPongMaxDurSec = 1.5,
    } = data as {
      windowSamples: Float32Array;
      windowStartSec: number;
      windowEndSec: number;
      fineTurns: FineDiarizationTurn[];
      posteriorMatrix?: FrameSpeakerPosteriorMatrix;
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

      const { chunks: rawChunks, isWordLevel } =
        await runWhisperWithWordTimestampsAndFallback(
          transcriber,
          audioInput,
          resolvedWhisperLang,
          30,
          5
        );

      const wordItems = normalizeWhisperChunksToWordItems(
        rawChunks,
        windowStartSec,
        windowEndSec,
        fineTurns || [],
        isWordLevel
      );

      if (wordItems.length === 0) {
        self.postMessage({
          type: 'WINDOW_TRANSCRIBED',
          segments: [],
          windowIndex,
        });
        return;
      }

      // 優先使用由 fineTurns 合成之銳利話輪後驗矩陣，確保每個微觀話輪邊界與語用拆句精準對應
      const activePosteriorMatrix: FrameSpeakerPosteriorMatrix =
        fineTurns && fineTurns.length > 0
          ? synthesizePosteriorMatrixFromFineTurns(
              fineTurns,
              windowStartSec,
              windowEndSec,
              0.02
            )
          : posteriorMatrix &&
            posteriorMatrix.numFrames > 0 &&
            posteriorMatrix.speakerIds.length > 0
          ? posteriorMatrix
          : synthesizePosteriorMatrixFromFineTurns(
              [],
              windowStartSec,
              windowEndSec,
              0.02
            );

      const alignedWordSlots = decodeWordLevelSpeakersViterbi(
        wordItems,
        activePosteriorMatrix,
        {
          lambdaTrans: 0.55,
          backchannelMarginThreshold,
          pingPongMaxDurSec,
        }
      );

      const finalSegments = groupAlignedWordsIntoSegments(
        alignedWordSlots,
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

      const { chunks: rawChunks } =
        await runWhisperWithWordTimestampsAndFallback(
          transcriber,
          audioInput,
          resolvedWhisperLang,
          20,
          3
        );

      const rawText = rawChunks
        .map((c: { text?: string }) => (c.text || '').trim())
        .filter(Boolean)
        .join('');

      const cleanText = restoreMultilingualPunctuation(rawText);
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
            reason: '經無失真線性增益與重聽校準',
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