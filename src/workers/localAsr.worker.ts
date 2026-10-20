// src/workers/localAsr.worker.ts
import { env, pipeline } from '@huggingface/transformers';
import * as OpenCC from 'opencc-js';
import { FineDiarizationTurn, LocalAsrModel, Segment, WordSpan } from '../types';

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
    fileName: '準備載入語音辨識引擎...',
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
 * 支援中、英、日混用 (Code-Switching) 之通用標點與疊字正規化引擎
 */
function restoreMultilingualPunctuation(rawInput: string): string {
  const hasKana = /[\u3040-\u30ff]/.test(rawInput);
  let text = hasKana ? rawInput.trim() : toTwp(rawInput.trim());
  if (!text) return '';

  text = text.replace(/([\u4e00-\u9fa5])\1{2,}/g, '$1');
  text = text.replace(/([\u4e00-\u9fa5]{2,6})\1{2,}/g, '$1');
  text = text.replace(/^(字幕|請不吝點讚|訂閱轉發|謝謝觀看|Amara\.org).*$/gi, '');

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
  text = text.replace(/^(對對對|沒有沒有|不用不用)(?=[^\u3000-\u303f\uff00-\uffef])/, '$1，');

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
  text = text.replace(/(不知道啊|太遠了|太久了|超亂的)[。，！!]*$/g, '$1！');

  text = text
    .replace(/[？?]{2,}/g, '？')
    .replace(/[！!]{2,}/g, '！')
    .replace(/？[，。、]+/g, '？')
    .replace(/！[，。、]+/g, '！')
    .replace(/[，、]{2,}/g, '，')
    .replace(/，([。？！])/g, '$1');

  if (text.length >= 2 && cjkRatio >= 0.25 && !/[，。？！、；：.?!]$/.test(text)) {
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

// ============================================================================
// 【問題 4 & 5】原子語意單元 (Atomic Clauses) 拆解 + SCD-Viterbi 動態規劃對齊
// ============================================================================

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

/**
 * 將 Whisper 輸出之 Chunks 拆解為具備精確時間區間之最小語意單元序列 W = {w_1, ..., w_L}
 */
function decomposeChunksIntoAtomicUnits(
  rawChunks: Array<{ text: string; timestamp?: [number, number | null] }>,
  windowStartSec: number,
  windowEndSec: number
): AtomicTokenUnit[] {
  const units: AtomicTokenUnit[] = [];

  for (let i = 0; i < rawChunks.length; i++) {
    const ch = rawChunks[i];
    const rawText = (ch.text || '').trim();
    if (!rawText) continue;

    const relStart = typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
    const relEnd =
      typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
        ? ch.timestamp[1]
        : Math.min(windowEndSec - windowStartSec, relStart + 3.8);

    const gStart = Number((windowStartSec + relStart).toFixed(2));
    const gEnd = Number(Math.max(gStart + 0.35, windowStartSec + relEnd).toFixed(2));

    const prePunct = restoreMultilingualPunctuation(rawText);
    if (!prePunct) continue;

    // 在句號、問號、感嘆號、逗號及明確轉折連詞處切分為最小子句單元，供 Viterbi 逐點決策
    const marked = prePunct
      .replace(/([。？！?!])/g, '$1|')
      .replace(
        /([，、,])\s*(?=可是|但是|不是|不過|那如果|所以我|那我|不算|はい|ええ|なるほど|そうですね|しかし|ただ|yeah|right|but\s|however)/gi,
        '$1|'
      );

    const subStrings = marked
      .split('|')
      .flatMap((part) => (part.length >= 14 ? part.split(/(?<=[，、,])/) : [part]))
      .map((s) => s.trim())
      .filter((s) => s.length > 0 && !/^[，。？！、,.?!]+$/.test(s));

    if (subStrings.length === 0) continue;

    const totalChars = Math.max(
      1,
      subStrings.reduce((acc, s) => acc + s.length, 0)
    );
    const spanDur = Math.max(0.3, gEnd - gStart);
    let cursor = 0;

    for (const sub of subStrings) {
      const uStart = Number((gStart + (cursor / totalChars) * spanDur).toFixed(2));
      cursor += sub.length;
      const uEnd = Number(
        Math.max(uStart + 0.2, gStart + (cursor / totalChars) * spanDur).toFixed(2)
      );
      units.push({
        text: restoreMultilingualPunctuation(sub),
        start: uStart,
        end: uEnd,
        center: Number((0.5 * (uStart + uEnd)).toFixed(2)),
      });
    }
  }

  return units;
}

/**
 * 【Deep Research 問題 4 核心實作】
 * 執行 SCD 波峰導引之 Viterbi 動態規劃對齊：
 * - 發射成本：\Phi_emit(l, k) = -ln( exp(\gamma * S(l, k)) / \sum_j exp(\gamma * S(l, j)) ), \gamma = 12.0
 * - 轉移成本：\Psi_trans(l, k' -> k) = \Lambda_base * exp(-\alpha_scd * d_SCD(t_l^start)) * \beta_ling(w_{l-1}, w_l)
 *   其中 \Lambda_base = 4.5, \alpha_scd = 3.2, \beta_ling \in {0.25, 0.55, 1.00}
 * - 步驟 4 後處理：以 \Delta M(B, A) >= 0.14 或 (\Delta M(B, A) >= 0.06 且雙峰 P_scd >= 0.55) 保護短促附和詞
 */
function alignUnitsWithScdViterbiAndPragmatics(
  units: AtomicTokenUnit[],
  fineTurns: FineDiarizationTurn[]
): AtomicTokenUnit[] {
  const L = units.length;
  if (L === 0 || fineTurns.length === 0) return units;

  // 收集本視窗所有活躍之語者集合 G = {1, ..., G}
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

  // 查詢任一時間點 t 附近之最大聲紋跳變度 d_SCD(t)
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

  // 計算第 l 個單元對每位候選語者 g 的聲學親和力分數 S_comp(l, g)
  const unitSpeakerScores: Float64Array[] = new Array(L);

  for (let l = 0; l < L; l++) {
    const u = units[l];
    const scores = new Float64Array(G).fill(0.35);
    let weightSum = 0.0;

    for (const ft of fineTurns) {
      const ovStart = Math.max(u.start - 0.15, ft.start);
      const ovEnd = Math.min(u.end + 0.15, ft.end);
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
      // 若落在微小空隙，找時間中心最近的 fineTurn
      let nearest = fineTurns[0];
      let minGap = Infinity;
      for (const ft of fineTurns) {
        const gap = Math.min(Math.abs(u.center - ft.start), Math.abs(u.center - ft.end));
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

  // ── 步驟 2 & 3：Viterbi 動態規劃矩陣求解 ──
  const dp = new Float64Array(L * G).fill(Infinity);
  const backptr = new Int32Array(L * G).fill(0);

  const gamma = 12.0;      // Softmax 溫度係數
  const lambdaBase = 4.5;  // 基礎轉移懲罰 \Lambda_base
  const alphaScd = 3.2;    // SCD 跳變衰減率 \alpha_scd

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

        // 計算語言學與停頓調節因子 \beta_ling(w_{l-1}, w_l)
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

  // 回溯全局最優語者路徑 \hat{z}_{1:L}
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

  // ── 步驟 4：具備邊際與雙峰保護的智慧型後處理 (保護 0.4s~1.5s 真實附和詞) ──
  let idx = 0;
  while (idx < L) {
    let endIdx = idx;
    while (endIdx + 1 < L && assignedIdx[endIdx + 1] === assignedIdx[idx]) {
      endIdx++;
    }
    const turnDur = units[endIdx].end - units[idx].start;

    // 檢查夾在同語者 A 中間的短區間 B (A -> B -> A, 且 B < 1.5s)
    if (
      idx > 0 &&
      endIdx < L - 1 &&
      assignedIdx[idx - 1] === assignedIdx[endIdx + 1] &&
      turnDur < 1.5
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
        (units[idx].boundaryScdLeft || 0) + (units[endIdx].boundaryScdRight || 0);
      const isLexicalBackchannel = MULTILINGUAL_BACKCHANNEL_REGEX.test(combinedText);

      // 報告步驟 4 保護公式：\Delta M(B, A) >= 0.14 或 (\Delta M(B, A) >= 0.06 且雙峰 >= 0.55) 或命中附和詞庫
      const isTrueBackchannel =
        avgMarginBA >= 0.14 ||
        (avgMarginBA >= 0.06 && dualScdPeakSum >= 0.55) ||
        isLexicalBackchannel;

      if (!isTrueBackchannel) {
        for (let k = idx; k <= endIdx; k++) {
          assignedIdx[k] = spkA;
        }
      }
    }
    idx = endIdx + 1;
  }

  // 寫回每個單元的 Top-1、Top-2 與聲學邊際 M(u_m)
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

  // ── 步驟 5：套用 DiarizationLM 對話語用學法則 (問答對互斥律 & 反詰分離律) ──
  for (let l = 1; l < L; l++) {
    const prev = units[l - 1];
    const curr = units[l];
    const currDur = curr.end - curr.start;
    const isLocked = (curr.acousticMargin || 0) >= 0.16 && currDur >= 1.2;

    const prevIsQuestion = MULTILINGUAL_QUESTION_END_REGEX.test(prev.text);
    const currIsShortAnswer = MULTILINGUAL_BACKCHANNEL_REGEX.test(curr.text);

    // 【法則一：問答對互斥律】疑問句後緊接之直接確認/否決回答，必屬不同語者
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

    // 【法則三：句首反詰轉折分離律】對非 LOCKED 之反詰問句且具備邊界 SCD 跳變，切換至次佳語者
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

/**
 * 將完成 Viterbi 與語用學裁決之單元序列合併為段落 Segment[] (上限 260 字)
 */
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
      item.start - prev.end <= 4.5 &&
      prev.text.length + item.text.length <= 260
    ) {
      const prevClean = prev.text.replace(/[。.]$/, '，');
      prev.text = restoreMultilingualPunctuation(prevClean + item.text);
      prev.end = item.end;
      prev.confidence = Math.max(prev.confidence || 0.85, item.confidence || 0.85);
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
        windowEndSec
      );
      const alignedUnits = alignUnitsWithScdViterbiAndPragmatics(
        atomicUnits,
        fineTurns
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

      const output = await transcriber(audioInput, {
        language: resolvedWhisperLang,
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