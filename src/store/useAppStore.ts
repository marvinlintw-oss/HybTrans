// src/store/useAppStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  AtomicCharToken,
  CorrectionPatchItem,
  DiarizationConfig,
  DictEntry,
  DictScope,
  EngineConfig,
  GranularityTier,
  MacroChunkState,
  ModelDownloadStatus,
  PhoneticMatchCandidate,
  Segment,
  Speaker,
  SuspectSpanNode,
  UncertainCandidateOption,
  UtteranceNode,
  WordSpan,
} from '../types';
import { TIER_PRESET_SPECS } from '../services/modelCacheManager';
import { DEFAULT_PUBLIC_DICTIONARY, mergeDictLists } from '../utils/dictIO';
import { toTaiwanFuzzyPhoneticSyllables } from '../utils/phoneticMatcher';

/**
 * 【問題五 5.2 核心演算法】：不可變 Token 圖譜之雙向音韻-字面重疊自動消融與無縫替換演算法
 */
export function applySeamlessSpanReplacement(
  utterance: UtteranceNode,
  targetSpanId: string,
  candidate: UncertainCandidateOption
): UtteranceNode {
  const flatTokens: AtomicCharToken[] = [];
  let spanStartIdx = -1;
  let spanEndIdx = -1;

  for (const seg of utterance.segments) {
    if ('spanId' in seg) {
      if (seg.spanId === targetSpanId) {
        spanStartIdx = flatTokens.length;
        flatTokens.push(...seg.activeTokens);
        spanEndIdx = flatTokens.length;
      } else {
        flatTokens.push(...seg.activeTokens);
      }
    } else {
      flatTokens.push(seg);
    }
  }

  if (spanStartIdx === -1) return utterance;

  const candChars = Array.from(candidate.text);
  const candPinyins =
    candidate.pinyinSequence && candidate.pinyinSequence.length === candChars.length
      ? candidate.pinyinSequence
      : toTaiwanFuzzyPhoneticSyllables(candidate.text);
  const M = candChars.length;

  const isTokenMatch = (
    tok: AtomicCharToken,
    char: string,
    pinyin: string
  ): boolean => {
    return tok.char === char || (tok.pinyinKey !== '' && tok.pinyinKey === pinyin);
  };

  let bestKL = 0;
  const maxCheckL = Math.min(M - 1, spanStartIdx);
  for (let k = maxCheckL; k >= 1; k--) {
    let matchAll = true;
    for (let i = 0; i < k; i++) {
      const leftTok = flatTokens[spanStartIdx - k + i];
      if (!isTokenMatch(leftTok, candChars[i], candPinyins[i])) {
        matchAll = false;
        break;
      }
    }
    if (matchAll) {
      bestKL = k;
      break;
    }
  }

  let bestKR = 0;
  const maxCheckR = Math.min(M - 1 - bestKL, flatTokens.length - spanEndIdx);
  for (let k = maxCheckR; k >= 1; k--) {
    let matchAll = true;
    for (let i = 0; i < k; i++) {
      const rightTok = flatTokens[spanEndIdx + i];
      const candIdx = M - k + i;
      if (!isTokenMatch(rightTok, candChars[candIdx], candPinyins[candIdx])) {
        matchAll = false;
        break;
      }
    }
    if (matchAll) {
      bestKR = k;
      break;
    }
  }

  const absorbedStart = spanStartIdx - bestKL;
  const absorbedEnd = spanEndIdx + bestKR;

  let effectiveCandChars = candChars;
  let effectiveCandPinyins = candPinyins;

  if (
    absorbedStart > 0 &&
    effectiveCandChars.length >= 2 &&
    flatTokens[absorbedStart - 1].char === effectiveCandChars[0] &&
    effectiveCandChars[0] === effectiveCandChars[1]
  ) {
    effectiveCandChars = effectiveCandChars.slice(1);
    effectiveCandPinyins = effectiveCandPinyins.slice(1);
  }

  if (
    absorbedEnd < flatTokens.length &&
    effectiveCandChars.length >= 2 &&
    flatTokens[absorbedEnd].char ===
      effectiveCandChars[effectiveCandChars.length - 1] &&
    effectiveCandChars[effectiveCandChars.length - 1] ===
      effectiveCandChars[effectiveCandChars.length - 2]
  ) {
    effectiveCandChars = effectiveCandChars.slice(0, -1);
    effectiveCandPinyins = effectiveCandPinyins.slice(0, -1);
  }

  const sliceStartTime = flatTokens[absorbedStart].startTimeMs;
  const sliceEndTime = flatTokens[absorbedEnd - 1].endTimeMs;
  const totalDuration = Math.max(
    sliceEndTime - sliceStartTime,
    effectiveCandChars.length * 40
  );
  const stepMs = totalDuration / Math.max(1, effectiveCandChars.length);
  const dominantSpeaker = flatTokens[spanStartIdx].speakerId;
  const avgConf =
    flatTokens
      .slice(absorbedStart, absorbedEnd)
      .reduce((acc, t) => acc + t.acousticConf, 0) /
    Math.max(1, absorbedEnd - absorbedStart);

  const newSpanTokens: AtomicCharToken[] = effectiveCandChars.map(
    (ch, idx) => ({
      tokenId: `${targetSpanId}_opt_${candidate.optionId}_${idx}`,
      char: ch,
      pinyinKey: effectiveCandPinyins[idx] || '',
      startTimeMs: Math.round(sliceStartTime + idx * stepMs),
      endTimeMs: Math.round(sliceStartTime + (idx + 1) * stepMs),
      acousticConf: avgConf,
      speakerId: dominantSpeaker,
    })
  );

  const updatedSegments: Array<AtomicCharToken | SuspectSpanNode> = [];
  let cursor = 0;

  for (const seg of utterance.segments) {
    if ('spanId' in seg) {
      const segLen = seg.activeTokens.length;
      if (seg.spanId === targetSpanId) {
        const expandedOriginal = flatTokens.slice(absorbedStart, absorbedEnd);
        updatedSegments.push({
          ...seg,
          status: 'USER_LOCKED',
          activeTokens: newSpanTokens,
          originalTokens:
            seg.originalTokens.length < expandedOriginal.length
              ? expandedOriginal
              : seg.originalTokens,
          selectedOptionId: candidate.optionId,
        });
      } else {
        const kept = seg.activeTokens.filter((_, idx) => {
          const pos = cursor + idx;
          return pos < absorbedStart || pos >= absorbedEnd;
        });
        if (kept.length > 0) {
          updatedSegments.push({ ...seg, activeTokens: kept });
        }
      }
      cursor += segLen;
    } else {
      if (cursor < absorbedStart || cursor >= absorbedEnd) {
        updatedSegments.push(seg);
      }
      cursor += 1;
    }
  }

  return {
    ...utterance,
    segments: updatedSegments,
  };
}

/**
 * 將雙向音韻-字面重疊自動消融演算法直接套用於 Segment.words (WordSpan[])！
 */
export function applySeamlessWordSpanReplacement(
  words: WordSpan[],
  startWordIdx: number,
  endWordIdx: number,
  replacementText: string,
  options?: {
    status?: WordSpan['status'];
    reason?: string;
    candidates?: string[];
    userLocked?: boolean;
  }
): WordSpan[] {
  if (
    words.length === 0 ||
    startWordIdx < 0 ||
    startWordIdx >= words.length
  ) {
    return words;
  }
  const safeEndWordIdx = Math.max(
    startWordIdx + 1,
    Math.min(words.length, endWordIdx)
  );

  interface FlatCharItem {
    char: string;
    pinyinKey: string;
    wordIdx: number;
    startSec: number;
    endSec: number;
    conf: number;
  }

  const flatChars: FlatCharItem[] = [];
  let charSpanStart = -1;
  let charSpanEnd = -1;

  for (let wIdx = 0; wIdx < words.length; wIdx++) {
    const w = words[wIdx];
    if (wIdx === startWordIdx) charSpanStart = flatChars.length;

    const chars = Array.from(w.text);
    const pinyins = toTaiwanFuzzyPhoneticSyllables(w.text);
    const dur = Math.max(0.04, w.end - w.start);
    const step = dur / Math.max(1, chars.length);

    chars.forEach((ch, cIdx) => {
      flatChars.push({
        char: ch,
        pinyinKey: pinyins[cIdx] || '',
        wordIdx: wIdx,
        startSec: w.start + cIdx * step,
        endSec: w.start + (cIdx + 1) * step,
        conf: w.confidence,
      });
    });

    if (wIdx === safeEndWordIdx - 1) charSpanEnd = flatChars.length;
  }

  if (charSpanStart === -1 || charSpanEnd === -1) return words;

  const candChars = Array.from(replacementText);
  const candPinyins = toTaiwanFuzzyPhoneticSyllables(replacementText);
  const M = candChars.length;

  const isCharOrPhoneticMatch = (
    item: FlatCharItem,
    ch: string,
    py: string
  ): boolean => {
    if (!/^[\u4e00-\u9fa5]$/.test(item.char) || !/^[\u4e00-\u9fa5]$/.test(ch)) {
      return item.char === ch;
    }
    return item.char === ch || (item.pinyinKey !== '' && item.pinyinKey === py);
  };

  let bestKL = 0;
  const maxCheckL = Math.min(M - 1, charSpanStart);
  for (let k = maxCheckL; k >= 1; k--) {
    let matchAll = true;
    for (let i = 0; i < k; i++) {
      const leftItem = flatChars[charSpanStart - k + i];
      if (!isCharOrPhoneticMatch(leftItem, candChars[i], candPinyins[i])) {
        matchAll = false;
        break;
      }
    }
    if (matchAll) {
      bestKL = k;
      break;
    }
  }

  let bestKR = 0;
  const maxCheckR = Math.min(M - 1 - bestKL, flatChars.length - charSpanEnd);
  for (let k = maxCheckR; k >= 1; k--) {
    let matchAll = true;
    for (let i = 0; i < k; i++) {
      const rightItem = flatChars[charSpanEnd + i];
      const candIdx = M - k + i;
      if (
        !isCharOrPhoneticMatch(rightItem, candChars[candIdx], candPinyins[candIdx])
      ) {
        matchAll = false;
        break;
      }
    }
    if (matchAll) {
      bestKR = k;
      break;
    }
  }

  const absorbedCharStart = charSpanStart - bestKL;
  const absorbedCharEnd = charSpanEnd + bestKR;

  let effectiveText = replacementText;
  if (
    absorbedCharStart > 0 &&
    effectiveText.length >= 2 &&
    flatChars[absorbedCharStart - 1].char === effectiveText[0]
  ) {
    effectiveText = effectiveText.slice(1);
  }
  if (
    absorbedCharEnd < flatChars.length &&
    effectiveText.length >= 2 &&
    flatChars[absorbedCharEnd].char === effectiveText[effectiveText.length - 1]
  ) {
    effectiveText = effectiveText.slice(0, -1);
  }

  const absorbedWordStart = flatChars[absorbedCharStart].wordIdx;
  const absorbedWordEnd = flatChars[absorbedCharEnd - 1].wordIdx + 1;

  const firstWordChars = flatChars.filter(
    (c, idx) => c.wordIdx === absorbedWordStart && idx < absorbedCharStart
  );
  const lastWordChars = flatChars.filter(
    (c, idx) => c.wordIdx === absorbedWordEnd - 1 && idx >= absorbedCharEnd
  );

  const prefixKeep = firstWordChars.map((c) => c.char).join('');
  const suffixKeep = lastWordChars.map((c) => c.char).join('');

  const expandedOriginalText = flatChars
    .slice(absorbedCharStart, absorbedCharEnd)
    .map((c) => c.char)
    .join('');

  const targetAnchorWord = words[startWordIdx];
  const preservedOriginal =
    targetAnchorWord.originalText &&
    targetAnchorWord.originalText.length >= expandedOriginalText.length
      ? targetAnchorWord.originalText
      : expandedOriginalText;

  const newStartSec = Number(flatChars[absorbedCharStart].startSec.toFixed(2));
  const newEndSec = Number(
    Math.max(
      newStartSec + 0.15,
      flatChars[absorbedCharEnd - 1].endSec
    ).toFixed(2)
  );

  const resultWords: WordSpan[] = [];

  for (let i = 0; i < absorbedWordStart; i++) {
    resultWords.push(words[i]);
  }

  if (prefixKeep) {
    resultWords.push({
      ...words[absorbedWordStart],
      id: `${words[absorbedWordStart].id}_pre`,
      text: prefixKeep,
      end: newStartSec,
    });
  }

  resultWords.push({
    ...targetAnchorWord,
    text: effectiveText || replacementText,
    originalText: preservedOriginal,
    start: newStartSec,
    end: newEndSec,
    confidence: options?.status === 'auto_corrected' ? 0.96 : targetAnchorWord.confidence,
    status: options?.status ?? 'normal',
    reason: options?.reason ?? targetAnchorWord.reason,
    candidates: options?.candidates ?? targetAnchorWord.candidates,
    userLocked: options?.userLocked ?? targetAnchorWord.userLocked,
  });

  if (suffixKeep) {
    resultWords.push({
      ...words[absorbedWordEnd - 1],
      id: `${words[absorbedWordEnd - 1].id}_suf`,
      text: suffixKeep,
      start: newEndSec,
    });
  }

  for (let i = absorbedWordEnd; i < words.length; i++) {
    resultWords.push(words[i]);
  }

  return resultWords;
}

/**
 * 依原始會議絕對時間軸排序並安全合併同語者連續段落。
 */
function smartConsolidateSegments(allSegments: Segment[]): Segment[] {
  if (allSegments.length <= 1) return allSegments;

  const sorted: Segment[] = allSegments
    .map((s) => ({
      ...s,
      words: [...s.words],
    }))
    .sort((a, b) => a.start - b.start || a.end - b.end);

  const merged: Segment[] = [];
  for (const seg of sorted) {
    if (seg.words.length === 0) continue;
    const prev = merged[merged.length - 1];
    const prevChars = prev
      ? prev.words.reduce((acc, w) => acc + w.text.length, 0)
      : 0;
    const currChars = seg.words.reduce((acc, w) => acc + w.text.length, 0);

    if (
      prev &&
      prev.chunkIndex === seg.chunkIndex &&
      prev.speakerId === seg.speakerId &&
      seg.start >= prev.end - 0.15 &&
      seg.start - prev.end <= 2.8 &&
      prevChars + currChars <= 260
    ) {
      const prevWords = [...prev.words];
      if (prevWords.length > 0) {
        const lastW = prevWords[prevWords.length - 1];
        if (!/[，。？！、；：,.?!]$/.test(lastW.text)) {
          prevWords[prevWords.length - 1] = {
            ...lastW,
            text: `${lastW.text}，`,
          };
        }
      }
      prev.words = [...prevWords, ...seg.words];
      prev.end = Math.max(prev.end, seg.end);
      prev.acousticMargin = Math.max(
        prev.acousticMargin ?? 0.15,
        seg.acousticMargin ?? 0.15
      );
    } else {
      merged.push(seg);
    }
  }

  return merged;
}

const DEFAULT_DIARIZATION_CONFIG: DiarizationConfig = {
  engineMode: 'mode_0_heritage_classic',
  clusterMethod: 'ahc_cosine',
  executionBackend: 'wasm',
  onnxModelFilename: 'wespeaker_voxceleb_resnet34_int8.onnx',
  localWasmBasePath: './wasm/',
  localModelBasePath: './models/',
  yieldEveryNUnits: 14,
  enableGlobalMeanCentering: true,
  languageHint: 'zh',
  shortTurnConfidenceKeepThreshold: 0.66,
  pingPongMaxDurSec: 1.6,
  enablePauseAndSemanticWeighting: true,
  scdPeakThreshold: 0.22,
  backchannelMarginThreshold: 0.12,
  enableAcousticDivergencePreproc: true,
  enableVirtualSpeakerTrackStitching: false,
};

interface AppState {
  config: EngineConfig;
  chunks: MacroChunkState[];
  segments: Segment[];
  speakers: Record<string, Speaker>;
  dictionary: DictEntry[];
  docContextSummary: string;
  rollingMeetingSummary: string;
  currentTime: number;
  phoneticCandidates: PhoneticMatchCandidate[];
  modelDownload: ModelDownloadStatus;

  updateConfig: (partial: Partial<EngineConfig>) => void;
  updateDiarizationConfig: (partial: Partial<DiarizationConfig>) => void;
  applyGranularityTier: (tier: GranularityTier) => void;
  initChunks: (totalDurationSec: number) => void;
  updateChunkState: (index: number, partial: Partial<MacroChunkState>) => void;
  setModelDownload: (status: Partial<ModelDownloadStatus>) => void;
  appendSegments: (newSegments: Segment[]) => void;
  replaceBatchSegments: (updatedSegments: Segment[]) => void;
  clearTranscript: (clearSessionDictOrEvent?: boolean | unknown) => void;
  applyCorrectionPatches: (
    patches: CorrectionPatchItem[],
    markSegmentIdsCorrected?: string[]
  ) => void;
  replaceSegmentAfterRelisten: (segmentId: string, newWords: WordSpan[]) => void;
  revertWord: (segmentId: string, wordIndex: number, newText?: string) => void;
  upsertSpeakers: (list: Speaker[]) => void;
  mergeSpeakers: (sourceId: string, targetId: string) => void;
  renameSpeaker: (speakerId: string, newName: string) => void;
  setPhoneticCandidates: (candidates: PhoneticMatchCandidate[]) => void;
  applyPhoneticReplacements: () => void;
  addDictEntries: (entries: DictEntry[]) => void;
  removeDictEntry: (id: string) => void;
  promoteDictEntryScope: (id: string, targetScope: DictScope) => void;
  setDocContextSummary: (summary: string) => void;
  setRollingMeetingSummary: (summary: string) => void;
  setCurrentTime: (sec: number) => void;
}

const defaultTierSpec =
  TIER_PRESET_SPECS.balanced_pro || TIER_PRESET_SPECS.eco_safe;

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      config: {
        mode: 'local_webgpu',
        granularityTier: 'balanced_pro',
        localAsrModel: 'Whisper-Large-v3-Turbo',
        localLlmModel: defaultTierSpec.recommendedLlm,
        groqApiKey: '',
        geminiApiKey: '',
        minSpeakers: 2,
        maxSpeakers: 6,
        acousticThreshold: 0.72,
        chunkDurationSec: defaultTierSpec.chunkDurationSec,
        keepModelsCoResident: defaultTierSpec.keepModelsCoResident,
        asrBeamSize: defaultTierSpec.asrBeamSize,
        highAttentionChars: defaultTierSpec.highAttentionChars,
        mediumAttentionChars: defaultTierSpec.mediumAttentionChars,
        maxRelistenQuota: defaultTierSpec.maxRelistenQuotaPerChunk,
        relistenPaddingSec: defaultTierSpec.relistenPaddingSec,
        diarizationConfig: DEFAULT_DIARIZATION_CONFIG,
      },
      chunks: [],
      segments: [],
      speakers: {},
      dictionary: DEFAULT_PUBLIC_DICTIONARY,
      docContextSummary: '',
      rollingMeetingSummary: '',
      currentTime: 0,
      phoneticCandidates: [],
      modelDownload: {
        active: false,
        modelName: '',
        fileName: '',
        progress: 0,
        statusText: '',
      },

      updateConfig: (partial) =>
        set((state) => ({
          config: { ...state.config, ...partial },
        })),

      updateDiarizationConfig: (partial) =>
        set((state) => ({
          config: {
            ...state.config,
            diarizationConfig: {
              ...(state.config.diarizationConfig || DEFAULT_DIARIZATION_CONFIG),
              ...partial,
            },
          },
        })),

      applyGranularityTier: (tier) => {
        const spec = TIER_PRESET_SPECS[tier];
        const currentDiar =
          get().config.diarizationConfig || DEFAULT_DIARIZATION_CONFIG;

        const tierDiarPatch: Partial<DiarizationConfig> =
          tier === 'eco_safe'
            ? {
                engineMode: 'mode_a_classic',
                clusterMethod: 'ahc_cosine',
                yieldEveryNUnits: 8,
                scdPeakThreshold: 0.28,
                backchannelMarginThreshold: 0.16,
                shortTurnConfidenceKeepThreshold: 0.72,
                pingPongMaxDurSec: 1.4,
                enableAcousticDivergencePreproc: true,
                enableVirtualSpeakerTrackStitching: false,
              }
            : tier === 'balanced_pro'
            ? {
                engineMode: 'mode_0_heritage_classic',
                clusterMethod: 'ahc_cosine',
                yieldEveryNUnits: 14,
                scdPeakThreshold: 0.22,
                backchannelMarginThreshold: 0.12,
                shortTurnConfidenceKeepThreshold: 0.66,
                pingPongMaxDurSec: 1.6,
                enableAcousticDivergencePreproc: true,
                enableVirtualSpeakerTrackStitching: false,
              }
            : {
                engineMode: 'mode_c_onnx_neural',
                clusterMethod: 'spectral_cosine',
                yieldEveryNUnits: 24,
                scdPeakThreshold: 0.19,
                backchannelMarginThreshold: 0.1,
                shortTurnConfidenceKeepThreshold: 0.6,
                pingPongMaxDurSec: 1.8,
                enableAcousticDivergencePreproc: true,
                enableVirtualSpeakerTrackStitching: false,
              };

        set((state) => ({
          config: {
            ...state.config,
            granularityTier: tier,
            localLlmModel: spec.recommendedLlm,
            chunkDurationSec: spec.chunkDurationSec,
            keepModelsCoResident: spec.keepModelsCoResident,
            asrBeamSize: spec.asrBeamSize,
            highAttentionChars: spec.highAttentionChars,
            mediumAttentionChars: spec.mediumAttentionChars,
            maxRelistenQuota: spec.maxRelistenQuotaPerChunk,
            relistenPaddingSec: spec.relistenPaddingSec,
            diarizationConfig: {
              ...currentDiar,
              ...tierDiarPatch,
            },
          },
        }));
      },

      initChunks: (totalDurationSec) => {
        const chunkSec = get().config.chunkDurationSec || 300;
        const count = Math.max(1, Math.ceil(totalDurationSec / chunkSec));
        const mode = get().config.mode;
        const chunks: MacroChunkState[] = Array.from(
          { length: count },
          (_, idx) => ({
            index: idx,
            startTime: idx * chunkSec,
            endTime: Math.min(totalDurationSec, (idx + 1) * chunkSec),
            stage: 'idle',
            engineUsed: mode,
            progress: 0,
            detailText: '等待處理',
            relistenUsed: 0,
          })
        );
        set({ chunks });
      },

      updateChunkState: (index, partial) =>
        set((state) => ({
          chunks: state.chunks.map((c) =>
            c.index === index ? { ...c, ...partial } : c
          ),
        })),

      setModelDownload: (partial) =>
        set((state) => ({
          modelDownload: { ...state.modelDownload, ...partial },
        })),

      appendSegments: (newSegments) =>
        set((state) => ({
          segments: smartConsolidateSegments([
            ...state.segments,
            ...newSegments,
          ]),
        })),

      replaceBatchSegments: (updatedSegments) =>
        set((state) => {
          const map = new Map(updatedSegments.map((s) => [s.id, s]));
          const replaced = state.segments.map((s) => map.get(s.id) || s);
          return {
            segments: replaced,
          };
        }),

      clearTranscript: (clearSessionDictOrEvent) => {
        const shouldClearSession = clearSessionDictOrEvent === true;
        set((state) => ({
          chunks: [],
          segments: [],
          speakers: {},
          phoneticCandidates: [],
          rollingMeetingSummary: '',
          currentTime: 0,
          modelDownload: {
            active: false,
            modelName: '',
            fileName: '',
            progress: 0,
            statusText: '',
          },
          dictionary: shouldClearSession
            ? state.dictionary.filter((d) => d.scope !== 'session')
            : state.dictionary,
        }));
      },

      applyCorrectionPatches: (patches, markSegmentIdsCorrected = []) =>
        set((state) => {
          const patchMap = new Map<string, CorrectionPatchItem[]>();
          for (const p of patches) {
            const list = patchMap.get(p.segmentId) || [];
            list.push(p);
            patchMap.set(p.segmentId, list);
          }
          const correctedSet = new Set(markSegmentIdsCorrected);
          let hasSpeakerTransfer = false;

          const updatedSegments: Segment[] = [];

          for (const seg of state.segments) {
            const segPatches = patchMap.get(seg.id);
            const shouldMarkCorrected =
              correctedSet.has(seg.id) || Boolean(segPatches?.length);
            if (!segPatches && !shouldMarkCorrected) {
              updatedSegments.push(seg);
              continue;
            }

            // 處理 LG-TPST 語者轉移或段內 split_before_anchor 拆句
            const speakerPatch = segPatches?.find(
              (p) =>
                Boolean(p.newSpeakerId) && p.newSpeakerId !== seg.speakerId
            );

            if (speakerPatch && speakerPatch.newSpeakerId) {
              hasSpeakerTransfer = true;
              const anchor = (speakerPatch.splitBeforeAnchor || '').trim();
              if (anchor.length >= 2) {
                const fullSegText = seg.words.map((w) => w.text).join('');
                const anchorCharIdx = fullSegText.indexOf(anchor);
                if (anchorCharIdx > 2) {
                  let charAcc = 0;
                  let splitWordIdx = -1;
                  for (let wIdx = 0; wIdx < seg.words.length; wIdx++) {
                    if (charAcc >= anchorCharIdx) {
                      splitWordIdx = wIdx;
                      break;
                    }
                    charAcc += seg.words[wIdx].text.length;
                  }
                  if (splitWordIdx > 0 && splitWordIdx < seg.words.length) {
                    const beforeWords = seg.words.slice(0, splitWordIdx);
                    const afterWords = seg.words.slice(splitWordIdx);
                    const splitTime = Number(
                      beforeWords[beforeWords.length - 1].end.toFixed(2)
                    );
                    updatedSegments.push(
                      {
                        ...seg,
                        end: splitTime,
                        words: beforeWords,
                        isCorrected: true,
                      },
                      {
                        ...seg,
                        id: `${seg.id}_tpst`,
                        speakerId: speakerPatch.newSpeakerId,
                        runnerUpSpeakerId: seg.speakerId,
                        start: splitTime,
                        end: Math.max(splitTime + 0.25, seg.end),
                        words: afterWords.map((w, idx) =>
                          idx === 0
                            ? {
                                ...w,
                                status: 'auto_corrected',
                                reason:
                                  speakerPatch.reason ||
                                  'LG-TPST 問答對語者拆句移轉',
                              }
                            : w
                        ),
                        isCorrected: true,
                      }
                    );
                    continue;
                  }
                }
              }
            }

            const nextSpeakerId = speakerPatch?.newSpeakerId || seg.speakerId;
            let currentWords = [...seg.words];

            // 依詞索引由後往前套用免索引槽位替換（確保多詞槽位消融不影響前方索引）
            const wordPatches = (segPatches || [])
              .filter((p) => p.wordIndex >= 0 && !p.newSpeakerId)
              .sort((a, b) => b.wordIndex - a.wordIndex);

            for (const wp of wordPatches) {
              if (wp.wordIndex >= currentWords.length) continue;
              if (currentWords[wp.wordIndex]?.userLocked) continue;

              const endIdx =
                typeof wp.endWordIndex === 'number' &&
                wp.endWordIndex > wp.wordIndex
                  ? wp.endWordIndex
                  : wp.wordIndex + 1;

              if (wp.status === 'auto_corrected' && wp.correctedText) {
                currentWords = applySeamlessWordSpanReplacement(
                  currentWords,
                  wp.wordIndex,
                  endIdx,
                  wp.correctedText,
                  {
                    status: 'auto_corrected',
                    reason: wp.reason,
                    candidates: wp.candidates,
                  }
                );
              } else if (wp.status === 'semantic_uncertain') {
                currentWords = currentWords.map((w, idx) =>
                  idx === wp.wordIndex
                    ? {
                        ...w,
                        status: 'semantic_uncertain',
                        candidates: wp.candidates || w.candidates,
                        reason: wp.reason || w.reason,
                      }
                    : w
                );
              }
            }

            updatedSegments.push({
              ...seg,
              speakerId: nextSpeakerId,
              words: currentWords,
              isCorrected: seg.isCorrected || shouldMarkCorrected,
            });
          }

          return {
            segments: hasSpeakerTransfer
              ? smartConsolidateSegments(updatedSegments)
              : updatedSegments,
          };
        }),

      replaceSegmentAfterRelisten: (segmentId, newWords) =>
        set((state) => ({
          segments: state.segments.map((seg) =>
            seg.id === segmentId
              ? {
                  ...seg,
                  words: newWords,
                  isCorrected: true,
                  relistenCount: (seg.relistenCount || 0) + 1,
                }
              : seg
          ),
        })),

      revertWord: (segmentId, wordIndex, newText) =>
        set((state) => ({
          segments: state.segments.map((seg) => {
            if (seg.id !== segmentId) return seg;
            const targetW = seg.words[wordIndex];
            if (!targetW) return seg;

            const resolvedText =
              newText !== undefined
                ? newText
                : targetW.originalText || targetW.text;

            // 調用問題五 5.2 之雙向音韻-字面重疊自動消融演算法！
            const newWords = applySeamlessWordSpanReplacement(
              seg.words,
              wordIndex,
              wordIndex + 1,
              resolvedText,
              {
                status: 'normal',
                userLocked: true,
              }
            );

            return { ...seg, words: newWords };
          }),
        })),

      upsertSpeakers: (list) =>
        set((state) => {
          const next = { ...state.speakers };
          for (const sp of list) {
            if (!next[sp.id]) {
              next[sp.id] = sp;
            } else {
              next[sp.id] = {
                ...next[sp.id],
                centroid: sp.centroid || next[sp.id].centroid,
                f0Hz: sp.f0Hz ?? next[sp.id].f0Hz,
                embedding: sp.embedding || next[sp.id].embedding,
                anchorBank: sp.anchorBank || next[sp.id].anchorBank,
                accumulatedMass:
                  sp.accumulatedMass ?? next[sp.id].accumulatedMass,
                featureMode: sp.featureMode || next[sp.id].featureMode,
              };
            }
          }
          return { speakers: next };
        }),

      mergeSpeakers: (sourceId, targetId) =>
        set((state) => {
          if (sourceId === targetId || !state.speakers[targetId]) return state;
          const nextSpeakers = { ...state.speakers };
          const target = nextSpeakers[targetId];
          nextSpeakers[targetId] = {
            ...target,
            mergedFrom: Array.from(new Set([...target.mergedFrom, sourceId])),
          };
          delete nextSpeakers[sourceId];

          const nextSegments = state.segments.map((seg) =>
            seg.speakerId === sourceId ? { ...seg, speakerId: targetId } : seg
          );
          return {
            speakers: nextSpeakers,
            segments: smartConsolidateSegments(nextSegments),
          };
        }),

      renameSpeaker: (speakerId, newName) =>
        set((state) => {
          const sp = state.speakers[speakerId];
          if (!sp) return state;
          return {
            speakers: {
              ...state.speakers,
              [speakerId]: { ...sp, name: newName },
            },
          };
        }),

      setPhoneticCandidates: (candidates) =>
        set({ phoneticCandidates: candidates }),

      applyPhoneticReplacements: () =>
        set((state) => {
          const selected = state.phoneticCandidates.filter((c) => c.selected);
          if (selected.length === 0) return state;

          const nextSegments = state.segments.map((seg) => {
            const matches = selected.filter((c) => c.segmentId === seg.id);
            if (matches.length === 0) return seg;
            const words = seg.words.map((w, idx) => {
              const m = matches.find((c) => c.wordIndex === idx);
              if (!m) return w;
              return {
                ...w,
                originalText: w.originalText ?? w.text,
                text: m.targetSpeakerName,
                status: 'auto_corrected' as const,
                reason: '語者姓名同音連動修正',
              };
            });
            return { ...seg, words };
          });

          return {
            segments: nextSegments,
            phoneticCandidates: state.phoneticCandidates.filter(
              (c) => !c.selected
            ),
          };
        }),

      addDictEntries: (entries) =>
        set((state) => ({
          dictionary: mergeDictLists(state.dictionary, entries),
        })),

      removeDictEntry: (id) =>
        set((state) => ({
          dictionary: state.dictionary.filter((d) => d.id !== id),
        })),

      promoteDictEntryScope: (id: string, targetScope: DictScope) =>
        set((state) => ({
          dictionary: state.dictionary.map((d) =>
            d.id === id ? { ...d, scope: targetScope } : d
          ),
        })),

      setDocContextSummary: (summary) => set({ docContextSummary: summary }),
      setRollingMeetingSummary: (summary) =>
        set({ rollingMeetingSummary: summary }),
      setCurrentTime: (sec: number) => set({ currentTime: sec }),
    }),
    {
      name: 'hybrid-transcriber-storage-v6',
      partialize: (state) => ({
        config: state.config,
        dictionary: mergeDictLists(
          DEFAULT_PUBLIC_DICTIONARY,
          state.dictionary.filter((d) => d.scope !== 'session')
        ),
      }),
    }
  )
);