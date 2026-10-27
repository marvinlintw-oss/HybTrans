// src/store/useAppStore.ts
// 升級與根因修正重點：
// 1. [解決句子順序錯位與跨語者黏合]：在 smartConsolidateSegments 中嚴格檢查兩段落之間是否夾有其他語者的插話，無插話時自動跨視窗合併同語者連續發言。
// 2. [解決 localStorage 舊快取強制開啟「專屬音軌轉寫」問題]：將 Zustand persist 儲存鍵升級並加入自動遷移 (merge)，強制預設關閉 enableVirtualSpeakerTrackStitching，直接走 12~20s 靜音谷底視窗 + 字級 Viterbi 對齊。
// 3. [新增會議 AI 摘要、重要結論與行動事項報告狀態 (meetingMinutes)]：支援一鍵生成、檢視與匯出結構化會議紀錄。

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  AtomicCharToken,
  AudioPreAnalysisProfile,
  AudioRecommendationOption,
  CorrectionPatchItem,
  DiarizationConfig,
  DictEntry,
  DictScope,
  EngineConfig,
  GranularityTier,
  MacroChunkState,
  MeetingMinutesReport,
  ModelDownloadStatus,
  PhoneticMatchCandidate,
  PipelineProgress,
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

export function syncSegmentDerivedFields(seg: Segment): Segment {
  const syncedWords: WordSpan[] = (seg.words || []).map((w) => ({
    ...w,
    word: w.text,
    originalWord: w.originalText,
    isPatched: w.status === 'auto_corrected',
    isCorrected: w.status === 'auto_corrected',
  }));
  const joinedText = syncedWords.map((w) => w.text).join('');
  const joinedRaw =
    seg.rawText ||
    syncedWords.map((w) => w.originalText || w.text).join('') ||
    joinedText;
  const integrityScore =
    seg.sentenceIntegrity?.score ?? seg.sentenceIntegrityScore ?? 100;
  const integrityIssues =
    seg.sentenceIntegrity?.issues ?? seg.sentenceIntegrityIssues ?? [];

  return {
    ...seg,
    words: syncedWords,
    text: joinedText,
    rawText: joinedRaw,
    polishedText: joinedText,
    isPolished: seg.isCorrected,
    ispolished: seg.isCorrected,
    overlapWarning:
      seg.overlapWarning ??
      ((seg.overlapRatio ?? 0) >= 0.25 ||
        seg.slotTier === 'tier3_overlap_split'),
    sentenceIntegrityScore: integrityScore,
    sentenceIntegrityIssues: integrityIssues,
  };
}

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
    candidate.pinyinSequence &&
    candidate.pinyinSequence.length === candChars.length
      ? candidate.pinyinSequence
      : toTaiwanFuzzyPhoneticSyllables(candidate.text);
  const M = candChars.length;

  const isTokenMatch = (
    tok: AtomicCharToken,
    char: string,
    pinyin: string
  ): boolean => {
    return (
      tok.char === char || (tok.pinyinKey !== '' && tok.pinyinKey === pinyin)
    );
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
      word: prefixKeep,
      end: newStartSec,
    });
  }

  const finalTxt = effectiveText || replacementText;
  resultWords.push({
    ...targetAnchorWord,
    text: finalTxt,
    word: finalTxt,
    originalText: preservedOriginal,
    originalWord: preservedOriginal,
    start: newStartSec,
    end: newEndSec,
    confidence:
      options?.status === 'auto_corrected'
        ? 0.96
        : targetAnchorWord.confidence,
    status: options?.status ?? 'normal',
    isPatched: options?.status === 'auto_corrected',
    isCorrected: options?.status === 'auto_corrected',
    reason: options?.reason ?? targetAnchorWord.reason,
    candidates: options?.candidates ?? targetAnchorWord.candidates,
    userLocked: options?.userLocked ?? targetAnchorWord.userLocked,
  });

  if (suffixKeep) {
    resultWords.push({
      ...words[absorbedWordEnd - 1],
      id: `${words[absorbedWordEnd - 1].id}_suf`,
      text: suffixKeep,
      word: suffixKeep,
      start: newEndSec,
    });
  }

  for (let i = absorbedWordEnd; i < words.length; i++) {
    resultWords.push(words[i]);
  }

  return resultWords;
}

const DISCOURSE_TOPIC_SHIFT_REGEX =
  /^(?:然後|那|另外|再來|此外|同時|最後)?(?:第[二三四五六七八九十][個點項部分步]|另外一個(?:部分|問題|重點|面向)|再來(?:就是|是|一個)|那另外(?:就是|一個)|接下來(?:我們|就是)|至於(?:在|說)?)/;

function analyzeCrossWindowBoundaryStitchMode(
  prevLastWordText: string,
  currFirstWordText: string,
  pauseGap = 0.5
): 'strip' | 'comma' | 'keep' {
  const prevClean = prevLastWordText.replace(/[，。？！、；：,.?!\s]+$/g, '');
  const currClean = currFirstWordText.replace(/^[，。？！、；：,.?!\s]+/g, '');
  if (!prevClean || !currClean) return 'keep';

  if (
    /(?:[的之比在跟對向把被讓給為從到]|那個|這個|一些|一個|我們有|他們有|協助他們去)$/.test(
      prevClean
    ) ||
    (/^[的之得地了嗎呢吧啊呀喔哦啦囉嘛齁吼]/.test(currClean) &&
      currClean.length <= 3) ||
    (/(?:預算|報價|規模|問題|部分|角色|活動|合作|費用|計畫|補助)的$/.test(
      prevClean
    ) &&
      /^(?:狀況|部分|問題|規模|方式|費用|做法|辦法|方向)/.test(currClean))
  ) {
    return 'strip';
  }

  if (
    /(?:要能夠|能夠|可以|應該|就是|因為|所以|而且|然後|例如說|比方說|像是|主要是|或者是|但如果|的時候|的話)$/.test(
      prevClean
    ) ||
    /^(?:就是|因為|所以|而且|但是|可是|不過|假設|如果|例如|比方|那目前|或者是|他們現在|現在就是|然後一個|另外一個)/.test(
      currClean
    ) ||
    (pauseGap <= 0.75 && !/[？！?!]$/.test(prevLastWordText.trim()))
  ) {
    return 'comma';
  }

  return 'keep';
}

function splitLongMonologueAtSemanticBoundaries(
  segments: Segment[]
): Segment[] {
  const result: Segment[] = [];

  for (const seg of segments) {
    const totalChars = seg.words.reduce((acc, w) => acc + w.text.length, 0);
    if (totalChars < 200 || seg.words.length < 12) {
      result.push(seg);
      continue;
    }

    let splitWordIdx = -1;
    let accChars = 0;

    for (let i = 0; i < seg.words.length - 3; i++) {
      accChars += seg.words[i].text.length;
      const remChars = totalChars - accChars;
      if (accChars < 70 || remChars < 60) continue;

      const currW = seg.words[i].text.trim();
      const lookaheadText = seg.words
        .slice(i + 1, Math.min(seg.words.length, i + 5))
        .map((w) => w.text)
        .join('')
        .trim();

      const prevEndsSentence = /[。？！.?!]$/.test(currW);
      const prevEndsTopicSummary =
        /(?:這是第[一二三]個|這是另一個|有點不太一樣|比較好|標示出來|先說明一下)[，。？！.?!]*$/.test(
          seg.words
            .slice(Math.max(0, i - 2), i + 1)
            .map((w) => w.text)
            .join('')
        );

      if (
        (prevEndsSentence || prevEndsTopicSummary) &&
        DISCOURSE_TOPIC_SHIFT_REGEX.test(lookaheadText)
      ) {
        splitWordIdx = i + 1;
        break;
      }
    }

    if (splitWordIdx > 0 && splitWordIdx < seg.words.length) {
      const firstWords = seg.words.slice(0, splitWordIdx).map((w, idx, arr) => {
        if (idx === arr.length - 1 && !/[。？！.?!]$/.test(w.text)) {
          const clean = w.text.replace(/[，、；：,:;\s]+$/g, '');
          return { ...w, text: `${clean}。`, word: `${clean}。` };
        }
        return w;
      });
      const secondWords = seg.words.slice(splitWordIdx);
      const splitTime = Number(secondWords[0].start.toFixed(2));

      const firstSeg = syncSegmentDerivedFields({
        ...seg,
        id: `${seg.id}_p1`,
        end: Math.max(seg.start + 0.2, splitTime),
        words: firstWords,
      });
      const secondSeg = syncSegmentDerivedFields({
        ...seg,
        id: `${seg.id}_p2`,
        start: splitTime,
        words: secondWords,
      });

      result.push(
        ...splitLongMonologueAtSemanticBoundaries([firstSeg, secondSeg])
      );
    } else {
      result.push(seg);
    }
  }

  return result;
}

function smartConsolidateSegments(allSegments: Segment[]): Segment[] {
  if (allSegments.length === 0) return [];
  if (allSegments.length === 1) return [syncSegmentDerivedFields(allSegments[0])];

  const sorted: Segment[] = allSegments
    .map((s) =>
      syncSegmentDerivedFields({
        ...s,
        words: [...s.words],
      })
    )
    .sort((a, b) => {
      if (Math.abs(a.start - b.start) > 0.08) return a.start - b.start;
      return a.end - b.end;
    });

  const merged: Segment[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const seg = sorted[i];
    if (seg.words.length === 0) continue;
    const prev = merged[merged.length - 1];
    const prevChars = prev
      ? prev.words.reduce((acc, w) => acc + w.text.length, 0)
      : 0;
    const currChars = seg.words.reduce((acc, w) => acc + w.text.length, 0);

    let hasInterveningOtherSpeaker = false;
    if (prev && prev.speakerId === seg.speakerId) {
      for (let j = 0; j < sorted.length; j++) {
        const other = sorted[j];
        if (other.id === prev.id || other.id === seg.id) continue;
        if (
          other.speakerId !== prev.speakerId &&
          other.end > prev.start + 0.05 &&
          other.start < seg.end - 0.05
        ) {
          hasInterveningOtherSpeaker = true;
          break;
        }
      }
    }

    const pauseGap = prev ? seg.start - prev.end : 999;
    const prevLastText =
      prev && prev.words.length > 0
        ? prev.words[prev.words.length - 1].text
        : '';
    const currFullText = seg.words.map((w) => w.text).join('');
    const currFirstText = seg.words[0]?.text || '';

    const stitchMode = prev
      ? analyzeCrossWindowBoundaryStitchMode(
          prevLastText,
          currFirstText,
          pauseGap
        )
      : 'keep';
    const isMidSentenceCut = stitchMode === 'strip' || stitchMode === 'comma';

    const isSemanticTopicBreak =
      !isMidSentenceCut &&
      prevChars >= 140 &&
      (DISCOURSE_TOPIC_SHIFT_REGEX.test(currFullText.trim()) ||
        (prevChars >= 240 && pauseGap >= 2.2) ||
        prevChars + currChars > 380);

    if (
      prev &&
      !hasInterveningOtherSpeaker &&
      prev.chunkIndex === seg.chunkIndex &&
      prev.speakerId === seg.speakerId &&
      seg.start >= prev.end - 0.25 &&
      pauseGap <= 12.0 &&
      (isMidSentenceCut || !isSemanticTopicBreak)
    ) {
      const prevWords = [...prev.words];
      if (prevWords.length > 0) {
        const lastW = prevWords[prevWords.length - 1];
        const cleanLast = lastW.text.replace(/[，。？！、；：,.?!\s]+$/g, '');
        if (stitchMode === 'strip') {
          prevWords[prevWords.length - 1] = {
            ...lastW,
            text: cleanLast,
            word: cleanLast,
          };
        } else if (stitchMode === 'comma') {
          prevWords[prevWords.length - 1] = {
            ...lastW,
            text: `${cleanLast}，`,
            word: `${cleanLast}，`,
          };
        } else if (!/[，。？！、；：,.?!]$/.test(lastW.text)) {
          prevWords[prevWords.length - 1] = {
            ...lastW,
            text: `${lastW.text}，`,
            word: `${lastW.text}，`,
          };
        }
      }
      prev.words = [...prevWords, ...seg.words];
      prev.end = Math.max(prev.end, seg.end);
      prev.acousticMargin = Math.max(
        prev.acousticMargin ?? 0.15,
        seg.acousticMargin ?? 0.15
      );
      merged[merged.length - 1] = syncSegmentDerivedFields(prev);
    } else {
      merged.push(syncSegmentDerivedFields(seg));
    }
  }

  // 消除因跨視窗或 Stage 1 合併後殘留之重疊子句重複段（如前段尾端已包含後段全文）
  const deduped: Segment[] = [];
  for (const seg of merged) {
    const prev = deduped[deduped.length - 1];
    if (prev && prev.chunkIndex === seg.chunkIndex) {
      const prevPure = (prev.text || '').replace(/[，。？！、；：,.?!\s]/g, '');
      const currPure = (seg.text || '').replace(/[，。？！、；：,.?!\s]/g, '');
      if (
        currPure.length >= 2 &&
        currPure.length <= prevPure.length &&
        seg.start <= prev.end + 0.8 &&
        prevPure.slice(-Math.min(prevPure.length, currPure.length + 18)).includes(currPure)
      ) {
        continue;
      }
    }
    deduped.push(seg);
  }

  return splitLongMonologueAtSemanticBoundaries(deduped);
}

const DEFAULT_DIARIZATION_CONFIG: DiarizationConfig = {
  engineMode: 'mode_0_heritage_classic',
  strategyMode: 'MODE_0_GOLDEN_TWO_STAGE',
  clusterMethod: 'ahc_cosine',
  clusteringMethod: 'ahc_cosine',
  executionBackend: 'wasm',
  backend: 'wasm',
  onnxModelFilename: 'wespeaker_voxceleb_resnet34_int8.onnx',
  embeddingModel: 'wespeaker_voxceleb_resnet34_int8.onnx',
  localWasmBasePath: './wasm/',
  localModelBasePath: './models/',
  yieldEveryNUnits: 14,
  enableGlobalMeanCentering: false,
  languageHint: 'zh',
  shortTurnConfidenceKeepThreshold: 0.66,
  pingPongMaxDurSec: 1.6,
  enablePauseAndSemanticWeighting: true,
  scdPeakThreshold: 0.22,
  backchannelMarginThreshold: 0.12,
  enableAcousticDivergencePreproc: true,
  enableVirtualSpeakerTrackStitching: false,
  embeddingExcludeOverlap: true,
};

export interface AppState {
  config: EngineConfig;
  engineConfig: EngineConfig;
  diarizationConfig: DiarizationConfig;
  activeDiarizationBackend: string;
  audioFile: File | null;
  audioUrl: string | null;
  duration: number;
  currentTime: number;
  isPlaying: boolean;
  chunks: MacroChunkState[];
  isRunning: boolean;
  progress: PipelineProgress;
  segments: Segment[];
  activeSegmentId: string | null;
  speakers: Record<string, Speaker>;
  dictionary: DictEntry[];
  docContextSummary: string;
  rollingMeetingSummary: string;
  briefingSummary: string;
  meetingMinutes: MeetingMinutesReport | null;
  isGeneratingMinutes: boolean;
  phoneticCandidates: PhoneticMatchCandidate[];
  modelDownload: ModelDownloadStatus;
  audioPreAnalysis: AudioPreAnalysisProfile | null;
  isPreAnalyzingAudio: boolean;
  preAnalysisProgress: { percent: number; detailText: string };
  selectedPresetId: string;

  updateConfig: (partial: Partial<EngineConfig>) => void;
  setEngineConfig: (partial: Partial<EngineConfig>) => void;
  updateDiarizationConfig: (partial: Partial<DiarizationConfig>) => void;
  setDiarizationConfig: (partial: Partial<DiarizationConfig>) => void;
  setActiveDiarizationBackend: (backend: string) => void;
  applyGranularityTier: (tier: GranularityTier) => void;
  applyRecommendedPreset: (preset: AudioRecommendationOption | string) => void;

  setAudioFile: (file: File | null, url?: string | null) => void;
  setDuration: (sec: number) => void;
  setCurrentTime: (sec: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setAudioPreAnalysis: (profile: AudioPreAnalysisProfile | null) => void;
  setIsPreAnalyzingAudio: (analyzing: boolean) => void;
  setPreAnalysisProgress: (percent: number, detailText: string) => void;

  initChunks: (totalDurationSec: number) => void;
  updateChunkState: (index: number, partial: Partial<MacroChunkState>) => void;
  setIsRunning: (running: boolean) => void;
  setProgress: (partial: Partial<PipelineProgress>) => void;
  setModelDownload: (status: Partial<ModelDownloadStatus>) => void;

  appendSegments: (newSegments: Segment[]) => void;
  appendSegment: (newSegment: Segment) => void;
  replaceBatchSegments: (
    updatedSegments: Segment[],
    sourceSegmentIds?: string[]
  ) => void;
  updateSegment: (segmentId: string, partial: Partial<Segment>) => void;
  updateSegmentText: (segmentId: string, newText: string) => void;
  updateSegmentSpeaker: (segmentId: string, newSpeakerId: string) => void;
  toggleSegmentLock: (segmentId: string) => void;
  applyCorrectionPatches: (
    patches: CorrectionPatchItem[],
    markSegmentIdsCorrected?: string[]
  ) => void;
  replaceSegmentAfterRelisten: (segmentId: string, newWords: WordSpan[]) => void;
  revertWord: (segmentId: string, wordIndex: number, newText?: string) => void;
  setActiveSegmentId: (segmentId: string | null) => void;
  clearTranscript: (clearSessionDictOrEvent?: boolean | unknown) => void;

  upsertSpeakers: (list: Speaker[] | Record<string, Speaker>) => void;
  mergeSpeakers: (sourceId: string, targetId: string) => void;
  renameSpeaker: (speakerId: string, newName: string) => void;
  setPhoneticCandidates: (candidates: PhoneticMatchCandidate[]) => void;
  applyPhoneticReplacements: () => void;
  addDictEntries: (entries: DictEntry[]) => void;
  removeDictEntry: (id: string) => void;
  promoteDictEntryScope: (id: string, targetScope: DictScope) => void;
  setDocContextSummary: (summary: string) => void;
  setRollingMeetingSummary: (summary: string) => void;
  setBriefingSummary: (summary: string) => void;
  setMeetingMinutes: (report: MeetingMinutesReport | null) => void;
  setIsGeneratingMinutes: (generating: boolean) => void;
}

const defaultTierSpec =
  TIER_PRESET_SPECS.balanced_pro || TIER_PRESET_SPECS.eco_safe;

const INITIAL_ENGINE_CONFIG: EngineConfig = {
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
  diarization: DEFAULT_DIARIZATION_CONFIG,
};

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      config: INITIAL_ENGINE_CONFIG,
      engineConfig: INITIAL_ENGINE_CONFIG,
      diarizationConfig: DEFAULT_DIARIZATION_CONFIG,
      activeDiarizationBackend: 'WeSpeaker 256D + Dual-Stage Anchor',
      audioFile: null,
      audioUrl: null,
      duration: 0,
      currentTime: 0,
      isPlaying: false,
      chunks: [],
      isRunning: false,
      progress: {
        active: false,
        currentChunkIndex: 0,
        totalChunks: 0,
        stage: 'idle',
        overallProgress: 0,
        detailText: '等待載入音檔',
      },
      segments: [],
      activeSegmentId: null,
      speakers: {},
      dictionary: DEFAULT_PUBLIC_DICTIONARY,
      docContextSummary: '',
      rollingMeetingSummary: '',
      briefingSummary: '',
      meetingMinutes: null,
      isGeneratingMinutes: false,
      phoneticCandidates: [],
      modelDownload: {
        active: false,
        modelName: '',
        fileName: '',
        progress: 0,
        statusText: '',
      },
      audioPreAnalysis: null,
      isPreAnalyzingAudio: false,
      preAnalysisProgress: {
        percent: 0,
        detailText: '',
      },
      selectedPresetId: 'preset_a_balanced',

      updateConfig: (partial) =>
        set((state) => {
          const nextDiar = partial.diarizationConfig
            ? {
                ...state.config.diarizationConfig,
                ...partial.diarizationConfig,
              }
            : state.config.diarizationConfig;
          const nextConfig: EngineConfig = {
            ...state.config,
            ...partial,
            diarizationConfig: nextDiar,
            diarization: nextDiar,
          };
          return {
            config: nextConfig,
            engineConfig: nextConfig,
            diarizationConfig: nextDiar,
          };
        }),

      setEngineConfig: (partial) => get().updateConfig(partial),

      updateDiarizationConfig: (partial) =>
        set((state) => {
          const nextDiar: DiarizationConfig = {
            ...(state.config.diarizationConfig || DEFAULT_DIARIZATION_CONFIG),
            ...partial,
          };
          const nextConfig: EngineConfig = {
            ...state.config,
            diarizationConfig: nextDiar,
            diarization: nextDiar,
          };
          return {
            config: nextConfig,
            engineConfig: nextConfig,
            diarizationConfig: nextDiar,
          };
        }),

      setDiarizationConfig: (partial) => get().updateDiarizationConfig(partial),

      setActiveDiarizationBackend: (backend) =>
        set({ activeDiarizationBackend: backend }),

      applyGranularityTier: (tier) => {
        const spec = TIER_PRESET_SPECS[tier];
        const currentDiar =
          get().config.diarizationConfig || DEFAULT_DIARIZATION_CONFIG;

        const tierDiarPatch: Partial<DiarizationConfig> =
          tier === 'eco_safe'
            ? {
                engineMode: 'mode_a_classic',
                strategyMode: 'MODE_A_CLASSIC_DSP',
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
                strategyMode: 'MODE_0_GOLDEN_TWO_STAGE',
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
                strategyMode: 'MODE_C_ONNX_NEURAL',
                clusterMethod: 'ahc_cosine',
                yieldEveryNUnits: 24,
                scdPeakThreshold: 0.19,
                backchannelMarginThreshold: 0.1,
                shortTurnConfidenceKeepThreshold: 0.6,
                pingPongMaxDurSec: 1.8,
                enableAcousticDivergencePreproc: true,
                enableVirtualSpeakerTrackStitching: false,
              };

        const preservedCalibration =
          currentDiar.formulaCalibration ||
          get().audioPreAnalysis?.formulaCalibration;

        const nextDiar: DiarizationConfig = {
          ...currentDiar,
          ...tierDiarPatch,
          ...(preservedCalibration
            ? { formulaCalibration: preservedCalibration }
            : {}),
        };

        set((state) => {
          const nextConfig: EngineConfig = {
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
            diarizationConfig: nextDiar,
            diarization: nextDiar,
          };
          return {
            config: nextConfig,
            engineConfig: nextConfig,
            diarizationConfig: nextDiar,
          };
        });
      },

      applyRecommendedPreset: (presetOrId) => {
        const state = get();
        const resolvedPreset: AudioRecommendationOption | undefined =
          typeof presetOrId === 'string'
            ? state.audioPreAnalysis?.recommendations.find(
                (r) => r.id === presetOrId
              )
            : presetOrId;

        if (!resolvedPreset) return;

        const nextDiar: DiarizationConfig = {
          ...(state.config.diarizationConfig || DEFAULT_DIARIZATION_CONFIG),
          ...resolvedPreset.diarizationPatch,
        };
        const nextConfig: EngineConfig = {
          ...state.config,
          ...(resolvedPreset.enginePatch || {}),
          minSpeakers: resolvedPreset.recommendedMinSpeakers,
          maxSpeakers: resolvedPreset.recommendedMaxSpeakers,
          diarizationConfig: nextDiar,
          diarization: nextDiar,
        };

        set({
          config: nextConfig,
          engineConfig: nextConfig,
          diarizationConfig: nextDiar,
          selectedPresetId: resolvedPreset.id,
        });
      },

      setAudioFile: (file, url) =>
        set((state) => {
          if (state.audioUrl && state.audioUrl !== url) {
            try {
              URL.revokeObjectURL(state.audioUrl);
            } catch {
              // ignore
            }
          }
          const nextUrl =
            url !== undefined
              ? url
              : file
              ? URL.createObjectURL(file)
              : null;
          return {
            audioFile: file,
            audioUrl: nextUrl,
          };
        }),

      setDuration: (sec) => set({ duration: sec }),
      setCurrentTime: (sec) => set({ currentTime: sec }),
      setIsPlaying: (playing) => set({ isPlaying: playing }),
      setAudioPreAnalysis: (profile) =>
        set({
          audioPreAnalysis: profile,
          selectedPresetId:
            profile?.recommendedPresetId || get().selectedPresetId,
        }),
      setIsPreAnalyzingAudio: (analyzing) =>
        set({ isPreAnalyzingAudio: analyzing }),
      setPreAnalysisProgress: (percent, detailText) =>
        set({
          preAnalysisProgress: { percent, detailText },
        }),

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
        set({
          chunks,
          duration: totalDurationSec,
          progress: {
            active: true,
            currentChunkIndex: 0,
            totalChunks: count,
            stage: 'idle',
            overallProgress: 0,
            detailText: '準備開始微批次處理',
          },
        });
      },

      updateChunkState: (index, partial) =>
        set((state) => {
          const nextChunks = state.chunks.map((c) =>
            c.index === index ? { ...c, ...partial } : c
          );
          const total = Math.max(1, nextChunks.length);
          const completed = nextChunks.filter(
            (c) => c.stage === 'completed' || c.stage === 'COMPLETED'
          ).length;
          const active = nextChunks[index];
          const overallProgress = Math.min(
            100,
            Math.round(
              ((completed * 100 +
                (active &&
                active.stage !== 'completed' &&
                active.stage !== 'COMPLETED'
                  ? active.progress
                  : 0)) /
                (total * 100)) *
                100
            )
          );
          return {
            chunks: nextChunks,
            progress: {
              active: completed < total,
              currentChunkIndex: index,
              totalChunks: total,
              stage: partial.stage || active?.stage || 'idle',
              overallProgress,
              detailText:
                partial.detailText || active?.detailText || '處理中...',
            },
          };
        }),

      setIsRunning: (running) => set({ isRunning: running }),
      setProgress: (partial) =>
        set((state) => ({
          progress: { ...state.progress, ...partial },
        })),

      setModelDownload: (partial) =>
        set((state) => ({
          modelDownload: { ...state.modelDownload, ...partial },
        })),

      appendSegments: (newSegments) =>
        set((state) => ({
          segments: smartConsolidateSegments([
            ...state.segments,
            ...newSegments.map(syncSegmentDerivedFields),
          ]),
        })),

      appendSegment: (newSegment) =>
        get().appendSegments([syncSegmentDerivedFields(newSegment)]),

      replaceBatchSegments: (updatedSegments, sourceSegmentIds) =>
        set((state) => {
          const syncedList = updatedSegments.map(syncSegmentDerivedFields);
          const map = new Map(syncedList.map((s) => [s.id, s]));
          const sourceSet = new Set(sourceSegmentIds || []);
          const existingIdSet = new Set(state.segments.map((s) => s.id));

          // 若某個原始段落 ID 位於 sourceSegmentIds 中，但在 Stage 1 合併後已併入前段而不在 map 中，必須移除而不可保留舊複本！
          const replaced: Segment[] = [];
          for (const s of state.segments) {
            if (map.has(s.id)) {
              replaced.push(map.get(s.id)!);
            } else if (sourceSet.has(s.id)) {
              continue;
            } else {
              // 檢查是否已被 syncedList 中時間跨度擴大之段落吞併
              const sPure = (s.text || '').replace(/[，。？！、；：,.?!\s]/g, '');
              const isSwallowed =
                sPure.length >= 2 &&
                syncedList.some((u) => {
                  if (u.id === s.id || u.chunkIndex !== s.chunkIndex) return false;
                  if (s.start < u.start - 0.15 || s.end > u.end + 0.25) return false;
                  const uPure = (u.text || '').replace(/[，。？！、；：,.?!\s]/g, '');
                  return uPure.length > sPure.length && uPure.includes(sPure);
                });
              if (!isSwallowed) {
                replaced.push(s);
              }
            }
          }

          const brandNew = syncedList.filter((s) => !existingIdSet.has(s.id));
          return {
            segments: smartConsolidateSegments([...replaced, ...brandNew]),
          };
        }),

      updateSegment: (segmentId, partial) =>
        set((state) => ({
          segments: state.segments.map((seg) =>
            seg.id === segmentId
              ? syncSegmentDerivedFields({ ...seg, ...partial })
              : seg
          ),
        })),

      updateSegmentText: (segmentId, newText) =>
        set((state) => ({
          segments: state.segments.map((seg) => {
            if (seg.id !== segmentId) return seg;
            const replacedWords = applySeamlessWordSpanReplacement(
              seg.words,
              0,
              seg.words.length,
              newText,
              { status: 'normal', userLocked: true }
            );
            return syncSegmentDerivedFields({
              ...seg,
              words: replacedWords,
              isCorrected: true,
            });
          }),
        })),

      updateSegmentSpeaker: (segmentId, newSpeakerId) =>
        set((state) => ({
          segments: state.segments.map((seg) =>
            seg.id === segmentId
              ? syncSegmentDerivedFields({ ...seg, speakerId: newSpeakerId })
              : seg
          ),
        })),

      toggleSegmentLock: (segmentId) =>
        set((state) => ({
          segments: state.segments.map((seg) =>
            seg.id === segmentId
              ? syncSegmentDerivedFields({ ...seg, isLocked: !seg.isLocked })
              : seg
          ),
        })),

      setActiveSegmentId: (segmentId) => set({ activeSegmentId: segmentId }),

      clearTranscript: (clearSessionDictOrEvent) => {
        const shouldClearSession = clearSessionDictOrEvent === true;
        set((state) => ({
          chunks: [],
          segments: [],
          speakers: {},
          phoneticCandidates: [],
          rollingMeetingSummary: '',
          briefingSummary: '',
          meetingMinutes: null,
          isGeneratingMinutes: false,
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
                      syncSegmentDerivedFields({
                        ...seg,
                        end: splitTime,
                        words: beforeWords,
                        isCorrected: true,
                      }),
                      syncSegmentDerivedFields({
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
                      })
                    );
                    continue;
                  }
                }
              }
            }

            const nextSpeakerId = speakerPatch?.newSpeakerId || seg.speakerId;
            let currentWords = [...seg.words];

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

            updatedSegments.push(
              syncSegmentDerivedFields({
                ...seg,
                speakerId: nextSpeakerId,
                words: currentWords,
                isCorrected: seg.isCorrected || shouldMarkCorrected,
              })
            );
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
              ? syncSegmentDerivedFields({
                  ...seg,
                  words: newWords,
                  isCorrected: true,
                  relistenCount: (seg.relistenCount || 0) + 1,
                })
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

            return syncSegmentDerivedFields({ ...seg, words: newWords });
          }),
        })),

      upsertSpeakers: (listOrRecord) =>
        set((state) => {
          const list = Array.isArray(listOrRecord)
            ? listOrRecord
            : Object.values(listOrRecord || {});
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
            seg.speakerId === sourceId
              ? syncSegmentDerivedFields({ ...seg, speakerId: targetId })
              : seg
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
                originalWord: w.originalText ?? w.text,
                text: m.targetSpeakerName,
                word: m.targetSpeakerName,
                status: 'auto_corrected' as const,
                isPatched: true,
                isCorrected: true,
                reason: '語者姓名同音連動修正',
              };
            });
            return syncSegmentDerivedFields({ ...seg, words });
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

      promoteDictEntryScope: (id, targetScope) =>
        set((state) => ({
          dictionary: state.dictionary.map((d) =>
            d.id === id ? { ...d, scope: targetScope } : d
          ),
        })),

      setDocContextSummary: (summary) => set({ docContextSummary: summary }),
      setRollingMeetingSummary: (summary) =>
        set({ rollingMeetingSummary: summary, briefingSummary: summary }),
      setBriefingSummary: (summary) =>
        set({ rollingMeetingSummary: summary, briefingSummary: summary }),
      setMeetingMinutes: (report) => set({ meetingMinutes: report }),
      setIsGeneratingMinutes: (generating) =>
        set({ isGeneratingMinutes: generating }),
    }),
    {
      name: 'hybtrans-rwd-storage-v10',
      partialize: (state) => ({
        config: state.config,
        dictionary: mergeDictLists(
          DEFAULT_PUBLIC_DICTIONARY,
          state.dictionary.filter((d) => d.scope !== 'session')
        ),
      }),
      merge: (persistedState, currentState) => {
        let p = (persistedState || {}) as Partial<AppState>;
        if (!p.config && typeof window !== 'undefined') {
          try {
            const legacyRaw = window.localStorage.getItem('hybrid-transcriber-storage-v9');
            if (legacyRaw) {
              const legacyParsed = JSON.parse(legacyRaw) as {
                state?: {
                  config?: {
                    geminiApiKey?: string;
                    groqApiKey?: string;
                  };
                };
              };
              const legacyCfg = legacyParsed?.state?.config;
              if (legacyCfg) {
                p = {
                  ...p,
                  config: {
                    ...currentState.config,
                    geminiApiKey:
                      legacyCfg.geminiApiKey || currentState.config.geminiApiKey,
                    groqApiKey:
                      legacyCfg.groqApiKey || currentState.config.groqApiKey,
                  },
                };
              }
            }
          } catch {
            // 忽略單向讀取異常
          }
        }
        const mergedDiar: DiarizationConfig = {
          ...DEFAULT_DIARIZATION_CONFIG,
          ...(p.config?.diarizationConfig || {}),
          enableVirtualSpeakerTrackStitching: false,
        };
        const mergedConfig: EngineConfig = {
          ...currentState.config,
          ...(p.config || {}),
          diarizationConfig: mergedDiar,
          diarization: mergedDiar,
        };
        const mergedDict = mergeDictLists(
          DEFAULT_PUBLIC_DICTIONARY,
          Array.isArray(p.dictionary) ? p.dictionary : []
        );
        return {
          ...currentState,
          ...p,
          dictionary: mergedDict,
          config: mergedConfig,
          engineConfig: mergedConfig,
          diarizationConfig: mergedDiar,
        };
      },
    }
  )
);
