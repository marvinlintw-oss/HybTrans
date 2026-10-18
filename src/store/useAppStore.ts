// src/store/useAppStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  CorrectionPatchItem,
  DictEntry,
  DictScope,
  EngineConfig,
  GranularityTier,
  MacroChunkState,
  ModelDownloadStatus,
  PhoneticMatchCandidate,
  Segment,
  Speaker,
  WordSpan,
} from '../types';
import { TIER_PRESET_SPECS } from '../services/modelCacheManager';
import { DEFAULT_PUBLIC_DICTIONARY, mergeDictLists } from '../utils/dictIO';

/**
 * 跨視窗與跨段落智慧整理器：
 * 1. 若後一段開頭是前一段的補充子句（如「沒有要支援店面翻修的」），自動將該子句搬回前一段語者！
 * 2. 若相鄰兩段為同一位語者且間隔 <= 5 秒、合併後 <= 260 字，自動合併為單一完整段落！
 */
function smartConsolidateSegments(allSegments: Segment[]): Segment[] {
  if (allSegments.length <= 1) return allSegments;

  const list: Segment[] = allSegments.map((s) => ({
    ...s,
    words: [...s.words],
  }));

  // Pass 1: 跨段落句子搬移 (若 curr 開頭包含「沒有要支援...，可是...」，把前半句搬給 prev！)
  for (let i = 1; i < list.length; i++) {
    const prev = list[i - 1];
    const curr = list[i];
    if (prev.speakerId !== curr.speakerId && curr.start - prev.end <= 2.0) {
      const currText = curr.words.map((w) => w.text).join('');
      const splitIdx = curr.words.findIndex((w, idx) =>
        idx > 0 && /^(可是|不是|不過|那如果)/.test(w.text)
      );

      if (
        splitIdx > 0 &&
        /^(沒有要支援|沒有要支持)/.test(currText)
      ) {
        const movedWords = curr.words.slice(0, splitIdx);
        const remainWords = curr.words.slice(splitIdx);
        if (movedWords.length > 0 && remainWords.length > 0) {
          // 將前半句句尾標點改為句號，搬回給上一位語者
          if (prev.words.length > 0) {
            const lastPrev = prev.words[prev.words.length - 1];
            if (lastPrev.text === '。') {
              prev.words[prev.words.length - 1] = { ...lastPrev, text: '，' };
            }
          }
          const lastMoved = movedWords[movedWords.length - 1];
          if (lastMoved.text === '，') {
            movedWords[movedWords.length - 1] = { ...lastMoved, text: '。' };
          }
          prev.words = [...prev.words, ...movedWords];
          prev.end = movedWords[movedWords.length - 1].end;
          curr.words = remainWords;
          curr.start = remainWords[0].start;
        }
      }
    }
  }

  // Pass 2: 同一個語者連續講的話自動合併成一段 (上限 260 字)
  const merged: Segment[] = [];
  for (const seg of list) {
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
      seg.start - prev.end <= 5.0 &&
      prevChars + currChars <= 260
    ) {
      const prevWords = [...prev.words];
      if (prevWords.length > 0) {
        const lastW = prevWords[prevWords.length - 1];
        if (lastW.text === '。') {
          prevWords[prevWords.length - 1] = { ...lastW, text: '，' };
        }
      }
      prev.words = [...prevWords, ...seg.words];
      prev.end = seg.end;
    } else {
      merged.push(seg);
    }
  }

  return merged;
}

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

const defaultTierSpec = TIER_PRESET_SPECS.eco_safe;

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      config: {
        mode: 'local_webgpu',
        granularityTier: 'eco_safe',
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

      applyGranularityTier: (tier) => {
        const spec = TIER_PRESET_SPECS[tier];
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
          },
        }));
      },

      initChunks: (totalDurationSec) => {
        const chunkSec = get().config.chunkDurationSec || 300;
        const count = Math.max(1, Math.ceil(totalDurationSec / chunkSec));
        const mode = get().config.mode;
        const chunks: MacroChunkState[] = Array.from({ length: count }, (_, idx) => ({
          index: idx,
          startTime: idx * chunkSec,
          endTime: Math.min(totalDurationSec, (idx + 1) * chunkSec),
          stage: 'idle',
          engineUsed: mode,
          progress: 0,
          detailText: '等待處理',
          relistenUsed: 0,
        }));
        set({ chunks });
      },

      updateChunkState: (index, partial) =>
        set((state) => ({
          chunks: state.chunks.map((c) => (c.index === index ? { ...c, ...partial } : c)),
        })),

      setModelDownload: (partial) =>
        set((state) => ({
          modelDownload: { ...state.modelDownload, ...partial },
        })),

      appendSegments: (newSegments) =>
        set((state) => ({
          segments: smartConsolidateSegments([...state.segments, ...newSegments]),
        })),

      replaceBatchSegments: (updatedSegments) =>
        set((state) => {
          const map = new Map(updatedSegments.map((s) => [s.id, s]));
          const replaced = state.segments.map((s) => map.get(s.id) || s);
          return {
            segments: smartConsolidateSegments(replaced),
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

          const updatedSegments = state.segments.map((seg) => {
            const segPatches = patchMap.get(seg.id);
            const shouldMarkCorrected =
              correctedSet.has(seg.id) || Boolean(segPatches?.length);
            if (!segPatches && !shouldMarkCorrected) return seg;

            const newWords = seg.words.map((w, idx) => {
              const found = segPatches?.find((p) => p.wordIndex === idx);
              if (!found) return w;
              return {
                ...w,
                originalText: w.originalText ?? w.text,
                text:
                  found.status === 'auto_corrected' && found.correctedText
                    ? found.correctedText
                    : w.text,
                status: found.status,
                candidates: found.candidates,
                reason: found.reason,
              };
            });

            return {
              ...seg,
              words: newWords,
              isCorrected: seg.isCorrected || shouldMarkCorrected,
            };
          });

          return { segments: smartConsolidateSegments(updatedSegments) };
        }),

      replaceSegmentAfterRelisten: (segmentId, newWords) =>
        set((state) => ({
          segments: smartConsolidateSegments(
            state.segments.map((seg) =>
              seg.id === segmentId
                ? {
                    ...seg,
                    words: newWords,
                    isCorrected: true,
                    relistenCount: (seg.relistenCount || 0) + 1,
                  }
                : seg
            )
          ),
        })),

      revertWord: (segmentId, wordIndex, newText) =>
        set((state) => ({
          segments: state.segments.map((seg) => {
            if (seg.id !== segmentId) return seg;
            const words = seg.words.map((w, idx) => {
              if (idx !== wordIndex) return w;
              const resolvedText =
                newText !== undefined ? newText : w.originalText || w.text;
              return {
                ...w,
                text: resolvedText,
                status: 'normal' as const,
              };
            });
            return { ...seg, words };
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

      promoteDictEntryScope: (id, targetScope) =>
        set((state) => ({
          dictionary: state.dictionary.map((d) =>
            d.id === id ? { ...d, scope: targetScope } : d
          ),
        })),

      setDocContextSummary: (summary) => set({ docContextSummary: summary }),
      setRollingMeetingSummary: (summary) => set({ rollingMeetingSummary: summary }),
      setCurrentTime: (sec) => set({ currentTime: sec }),
    }),
    {
      name: 'hybrid-transcriber-storage-v3',
      partialize: (state) => ({
        config: state.config,
        dictionary: state.dictionary.filter((d) => d.scope !== 'session'),
      }),
    }
  )
);