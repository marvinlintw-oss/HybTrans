// src/store/useAppStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
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
  WordSpan,
} from '../types';
import { TIER_PRESET_SPECS } from '../services/modelCacheManager';
import { DEFAULT_PUBLIC_DICTIONARY, mergeDictLists } from '../utils/dictIO';

function smartConsolidateSegments(allSegments: Segment[]): Segment[] {
  if (allSegments.length <= 1) return allSegments;

  const list: Segment[] = allSegments.map((s) => ({
    ...s,
    words: [...s.words],
  }));

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

const DEFAULT_DIARIZATION_CONFIG: DiarizationConfig = {
  engineMode: 'mode_b_advanced_dsp',
  clusterMethod: 'spectral_cosine',
  executionBackend: 'wasm',
  onnxModelFilename: 'nemo_en_titanet_small.int8.onnx',
  localWasmBasePath: './wasm/',
  localModelBasePath: './models/',
  yieldEveryNUnits: 12,
  enableGlobalMeanCentering: true,
  languageHint: 'zh',
  shortTurnConfidenceKeepThreshold: 0.68,
  pingPongMaxDurSec: 2.2,
  enablePauseAndSemanticWeighting: true,
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

      /**
       * 切換頂部「輕量穩定 / 進階平衡 / 旗艦精修」時，同步連動語者分離演算法深度與重聽配額！
       */
      applyGranularityTier: (tier) => {
        const spec = TIER_PRESET_SPECS[tier];
        const currentDiar = get().config.diarizationConfig || DEFAULT_DIARIZATION_CONFIG;

        const tierDiarPatch: Partial<DiarizationConfig> =
          tier === 'eco_safe'
            ? {
                engineMode: 'mode_a_classic',
                clusterMethod: 'ahc_cosine',
                yieldEveryNUnits: 8,
              }
            : tier === 'balanced_pro'
            ? {
                engineMode: 'mode_b_advanced_dsp',
                clusterMethod: 'spectral_cosine',
                yieldEveryNUnits: 14,
              }
            : {
                engineMode: 'mode_b_advanced_dsp',
                clusterMethod: 'spectral_cosine',
                yieldEveryNUnits: 24,
                shortTurnConfidenceKeepThreshold: 0.65,
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
                f0Hz: sp.f0Hz ?? next[sp.id].f0Hz,
                embedding: sp.embedding || next[sp.id].embedding,
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
      setRollingMeetingSummary: (summary) => set({ rollingMeetingSummary: summary }),
      setCurrentTime: (sec) => set({ currentTime: sec }),
    }),
    {
      name: 'hybrid-transcriber-storage-v4',
      partialize: (state) => ({
        config: state.config,
        dictionary: state.dictionary.filter((d) => d.scope !== 'session'),
      }),
    }
  )
);