// src/types/index.ts
// [SSOT] 專案唯一型別定義來源，禁止其他檔案私自定義共用介面

export type EngineMode = 'local_webgpu' | 'byok_cloud';

export type LocalAsrModel =
  | 'SenseVoice-Small'
  | 'Whisper-Large-v3-Turbo'
  | 'Breeze-ASR';

export type LocalLlmModel =
  | 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  | 'gemma-2-2b-it-q4f16_1-MLC';

export type GranularityTier = 'eco_safe' | 'balanced_pro' | 'ultra_studio';

export type ChunkStage =
  | 'idle'
  | 'preprocessing'
  | 'diarizing'
  | 'transcribing'
  | 'correcting'
  | 'relistening'
  | 'completed'
  | 'error';

export type WordUncertaintyStatus =
  | 'normal'
  | 'auto_corrected'
  | 'acoustic_uncertain'
  | 'semantic_uncertain';

export type DictScope = 'public' | 'personal' | 'session' | 'global';
export type DictCategory = 'person' | 'project' | 'jargon';

/** 背景下載或初始化 AI 模型之即時進度狀態 */
export interface ModelDownloadStatus {
  active: boolean;
  modelName: string;
  fileName: string;
  progress: number; // 0 ~ 100
  statusText: string;
}

export interface WordSpan {
  id: string;
  text: string;
  originalText?: string;
  start: number;
  end: number;
  confidence: number;
  status: WordUncertaintyStatus;
  reason?: string;
  candidates?: string[];
}

export interface Segment {
  id: string;
  chunkIndex: number;
  speakerId: string;
  start: number;
  end: number;
  words: WordSpan[];
  isCorrected: boolean;
  relistenCount?: number;
}

export interface CorrectionPatchItem {
  segmentId: string;
  wordIndex: number;
  originalText: string;
  status: 'auto_corrected' | 'semantic_uncertain';
  correctedText?: string;
  candidates?: string[];
  reason: string;
  requireRelisten?: boolean;
}

export interface TriLevelContextPayload {
  lowAttentionGlobalMemory: {
    docSummary: string;
    verifiedTerms: string[];
    earlierTopicDigest: string;
  };
  mediumAttentionRollingText: string;
  highAttentionTargetSegments: Segment[];
}

export interface TierPresetSpec {
  tier: GranularityTier;
  label: string;
  badgeText: string;
  hardwareHint: string;
  chunkDurationSec: number;
  keepModelsCoResident: boolean;
  recommendedLlm: LocalLlmModel;
  asrBeamSize: number;
  highAttentionChars: number;
  mediumAttentionChars: number;
  maxRelistenQuotaPerChunk: number;
  relistenPaddingSec: number;
}

export interface HardwareProbeResult {
  hasWebGPU: boolean;
  adapterName: string;
  isDiscreteGpu: boolean;
  maxBufferSizeMB: number;
  estimatedCacheUsageMB: number;
  recommendedTier: GranularityTier;
  warningMessage?: string;
}

export interface Speaker {
  id: string;
  name: string;
  color: string;
  centroid?: number[];
  mergedFrom: string[];
}

export interface PhoneticMatchCandidate {
  segmentId: string;
  wordIndex: number;
  matchedText: string;
  targetSpeakerName: string;
  contextPreview: string;
  selected: boolean;
}

export interface DictEntry {
  id: string;
  term: string;
  category: DictCategory;
  scope: DictScope;
  aliases?: string[];
}

export interface MacroChunkState {
  index: number;
  startTime: number;
  endTime: number;
  stage: ChunkStage;
  engineUsed: EngineMode;
  progress: number;
  /** 當前區塊內細部狀態文字 (如：「語音轉寫中 (14/67 句)」) */
  detailText?: string;
  relistenUsed?: number;
  error?: string;
}

export interface EngineConfig {
  mode: EngineMode;
  granularityTier: GranularityTier;
  localAsrModel: LocalAsrModel;
  localLlmModel: LocalLlmModel;
  groqApiKey: string;
  geminiApiKey: string;
  minSpeakers: number;
  maxSpeakers: number;
  acousticThreshold: number;
  chunkDurationSec: number;
  keepModelsCoResident: boolean;
  asrBeamSize: number;
  highAttentionChars: number;
  mediumAttentionChars: number;
  maxRelistenQuota: number;
  relistenPaddingSec: number;
}