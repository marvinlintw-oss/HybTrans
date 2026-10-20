// src/types/index.ts
// [SSOT] 專案唯一型別定義來源

export type EngineMode = 'local_webgpu' | 'byok_cloud';

export type LocalAsrModel =
  | 'SenseVoice-Small'
  | 'Whisper-Large-v3-Turbo'
  | 'Breeze-ASR';

export type LocalLlmModel =
  | 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  | 'gemma-2-2b-it-q4f16_1-MLC';

export type GranularityTier = 'eco_safe' | 'balanced_pro' | 'ultra_studio';

export type DiarizationEngineMode =
  | 'mode_a_classic'
  | 'mode_b_advanced_dsp'
  | 'mode_c_onnx_neural';

export type DiarizationClusterMethod = 'ahc_cosine' | 'spectral_cosine';

export type DiarizationExecutionBackend = 'wasm' | 'webgpu';

export type CompliantOnnxSpeakerModel =
  | 'nemo_en_titanet_small.int8.onnx'
  | 'wespeaker_voxceleb_resnet34_int8.onnx';

export type MeetingLanguageHint = 'auto_multilingual' | 'zh' | 'en' | 'ja';

export interface DiarizationConfig {
  engineMode: DiarizationEngineMode;
  clusterMethod: DiarizationClusterMethod;
  executionBackend: DiarizationExecutionBackend;
  onnxModelFilename: CompliantOnnxSpeakerModel;
  localWasmBasePath: string;
  localModelBasePath: string;
  yieldEveryNUnits: number;
  enableGlobalMeanCentering: boolean;
  languageHint: MeetingLanguageHint;
  shortTurnConfidenceKeepThreshold: number;
  pingPongMaxDurSec: number;
  enablePauseAndSemanticWeighting: boolean;
}

/** 語者分離細部時間軸單元 (含 SCD 跳變值、微觀聲紋邊際與多原型對齊資訊) */
export interface FineDiarizationTurn {
  start: number;
  end: number;
  speakerId: string;
  runnerUpSpeakerId?: string;
  f0Hz: number;
  speakerConfidence?: number;
  speakerMargin?: number;
  /** 該時間點左側邊界之瞬時聲紋跳變度 d_SCD(t) (0.0 ~ 2.0) */
  boundaryScdLeft?: number;
  /** 該時間點右側邊界之瞬時聲紋跳變度 d_SCD(t) (0.0 ~ 2.0) */
  boundaryScdRight?: number;
  isProtectedBackchannel?: boolean;
}

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

export interface ModelDownloadStatus {
  active: boolean;
  modelName: string;
  fileName: string;
  progress: number;
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
  /** 聲學餘弦邊際 M(u_m)：>= 0.16 且 >= 1.2s 為 LOCKED 錨點，< 0.16 為 REVIEW 候選 */
  acousticMargin?: number;
  runnerUpSpeakerId?: string;
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
  f0Hz?: number;
  /** 品質加權主重心 Master Centroid (L2-normalized) */
  embedding?: number[];
  /** 多原型代表性錨點庫 Anchor Exemplar Sub-Bank (上限 M=6 個高純度向量) */
  anchorBank?: number[][];
  /** 累積有效高純度發聲秒數 W_g */
  accumulatedMass?: number;
  featureMode?: DiarizationEngineMode;
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
  diarizationConfig: DiarizationConfig;
}