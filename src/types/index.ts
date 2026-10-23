// src/types/index.ts
// [SSOT] 專案唯一型別定義來源（含 Deep Research 多階段 AI 語意推敲、免索引槽位協定與不可變 Token 圖譜型別）

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
  | 'mode_0_heritage_classic'
  | 'mode_a_classic'
  | 'mode_b_advanced_dsp'
  | 'mode_c_onnx_neural';

export type DiarizationClusterMethod = 'ahc_cosine' | 'spectral_cosine';

export type DiarizationExecutionBackend = 'wasm' | 'webgpu';

export type CompliantOnnxSpeakerModel =
  | 'wespeaker_voxceleb_resnet34_int8.onnx'
  | 'nemo_en_titanet_small.int8.onnx';

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
  /** 微觀聲紋跳變曲線 d_SCD(t) 波峰切分敏感度門檻 (預設 0.22；越小切分越細敏，範圍 0.16 ~ 0.35) */
  scdPeakThreshold?: number;
  /** 短促附和詞保護之最小聲學邊際門檻 \Delta M(B, A) (預設 0.12；雙峰伴隨時自動折半) */
  backchannelMarginThreshold?: number;
  /** 雙軌無損聲紋差異強化預處理 (僅作用於聲紋分離軌，完全不影響 Whisper 原音辨識) */
  enableAcousticDivergencePreproc?: boolean;
  /**
   * 啟用「長短句分流定錨 + 語者專屬虛擬音軌拼接轉寫 (Virtual Speaker-Track Stitching)」
   * 預設為 false 以保持 Whisper 12~18s 連續自然語音視窗與句子完整度
   */
  enableVirtualSpeakerTrackStitching?: boolean;
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
  /** 是否為參與第一輪聚類定錨之黃金長句 (Anchor Segment) */
  isAnchorTurn?: boolean;
}

/**
 * 語者專屬虛擬音軌內之單一時間區段映射項 (Piecewise Linear Time Map Item)
 */
export interface VirtualTrackMapItem {
  /** 在虛擬拼接音軌中的相對起始秒數 */
  virtualStartSec: number;
  /** 在虛擬拼接音軌中的相對結束秒數 */
  virtualEndSec: number;
  /** 在原始會議音檔中的絕對起始秒數 */
  realStartSec: number;
  /** 在原始會議音檔中的絕對結束秒數 */
  realEndSec: number;
  speakerId: string;
  runnerUpSpeakerId?: string;
  speakerConfidence: number;
  speakerMargin: number;
  boundaryScdLeft: number;
  boundaryScdRight: number;
}

/**
 * 單一語者之虛擬音軌轉寫批次 (約 12~22 秒純同語者語音，中間以 0.35s 靜音隔板串接)
 */
export interface SpeakerTrackBatch {
  batchIndex: number;
  speakerId: string;
  /** 本批次涵蓋之最早原始時間 (用於進度顯示與排序) */
  earliestRealStartSec: number;
  /** 本批次涵蓋之最晚原始時間 */
  latestRealEndSec: number;
  /** 拼接後之總虛擬音軌秒數 */
  totalVirtualDurationSec: number;
  /** 片段間插入之零值靜音隔板秒數 (預設 0.35s) */
  spacerDurationSec: number;
  /** 虛擬時間軸與真實會議時間軸之分段線性映射表 */
  timeMap: VirtualTrackMapItem[];
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
  /** 若該 Token 已被使用者手動點選確認，標記為鎖定以維持冪等性 */
  userLocked?: boolean;
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

/**
 * 【問題一 & 問題二】：ISIAL 圈選出之免索引懷疑區段槽位映射項 (SlotMap Entry)
 */
export interface SuspectSlotItem {
  /** 槽位唯一識別碼，例如 "#S1" ~ "#S5" */
  slotId: string;
  segmentId: string;
  /** 對應 Segment.words 之起始索引 (含) */
  startWordIdx: number;
  /** 對應 Segment.words 之結束索引 (不含) */
  endWordIdx: number;
  originalText: string;
  leftContext: string;
  rightContext: string;
  anomalyScore: number;
  anomalyType:
    | 'orphan_char_chain'
    | 'phonetic_gravity_tension'
    | 'collocation_surprisal'
    | 'contraction_split';
  /** 三源混淆集候選網格 (Confusion Set Lattice) Top-K 候選詞 */
  candidates: string[];
}

/**
 * 【問題二 2.3】：XGrammar / Gemini 結構化槽位填空與語用拆句輸出格式
 */
export interface SlotCorrectionOutputItem {
  slot_id: string;
  /** 15 字以內微推理 (Micro-CoT)，必須在 action 與 selected_text 前生成 */
  brief_reason: string;
  action: 'KEEP_ORIGINAL' | 'APPLY_HIGH_CONF' | 'MARK_UNCERTAIN';
  selected_text: string;
  uncertain_candidates: string[];
}

export interface TurnSplitOutputItem {
  utterance_id: string;
  pragmatic_type:
    | 'NO_SPLIT'
    | 'QA_SPEAKER_CHANGE'
    | 'RHETORICAL_SELF_ANSWER'
    | 'BACKCHANNEL_INTERJECTION';
  /** 若為 QA_SPEAKER_CHANGE，填寫回答句開頭的精確 2~4 個字作為切分錨點 */
  split_before_anchor: string;
  new_speaker_id: string;
}

export interface StructuredSlotGerResponse {
  slot_corrections: SlotCorrectionOutputItem[];
  turn_splits: TurnSplitOutputItem[];
}

/**
 * 【問題五 5.1】：不可變 Token 圖譜資料結構定義 (Immutable Token Graph)
 */
export interface AtomicCharToken {
  readonly tokenId: string; // 唯一不可變 ID，如 "u102_t04"
  char: string; // 單一漢字或英文單詞
  pinyinKey: string; // 台式模糊音韻雜湊碼，如 "dian"
  startTimeMs: number; // 聲學起始時間戳記
  endTimeMs: number; // 聲學結束時間戳記
  acousticConf: number; // Whisper 原始信心度
  speakerId: string; // 經 LG-TPST 對齊後之語者 ID
}

export interface UncertainCandidateOption {
  readonly optionId: string;
  text: string; // 候選詞字串，如 "店面翻修"
  pinyinSequence: string[]; // 候選詞各字模糊音韻碼 ["dian", "mian", "fan", "xiu"]
  source: 'ORIGINAL' | 'LLM_PASS2' | 'BOOTSTRAPPED_HOTWORD' | 'DOMAIN_DICT';
  reason?: string;
}

export interface SuspectSpanNode {
  readonly spanId: string; // 對應 "#S1"
  status: 'CLEAN' | 'AUTO_CORRECTED' | 'SEMANTIC_UNCERTAIN' | 'USER_LOCKED';
  // 直接持有當前涵蓋之 AtomicCharToken 陣列，完全不依賴字元整數索引！
  activeTokens: AtomicCharToken[];
  originalTokens: AtomicCharToken[];
  candidates: UncertainCandidateOption[];
  selectedOptionId: string;
}

export interface UtteranceNode {
  readonly utteranceId: string;
  // 句子由一般 AtomicCharToken 與 SuspectSpanNode 交錯組成之有序序列
  segments: Array<AtomicCharToken | SuspectSpanNode>;
}

export interface CorrectionPatchItem {
  segmentId: string;
  /** 若為詞彙修正則為 >= 0 之起始詞索引；若為整段語者轉移或拆句 (LG-TPST) 則為 -1 */
  wordIndex: number;
  /** 槽位涵蓋之結束詞索引 (不含)，供多詞無縫消融替換使用 */
  endWordIndex?: number;
  originalText: string;
  status: 'auto_corrected' | 'semantic_uncertain';
  correctedText?: string;
  /** 當 [REVIEW] 段落經對話語用學判定需切換語者時，填入新語者 ID (例如 runnerUpSpeakerId) */
  newSpeakerId?: string;
  /** 若為段內問答對拆句 (QA_SPEAKER_CHANGE)，指定回答句開頭錨點字串 */
  splitBeforeAnchor?: string;
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
  /** 免索引槽位映射表 (#S1 ~ #S5) */
  suspectSlots?: SuspectSlotItem[];
  /** 已嵌入 ⟪#S1: 原詞 | 候選: ...⟫ 槽位標記之待校對文本 */
  slottedPromptTranscript?: string;
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