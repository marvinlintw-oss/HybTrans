// src/types/index.ts
// [SSOT] 專案唯一型別定義來源（含 Deep Research 多階段 AI 語意推敲、免索引槽位協定、不可變 Token 圖譜與次世代字級 Viterbi / Powerset EEND 耦合型別）

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

export type DiarizationClusterMethod =
  | 'ahc_cosine'
  | 'spectral_cosine'
  | 'vbx_plda'
  | 'e2cp_constrained_spectral';

export type DiarizationExecutionBackend = 'wasm' | 'webgpu';

export type CompliantOnnxSpeakerModel =
  | 'wespeaker_voxceleb_resnet34_int8.onnx'
  | 'nemo_en_titanet_small.int8.onnx';

export type MeetingLanguageHint = 'auto_multilingual' | 'zh' | 'en' | 'ja';

/**
 * 三級聲學信心度閘控等級 (3-Tier Acoustic Confidence Gating Protocol)
 * - Tier 1 (locked_anchor): \Delta L > 1.8 且 \bar{\Omega} < 0.15，唯讀鎖定語者標籤，禁止 LLM 修改語者
 * - Tier 2 (candidate_constrained): 0.45 <= \Delta L <= 1.8 且 \bar{\Omega} < 0.25，僅允許在 Top-1 與 Top-2 候選語者間依語用切換
 * - Tier 3 (overlap_split): \Delta L < 0.45 或 \bar{\Omega} >= 0.25，雙人重疊搶話或插話附和，允許拆分為兩個語者槽位
 */
export type AcousticSlotTier =
  | 'tier1_locked_anchor'
  | 'tier2_candidate_constrained'
  | 'tier3_overlap_split';

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
   * 預設為 false 以保持 Whisper 12~22s 靜音谷底連續自然語音視窗與 DTW 字級時間戳對齊
   */
  enableVirtualSpeakerTrackStitching?: boolean;
  /** 啟用重疊幀排除聲紋純化 (embeddingExcludeOverlap: true，剔除 Powerset 類別 4,5,6 重疊幀) */
  embeddingExcludeOverlap?: boolean;
  /** 啟用 Silero VAD 閘控之零相位前瞻包絡響度正規化 (預設 true) */
  enableZeroPhaseLoudnessNorm?: boolean;
  /** Silero VAD 本地模型檔名 (預設 'silero_vad_v5.onnx'，未安裝時自動降級自適應能量 VAD) */
  sileroVadModelFilename?: string;
  /** Pyannote Segmentation 3.0 本地模型檔名 (預設 'segmentation-3.0.onnx'，未安裝時自動降級微觀 SCD) */
  pyannoteSegModelFilename?: string;
  /** VBx PLDA 轉換矩陣參數檔名 (預設 'plda-params-vbx.json'，未安裝時自動降級 E2CP 受限譜聚類) */
  pldaParamsFilename?: string;
  /** VBx 聲學似然縮放因子 Fa (選填) */
  vbxFa?: number;
  /** VBx 語者先驗正規化因子 Fb (選填) */
  vbxFb?: number;
}

/** 語者分離細部時間軸單元 (含 SCD 跳變值、微觀聲紋邊際、對數勝算邊界與重疊指標) */
export interface FineDiarizationTurn {
  start: number;
  end: number;
  speakerId: string;
  runnerUpSpeakerId?: string;
  f0Hz: number;
  speakerConfidence?: number;
  speakerMargin?: number;
  /** 對數勝算邊界 \Delta L = \ln(P_1 + \epsilon) - \ln(P_2 + \epsilon) */
  logOddsMargin?: number;
  /** 區段內重疊語音比例 \bar{\Omega} \in [0, 1] */
  overlapRatio?: number;
  /** 該時間點左側邊界之瞬時聲紋跳變度 d_SCD(t) 或語者機率通量 \Phi (0.0 ~ 2.0) */
  boundaryScdLeft?: number;
  /** 該時間點右側邊界之瞬時聲紋跳變度 d_SCD(t) 或語者機率通量 \Phi (0.0 ~ 2.0) */
  boundaryScdRight?: number;
  isProtectedBackchannel?: boolean;
  /** 是否為參與第一輪聚類定錨之黃金長句 (Anchor Segment, 純淨非重疊時長 >= 1.2s ~ 1.5s) */
  isAnchorTurn?: boolean;
}

/**
 * 經重疊相加 (OLA) 重建後之全域幀級語者後驗機率矩陣封裝
 * 供字級 Viterbi 解碼器 (`viterbiWordAligner.ts`) 進行 ~17ms 解析度中心加權發射機率計算
 */
export interface FrameSpeakerPosteriorMatrix {
  /** 該矩陣對應之原始音檔起始秒數 (通常等於 chunkStartSec) */
  chunkStartSec: number;
  /** 相鄰兩幀之時間步長 (Pyannote Segmentation 3.0 為 10 / 589 ≈ 0.016978s；Fallback SCD 為 0.02s) */
  frameStepSec: number;
  /** 總幀數 T_{frames} */
  numFrames: number;
  /** 對應每一列索引 k \in [0, K-1] 之全域語者 ID 列表 (例如 ['Speaker_01', 'Speaker_02', ...]) */
  speakerIds: string[];
  /**
   * 扁平化之全域語者幀級活躍後驗機率矩陣，長度為 numFrames * speakerIds.length
   * 第 f 幀、第 k 位語者之機率位於索引 `f * speakerIds.length + k`
   */
  flatPosteriors: Float32Array;
  /** 長度為 numFrames 之幀級重疊語音機率 P_{OSD}(t) = p_4 + p_5 + p_6 */
  osdProbs: Float32Array;
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
  /** 字級 Viterbi 解碼指派之最佳語者 ID */
  speakerId?: string;
  /** 字級聲學發射機率 Top-2 競爭語者 ID */
  runnerUpSpeakerId?: string;
  /** 字級聲學對數勝算邊界 \Delta L_i = \ln(E_i(k_1) + \epsilon) - \ln(E_i(k_2) + \epsilon) */
  logOddsMargin?: number;
  /** 字級重疊語音比例 \Omega_i \in [0, 1] */
  overlapRatio?: number;
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
  /** 子句平均聲學對數勝算邊界 \Delta L：> 1.8 為 Tier 1 鎖定，0.45~1.8 為 Tier 2 候選，< 0.45 為 Tier 3 重疊 */
  logOddsMargin?: number;
  /** 子句內平均重疊語音比例 \bar{\Omega} \in [0, 1] */
  overlapRatio?: number;
  /** 三級聲學信心度閘控等級，供 ISIAL + LG-TPST LLM 決定是否允許變更語者或拆解重疊句 */
  slotTier?: AcousticSlotTier;
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
  /** 若為 QA_SPEAKER_CHANGE 或 BACKCHANNEL_INTERJECTION，填寫回答句/附和句開頭的精確 2~4 個字作為切分錨點 */
  split_before_anchor: string;
  new_speaker_id: string;
}

export interface ContextualFixOutputItem {
  utterance_id: string;
  /** 15 字以內說明前後文語意呼應或標點修正理由 (Micro-CoT) */
  brief_reason: string;
  /** 原句中之同音誤植、語意不合脈絡或標點缺漏片段 (2~10 字，須精確出現於原句中) */
  wrong_span: string;
  /** 根據上下文脈絡修正後之詞句或含標點片段 */
  corrected_span: string;
}

export interface StructuredSlotGerResponse {
  slot_corrections: SlotCorrectionOutputItem[];
  contextual_fixes?: ContextualFixOutputItem[];
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
  /** 當 [REVIEW] / [OVERLAP] 段落經對話語用學判定需切換語者時，填入新語者 ID (例如 runnerUpSpeakerId) */
  newSpeakerId?: string;
  /** 若為段內問答對拆句 (QA_SPEAKER_CHANGE) 或重疊附和拆解，指定回答句開頭錨點字串 */
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
  /** 多原型代表性錨點庫 Anchor Exemplar Sub-Bank (上限 M=15 個高純度向量，防止跨 5 分鐘區塊質心漂移) */
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