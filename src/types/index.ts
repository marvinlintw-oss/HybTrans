// src/types/index.ts
// [SSOT] 專案唯一型別定義來源（含 Deep Research 多階段 AI 語意推敲、方向一 3~5秒深度全文聲學預分析與底層聲學公式動態校準層、方向二 先句後詞兩階段語意修復、免索引槽位協定、不可變 Token 圖譜與次世代字級 Viterbi / Powerset EEND 耦合型別）

export type EngineMode = 'local_webgpu' | 'byok_cloud';

export type LocalAsrModel =
  | 'SenseVoice-Small'
  | 'Whisper-Large-v3-Turbo'
  | 'Breeze-ASR'
  | 'whisper-large-v3-turbo'
  | 'whisper-base'
  | 'breeze-asr-25';

export type LocalLlmModel =
  | 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  | 'gemma-2-2b-it-q4f16_1-MLC';

export type GranularityTier = 'eco_safe' | 'balanced_pro' | 'ultra_studio';

export type DiarizationEngineMode =
  | 'mode_0_heritage_classic'
  | 'mode_a_classic'
  | 'mode_b_advanced_dsp'
  | 'mode_c_onnx_neural';

export type DiarizationStrategyMode =
  | 'MODE_0_GOLDEN_TWO_STAGE'
  | 'MODE_A_CLASSIC_DSP'
  | 'MODE_B_ADVANCED_SPECTRAL'
  | 'MODE_C_ONNX_NEURAL'
  | DiarizationEngineMode;

export type DiarizationStrategy = DiarizationStrategyMode;

export type DiarizationClusterMethod =
  | 'ahc_cosine'
  | 'spectral_cosine'
  | 'vbx_plda'
  | 'e2cp_constrained_spectral';

export type ClusterMethod = DiarizationClusterMethod;
export type ClusteringMethod = DiarizationClusterMethod;

export type DiarizationExecutionBackend = 'wasm' | 'webgpu';
export type EmbeddingBackendType = DiarizationExecutionBackend;
export type EmbeddingBackend = DiarizationExecutionBackend;

export type CompliantOnnxSpeakerModel =
  | 'wespeaker_voxceleb_resnet34_int8.onnx'
  | 'nemo_en_titanet_small.int8.onnx'
  | 'wespeaker_en_voxceleb_cam++.onnx'
  | 'wespeaker-voxceleb-resnet34-LM.onnx';

export type EmbeddingModelType = CompliantOnnxSpeakerModel;

export type MeetingLanguageHint = 'auto_multilingual' | 'zh' | 'en' | 'ja';
export type AcousticLanguageProfile = MeetingLanguageHint;

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

/**
 * 【方向一深化擴充】：啟動階段 3~5 秒全文聲學預分析下放至「底層聲學公式動態校準層 (Acoustic Formula Calibration Layer)」
 * 根據單一音檔之真實物理分佈（底噪分位數、有聲母音 P15 諧波清晰度、話輪轉換頻率、基頻跨度）動態校準底層公式閥值，
 * 解決單一全局預設常數無法同時應對「高頻搶話短句」與「遠距微弱尾音」之痛點。
 */
export interface AcousticFormulaCalibration {
  /** VAD 底噪估計分位數 (預設 0.05，範圍 0.03 ~ 0.08；依全場靜音比例動態校準) */
  vadNoiseQuantile: number;
  /** VAD 語音啟動乘數 (預設 1.65，範圍 1.50 ~ 1.95；依全場 SNR 動態校準) */
  vadSpeechMultiplier: number;
  /** VAD 動態範圍補償權重 (預設 0.06，範圍 0.04 ~ 0.09) */
  vadDynamicRangeWeight: number;
  /** 自相關基頻與有聲母音認定門檻 (預設 0.36，範圍 0.32 ~ 0.45；由全場有聲幀 P15 諧波度自動校準，保護遠距小聲語者如女聲2不被誤判為 0Hz) */
  harmonicityGateThreshold: number;
  /** 翻紙與桌面摩擦雜音抑制上限 (預設 0.30，範圍 0.26 ~ 0.38；低於此值之非諧波寬頻噪音壓低 VAD) */
  frictionHarmonicCeiling: number;
  /** 邊界曲線之 150ms 音節包絡能量谷底權重 w_dip (預設 0.45，範圍 0.38 ~ 0.52) */
  boundaryDipWeight: number;
  /** 邊界曲線之聲紋/基頻/響度跳變複合權重 w_voice (預設 0.55，範圍 0.48 ~ 0.62) */
  boundaryVoiceWeight: number;
  /** 左右視窗分貝階躍起算門檻 dB (預設 3.2 dB，捕捉遠距小聲語者切換至近距大聲語者之零停頓換人點) */
  stepDbDiffFloor: number;
  /** 有聲母音 F1-F3 共振峰餘弦距離起算門檻 (預設 0.21，排除無聲子音摩擦干擾) */
  formantCosDistFloor: number;
  /** Stage 2 微觀話輪邊界波峰切分門檻 finePeakCutoff (預設 0.52，範圍 0.46 ~ 0.58) */
  scdPeakCutoff: number;
  /** Stage 2 連續無停頓發聲之微觀切分上限秒數 (預設 1.60s，範圍 1.35s ~ 1.95s；依每分鐘換人頻率自適應) */
  maxMicroTurnSec: number;
  /** Stage 1 超球面 AHC 聚類停止合併門檻 (256維 ONNX 預設 0.28，範圍 0.24 ~ 0.33) */
  ahcStopSimThreshold: number;
  /** 軟性基頻差異免罰容忍半音數 (預設 3.8 半音，範圍 3.2 ~ 4.4 半音；依全場音域跨度校準) */
  pitchPenaltySemitoneTolerance: number;
  /** 核心成員閘控 VBx 變分精煉之逆溫度係數 (預設 22.0，防止少數語者質心被主講人稀釋) */
  vbxInvTemp: number;
  /** VBx 質心更新之核心成員責任門檻 (預設 0.45，僅允許責任 >= 0.45 之核心錨點參與該語者質心計算) */
  vbxCoreResponsibilityGate: number;
}

export interface DiarizationConfig {
  engineMode: DiarizationEngineMode;
  strategyMode?: DiarizationStrategyMode;
  clusterMethod: DiarizationClusterMethod;
  clusteringMethod?: DiarizationClusterMethod;
  executionBackend: DiarizationExecutionBackend;
  backend?: DiarizationExecutionBackend;
  onnxModelFilename: CompliantOnnxSpeakerModel;
  embeddingModel?: CompliantOnnxSpeakerModel;
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
  /** 啟用 Silero VAD 閘控之零相位前瞻包絡響度正規化 (預設 true，僅作用於 Whisper ASR 音軌) */
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
  /** 【方向一深化】：由啟動階段 3~5 秒全文聲學預分析自動推導並注入之底層聲學公式動態閥值 */
  formulaCalibration?: AcousticFormulaCalibration;
}

/**
 * 【方向一】：全文深度聲學預分析之情境推薦方案項目
 * 雙向相容別名：AudioRecommendationOption / RecommendedConfigOption
 */
export interface AudioRecommendationOption {
  id: 'preset_a_balanced' | 'preset_b_sensitive' | 'preset_c_long_speech' | string;
  badge: string;
  title: string;
  label?: string;
  subtitle: string;
  whyRecommend: string;
  bestFor: string;
  notIdealFor: string;
  isRecommended: boolean;
  recommendedMinSpeakers: number;
  recommendedMaxSpeakers: number;
  diarizationPatch: Partial<DiarizationConfig>;
  enginePatch?: Partial<EngineConfig>;
}

export type RecommendedConfigOption = AudioRecommendationOption;

/**
 * 【方向一】：取檔後 3~5 秒深度多點全文聲學預分析設定檔
 * 雙向相容別名：AudioPreAnalysisProfile / AudioAcousticPreAnalysis
 */
export interface AudioPreAnalysisProfile {
  /** 信噪比 (dB) */
  snrDb: number;
  /** 環境底噪分貝 (dBFS) */
  noiseFloorDb: number;
  /** 噪音等級分類 */
  noiseLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  /** 全場語音基頻跨度 (Hz, P90 - P10) */
  f0SpreadHz: number;
  /** 全場語音基頻半音跨度 (Semitones, 12 * log2(P90 / P10)) */
  pitchSpreadSemitones: number;
  /** 基頻 P10 低頻端點 (Hz) */
  f0P10Hz: number;
  /** 基頻 P90 高頻端點 (Hz) */
  f0P90Hz: number;
  /** 中位諧波清晰度 (0.0 ~ 1.0) */
  harmonicityMedian: number;
  /** 與會者聲線相似度評估 */
  voiceSimilarity: 'HIGH' | 'MEDIUM' | 'LOW';
  /** 每分鐘話輪轉換率 (次/分) */
  turnSwitchRatePerMin: number;
  /** 重疊搶話比例 (0.0 ~ 1.0) */
  overlapRatio: number;
  /** 重疊搶話風險等級 */
  overlapRisk: 'LOW' | 'MEDIUM' | 'HIGH';
  /** 預估最少語者人數 */
  estimatedMinSpeakers: number;
  /** 預估最多語者人數 */
  estimatedMaxSpeakers: number;
  /** 實際抽樣掃描之視窗數量 */
  sampledWindowsCount: number;
  /** 實際抽樣分析之語音總秒數 */
  sampledDurationSec: number;
  /** 抽樣涵蓋率 (0.0 ~ 1.0) */
  coverageRatio: number;
  /** 預分析實際耗時 (ms) */
  analysisDurationMs: number;
  /** 白話情境診斷摘要 */
  plainSummary: string;
  /** 系統推薦之最佳方案 ID */
  recommendedPresetId: string;
  /** 3 組白話情境化建議方案 */
  recommendations: AudioRecommendationOption[];
  /** 相容別名：指向同一陣列 */
  recommendedOptions: RecommendedConfigOption[];
  /** 【方向一深化】：根據本音檔實測分佈自動校準之底層聲學公式動態閥值 */
  formulaCalibration?: AcousticFormulaCalibration;
}

export type AudioAcousticPreAnalysis = AudioPreAnalysisProfile;

/**
 * 【方向二】：句子語意與結構完整性問題類型（同時支援大小寫枚舉值）
 */
export type SentenceIntegrityIssueType =
  | 'TAIL_DROP_TO_NEXT'
  | 'HEAD_STOLEN_FROM_PREV'
  | 'INCOMPLETE_SYNTAX'
  | 'SEMANTIC_ANOMALY'
  | 'HALLUCINATION_LOOP'
  | 'NONE'
  | 'tail_character_drop'
  | 'head_stolen_from_prev'
  | 'broken_syntax_split'
  | 'repetition_noise_hallucination'
  | 'unreasonable_self_qa'
  | 'none';

/**
 * 【方向二】：單一段落之句子語意與結構完整性診斷紀錄
 */
export interface SentenceIntegrityDiagnostic {
  segmentId: string;
  /** 句子結構完整度分數 (0 ~ 100) */
  score: number;
  issues: SentenceIntegrityIssueType[];
  /** 是否已在 Stage 1 自動執行結構修補 */
  autoRepaired: boolean;
  suggestedRepair?: string;
  suggestedRepairHint?: string;
  relatedSegmentId?: string;
  diagnosticNote?: string;
}

/**
 * 【方向二】：Stage 1 跨句結構修復動作紀錄
 */
export interface SentenceIntegrityRepair {
  segmentId: string;
  relatedSegmentId?: string;
  issueType: SentenceIntegrityIssueType;
  beforeText: string;
  afterText: string;
  description: string;
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

export type DiarizationTurn = FineDiarizationTurn;

export interface OptimalAsrWindow {
  start: number;
  end: number;
}

export type AsrMacroWindow = OptimalAsrWindow;

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

export type VirtualSpeakerTrackBatch = SpeakerTrackBatch;

export type ChunkStage =
  | 'idle'
  | 'preprocessing'
  | 'diarizing'
  | 'transcribing'
  | 'correcting'
  | 'refining'
  | 'relistening'
  | 'completed'
  | 'error'
  | 'DIARIZING'
  | 'TRANSCRIBING_ASR'
  | 'LLM_REFINING'
  | 'COMPLETED';

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
  /** 相容別名：與 text 同步 */
  word?: string;
  originalText?: string;
  originalWord?: string;
  start: number;
  end: number;
  confidence: number;
  status: WordUncertaintyStatus;
  isPatched?: boolean;
  isCorrected?: boolean;
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

export type WordTimestamp = WordSpan;

export interface Segment {
  id: string;
  chunkIndex: number;
  speakerId: string;
  start: number;
  end: number;
  words: WordSpan[];
  /** 雙向相容同步衍生欄位 */
  text?: string;
  rawText?: string;
  polishedText?: string;
  isCorrected: boolean;
  isPolished?: boolean;
  ispolished?: boolean;
  isLocked?: boolean;
  matchedTerms?: string[];
  relistenCount?: number;
  /** 聲學餘弦邊際 M(u_m)：>= 0.16 且 >= 1.2s 為 LOCKED 錨點，< 0.16 為 REVIEW 候選 */
  acousticMargin?: number;
  /** 子句平均聲學對數勝算邊界 \Delta L：> 1.8 為 Tier 1 鎖定，0.45~1.8 為 Tier 2 候選，< 0.45 為 Tier 3 重疊 */
  logOddsMargin?: number;
  /** 子句內平均重疊語音比例 \bar{\Omega} \in [0, 1] */
  overlapRatio?: number;
  overlapWarning?: boolean;
  /** 三級聲學信心度閘控等級，供 ISIAL + LG-TPST LLM 決定是否允許變更語者或拆解重疊句 */
  slotTier?: AcousticSlotTier;
  runnerUpSpeakerId?: string;
  /** 【方向二】：Stage 1 句子語意與結構完整性診斷 */
  sentenceIntegrity?: SentenceIntegrityDiagnostic;
  sentenceIntegrityScore?: number;
  sentenceIntegrityIssues?: SentenceIntegrityIssueType[];
}

export type TranscriptSegment = Segment;

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

export type SuspectSlot = SuspectSlotItem;

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

export type LlmSlotCorrection = SlotCorrectionOutputItem;

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

export type LlmTurnSplitSuggestion = TurnSplitOutputItem;
export type LlmTurnSplitProposal = TurnSplitOutputItem;

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
  readonly tokenId: string;
  char: string;
  pinyinKey: string;
  startTimeMs: number;
  endTimeMs: number;
  acousticConf: number;
  speakerId: string;
}

export interface UncertainCandidateOption {
  readonly optionId: string;
  text: string;
  pinyinSequence: string[];
  source: 'ORIGINAL' | 'LLM_PASS2' | 'BOOTSTRAPPED_HOTWORD' | 'DOMAIN_DICT';
  reason?: string;
}

export interface SuspectSpanNode {
  readonly spanId: string;
  status: 'CLEAN' | 'AUTO_CORRECTED' | 'SEMANTIC_UNCERTAIN' | 'USER_LOCKED';
  activeTokens: AtomicCharToken[];
  originalTokens: AtomicCharToken[];
  candidates: UncertainCandidateOption[];
  selectedOptionId: string;
}

export interface UtteranceNode {
  readonly utteranceId: string;
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

export type LlmCorrectionPatch = CorrectionPatchItem;

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
  weight?: number;
  boostWeight?: number;
  phonetic?: string;
  phoneticKey?: string;
  notes?: string;
  definition?: string;
}

export type DictionaryEntry = DictEntry;

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

export interface PipelineProgress {
  active: boolean;
  currentChunkIndex: number;
  totalChunks: number;
  stage: ChunkStage;
  overallProgress: number;
  detailText: string;
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
  /** 相容別名：與 diarizationConfig 同步 */
  diarization?: DiarizationConfig;
}