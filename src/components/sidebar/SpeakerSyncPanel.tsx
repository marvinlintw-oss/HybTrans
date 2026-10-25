// src/components/sidebar/SpeakerSyncPanel.tsx
import React, { useState } from 'react';
import {
  Users,
  Merge,
  CheckCircle2,
  Sliders,
  Sparkles,
  Info,
  Globe2,
  Activity,
  ChevronDown,
  ChevronUp,
  HelpCircle,
  Settings2,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import {
  CompliantOnnxSpeakerModel,
  DiarizationClusterMethod,
  DiarizationEngineMode,
  MeetingLanguageHint,
} from '../../types';
import * as phoneticModule from '../../utils/phoneticMatcher';

interface ModeScenarioGuide {
  shortBadge: string;
  optionLabel: string;
  whyChoose: string;
  bestFor: string;
  notFor: string;
  techPrincipleTitle: string;
  techPrincipleDetail: string;
}

const ENGINE_STRATEGY_GUIDES: Record<DiarizationEngineMode, ModeScenarioGuide> =
  {
    mode_0_heritage_classic: {
      shortBadge: '標準會議 (推薦)',
      optionLabel: '★ 標準會議模式（預設推薦・多數會議首選）',
      whyChoose: '兼顧「多人聲音區分」與「對話先後順序」，開箱即用最穩定。',
      bestFor: '2～6 人一般部門會議、專案討論、男女與同性別混合且有來回接話。',
      notFor: '超過 2 小時以上長錄音且使用無獨顯舊筆電（建議改選快速省電模式）。',
      techPrincipleTitle: '兩階段黃金長句定錨 + WeSpeaker 256維聲紋',
      techPrincipleDetail:
        '第一階段先抓取 1.2～2.2 秒高信噪比長句提取 256 維神經聲紋定錨語者；第二階段結合「基頻跳變 + 能量谷底 + 共振峰通量」切出微觀話輪，並以 12～20 秒靜音谷底視窗搭配字級 Viterbi 對齊，確保同性別可分開且句子順序不顛倒。',
    },
    mode_c_onnx_neural: {
      shortBadge: '同性別精細區分',
      optionLabel: '精細音色模式（多位同性別與會者）',
      whyChoose: '針對聲音相近的同性別同事，強化細微腔調與音色共鳴的區分度。',
      bestFor: '多位同性別與會者（如 3 女 1 男、全男或全女）且需要細分每個人時。',
      notFor: '現場環境雜音極大、麥克風離太遠，或想最快拿到初稿時。',
      techPrincipleTitle: 'WeSpeaker 256維雙尺度深度聲紋 + 短句延展',
      techPrincipleDetail:
        '融合宏觀與微觀雙尺度 256 維超球面神經聲紋，並針對短於 1 秒的附和詞執行時長自適應混合 CMN 與張量週期延展，最大化同性別不同語者的超球面餘弦邊際。',
    },
    mode_b_advanced_dsp: {
      shortBadge: '主講與簡報',
      optionLabel: '主講與簡報模式（一人主講、少數人提問）',
      whyChoose: '保護偶爾才開口提問的人，不會因為主講人講太久而被系統吞併。',
      bestFor: '教育訓練、專題發表、1 人主講佔 70% 以上時間、其他人僅偶爾插話。',
      notFor: '每個人都在頻繁搶話、每句話都只有兩三個字的激烈討論。',
      techPrincipleTitle: '密度感知譜聚類 (NME-SC) + 溫和先驗 VBx',
      techPrincipleDetail:
        '結合正規化拉普拉斯算子特徵間隙與阻尼先驗變分貝氏聚類（Tempered-Prior VBx），防止高發言量主講人的先驗機率過度膨脹而吸走少數發言者的短句。',
    },
    mode_a_classic: {
      shortBadge: '快速省電初稿',
      optionLabel: '快速省電模式（長錄音 / 一般文書筆電）',
      whyChoose: '轉檔速度最快、最省電，適合先快速產出文字初稿再手動微調。',
      bestFor: '1～2 小時以上長錄音、輕薄文書筆電，或不在意同性別細分的情境。',
      notFor: '有多位同性別同事且希望系統自動精準分出每個人時。',
      techPrincipleTitle: '純 DSP 64維共振峰統計池化 (無須 ONNX 模型)',
      techPrincipleDetail:
        '全程使用輕量級 64 維 Kaldi Fbank 共振峰包絡與時序統計池化（TSTP），CPU 與記憶體負擔最低，老舊筆電也能流暢執行。',
    },
  };

/**
 * 浮出式說明視窗組件 (內建於組件內，零外部路徑依賴，避免 ts(2307) 報錯)
 */
const HoverInfoPopover: React.FC<{
  label?: string;
  title: string;
  children: React.ReactNode;
  align?: 'left' | 'right';
}> = ({ label, title, children, align = 'right' }) => {
  return (
    <div className="relative inline-flex items-center group">
      <button
        type="button"
        className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-600 transition-colors cursor-help focus:outline-none"
      >
        <HelpCircle className="w-3.5 h-3.5 shrink-0" />
        {label && (
          <span className="underline decoration-dotted underline-offset-2">
            {label}
          </span>
        )}
      </button>

      <div
        className={`absolute ${
          align === 'right' ? 'right-0' : 'left-0'
        } top-full mt-1.5 z-50 w-72 p-3 rounded-xl bg-slate-900/95 text-slate-100 text-[11px] leading-relaxed shadow-xl border border-slate-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible group-focus-within:opacity-100 group-focus-within:visible transition-all duration-150 pointer-events-none group-hover:pointer-events-auto`}
      >
        <div className="font-semibold text-white mb-1 flex items-center gap-1.5 border-b border-slate-700/80 pb-1">
          <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span>{title}</span>
        </div>
        <div className="text-slate-300 space-y-1">{children}</div>
      </div>
    </div>
  );
};

export const SpeakerSyncPanel: React.FC = () => {
  const {
    config,
    updateConfig,
    updateDiarizationConfig,
    speakers,
    segments,
    renameSpeaker,
    mergeSpeakers,
    phoneticCandidates,
    setPhoneticCandidates,
    applyPhoneticReplacements,
    audioPreAnalysis,
    selectedPresetId,
    applyRecommendedPreset,
  } = useAppStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [nameInput, setNameInput] = useState('');
  const [mergeSourceId, setMergeSourceId] = useState<string>('');
  const [mergeTargetId, setMergeTargetId] = useState<string>('');
  const [showAdvancedTech, setShowAdvancedTech] = useState<boolean>(false);

  const diarCfg = config.diarizationConfig || {
    engineMode: 'mode_0_heritage_classic',
    clusterMethod: 'ahc_cosine',
    executionBackend: 'wasm',
    onnxModelFilename: 'wespeaker_voxceleb_resnet34_int8.onnx',
    languageHint: 'zh',
    scdPeakThreshold: 0.22,
    backchannelMarginThreshold: 0.12,
    enableAcousticDivergencePreproc: true,
    enableVirtualSpeakerTrackStitching: false,
  };

  const currentGuide =
    ENGINE_STRATEGY_GUIDES[diarCfg.engineMode] ||
    ENGINE_STRATEGY_GUIDES.mode_0_heritage_classic;

  const speakerList = Object.values(speakers);
  const hasMultipleSpeakers = speakerList.length >= 2;
  const hasPhoneticCandidates = phoneticCandidates.length > 0;
  const scdThreshold = diarCfg.scdPeakThreshold ?? 0.22;
  const preprocEnabled = diarCfg.enableAcousticDivergencePreproc ?? true;
  const virtualTrackEnabled =
    diarCfg.enableVirtualSpeakerTrackStitching ?? false;

  const handleStartEdit = (id: string, currentName: string) => {
    setEditingId(id);
    setNameInput(currentName);
  };

  const handleConfirmRename = (id: string) => {
    const trimmed = nameInput.trim();
    if (trimmed) {
      renameSpeaker(id, trimmed);

      try {
        const matcherFn = (phoneticModule as Record<string, unknown>)
          .scanHomophoneSpeakerMentions as (...args: unknown[]) => unknown;

        if (typeof matcherFn === 'function') {
          let result: unknown;
          try {
            result = matcherFn(trimmed, segments);
          } catch {
            const fullText = segments
              .map((seg) => seg.words.map((w) => w.text).join(''))
              .join('\n');
            result = matcherFn(fullText, trimmed, segments);
          }

          if (Array.isArray(result)) {
            setPhoneticCandidates(result);
          }
        }
      } catch (err) {
        console.warn('掃描同音稱謂略過:', err);
      }
    }
    setEditingId(null);
  };

  const handleMinChange = (val: number) => {
    const safeMin = Math.max(1, Math.min(10, val));
    const safeMax = Math.max(safeMin, config.maxSpeakers);
    updateConfig({ minSpeakers: safeMin, maxSpeakers: safeMax });
  };

  const handleMaxChange = (val: number) => {
    const safeMax = Math.max(1, Math.min(10, val));
    const safeMin = Math.min(config.minSpeakers, safeMax);
    updateConfig({ minSpeakers: safeMin, maxSpeakers: safeMax });
  };

  const handleMergeConfirm = () => {
    if (mergeSourceId && mergeTargetId && mergeSourceId !== mergeTargetId) {
      mergeSpeakers(mergeSourceId, mergeTargetId);
      setMergeSourceId('');
      setMergeTargetId('');
    }
  };

  const handleSelectEngineMode = (mode: DiarizationEngineMode) => {
    if (mode === 'mode_0_heritage_classic') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.21,
        backchannelMarginThreshold: 0.11,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
      });
    } else if (mode === 'mode_a_classic') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.27,
        backchannelMarginThreshold: 0.15,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
      });
    } else if (mode === 'mode_b_advanced_dsp') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'spectral_cosine',
        scdPeakThreshold: 0.22,
        backchannelMarginThreshold: 0.12,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
      });
    } else {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.19,
        backchannelMarginThreshold: 0.1,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
      });
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-sm space-y-3">
      {/* 標題列 */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <div className="flex items-center gap-1.5">
          <Users className="w-4 h-4 text-blue-600" />
          <h3 className="font-semibold text-slate-800 text-sm">
            與會語者與會議情境
          </h3>
        </div>
        <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full font-medium">
          {speakerList.length} 人
        </span>
      </div>

      {/* 0. 全場深度聲學預分析快捷方案切換 */}
      {audioPreAnalysis && (
        <div className="bg-indigo-50/70 rounded-lg p-2.5 border border-indigo-200/80 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-indigo-900 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
              全場聲學預分析建議方案
            </span>
            <span className="text-[10px] text-indigo-700 font-medium">
              SNR {audioPreAnalysis.snrDb}dB · {audioPreAnalysis.turnSwitchRatePerMin}次/分
            </span>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {audioPreAnalysis.recommendations.map((opt) => {
              const isSelected = selectedPresetId === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => applyRecommendedPreset(opt)}
                  className={`text-[10px] px-1.5 py-1 rounded border text-center transition-all cursor-pointer truncate ${
                    isSelected
                      ? 'bg-indigo-600 text-white border-indigo-600 font-semibold shadow-2xs'
                      : 'bg-white text-slate-700 border-indigo-200 hover:bg-indigo-100/60'
                  }`}
                  title={`${opt.title}：${opt.whyRecommend}`}
                >
                  {opt.isRecommended ? '★ ' : ''}
                  {opt.badge.split('・')[0]}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* 1. 預估會議人數 */}
      <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-200/80 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Sliders className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-xs font-medium text-slate-700">
              預估會議人數
            </span>
            <HoverInfoPopover
              title="如何設定會議人數？"
              align="left"
            >
              <p>
                • <strong>點選上方快捷鍵</strong>即可套用常見人數範圍。
              </p>
              <p>
                • 若不確定實際有幾人開口，建議維持 <strong>2～6 人</strong>，系統會根據聲音特徵自動判斷實際人數。
              </p>
            </HoverInfoPopover>
          </div>

          <div className="flex items-center gap-1">
            {[
              { label: '2人', min: 2, max: 2 },
              { label: '2~4人', min: 2, max: 4 },
              { label: '2~6人', min: 2, max: 6 },
              { label: '4~8人', min: 4, max: 8 },
            ].map((preset) => {
              const active =
                config.minSpeakers === preset.min &&
                config.maxSpeakers === preset.max;
              return (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() =>
                    updateConfig({
                      minSpeakers: preset.min,
                      maxSpeakers: preset.max,
                    })
                  }
                  className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
                    active
                      ? 'bg-blue-600 text-white border-blue-600 font-medium'
                      : 'bg-white text-slate-600 border-slate-200 hover:bg-blue-50'
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="bg-white px-2 py-1 rounded border border-slate-200 flex items-center justify-between">
            <span className="text-xs text-slate-500">最少人數</span>
            <input
              type="number"
              min={1}
              max={10}
              value={config.minSpeakers}
              onChange={(e) => handleMinChange(Number(e.target.value))}
              className="w-10 text-right font-semibold text-xs text-slate-800 focus:outline-none border-b border-slate-300"
            />
          </div>
          <div className="bg-white px-2 py-1 rounded border border-slate-200 flex items-center justify-between">
            <span className="text-xs text-slate-500">最多人數</span>
            <input
              type="number"
              min={1}
              max={10}
              value={config.maxSpeakers}
              onChange={(e) => handleMaxChange(Number(e.target.value))}
              className="w-10 text-right font-semibold text-xs text-blue-700 focus:outline-none border-b border-blue-300"
            />
          </div>
        </div>
      </div>

      {/* 2. 會議情境模式選擇 (白話直覺版 + 浮出式技術說明) */}
      <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-2.5 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-700">
            會議情境模式
          </span>
          <HoverInfoPopover
            label="技術原理說明"
            title={currentGuide.techPrincipleTitle}
            align="right"
          >
            <p>{currentGuide.techPrincipleDetail}</p>
            <p className="text-indigo-300 pt-1 border-t border-slate-700/80">
              💡 系統已自動為此情境配好最佳分群與聲紋強化參數，一般使用無須手動調整。
            </p>
          </HoverInfoPopover>
        </div>

        <select
          value={diarCfg.engineMode}
          onChange={(e) =>
            handleSelectEngineMode(e.target.value as DiarizationEngineMode)
          }
          className="w-full bg-white border border-slate-300 rounded-md px-2.5 py-1.5 text-xs font-semibold text-slate-800 focus:outline-none focus:border-blue-500 shadow-2xs"
        >
          <option value="mode_0_heritage_classic">
            {ENGINE_STRATEGY_GUIDES.mode_0_heritage_classic.optionLabel}
          </option>
          <option value="mode_c_onnx_neural">
            {ENGINE_STRATEGY_GUIDES.mode_c_onnx_neural.optionLabel}
          </option>
          <option value="mode_b_advanced_dsp">
            {ENGINE_STRATEGY_GUIDES.mode_b_advanced_dsp.optionLabel}
          </option>
          <option value="mode_a_classic">
            {ENGINE_STRATEGY_GUIDES.mode_a_classic.optionLabel}
          </option>
        </select>

        {/* 簡潔三行情境提示：為什麼選 / 適合 / 不適合 */}
        <div className="rounded-md bg-white border border-slate-200/90 p-2 text-[11px] leading-relaxed space-y-1">
          <div className="text-slate-800">
            <span className="font-semibold text-indigo-700">💡 為什麼選：</span>
            {currentGuide.whyChoose}
          </div>
          <div className="text-slate-700">
            <span className="font-semibold text-emerald-700">✅ 適合情境：</span>
            {currentGuide.bestFor}
          </div>
          <div className="text-slate-500">
            <span className="font-semibold text-amber-700">⚠️ 不適合：</span>
            {currentGuide.notFor}
          </div>
        </div>

        {/* 會議語言與插話敏感度 (精簡直覺版) */}
        <div className="grid grid-cols-2 gap-1.5 pt-0.5">
          <div className="flex items-center bg-white px-2 py-1 rounded border border-slate-200">
            <Globe2 className="w-3.5 h-3.5 text-slate-400 shrink-0 mr-1.5" />
            <select
              value={diarCfg.languageHint}
              onChange={(e) =>
                updateDiarizationConfig({
                  languageHint: e.target.value as MeetingLanguageHint,
                })
              }
              className="text-[11px] font-medium text-slate-700 bg-transparent focus:outline-none w-full"
            >
              <option value="zh">華語 / 中英夾雜</option>
              <option value="auto_multilingual">中 / 英 / 日混用</option>
              <option value="ja">日本語會議</option>
              <option value="en">English</option>
            </select>
          </div>

          <div className="flex items-center justify-between bg-white px-2 py-1 rounded border border-slate-200">
            <div className="flex items-center gap-1 min-w-0">
              <Activity className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
              <select
                value={
                  scdThreshold <= 0.19
                    ? 'high'
                    : scdThreshold <= 0.24
                    ? 'normal'
                    : 'low'
                }
                onChange={(e) => {
                  const val = e.target.value;
                  updateDiarizationConfig({
                    scdPeakThreshold:
                      val === 'high' ? 0.18 : val === 'normal' ? 0.22 : 0.27,
                  });
                }}
                className="text-[11px] font-medium text-slate-700 bg-transparent focus:outline-none"
              >
                <option value="high">換人：積極切分</option>
                <option value="normal">換人：標準平衡</option>
                <option value="low">換人：減少碎切</option>
              </select>
            </div>
            <HoverInfoPopover
              title="換人切分敏感度說明"
              align="right"
            >
              <p>
                • <strong>標準平衡（預設）</strong>：適合絕大多數會議。
              </p>
              <p>
                • <strong>積極切分</strong>：當大家頻繁插話（「對」、「沒錯」），覺得兩個人常被黏在同一句時選這個。
              </p>
              <p>
                • <strong>減少碎切</strong>：當同一個人講話常停頓喘氣，卻被誤切成兩個人時選這個。
              </p>
            </HoverInfoPopover>
          </div>
        </div>

        {/* 進階工程微調 (預設收起，不干擾一般同事) */}
        <div className="pt-1 border-t border-slate-200/70">
          <button
            type="button"
            onClick={() => setShowAdvancedTech((prev) => !prev)}
            className="w-full flex items-center justify-between text-[11px] text-slate-400 hover:text-slate-600 py-0.5 transition-colors"
          >
            <span className="flex items-center gap-1">
              <Settings2 className="w-3 h-3" />
              進階工程微調（一般同事無須調整）
            </span>
            {showAdvancedTech ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>

          {showAdvancedTech && (
            <div className="mt-2 pt-2 border-t border-slate-200/60 space-y-2 text-[11px]">
              {/* 細部滑桿與演算法選項，皆附帶浮出視窗說明 */}
              <div className="flex items-center justify-between">
                <span className="text-slate-600 flex items-center gap-1">
                  分群演算法
                  <HoverInfoPopover
                    title="分群演算法差異"
                    align="left"
                  >
                    <p>
                      • <strong>AHC 階層凝聚</strong>：對同性別不同音色的切分較果斷，適合 2～6 人商務會議。
                    </p>
                    <p>
                      • <strong>NME-SC 譜聚類</strong>：利用全局相似度圖譜平滑分群，適合長篇演講。
                    </p>
                  </HoverInfoPopover>
                </span>
                <select
                  value={diarCfg.clusterMethod}
                  onChange={(e) =>
                    updateDiarizationConfig({
                      clusterMethod:
                        e.target.value as DiarizationClusterMethod,
                    })
                  }
                  className="bg-white border border-slate-200 rounded px-1.5 py-0.5 text-[11px] text-slate-700"
                >
                  <option value="ahc_cosine">AHC 階層凝聚 (果斷)</option>
                  <option value="spectral_cosine">NME-SC 譜聚類 (平滑)</option>
                </select>
              </div>

              {diarCfg.engineMode !== 'mode_a_classic' && (
                <div className="flex items-center justify-between">
                  <span className="text-slate-600 flex items-center gap-1">
                    聲紋模型與運算後端
                    <HoverInfoPopover
                      title="聲紋模型與執行後端"
                      align="left"
                    >
                      <p>
                        • <strong>WeSpeaker ResNet34</strong>：256 維神經聲紋模型（存放於 <code>/public/models/</code>，未放置時自動退回 DSP）。
                      </p>
                      <p>
                        • <strong>WASM 後端（推薦）</strong>：使用 CPU SIMD 運算，0 顯存佔用，把顯卡記憶體全部留給 Whisper 與 LLM。
                      </p>
                    </HoverInfoPopover>
                  </span>
                  <div className="flex items-center gap-1">
                    <select
                      value={diarCfg.onnxModelFilename}
                      onChange={(e) =>
                        updateDiarizationConfig({
                          onnxModelFilename:
                            e.target.value as CompliantOnnxSpeakerModel,
                        })
                      }
                      className="bg-white border border-slate-200 rounded px-1.5 py-0.5 text-[10px] text-slate-700 max-w-[115px] truncate"
                    >
                      <option value="wespeaker_voxceleb_resnet34_int8.onnx">
                        WeSpeaker ResNet34
                      </option>
                      <option value="nemo_en_titanet_small.int8.onnx">
                        NeMo TitaNet
                      </option>
                    </select>
                    <select
                      value={diarCfg.executionBackend}
                      onChange={(e) =>
                        updateDiarizationConfig({
                          executionBackend: e.target.value as 'wasm' | 'webgpu',
                        })
                      }
                      className="bg-white border border-slate-200 rounded px-1.5 py-0.5 text-[10px] text-slate-700"
                    >
                      <option value="wasm">WASM</option>
                      <option value="webgpu">WebGPU</option>
                    </select>
                  </div>
                </div>
              )}

              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-slate-600 flex items-center gap-1">
                  雙軌聲紋差異強化
                  <HoverInfoPopover
                    title="雙軌無損聲紋差異強化"
                    align="left"
                  >
                    <p>
                      僅在「語者分離軌道」強化個人聲道共振峰（F2～F4）與時長自適應去均值；送給 Whisper 轉寫的音訊保持 100% 原始自然音質。建議保持開啟。
                    </p>
                  </HoverInfoPopover>
                </span>
                <input
                  type="checkbox"
                  checked={preprocEnabled}
                  onChange={(e) =>
                    updateDiarizationConfig({
                      enableAcousticDivergencePreproc: e.target.checked,
                    })
                  }
                  className="rounded text-blue-600 focus:ring-0 cursor-pointer"
                />
              </label>

              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-slate-500 flex items-center gap-1">
                  實驗性：同語者音軌拼接
                  <HoverInfoPopover
                    title="實驗性：語者專屬虛擬音軌拼接"
                    align="left"
                  >
                    <p>
                      將同一位語者散落在不同時間的片段拼成同一條音軌送給 Whisper。在多人快速問答時較容易發生時間錯位，<strong>強烈建議保持關閉</strong>（預設採用 12～20 秒靜音谷底視窗 + 字級 Viterbi 對齊）。
                    </p>
                  </HoverInfoPopover>
                </span>
                <input
                  type="checkbox"
                  checked={virtualTrackEnabled}
                  onChange={(e) =>
                    updateDiarizationConfig({
                      enableVirtualSpeakerTrackStitching: e.target.checked,
                    })
                  }
                  className="rounded text-indigo-600 focus:ring-0 cursor-pointer"
                />
              </label>
            </div>
          )}
        </div>
      </div>

      {/* 3. 語者清單與更名 */}
      <div className="space-y-1.5 max-h-48 overflow-y-auto pr-0.5">
        {speakerList.length === 0 ? (
          <div className="text-xs text-slate-400 text-center py-3 bg-slate-50 rounded-lg border border-dashed border-slate-200">
            上傳音檔開始轉寫後將自動識別與會語者
          </div>
        ) : (
          speakerList.map((sp) => (
            <div
              key={sp.id}
              className="flex items-center justify-between px-2.5 py-1.5 rounded-lg border border-slate-200/80 bg-slate-50/60 hover:bg-slate-100/70 transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0 flex-1">
                <span
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: sp.color }}
                />
                {editingId === sp.id ? (
                  <div className="flex items-center gap-1 flex-1">
                    <input
                      type="text"
                      value={nameInput}
                      onChange={(e) => setNameInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleConfirmRename(sp.id);
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                      autoFocus
                      className="text-xs px-1.5 py-0.5 border border-blue-400 rounded bg-white text-slate-800 w-full focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => handleConfirmRename(sp.id)}
                      className="text-[11px] px-1.5 py-0.5 bg-blue-600 text-white rounded shrink-0 hover:bg-blue-700"
                    >
                      確定
                    </button>
                  </div>
                ) : (
                  <span className="text-xs font-medium text-slate-800 truncate">
                    {sp.name}
                  </span>
                )}
              </div>

              {editingId !== sp.id && (
                <button
                  type="button"
                  onClick={() => handleStartEdit(sp.id, sp.name)}
                  className="text-[11px] text-slate-500 hover:text-blue-600 ml-2 shrink-0"
                >
                  更名
                </button>
              )}
            </div>
          ))
        )}
      </div>

      {/* 4. 同音稱謂連動修正提示 */}
      {hasPhoneticCandidates && (
        <div className="rounded-lg border border-amber-200 bg-amber-50/80 p-2.5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-amber-900 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-600" />
              發現 {phoneticCandidates.length} 處同音稱謂可一鍵修正
            </span>
            <button
              type="button"
              onClick={applyPhoneticReplacements}
              className="text-[11px] px-2 py-0.5 bg-amber-600 text-white rounded font-medium hover:bg-amber-700 flex items-center gap-1"
            >
              <CheckCircle2 className="w-3 h-3" />
              套用修正
            </button>
          </div>
          <div className="max-h-28 overflow-y-auto space-y-1 text-[11px] text-amber-800">
            {phoneticCandidates.map((cand, idx) => (
              <div
                key={`${cand.segmentId}_${cand.wordIndex}_${idx}`}
                className="flex items-center justify-between bg-white/80 px-2 py-1 rounded border border-amber-200/60"
              >
                <span className="truncate">
                  {cand.contextPreview} →{' '}
                  <strong className="text-amber-950">
                    {cand.targetSpeakerName}
                  </strong>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. 合併同一人之語者標籤 */}
      {hasMultipleSpeakers && (
        <div className="pt-2 border-t border-slate-100 space-y-1.5">
          <div className="text-xs font-medium text-slate-600 flex items-center gap-1">
            <Merge className="w-3.5 h-3.5 text-slate-400" />
            <span>合併同一人之語者標籤</span>
            <HoverInfoPopover
              title="什麼時候需要合併語者？"
              align="left"
            >
              <p>
                若同一位同事因為某幾句笑聲或音量變化被分成兩個標籤（例如「語者 01」與「語者 04」其實是同一人），可在此將「語者 04」併入「語者 01」，全文會自動同步合併。
              </p>
            </HoverInfoPopover>
          </div>
          <div className="flex items-center gap-1.5">
            <select
              value={mergeSourceId}
              onChange={(e) => setMergeSourceId(e.target.value)}
              className="flex-1 text-xs border border-slate-200 rounded px-1.5 py-1 bg-slate-50 text-slate-700 focus:outline-none"
            >
              <option value="">選擇來源...</option>
              {speakerList.map((sp) => (
                <option key={sp.id} value={sp.id}>
                  {sp.name}
                </option>
              ))}
            </select>
            <span className="text-xs text-slate-400">→</span>
            <select
              value={mergeTargetId}
              onChange={(e) => setMergeTargetId(e.target.value)}
              className="flex-1 text-xs border border-slate-200 rounded px-1.5 py-1 bg-slate-50 text-slate-700 focus:outline-none"
            >
              <option value="">併入目標...</option>
              {speakerList
                .filter((sp) => sp.id !== mergeSourceId)
                .map((sp) => (
                  <option key={sp.id} value={sp.id}>
                    {sp.name}
                  </option>
                ))}
            </select>
            <button
              type="button"
              disabled={!mergeSourceId || !mergeTargetId}
              onClick={handleMergeConfirm}
              className="px-2 py-1 text-xs bg-slate-800 text-white rounded hover:bg-slate-700 disabled:opacity-40 shrink-0"
            >
              合併
            </button>
          </div>
        </div>
      )}
    </div>
  );
};