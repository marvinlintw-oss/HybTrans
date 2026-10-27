// src/components/sidebar/SpeakerSyncPanel.tsx
import React, { useState } from 'react';
import {
  Users,
  Merge,
  Sparkles,
  CheckCircle2,
  Sliders,
  Settings2,
  ChevronDown,
  ChevronUp,
  Globe2,
  Activity,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import {
  AudioRecommendationOption,
  CompliantOnnxSpeakerModel,
  DiarizationClusterMethod,
  DiarizationEngineMode,
  MeetingLanguageHint,
} from '../../types';
import {
  computeTaiwanPhoneticDistance,
  toPhoneticSignature,
} from '../../utils/phoneticMatcher';
import { HoverInfoPopover } from '../common/HoverInfoPopover';

interface StrategyGuideCard {
  mode: DiarizationEngineMode;
  optionLabel: string;
  badge: string;
  title: string;
  whyChoose: string;
  bestFor: string;
  notFor: string;
  techPrincipleTitle: string;
  techPrincipleDetail: string;
}

const ENGINE_STRATEGY_GUIDES: Record<DiarizationEngineMode, StrategyGuideCard> =
  {
    mode_0_heritage_classic: {
      mode: 'mode_0_heritage_classic',
      optionLabel: '★ 標準會議模式（2～6 人推薦）',
      badge: '預設推薦',
      title: '標準會議模式（兼顧長句與快速問答）',
      whyChoose:
        '先抓取清楚的長句建立每個人的聲音檔案，再透過 YIN 基頻與話輪 Viterbi 平滑切分短句與快速應答（「24號」、「不算」、「對」），最不易把同一人切碎。',
      bestFor:
        '2～6 人一般部門會議、專案進度檢討、男女與同性別混合且有來回接話的討論',
      notFor: '10 人以上且全程同時重疊喧嘩的大型活動',
      techPrincipleTitle: '兩階段雙尺度定錨 + YIN 基頻與話輪 Viterbi 平滑',
      techPrincipleDetail:
        '第一階段僅取 >=1.2 秒高諧波有聲長句提取 WeSpeaker 256 維聲紋建立語者中心；第二階段以 YIN 累積均值差分基頻、分貝階躍與話輪級 Viterbi 平滑歸屬微觀話輪。',
    },
    mode_c_onnx_neural: {
      mode: 'mode_c_onnx_neural',
      optionLabel: '同性別與密集搶話強化（深度聲紋模式）',
      badge: '高敏感度',
      title: '同性別與密集搶話強化模式',
      whyChoose:
        '當與會者全部是同性別（例如全男或全女）、聲音聽起來很像，或是大家頻繁插話搶話時，加強區分細微的聲音腔體差異。',
      bestFor: '全同性別與會者且音色相近、高頻率交叉問答、腦力激盪會議',
      notFor: '講者習慣在句中長時間停頓思考（高敏感度較容易將停頓切開）',
      techPrincipleTitle: 'WeSpeaker ResNet34 256D 神經聲紋 + AHC 凝聚分群',
      techPrincipleDetail:
        '直接透過 80-dim Kaldi Fbank 擷取 256 維深度神經聲紋向量，針對 F2~F4 個人聲道共振峰差異加權，專攻基頻接近但腔體不同的同性別語者。',
    },
    mode_b_advanced_dsp: {
      mode: 'mode_b_advanced_dsp',
      optionLabel: '長篇簡報與訪談模式（防句中斷裂）',
      badge: '長句保護',
      title: '長篇簡報與專題報告模式',
      whyChoose:
        '拉高換人判定門檻，就算主講人在句中停頓喘氣、翻閱資料或思考，系統也會盡量保持為同一個人，避免把一段話切成好幾段。',
      bestFor: '專題簡報、教育訓練、法說會、單人主講搭配少數提問',
      notFor: '多人每秒快速一問一答的激烈討論',
      techPrincipleTitle: 'NME-SC 自調諧譜聚類 + 高門檻防碎切保護',
      techPrincipleDetail:
        '建構全局餘弦親和力拉普拉斯圖譜，透過特徵間隙自動估計語者數，並提高邊界波峰門檻以維持長篇論述的語者連貫性。',
    },
    mode_a_classic: {
      mode: 'mode_a_classic',
      optionLabel: '輕量省電模式（低階筆電 / 行動裝置適用）',
      badge: '極速低耗',
      title: '輕量省電模式（純聲學特徵快速分群）',
      whyChoose:
        '不載入額外的神經聲紋模型，把系統記憶體與顯卡資源全部留給語音轉文字引擎，在無獨立顯卡的筆電或平板上最流暢。',
      bestFor: '1～3 人單純對談、長時數錄音快速初稿、記憶體較小的文書筆電',
      notFor: '4 人以上且聲音極度相近的複雜多人會議',
      techPrincipleTitle: '64 維共振峰時頻統計池化 (Formant-Preserving TSTP)',
      techPrincipleDetail:
        '完全在 CPU 上高速計算 MFCC 倒頻譜、頻譜斜率與 YIN 基頻特徵，零額外記憶體佔用，推論速度最快。',
    },
  };

export const SpeakerSyncPanel: React.FC = () => {
  const speakers = useAppStore((s) => s.speakers);
  const segments = useAppStore((s) => s.segments);
  const config = useAppStore((s) => s.config);
  const updateConfig = useAppStore((s) => s.updateConfig);
  const updateDiarizationConfig = useAppStore((s) => s.updateDiarizationConfig);
  const renameSpeaker = useAppStore((s) => s.renameSpeaker);
  const mergeSpeakers = useAppStore((s) => s.mergeSpeakers);
  const phoneticCandidates = useAppStore((s) => s.phoneticCandidates);
  const setPhoneticCandidates = useAppStore((s) => s.setPhoneticCandidates);
  const applyPhoneticReplacements = useAppStore(
    (s) => s.applyPhoneticReplacements
  );
  const audioPreAnalysis = useAppStore((s) => s.audioPreAnalysis);
  const selectedPresetId = useAppStore((s) => s.selectedPresetId);
  const applyRecommendedPreset = useAppStore((s) => s.applyRecommendedPreset);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [nameInput, setNameInput] = useState('');
  const [mergeSourceId, setMergeSourceId] = useState<string>('');
  const [mergeTargetId, setMergeTargetId] = useState<string>('');
  const [showScenarioGuide, setShowScenarioGuide] = useState(false);
  const [showAdvancedTech, setShowAdvancedTech] = useState(false);

  const speakerList = Object.values(speakers);
  const diarCfg = config.diarizationConfig || {
    engineMode: 'mode_0_heritage_classic',
    clusterMethod: 'ahc_cosine',
    executionBackend: 'wasm',
    onnxModelFilename: 'wespeaker_voxceleb_resnet34_int8.onnx',
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
  };

  const currentGuide =
    ENGINE_STRATEGY_GUIDES[diarCfg.engineMode] ||
    ENGINE_STRATEGY_GUIDES.mode_0_heritage_classic;

  const scdThreshold = diarCfg.scdPeakThreshold ?? 0.22;
  const preprocEnabled = diarCfg.enableAcousticDivergencePreproc ?? true;
  const virtualTrackEnabled =
    diarCfg.enableVirtualSpeakerTrackStitching ?? false;

  const handleStartEdit = (id: string, currentName: string) => {
    setEditingId(id);
    setNameInput(currentName);
  };

  const scanPhoneticMatchesForSpeaker = (newName: string) => {
    const cleanName = newName
      .replace(/[\(（].*?[\)）]/g, '')
      .replace(/語者\s*\d+/g, '')
      .trim();
    if (cleanName.length < 2) return;

    const targetSig = toPhoneticSignature(cleanName);
    const len = cleanName.length;
    const matches: typeof phoneticCandidates = [];

    for (const seg of segments) {
      for (let wIdx = 0; wIdx < seg.words.length; wIdx++) {
        const w = seg.words[wIdx];
        const txt = w.text.trim();
        if (txt.length === len && txt !== cleanName && /^[\u4e00-\u9fa5]+$/.test(txt)) {
          const dist = computeTaiwanPhoneticDistance(txt, cleanName);
          if (toPhoneticSignature(txt) === targetSig || dist <= 0.24) {
            const left = seg.words
              .slice(Math.max(0, wIdx - 2), wIdx)
              .map((x) => x.text)
              .join('');
            const right = seg.words
              .slice(wIdx + 1, Math.min(seg.words.length, wIdx + 3))
              .map((x) => x.text)
              .join('');
            matches.push({
              segmentId: seg.id,
              wordIndex: wIdx,
              matchedText: txt,
              targetSpeakerName: cleanName,
              contextPreview: `...${left}【${txt}】${right}...`,
              selected: true,
            });
          }
        }
      }
    }

    if (matches.length > 0) {
      setPhoneticCandidates(matches);
    }
  };

  const handleConfirmRename = (id: string) => {
    const trimmed = nameInput.trim();
    if (trimmed) {
      renameSpeaker(id, trimmed);
      scanPhoneticMatchesForSpeaker(trimmed);
    }
    setEditingId(null);
  };

  const handleMergeConfirm = () => {
    if (!mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId) {
      return;
    }
    mergeSpeakers(mergeSourceId, mergeTargetId);
    setMergeSourceId('');
    setMergeTargetId('');
  };

  const handleMinChange = (val: number) => {
    const safeMin = Math.max(1, Math.min(10, val));
    updateConfig({
      minSpeakers: safeMin,
      maxSpeakers: Math.max(safeMin, config.maxSpeakers),
    });
  };

  const handleMaxChange = (val: number) => {
    const safeMax = Math.max(1, Math.min(10, val));
    updateConfig({
      minSpeakers: Math.min(config.minSpeakers, safeMax),
      maxSpeakers: safeMax,
    });
  };

  const hasMultipleSpeakers = speakerList.length >= 2;
  const hasPhoneticCandidates = phoneticCandidates.length > 0;

  const handleSelectEngineMode = (mode: DiarizationEngineMode) => {
    if (mode === 'mode_0_heritage_classic') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.22,
        backchannelMarginThreshold: 0.12,
        enableAcousticDivergencePreproc: true,
        enableVirtualSpeakerTrackStitching: false,
      });
    } else if (mode === 'mode_a_classic') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.25,
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

  const handleQuickAutoMergeSameGender = () => {
    if (speakerList.length < 2) return;
    // 自動尋找同性別標籤（如多個「男聲/中低音」或多個「女聲/高音」）並設定至下拉選單供使用者一鍵確認
    const males = speakerList.filter((s) => s.name.includes('男聲'));
    const females = speakerList.filter((s) => s.name.includes('女聲'));
    if (males.length >= 2) {
      setMergeTargetId(males[0].id);
      setMergeSourceId(males[males.length - 1].id);
    } else if (females.length >= 2) {
      setMergeTargetId(females[0].id);
      setMergeSourceId(females[females.length - 1].id);
    } else {
      setMergeTargetId(speakerList[0].id);
      setMergeSourceId(speakerList[speakerList.length - 1].id);
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
            {audioPreAnalysis.recommendations.map((opt: AudioRecommendationOption) => {
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
                • 建議將「最少人數」設為 <strong>2 人</strong>、「最多人數」設為 <strong>4～6 人</strong>，讓系統在雙人問答與多人補充之間自動調節，避免把同一人切碎。
              </p>
            </HoverInfoPopover>
          </div>

          <div className="flex items-center gap-1">
            {[
              { label: '2人', min: 2, max: 2 },
              { label: '2~4人', min: 2, max: 4 },
              { label: '2~6人', min: 2, max: 6 },
              { label: '4~8人', min: 2, max: 8 },
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

      {/* 2. 會議情境模式選擇 */}
      <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-2.5 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-slate-700">
              會議情境模式
            </span>
            <button
              type="button"
              onClick={() => setShowScenarioGuide((prev) => !prev)}
              className="inline-flex items-center gap-0.5 text-[11px] text-indigo-600 hover:text-indigo-800 font-medium cursor-pointer transition-colors"
            >
              <span>{showScenarioGuide ? '收起情境說明' : '適用情境說明'}</span>
              {showScenarioGuide ? (
                <ChevronUp className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </button>
          </div>
          <HoverInfoPopover
            label="技術原理"
            title={currentGuide.techPrincipleTitle}
            align="right"
          >
            <p>{currentGuide.techPrincipleDetail}</p>
            <p className="text-indigo-300 pt-1 border-t border-slate-700/80">
              💡 系統已自動為此情境配置最佳分群與聲紋強化參數，一般使用無須手動調整。
            </p>
          </HoverInfoPopover>
        </div>

        <select
          value={diarCfg.engineMode}
          onFocus={() => setShowScenarioGuide(true)}
          onChange={(e) => {
            handleSelectEngineMode(e.target.value as DiarizationEngineMode);
            setShowScenarioGuide(true);
          }}
          className="w-full bg-white border border-slate-300 rounded-md px-2.5 py-1.5 text-xs font-semibold text-slate-800 focus:outline-none focus:border-blue-500 shadow-2xs cursor-pointer"
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

        {showScenarioGuide && (
          <div className="rounded-md bg-white border border-slate-200/90 p-2 text-[11px] leading-relaxed space-y-1 relative">
            <button
              type="button"
              onClick={() => setShowScenarioGuide(false)}
              className="float-right text-[10px] text-slate-400 hover:text-slate-600 ml-2 cursor-pointer"
              title="收起說明"
            >
              收起 ✕
            </button>
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
        )}

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
                • <strong>減少碎切</strong>：當同一個人講話常停頓翻頁，卻被誤切成兩個人時選這個。
              </p>
            </HoverInfoPopover>
          </div>
        </div>

        <div className="pt-1 border-t border-slate-200/70">
          <button
            type="button"
            onClick={() => setShowAdvancedTech((prev) => !prev)}
            className="w-full flex items-center justify-between text-[11px] text-slate-400 hover:text-slate-600 py-0.5 transition-colors"
          >
            <span className="flex items-center gap-1">
              <Settings2 className="w-3 h-3" />
              進階聲學參數設定（預設已最佳化，一般無須調整）
            </span>
            {showAdvancedTech ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>

          {showAdvancedTech && (
            <div className="mt-2 pt-2 border-t border-slate-200/60 space-y-2 text-[11px]">
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
                      僅在「語者分離軌道」強化個人聲道共振峰（F2～F4）與通道錨定去均值；送給 Whisper 轉寫的音訊保持 100% 原始自然音質。建議保持開啟。
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
      <div className="space-y-1.5 max-h-64 lg:max-h-48 overflow-y-auto pr-0.5">
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
                  className="text-[11px] text-slate-500 hover:text-blue-600 ml-2 shrink-0 cursor-pointer"
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
              className="text-[11px] px-2 py-0.5 bg-amber-600 text-white rounded font-medium hover:bg-amber-700 flex items-center gap-1 cursor-pointer"
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

      {/* 5. 合併同一人之語者標籤 (修正版：雙欄自適應下拉選單 + 獨立滿版高對比「合併語者標籤」按鈕，永不裁切超出邊界) */}
      {hasMultipleSpeakers && (
        <div className="pt-2.5 border-t border-slate-200/80 space-y-2 bg-slate-50/70 -mx-1 px-2.5 py-2 rounded-lg">
          <div className="flex items-center justify-between gap-1">
            <div className="text-xs font-semibold text-slate-700 flex items-center gap-1">
              <Merge className="w-3.5 h-3.5 text-blue-600 shrink-0" />
              <span>合併同一人之語者標籤</span>
              <HoverInfoPopover
                title="什麼時候需要合併語者？"
                align="left"
              >
                <p>
                  若同一位與會者因遠近音量變化或翻閱文件被分成兩個標籤（例如「語者 04」與「語者 02」其實是同一人），可在此選擇來源與目標後點擊下方<strong>「合併語者標籤」</strong>，全文會自動同步合併並重新順暢接句。
                </p>
              </HoverInfoPopover>
            </div>
            {speakerList.length > 2 && (
              <button
                type="button"
                onClick={handleQuickAutoMergeSameGender}
                className="text-[10px] text-indigo-600 hover:text-indigo-800 font-medium underline cursor-pointer shrink-0"
                title="自動預選同性別之重複語者標籤"
              >
                預選同性別
              </button>
            )}
          </div>

          <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-1.5">
            <select
              value={mergeSourceId}
              onChange={(e) => setMergeSourceId(e.target.value)}
              className="w-full min-w-0 text-xs border border-slate-300 rounded-md px-2 py-1.5 bg-white text-slate-800 font-medium focus:outline-none focus:border-blue-500 truncate"
            >
              <option value="">選擇被合併語者...</option>
              {speakerList.map((sp) => (
                <option key={sp.id} value={sp.id}>
                  {sp.name}
                </option>
              ))}
            </select>
            <span className="text-xs font-bold text-slate-400 px-0.5">→</span>
            <select
              value={mergeTargetId}
              onChange={(e) => setMergeTargetId(e.target.value)}
              className="w-full min-w-0 text-xs border border-slate-300 rounded-md px-2 py-1.5 bg-white text-slate-800 font-medium focus:outline-none focus:border-blue-500 truncate"
            >
              <option value="">併入目標語者...</option>
              {speakerList
                .filter((sp) => sp.id !== mergeSourceId)
                .map((sp) => (
                  <option key={sp.id} value={sp.id}>
                    {sp.name}
                  </option>
                ))}
            </select>
          </div>

          <button
            type="button"
            disabled={!mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId}
            onClick={handleMergeConfirm}
            className="w-full py-1.5 px-3 text-xs font-semibold bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white rounded-md shadow-2xs disabled:bg-slate-300 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Merge className="w-3.5 h-3.5 shrink-0" />
            <span>
              {mergeSourceId && mergeTargetId
                ? `確認將「${speakers[mergeSourceId]?.name.split(' ')[0] || mergeSourceId}」併入「${speakers[mergeTargetId]?.name.split(' ')[0] || mergeTargetId}」`
                : '請選擇上方兩個語者後點擊合併'}
            </span>
          </button>
        </div>
      )}
    </div>
  );
};
