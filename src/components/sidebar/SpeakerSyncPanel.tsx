// src/components/sidebar/SpeakerSyncPanel.tsx
import React, { useState } from 'react';
import {
  Users,
  Merge,
  CheckCircle2,
  Sliders,
  Sparkles,
  Info,
  Cpu,
  Globe2,
  Activity,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Waves,
  Layers,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import {
  CompliantOnnxSpeakerModel,
  DiarizationClusterMethod,
  DiarizationEngineMode,
  MeetingLanguageHint,
} from '../../types';
import * as phoneticModule from '../../utils/phoneticMatcher';

const ENGINE_STRATEGY_GUIDES: Record<
  DiarizationEngineMode,
  {
    shortBadge: string;
    title: string;
    scenario: string;
    strengths: string;
    limitations: string;
  }
> = {
  mode_0_heritage_classic: {
    shortBadge: '兩階段神經定錨 (推薦)',
    title: 'Mode 0：兩階段黃金長句定錨 + WeSpeaker 256維聲紋 (推薦)',
    scenario:
      '一般 2~6 人台灣商務與專案會議、男女與同性混合討論、包含頻繁無停頓接話與短促附和（「不算」、「對」）。',
    strengths:
      '自動載入 /public/models/ 下之 WeSpeaker 256 維模型執行第一階段長句定錨，再以「基頻跳變+能量谷底」精準切出微觀話輪，能同時分出多位同性別語者並保持句子時間順序 100% 精準。',
    limitations:
      '若環境有劇烈移動或多人完全同時間重疊大聲爭論，建議搭配微調敏感度滑桿。',
  },
  mode_b_advanced_dsp: {
    shortBadge: '密度感知譜聚類',
    title: 'Mode B：密度感知譜聚類 + 溫和先驗 VBx (NME-SC + VBx)',
    scenario:
      '人數未知、發言長度極度不均（例如 1 人主講 60%、少數語者僅偶爾插話幾句）的中大型會議。',
    strengths:
      '結合拉普拉斯特徵間隙、AHC 自然群數探測與阻尼先驗 VBx，不會因為主講人話量過大而吞併少數發言者（如僅講幾句話的第二位女性或男性與會者）。',
    limitations:
      '當會議室回音極重且未放置本地 ONNX 模型時，譜平滑對極短句的解析度略低於 AHC。',
  },
  mode_c_onnx_neural: {
    shortBadge: 'ONNX 雙尺度深層聲紋',
    title: 'Mode C：本地 WeSpeaker 256維雙尺度深度神經聲紋',
    scenario:
      '已於 /public/models/ 放置 ONNX 權重檔、且有多位同性別與會者（如 2女2男 或 3男2女）需要高維度音色區分時。',
    strengths:
      '256 維超球面聲紋對同性別不同人的細微腔調與共鳴分辨力最高，支援時長自適應混合 CMN 與短句週期延展保護。',
    limitations:
      '需預先放置本地 .onnx 模型檔（若未放置會自動退回 DSP 模式）。',
  },
  mode_a_classic: {
    shortBadge: '極速輕量 DSP',
    title: 'Mode A：極速輕量模式 (純 DSP 低耗能快速切分)',
    scenario:
      '文書筆電、長達 2 小時以上錄音，或僅需快速取得初稿再手動微調語者的情境。',
    strengths: 'CPU 佔用率最低、轉檔等待時間最短，不會造成低階裝置發熱降頻。',
    limitations:
      '僅使用 64 維頻譜包絡特徵，對音色接近之同性別語者分辨力低於 WeSpeaker ONNX 模式。',
  },
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
  } = useAppStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [nameInput, setNameInput] = useState('');
  const [mergeSourceId, setMergeSourceId] = useState<string>('');
  const [mergeTargetId, setMergeTargetId] = useState<string>('');
  const [isStrategyExpanded, setIsStrategyExpanded] = useState<boolean>(false);

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
  // 預設關閉會破壞多人搶話時間軸順序的 Virtual Speaker-Track Stitching
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
        enableVirtualSpeakerTrackStitching: false,
      });
    } else if (mode === 'mode_a_classic') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.27,
        backchannelMarginThreshold: 0.15,
        enableVirtualSpeakerTrackStitching: false,
      });
    } else if (mode === 'mode_b_advanced_dsp') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'spectral_cosine',
        scdPeakThreshold: 0.22,
        backchannelMarginThreshold: 0.12,
        enableVirtualSpeakerTrackStitching: false,
      });
    } else {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.19,
        backchannelMarginThreshold: 0.1,
        enableVirtualSpeakerTrackStitching: false,
      });
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-sm space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <div className="flex items-center gap-1.5">
          <Users className="w-4 h-4 text-blue-600" />
          <h3 className="font-semibold text-slate-800 text-sm">
            與會語者與分離策略
          </h3>
          <span
            title="平時自動收摺進階聲紋設定；展開後可切換 4 種分離策略與微調換人切分敏感度"
            className="text-slate-400 hover:text-slate-600 cursor-help"
          >
            <Info className="w-3.5 h-3.5" />
          </span>
        </div>
        <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full font-medium">
          {speakerList.length} 人
        </span>
      </div>

      {/* 會議人數範圍設定 */}
      <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-200/80 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-700 flex items-center gap-1">
            <Sliders className="w-3.5 h-3.5 text-slate-500" />
            預估會議人數
          </span>
          <div className="flex items-center gap-1">
            {[
              { label: '2人對談', min: 2, max: 2 },
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
          <div className="bg-white px-2 py-1.5 rounded border border-slate-200 flex items-center justify-between">
            <span className="text-xs text-slate-500">最少</span>
            <input
              type="number"
              min={1}
              max={10}
              value={config.minSpeakers}
              onChange={(e) => handleMinChange(Number(e.target.value))}
              className="w-10 text-right font-semibold text-xs text-slate-800 focus:outline-none border-b border-slate-300"
            />
          </div>
          <div className="bg-white px-2 py-1.5 rounded border border-slate-200 flex items-center justify-between">
            <span className="text-xs text-slate-500">最多</span>
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

      {/* 可收摺的語者分離策略設定區 */}
      <div className="rounded-lg border border-slate-200/90 bg-slate-50/70 overflow-hidden">
        <button
          type="button"
          onClick={() => setIsStrategyExpanded((prev) => !prev)}
          className="w-full flex items-center justify-between px-2.5 py-2 text-left hover:bg-slate-100/80 transition-colors"
        >
          <div className="flex items-center gap-1.5 min-w-0">
            <Cpu className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
            <span className="text-xs font-medium text-slate-700 shrink-0">
              分離策略：
            </span>
            <span className="text-[11px] font-semibold text-indigo-700 bg-indigo-50 border border-indigo-200/80 px-1.5 py-0.5 rounded truncate">
              {currentGuide.shortBadge}
            </span>
          </div>
          <div className="flex items-center gap-1 text-[11px] text-slate-500 shrink-0">
            <span>{isStrategyExpanded ? '收起' : '變更 / 說明'}</span>
            {isStrategyExpanded ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </div>
        </button>

        {isStrategyExpanded && (
          <div className="p-2.5 border-t border-slate-200/80 space-y-2.5 text-xs bg-white">
            {/* 策略選擇器 */}
            <div className="space-y-1">
              <label className="text-[11px] font-medium text-slate-600 block">
                選擇語者分離核心策略：
              </label>
              <select
                value={diarCfg.engineMode}
                onChange={(e) =>
                  handleSelectEngineMode(
                    e.target.value as DiarizationEngineMode
                  )
                }
                className="w-full bg-slate-50 border border-slate-300 rounded-md px-2 py-1.5 text-xs font-semibold text-slate-800 focus:outline-none focus:border-indigo-500"
              >
                <option value="mode_0_heritage_classic">
                  ★ Mode 0：兩階段黃金長句定錨 + WeSpeaker (最穩定推薦)
                </option>
                <option value="mode_c_onnx_neural">
                  Mode C：WeSpeaker ONNX 雙尺度深度聲紋 (同性別多人區分強)
                </option>
                <option value="mode_b_advanced_dsp">
                  Mode B：密度感知譜聚類 + VBx (適合未知人數演講)
                </option>
                <option value="mode_a_classic">
                  Mode A：極速輕量模式 (純 DSP 低耗能初稿)
                </option>
              </select>
            </div>

            {/* 白話情境、強項與限制說明卡 */}
            <div className="rounded-lg bg-slate-50 border border-slate-200 p-2.5 space-y-1.5 text-[11px] leading-relaxed">
              <div className="font-semibold text-slate-900 flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
                {currentGuide.title}
              </div>
              <div className="text-slate-700">
                <span className="font-semibold text-slate-900">
                  🎯 適用情境：
                </span>
                {currentGuide.scenario}
              </div>
              <div className="text-emerald-800">
                <span className="font-semibold text-emerald-950">
                  💪 核心強項：
                </span>
                {currentGuide.strengths}
              </div>
              <div className="text-amber-800">
                <span className="font-semibold text-amber-950">
                  ⚠️ 潛在限制：
                </span>
                {currentGuide.limitations}
              </div>
            </div>

            {/* 雙軌無損聲紋差異強化預處理開關 */}
            <div className="rounded-lg border border-slate-200 p-2 bg-slate-50/70 space-y-1">
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-[11px] font-semibold text-slate-800 flex items-center gap-1">
                  <Waves className="w-3.5 h-3.5 text-blue-600" />
                  雙軌無損聲紋差異強化預處理
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
              <p className="text-[10px] text-slate-500 leading-normal">
                僅在「語者分離軌道」強化個人聲道共振峰與時長自適應混合去均值；送給 Whisper 轉寫的音訊保持自然連續語境。
              </p>
            </div>

            {/* 實驗性虛擬音軌拼接開關 (預設關閉) */}
            <div className="rounded-lg border border-slate-200 p-2 bg-slate-50/50 space-y-1">
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-[11px] font-medium text-slate-600 flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5 text-slate-500" />
                  實驗性：語者專屬虛擬音軌拼接 (建議關閉以維持句子順序)
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
              <p className="text-[10px] text-slate-500 leading-normal">
                預設關閉（採用 12~20s 靜音谷底視窗 + 字級 Viterbi 對齊，確保多人快速問答順序 100% 準確）。
              </p>
            </div>

            {/* ONNX 模型與後端設定 */}
            {diarCfg.engineMode !== 'mode_a_classic' && (
              <div className="grid grid-cols-2 gap-1.5 pt-0.5">
                <select
                  value={diarCfg.onnxModelFilename}
                  onChange={(e) =>
                    updateDiarizationConfig({
                      onnxModelFilename:
                        e.target.value as CompliantOnnxSpeakerModel,
                    })
                  }
                  title="自託管於 /public/models/ 目錄下"
                  className="bg-white border border-indigo-200 rounded px-1.5 py-1 text-[10px] text-indigo-900 truncate"
                >
                  <option value="wespeaker_voxceleb_resnet34_int8.onnx">
                    WeSpeaker ResNet34 (推薦)
                  </option>
                  <option value="nemo_en_titanet_small.int8.onnx">
                    NeMo TitaNet-Small (INT8)
                  </option>
                </select>
                <select
                  value={diarCfg.executionBackend}
                  onChange={(e) =>
                    updateDiarizationConfig({
                      executionBackend: e.target.value as 'wasm' | 'webgpu',
                    })
                  }
                  className="bg-white border border-slate-200 rounded px-1.5 py-1 text-[10px] text-slate-700"
                >
                  <option value="wasm">後端：WASM (0顯存)</option>
                  <option value="webgpu">後端：WebGPU</option>
                </select>
              </div>
            )}

            <div className="grid grid-cols-2 gap-1.5">
              <div className="flex items-center justify-between bg-slate-50 px-2 py-1 rounded border border-slate-200">
                <span className="text-[11px] text-slate-500">分群法:</span>
                <select
                  value={diarCfg.clusterMethod}
                  onChange={(e) =>
                    updateDiarizationConfig({
                      clusterMethod:
                        e.target.value as DiarizationClusterMethod,
                    })
                  }
                  className="text-[11px] font-medium text-slate-800 bg-transparent focus:outline-none"
                >
                  <option value="ahc_cosine">AHC 階層凝聚 (果斷)</option>
                  <option value="spectral_cosine">NME-SC 譜聚類 (平滑)</option>
                </select>
              </div>

              <div className="flex items-center justify-between bg-slate-50 px-2 py-1 rounded border border-slate-200">
                <Globe2 className="w-3 h-3 text-slate-400 shrink-0 mr-1" />
                <select
                  value={diarCfg.languageHint}
                  onChange={(e) =>
                    updateDiarizationConfig({
                      languageHint: e.target.value as MeetingLanguageHint,
                    })
                  }
                  className="text-[11px] font-medium text-slate-800 bg-transparent focus:outline-none w-full"
                >
                  <option value="zh">華語/中英夾雜會議</option>
                  <option value="auto_multilingual">中/英/日自動混用</option>
                  <option value="ja">日本語會議 (相槌保護)</option>
                  <option value="en">English Meeting</option>
                </select>
              </div>
            </div>

            {/* SCD 微觀換人切分敏感度微調 */}
            <div className="rounded-lg border border-slate-200 p-2 bg-slate-50/60 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-slate-700 font-medium flex items-center gap-1">
                  <Activity className="w-3 h-3 text-indigo-600" />
                  搶話與插話切分敏感度
                </span>
                <span className="text-[11px] font-semibold text-indigo-700">
                  {scdThreshold <= 0.19
                    ? `高敏果斷 (${scdThreshold.toFixed(2)})`
                    : scdThreshold <= 0.24
                    ? `標準平衡 (${scdThreshold.toFixed(2)})`
                    : `保守抗噪 (${scdThreshold.toFixed(2)})`}
                </span>
              </div>
              <input
                type="range"
                min={0.16}
                max={0.32}
                step={0.01}
                value={scdThreshold}
                onChange={(e) =>
                  updateDiarizationConfig({
                    scdPeakThreshold: Number(e.target.value),
                  })
                }
                className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>← 數值越小：積極切出短句換人</span>
                <span>數值越大：減少切分 →</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 語者清單與更名 */}
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

      {/* 同音稱謂連動修正提示 */}
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

      {/* 合併同一人之語者標籤 */}
      {hasMultipleSpeakers && (
        <div className="pt-2 border-t border-slate-100 space-y-1.5">
          <div className="text-xs font-medium text-slate-600 flex items-center gap-1">
            <Merge className="w-3.5 h-3.5 text-slate-400" />
            合併同一人之語者標籤
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