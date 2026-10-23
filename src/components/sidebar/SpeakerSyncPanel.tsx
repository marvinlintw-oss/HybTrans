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
    shortBadge: '初代經典穩健 (推薦)',
    title: 'Mode 0：初代經典穩健引擎 (全域音色保留 + AHC)',
    scenario:
      '一般 2~6 人部門會議、訪談、男女與同性混合討論（最接近初代 HybTrans 的穩定分群表現）。',
    strengths:
      '保留完整靜態聲道共振峰（不因短句減均值而洗掉音色），搭配黃金長句定錨，換人邊界果斷、同性語者不易被強制黏合。',
    limitations:
      '若環境有劇烈移動或中途切換不同麥克風，對通道變化的抵抗力略低於 ONNX 深度模型。',
  },
  mode_b_advanced_dsp: {
    shortBadge: '雙流形譜聚類',
    title: 'Mode B：進階雙流形譜聚類 (Auto-Tuning NME-SC)',
    scenario:
      '人數未知、發言長度極度不均（例如 1 人主講、多人偶爾插話提問）的中大型會議。',
    strengths:
      '透過拉普拉斯特徵間隙自動推估各性別流形人數，擅長處理長時段演講穿插問答。',
    limitations:
      '當兩位同性別語者音色極度接近且頻繁短句搶話時，譜平滑可能偏向合併為同一人。',
  },
  mode_c_onnx_neural: {
    shortBadge: 'ONNX 深度聲紋',
    title: 'Mode C：本地 WeSpeaker 256維深度神經聲紋',
    scenario:
      '已於 /public/models/ 放置 ONNX 權重檔、且有多位同性別與會者需要高維度音色區分時。',
    strengths:
      '256 維超球面聲紋對同性別不同人的細微腔調與共鳴分辨力最高，支援雙尺度融合。',
    limitations:
      '需預先放置本地 .onnx 模型檔（若未放置會自動退回 DSP 模式），且 CPU 運算耗時較長。',
  },
  mode_a_classic: {
    shortBadge: '極速輕量 DSP',
    title: 'Mode A：極速輕量模式 (低耗能快速切分)',
    scenario:
      '文書筆電、長達 2 小時以上錄音，或僅需快速取得初稿再手動微調語者的情境。',
    strengths: 'CPU 佔用率最低、轉檔等待時間最短，不會造成低階裝置發熱降頻。',
    limitations:
      '切分粒度較粗，對於 1 秒內的快速附和詞（「對」、「好」）較容易併入前一位語者。',
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
    enableVirtualSpeakerTrackStitching: true,
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
    diarCfg.enableVirtualSpeakerTrackStitching ?? true;

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
          const fullText = segments
            .map((seg) => seg.words.map((w) => w.text).join(''))
            .join('\n');

          let result: unknown;
          try {
            result = matcherFn(fullText, trimmed, segments);
          } catch {
            result = matcherFn(segments, trimmed);
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
      });
    } else if (mode === 'mode_a_classic') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'ahc_cosine',
        scdPeakThreshold: 0.27,
        backchannelMarginThreshold: 0.15,
      });
    } else if (mode === 'mode_b_advanced_dsp') {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'spectral_cosine',
        scdPeakThreshold: 0.23,
        backchannelMarginThreshold: 0.13,
      });
    } else {
      updateDiarizationConfig({
        engineMode: mode,
        clusterMethod: 'spectral_cosine',
        scdPeakThreshold: 0.19,
        backchannelMarginThreshold: 0.1,
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
            title="平時自動收摺進階聲紋設定；展開後可切換「語者專屬虛擬音軌拼接轉寫 (Track-First ASR)」與 4 種分離策略"
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
            {/* 【核心升級開關】長短句定錨 + 語者專屬虛擬音軌拼接轉寫 */}
            <div className="rounded-lg border border-indigo-200 p-2 bg-indigo-50/50 space-y-1">
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-[11px] font-semibold text-indigo-950 flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5 text-indigo-600" />
                  語者專屬虛擬音軌拼接轉寫 (Track-First)
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
              <p className="text-[10px] text-indigo-800/90 leading-normal">
                先用黃金長句定錨分出語者，再將同語者片段（夾 0.35s 斷句隔板）拼接為純淨同語者音軌送入 Whisper，物理級杜絕跨語者混聽與長靜音幻覺，並自動還原真實時間軸！
              </p>
            </div>

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
                  ★ Mode 0：初代經典穩健模式 (全域音色 + AHC，最穩定推薦)
                </option>
                <option value="mode_b_advanced_dsp">
                  Mode B：雙流形自動調參譜聚類 (NME-SC，適合未知人數演講)
                </option>
                <option value="mode_c_onnx_neural">
                  Mode C：WeSpeaker ONNX 深度聲紋 (需本地模型檔，同性區分強)
                </option>
                <option value="mode_a_classic">
                  Mode A：極速輕量模式 (低耗能快速初稿)
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
                僅在「語者分離軌道」強化個人聲道共振峰（F2~F4）與全場基底去均值；送給 Whisper 轉寫的音訊保持 100% 原始無損音質。
              </p>
            </div>

            {/* Mode C ONNX 專屬模型與後端設定 */}
            {diarCfg.engineMode === 'mode_c_onnx_neural' && (
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
            <div className="bg-slate-50 px-2 py-1.5 rounded border border-slate-200 space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-600 flex items-center gap-1">
                  <Activity className="w-3 h-3 text-indigo-500" />
                  搶話與插話切分敏感度
                </span>
                <span className="font-semibold text-indigo-700">
                  {scdThreshold <= 0.21
                    ? `高敏果斷 (${scdThreshold.toFixed(2)})`
                    : scdThreshold >= 0.26
                    ? `保守穩健 (${scdThreshold.toFixed(2)})`
                    : `標準平衡 (${scdThreshold.toFixed(2)})`}
                </span>
              </div>
              <input
                type="range"
                min={0.18}
                max={0.3}
                step={0.02}
                value={scdThreshold}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  updateDiarizationConfig({
                    scdPeakThreshold: val,
                    backchannelMarginThreshold: Number(
                      Math.max(0.09, val * 0.55).toFixed(2)
                    ),
                  });
                }}
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

      {/* 語者清單 */}
      <div className="space-y-1.5 max-h-52 overflow-y-auto pr-1">
        {speakerList.length === 0 ? (
          <p className="text-xs text-slate-400 text-center py-2.5">
            上傳音檔後將自動識別發言者與聲紋錨點
          </p>
        ) : (
          speakerList.map((sp) => {
            const isEditing = editingId === sp.id;
            const f0Val = sp.f0Hz
              ? Math.round(sp.f0Hz)
              : sp.centroid?.[0]
              ? Math.round(sp.centroid[0])
              : null;
            const anchorCount = sp.anchorBank?.length || 0;
            return (
              <div
                key={sp.id}
                title={
                  f0Val
                    ? `F0 基準：約 ${f0Val}Hz | 錨點數：${anchorCount} | 累積發聲：${Math.round(
                        sp.accumulatedMass || 0
                      )}s`
                    : undefined
                }
                className="flex items-center justify-between p-2 rounded-lg border border-slate-100 hover:border-slate-200 bg-slate-50/50"
              >
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: sp.color }}
                  />
                  {isEditing ? (
                    <input
                      type="text"
                      value={nameInput}
                      onChange={(e) => setNameInput(e.target.value)}
                      onKeyDown={(e) =>
                        e.key === 'Enter' && handleConfirmRename(sp.id)
                      }
                      placeholder="輸入姓名或職稱"
                      className="text-xs px-2 py-0.5 border border-blue-400 rounded w-full focus:outline-none"
                      autoFocus
                    />
                  ) : (
                    <span className="text-xs font-medium text-slate-800 truncate">
                      {sp.name}
                    </span>
                  )}
                </div>

                {isEditing ? (
                  <button
                    type="button"
                    onClick={() => handleConfirmRename(sp.id)}
                    className="text-xs text-blue-600 font-medium px-2 py-0.5 hover:bg-blue-50 rounded"
                  >
                    確定
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleStartEdit(sp.id, sp.name)}
                    className="text-[11px] text-slate-500 hover:text-blue-600 px-1.5 py-0.5"
                  >
                    更名
                  </button>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* 語者手動合併工具 */}
      {hasMultipleSpeakers && (
        <div className="pt-2 border-t border-slate-100 space-y-1.5">
          <div className="flex items-center gap-1 text-xs font-medium text-slate-600">
            <Merge className="w-3.5 h-3.5" />
            <span>合併同一人之語者標籤</span>
          </div>
          <div className="flex items-center gap-1.5">
            <select
              value={mergeSourceId}
              onChange={(e) => setMergeSourceId(e.target.value)}
              className="text-xs border border-slate-200 rounded px-1.5 py-1 flex-1 bg-white truncate"
            >
              <option value="">選擇來源...</option>
              {speakerList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <span className="text-xs text-slate-400">➔</span>
            <select
              value={mergeTargetId}
              onChange={(e) => setMergeTargetId(e.target.value)}
              className="text-xs border border-slate-200 rounded px-1.5 py-1 flex-1 bg-white truncate"
            >
              <option value="">併入目標...</option>
              {speakerList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleMergeConfirm}
              disabled={
                !mergeSourceId ||
                !mergeTargetId ||
                mergeSourceId === mergeTargetId
              }
              className="text-xs bg-slate-800 text-white px-2 py-1 rounded disabled:opacity-40 shrink-0"
            >
              合併
            </button>
          </div>
        </div>
      )}

      {/* 內文同音稱謂連動修正清單 */}
      {hasPhoneticCandidates && (
        <div className="p-2.5 bg-amber-50/80 border border-amber-200 rounded-lg space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-amber-900 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-600" />
              同音稱謂建議 ({phoneticCandidates.length})
            </span>
            <button
              type="button"
              onClick={applyPhoneticReplacements}
              className="text-[11px] bg-amber-600 text-white px-2 py-0.5 rounded flex items-center gap-1 hover:bg-amber-700"
            >
              <CheckCircle2 className="w-3 h-3" />
              全部替換
            </button>
          </div>
          <div className="max-h-24 overflow-y-auto space-y-1 text-[11px] text-amber-800">
            {phoneticCandidates.map((c, idx) => (
              <div key={`${c.segmentId}-${idx}`} className="truncate">
                「{c.matchedText}」➔ <strong>{c.targetSpeakerName}</strong>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};