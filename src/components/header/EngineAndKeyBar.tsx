// src/components/header/EngineAndKeyBar.tsx
import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Cloud,
  Key,
  Gauge,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  Info,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { GranularityTier, HardwareProbeResult } from '../../types';
import {
  probeHardwareAndCache,
  TIER_PRESET_SPECS,
  validateTierAgainstHardware,
} from '../../services/modelCacheManager';

/**
 * 內建互動式說明浮出視窗（支援滑鼠懸停與點擊展開，確保所有 ⓘ 圖示皆有豐富且即時的說明內容）
 */
const HoverInfoPopover: React.FC<{
  label?: string;
  title: string;
  children: React.ReactNode;
  align?: 'left' | 'right';
}> = ({ label, title, children, align = 'left' }) => {
  const [pinned, setPinned] = useState(false);

  return (
    <div
      className="relative inline-flex items-center group"
      onMouseLeave={() => setPinned(false)}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setPinned((prev) => !prev);
        }}
        aria-label={title}
        className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-blue-600 focus:text-blue-600 transition-colors cursor-pointer focus:outline-none"
      >
        <Info className="w-3.5 h-3.5 shrink-0" />
        {label && (
          <span className="underline decoration-dotted underline-offset-2">
            {label}
          </span>
        )}
      </button>

      <div
        className={`absolute ${
          align === 'right' ? 'right-0' : 'left-0'
        } top-full mt-1.5 z-50 w-80 p-3 rounded-xl bg-slate-900/95 text-slate-100 text-[11px] leading-relaxed shadow-xl border border-slate-700 transition-all duration-150 ${
          pinned
            ? 'opacity-100 visible pointer-events-auto'
            : 'opacity-0 invisible pointer-events-none group-hover:opacity-100 group-hover:visible group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:visible'
        }`}
      >
        <div className="font-semibold text-white mb-1.5 flex items-center gap-1.5 border-b border-slate-700/80 pb-1">
          <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span>{title}</span>
        </div>
        <div className="text-slate-300 space-y-1.5 font-normal">{children}</div>
      </div>
    </div>
  );
};

export const EngineAndKeyBar: React.FC = () => {
  const { config, updateConfig, applyGranularityTier } = useAppStore();
  const [probe, setProbe] = useState<HardwareProbeResult | null>(null);
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [showHelpPopover, setShowHelpPopover] = useState(false);

  useEffect(() => {
    let mounted = true;
    probeHardwareAndCache().then((res) => {
      if (mounted) setProbe(res);
    });
    return () => {
      mounted = false;
    };
  }, []);

  const isLocalMode = config.mode === 'local_webgpu';
  const hasGeminiKey = Boolean(config.geminiApiKey.trim());
  const currentSpec =
    TIER_PRESET_SPECS[config.granularityTier || 'balanced_pro'] ||
    TIER_PRESET_SPECS.balanced_pro;
  const hardwareWarning = validateTierAgainstHardware(
    config.granularityTier,
    probe,
    hasGeminiKey
  );

  return (
    <header className="bg-white border-b border-slate-200 px-4 py-2.5 shadow-sm">
      <div className="w-full flex flex-wrap items-center justify-start gap-4">
        {/* 1. 系統標題與硬體環境說明 Popover */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">
            HT
          </div>
          <div className="flex items-center gap-1.5">
            <h1 className="font-bold text-slate-800 text-sm">
              HybridTranscriber
            </h1>
            <HoverInfoPopover
              title="系統架構與本機硬體環境狀態"
              align="left"
            >
              <p>
                • <strong>偵測硬體：</strong>
                {probe ? probe.adapterName : '正在偵測 WebGPU 顯示卡環境...'}
              </p>
              <p>
                • <strong>模型快取佔用：</strong>
                {probe
                  ? `約 ${probe.estimatedCacheUsageMB} MB（首次下載後常駐瀏覽器快取，後續離線秒開）`
                  : '計算中...'}
              </p>
              <p>
                • <strong>長音檔微批次流水線：</strong>
                自動以 5 分鐘為單位切片處理，支援 2～4 小時大型會議錄音且不佔滿記憶體。
              </p>
            </HoverInfoPopover>
          </div>
        </div>

        {/* 2. 電腦效能與細膩度規格選單 */}
        <div className="flex items-center gap-1.5 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200">
          <Gauge className="w-3.5 h-3.5 text-blue-600 shrink-0" />
          <div className="text-xs flex items-center gap-1.5">
            <label
              htmlFor="tier-select"
              className="text-slate-500 font-medium"
            >
              效能模式:
            </label>
            <select
              id="tier-select"
              value={config.granularityTier}
              onChange={(e) =>
                applyGranularityTier(e.target.value as GranularityTier)
              }
              className="font-semibold text-slate-800 bg-transparent focus:outline-none cursor-pointer"
            >
              {(Object.keys(TIER_PRESET_SPECS) as GranularityTier[]).map(
                (tKey) => {
                  const spec = TIER_PRESET_SPECS[tKey];
                  const isRecommended = probe?.recommendedTier === tKey;
                  return (
                    <option key={tKey} value={tKey}>
                      {spec.label}
                      {isRecommended ? ' (硬體推薦)' : ''}
                    </option>
                  );
                }
              )}
            </select>
            <span className="hidden sm:inline-block text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 font-medium">
              Beam={config.asrBeamSize || currentSpec.asrBeamSize} · 重聽
              {config.maxRelistenQuota ?? currentSpec.maxRelistenQuotaPerChunk}段
            </span>
          </div>
          <HoverInfoPopover
            title="三段效能模式實際運算差異說明"
            align="left"
          >
            <p className="text-indigo-300 font-medium">
              目前套用：{currentSpec.label}（{currentSpec.badgeText}）
            </p>
            <p>
              • <strong>輕量省電：</strong>純 DSP 64維快速分群 + Whisper 貪婪解碼 (
              <code>Beam=1</code>) + 每 5 分鐘重聽 1 段，速度最快（約省時 35%）。
            </p>
            <p>
              • <strong>標準平衡（預設）：</strong>黃金兩階段定錨 + WeSpeaker 256維聲紋 + Whisper 雙束搜尋 (
              <code>Beam=2</code>) + 每 5 分鐘重聽 4 段。
            </p>
            <p>
              • <strong>深度精修：</strong>雙尺度 256維深度聲紋 + Whisper 三束深層解碼 (
              <code>Beam=3</code>，高重疊步長) + 180字高密度 AI 語境精修 + 每 5 分鐘重聽 8 段。
            </p>
          </HoverInfoPopover>
        </div>

        {/* 3. 引擎切換與 API Key */}
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-100">
            <button
              type="button"
              onClick={() => updateConfig({ mode: 'local_webgpu' })}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                isLocalMode
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              本機免金鑰
            </button>
            <button
              type="button"
              onClick={() => updateConfig({ mode: 'byok_cloud' })}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                !isLocalMode
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              <Cloud className="w-3.5 h-3.5" />
              雲端高速
            </button>
          </div>

          <button
            type="button"
            onClick={() => setShowKeyModal(!showKeyModal)}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
              hasGeminiKey
                ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
            }`}
          >
            {hasGeminiKey ? (
              <CheckCircle className="w-3.5 h-3.5" />
            ) : (
              <Key className="w-3.5 h-3.5" />
            )}
            {hasGeminiKey ? 'API 已連線' : 'API Key'}
          </button>

          {/* 4. 快速上手指南 Popover */}
          <div
            className="relative"
            onMouseEnter={() => setShowHelpPopover(true)}
            onMouseLeave={() => setShowHelpPopover(false)}
          >
            <button
              type="button"
              onClick={() => setShowHelpPopover(!showHelpPopover)}
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs text-slate-500 hover:text-blue-600 hover:bg-blue-50 border border-transparent hover:border-blue-100 transition-colors"
            >
              <HelpCircle className="w-4 h-4" />
              <span>使用說明</span>
            </button>

            {showHelpPopover && (
              <div className="absolute left-0 top-full mt-1.5 w-80 bg-white border border-slate-200 rounded-xl shadow-lg p-3.5 z-50 text-xs text-slate-700 space-y-2">
                <div className="font-semibold text-slate-900 border-b border-slate-100 pb-1.5 flex items-center justify-between">
                  <span>快速上手指南（三步驟）</span>
                  <span className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                    支援 2~4 小時音檔
                  </span>
                </div>
                <ol className="space-y-1.5 text-[11px] text-slate-600 list-decimal list-inside">
                  <li>
                    <strong className="text-slate-800">確認與會人數：</strong>
                    於左側面板點選預估的會議人數範圍（例如 2～6 人）。
                  </li>
                  <li>
                    <strong className="text-slate-800">載入字典或簡報：</strong>
                    可同步團隊 Google Sheet 專有名詞表，或上傳本次會議簡報作為背景參考。
                  </li>
                  <li>
                    <strong className="text-slate-800">選取會議錄音檔：</strong>
                    點擊「載入會議錄音檔」，系統會先花 3～5 秒自動檢測音質並推薦最佳方案，點擊啟動即可產出逐字稿。
                  </li>
                </ol>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 內顯硬體防呆提示列 */}
      {hardwareWarning && (
        <div className="mt-2 px-3 py-1 bg-amber-50 border border-amber-200 rounded-lg inline-flex items-center gap-1.5 text-xs text-amber-800">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
          <span>{hardwareWarning}</span>
        </div>
      )}

      {/* API Key 設定折疊面板 */}
      {showKeyModal && (
        <div className="max-w-3xl mt-2.5 p-3 bg-slate-50 rounded-lg border border-slate-200 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Google Gemini API Key（選填，啟用雲端深度語意校正與重聽）
            </label>
            <input
              type="password"
              value={config.geminiApiKey}
              onChange={(e) => updateConfig({ geminiApiKey: e.target.value })}
              placeholder="AIzaSy..."
              className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-300 bg-white"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Groq API Key（選填，啟用雲端高速語音轉寫）
            </label>
            <input
              type="password"
              value={config.groqApiKey}
              onChange={(e) => updateConfig({ groqApiKey: e.target.value })}
              placeholder="gsk_..."
              className="w-full text-xs px-2.5 py-1.5 rounded border border-slate-300 bg-white"
            />
          </div>
        </div>
      )}
    </header>
  );
};