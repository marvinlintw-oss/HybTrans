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
  const currentSpec = TIER_PRESET_SPECS[config.granularityTier || 'eco_safe'];
  const hardwareWarning = validateTierAgainstHardware(
    config.granularityTier,
    probe,
    hasGeminiKey
  );

  return (
    <header className="bg-white border-b border-slate-200 px-4 py-2.5 shadow-sm">
      {/* 全寬靠左對齊 (移除 max-w-7xl mx-auto 置中) */}
      <div className="w-full flex flex-wrap items-center justify-start gap-4">
        {/* 1. 系統標題與輕量資訊提示 */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">
            HT
          </div>
          <div className="flex items-center gap-1.5">
            <h1 className="font-bold text-slate-800 text-sm">HybridTranscriber</h1>
            <span
              title={
                probe
                  ? `硬體環境：${probe.adapterName} (已快取 ${probe.estimatedCacheUsageMB}MB) · 5分鐘微批次切片 · 支援 2~4 小時長音檔`
                  : '正在偵測硬體環境...'
              }
              className="text-slate-400 hover:text-slate-600 cursor-help inline-flex items-center"
            >
              <Info className="w-3.5 h-3.5" />
            </span>
          </div>
        </div>

        {/* 2. 電腦效能與細膩度規格選單 */}
        <div className="flex items-center gap-1.5 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200">
          <Gauge className="w-3.5 h-3.5 text-blue-600 shrink-0" />
          <div className="text-xs flex items-center">
            <label htmlFor="tier-select" className="text-slate-500 mr-1 font-medium">
              效能模式:
            </label>
            <select
              id="tier-select"
              value={config.granularityTier}
              onChange={(e) => applyGranularityTier(e.target.value as GranularityTier)}
              className="font-semibold text-slate-800 bg-transparent focus:outline-none cursor-pointer"
            >
              {(Object.keys(TIER_PRESET_SPECS) as GranularityTier[]).map((tKey) => {
                const spec = TIER_PRESET_SPECS[tKey];
                const isRecommended = probe?.recommendedTier === tKey;
                return (
                  <option key={tKey} value={tKey}>
                    {spec.label}{isRecommended ? ' (推薦)' : ''}
                  </option>
                );
              })}
            </select>
          </div>
          <span
            title={`高關注度視窗 ${currentSpec.highAttentionChars}字 / 中關注度前文 ${currentSpec.mediumAttentionChars}字 / 每5分鐘自動重聽上限 ${currentSpec.maxRelistenQuotaPerChunk}段 (${currentSpec.hardwareHint})`}
            className="text-slate-400 hover:text-blue-600 cursor-help pl-1"
          >
            <Info className="w-3.5 h-3.5" />
          </span>
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

          {/* 4. 收納進問號圖示的「快速上手指南」Popover */}
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
                  <span>同仁快速上手三步驟</span>
                  <span className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                    支援 2~4 小時音檔
                  </span>
                </div>
                <ol className="space-y-1.5 text-[11px] text-slate-600 list-decimal list-inside">
                  <li>
                    <strong className="text-slate-800">設定人數：</strong>
                    於左側欄點擊「鎖定 5~6 人」或輸入會議人數上限。
                  </li>
                  <li>
                    <strong className="text-slate-800">同步字典/簡報：</strong>
                    可貼上團隊 Google Sheet 字典網址自動同步，或上傳會議簡報。
                  </li>
                  <li>
                    <strong className="text-slate-800">載入錄音檔：</strong>
                    點擊藍色按鈕選取音檔即可自動完成（首次載入模型後即保存在瀏覽器免重載）。
                  </li>
                </ol>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 內顯硬體防呆提示列 (靠左對齊) */}
      {hardwareWarning && (
        <div className="mt-2 px-3 py-1 bg-amber-50 border border-amber-200 rounded-lg inline-flex items-center gap-1.5 text-xs text-amber-800">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
          <span>{hardwareWarning}</span>
        </div>
      )}

      {/* API Key 簡易設定折疊面板 (靠左對齊) */}
      {showKeyModal && (
        <div className="max-w-3xl mt-2.5 p-3 bg-slate-50 rounded-lg border border-slate-200 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Google Gemini API Key（選填，啟用雲端深度校正與重聽）
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
              Groq API Key（選填，啟用雲端極速語音轉寫）
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