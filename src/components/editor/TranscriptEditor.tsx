// src/components/editor/TranscriptEditor.tsx
// 修正重點：
// 1. 解決轉寫結束後「3 張聲學預分析大卡片自動彈出佔滿畫面，導致看不到逐字稿結果」之 UX 問題：
//    - 僅在「剛選取音檔、尚未開始轉寫 (`segments.length === 0 && !isRunning`)」或使用者主動點擊展開時，才顯示完整 3 方案大面板。
//    - 一旦進入轉寫或已有逐字稿結果 (`segments.length > 0`)，自動收合為「單行精簡狀態列」，將 85% 以上垂直空間留給逐字稿檢視！
// 2. 明確區分兩大後續動作，消除「不知道是要重來一次還是要順稿」的混淆：
//    - 按鈕一：【✨ AI 全文順稿與詞彙統一（不重跑音訊）】 -> 直接針對現有逐字稿執行 Stage 1 跨句修復與同音詞統一（1~2 秒完成，保留所有修改與時間軸）。
//    - 按鈕二：【⚙️ 換方案重新轉寫（展開聲學體檢卡）】 -> 展開方案 A/B/C 面板，供想切換語者分離參數並從頭重跑音檔的使用者使用。

import React, { useState } from 'react';
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  HelpCircle,
  Info,
  Loader2,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Volume2,
  Wand2,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { usePipelineScheduler } from '../../hooks/usePipelineScheduler';
import { useAudioPlayerSync } from '../../hooks/useAudioPlayerSync';
import { WordTokenNode } from './WordTokenNode';
import {
  exportAsDocx,
  exportAsMarkdown,
  exportAsSrt,
  exportProjectJson,
} from '../../utils/exporter';
import { AudioRecommendationOption, Segment } from '../../types';

/**
 * 內建浮出式說明視窗組件 (零外部路徑依賴，徹底避免 ts(2307) 找不到 ../common/HoverInfoPopover)
 */
const HoverInfoPopover: React.FC<{
  label?: string;
  title: string;
  children: React.ReactNode;
  align?: 'left' | 'right';
}> = ({ label, title, children, align = 'right' }) => {
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
        className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-600 focus:text-indigo-600 transition-colors cursor-pointer focus:outline-none"
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
        } top-full mt-1.5 z-50 w-72 p-3 rounded-xl bg-slate-900/95 text-slate-100 text-[11px] leading-relaxed shadow-xl border border-slate-700 transition-all duration-150 ${
          pinned
            ? 'opacity-100 visible pointer-events-auto'
            : 'opacity-0 invisible pointer-events-none group-hover:opacity-100 group-hover:visible group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:visible'
        }`}
      >
        <div className="font-semibold text-white mb-1 flex items-center gap-1.5 border-b border-slate-700/80 pb-1">
          <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span>{title}</span>
        </div>
        <div className="text-slate-300 space-y-1 font-normal">{children}</div>
      </div>
    </div>
  );
};

function renderSentenceIntegrityBadge(seg: Segment) {
  const diag = seg.sentenceIntegrity;
  const issues = diag?.issues || seg.sentenceIntegrityIssues || [];
  const score = diag?.score ?? seg.sentenceIntegrityScore ?? 100;

  const hasTailDrop =
    issues.includes('TAIL_DROP_TO_NEXT') ||
    issues.includes('tail_character_drop');
  const hasHeadStolen =
    issues.includes('HEAD_STOLEN_FROM_PREV') ||
    issues.includes('head_stolen_from_prev');
  const hasSyntaxMerge =
    issues.includes('INCOMPLETE_SYNTAX') ||
    issues.includes('broken_syntax_split');
  const hasSelfQa =
    issues.includes('SEMANTIC_ANOMALY') ||
    issues.includes('unreasonable_self_qa');
  const hasHallucination =
    issues.includes('HALLUCINATION_LOOP') ||
    issues.includes('repetition_noise_hallucination');

  if (
    !hasTailDrop &&
    !hasHeadStolen &&
    !hasSyntaxMerge &&
    !hasSelfQa &&
    !hasHallucination
  ) {
    return null;
  }

  let badgeLabel = '✨ Stage 1 句意已修復';
  if (hasTailDrop) badgeLabel = '✨ 已歸還跨段掉落尾字';
  else if (hasHeadStolen) badgeLabel = '✨ 已取回前段誤搶句首詞';
  else if (hasSelfQa) badgeLabel = '✨ 已拆解自問自答並移轉語者';
  else if (hasSyntaxMerge) badgeLabel = '✨ 已合併換氣斷裂殘句';
  else if (hasHallucination) badgeLabel = '✨ 已剝離摩擦重複幻覺';

  return (
    <div className="inline-flex items-center gap-1 mt-1">
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px] font-medium">
        <ShieldCheck className="w-3 h-3 shrink-0" />
        {badgeLabel}
      </span>
      <HoverInfoPopover
        title={`先句後詞檢核 (句意完整度 ${score}%)`}
        align="left"
      >
        <p>
          • <strong>檢核診斷</strong>：
          {diag?.diagnosticNote || '已於單詞檢查前完成跨句結構修復'}
        </p>
        <p className="text-indigo-300 pt-1 border-t border-slate-700/80">
          💡 系統先修復跨段尾字掉落、換氣斷句與問答歸屬，確認句子語意完整後才進行單詞槽位校對，且 100% 保留每個字的點擊播放時間戳。
        </p>
      </HoverInfoPopover>
    </div>
  );
}

export const TranscriptEditor: React.FC = () => {
  const {
    segments,
    speakers,
    currentTime,
    clearTranscript,
    audioFile,
    audioUrl,
    audioPreAnalysis,
    isPreAnalyzingAudio,
    preAnalysisProgress,
    selectedPresetId,
    applyRecommendedPreset,
    config,
  } = useAppStore();

  const {
    isRunning,
    preAnalyzeAudioFile,
    startPipeline,
    triggerRetroactivePass,
  } = usePipelineScheduler();
  const { audioRef, seekToWord, handleTimeUpdate } = useAudioPlayerSync();

  // 控制「全場聲學體檢與 3 方案大面板」是否手動展開：
  // 當已有逐字稿結果 (segments.length > 0) 或正在轉寫時，預設自動收合為單行精簡橫條，不遮擋逐字稿！
  const [manualExpandPreAnalysis, setManualExpandPreAnalysis] =
    useState<boolean>(false);
  const [isPolishingText, setIsPolishingText] = useState<boolean>(false);
  const [polishFeedback, setPolishFeedback] = useState<string | null>(null);

  const hasTranscript = segments.length > 0;

  const handleAudioSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setManualExpandPreAnalysis(true);
    setPolishFeedback(null);
    // 【方向一】：取得音檔後立刻進行 3~5 秒全場多點深度聲學預分析，找出最佳參數建議供使用者選擇後再啟動！
    await preAnalyzeAudioFile(file);
  };

  const handleConfirmAndLaunch = (preset?: AudioRecommendationOption) => {
    if (preset) {
      applyRecommendedPreset(preset);
    }
    setManualExpandPreAnalysis(false);
    setPolishFeedback(null);
    startPipeline();
  };

  const handlePolishOnlyWithoutRerun = async () => {
    if (isPolishingText || isRunning || segments.length === 0) return;
    setIsPolishingText(true);
    setPolishFeedback('正在執行「Stage 1 跨句結構修復 + Stage 2 全場詞彙統一順稿」（不重跑音訊）...');
    try {
      const fixedCount = await triggerRetroactivePass();
      setPolishFeedback(
        fixedCount > 0
          ? `✅ 全文順稿完成！共自動修復 ${fixedCount} 處跨句結構與同音詞彙（可直接點擊下方橘底/紅底詞彙微調）。`
          : '✅ 全文順稿檢查完成！目前跨句語意與全場詞彙已全數一致。'
      );
      setTimeout(() => setPolishFeedback(null), 5000);
    } finally {
      setIsPolishingText(false);
    }
  };

  // 是否展開完整 3 卡片面板：
  // 1. 正在執行 3~5 秒預分析時 -> 顯示預分析進度卡
  // 2. 尚未產生逐字稿 (!hasTranscript) 且未在轉寫中 (!isRunning) -> 自動顯示完整 3 方案卡片供啟動前選擇
  // 3. 已產生逐字稿 (hasTranscript) 或轉寫中 (isRunning) -> 預設收合為 1 行精簡導覽列，除非使用者主動點「換方案重新轉寫」展開
  const showFullPreAnalysisCards =
    Boolean(audioFile) &&
    (isPreAnalyzingAudio ||
      (Boolean(audioPreAnalysis) &&
        ((!hasTranscript && !isRunning) || manualExpandPreAnalysis)));

  const activePresetOption = audioPreAnalysis?.recommendations.find(
    (r) => r.id === selectedPresetId
  );

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* 頂部工具列 */}
      <div className="flex items-center justify-between px-5 py-2.5 bg-white border-b border-slate-200 gap-2 flex-wrap">
        <div className="flex items-center gap-2.5 flex-wrap">
          <label className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-3.5 py-1.5 rounded cursor-pointer shadow-xs transition-colors">
            {isRunning
              ? '微批次流水線轉寫中...'
              : isPreAnalyzingAudio
              ? '正在全場聲學深度預分析...'
              : audioFile
              ? '更換會議錄音檔'
              : '載入會議錄音檔 (MP3/WAV/M4A)'}
            <input
              type="file"
              accept="audio/*"
              onChange={handleAudioSelect}
              disabled={isRunning || isPreAnalyzingAudio}
              className="hidden"
            />
          </label>

          {/* 尚未產生逐字稿時：顯示主啟動按鈕 */}
          {audioFile && !hasTranscript && !isRunning && !isPreAnalyzingAudio && (
            <button
              type="button"
              onClick={() => handleConfirmAndLaunch()}
              className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold px-3.5 py-1.5 rounded shadow-xs transition-colors cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              套用推薦方案並開始轉寫
            </button>
          )}

          {/* 已有逐字稿結果且非轉寫中時：明確並列「✨ AI 全文順稿 (不重跑音訊)」與「🔄 換方案重新轉寫」 */}
          {hasTranscript && !isRunning && (
            <>
              <button
                type="button"
                onClick={handlePolishOnlyWithoutRerun}
                disabled={isPolishingText}
                className="inline-flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white text-xs font-semibold px-3.5 py-1.5 rounded shadow-xs transition-colors cursor-pointer"
                title="保留目前逐字稿與時間軸，僅針對文字重新執行 Stage 1 跨句修復與全場詞彙統一順稿（約 1~2 秒）"
              >
                {isPolishingText ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Wand2 className="w-3.5 h-3.5" />
                )}
                ✨ AI 全文順稿（不重跑音訊）
              </button>

              <button
                type="button"
                onClick={() =>
                  setManualExpandPreAnalysis((prev) => !prev)
                }
                className={`inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1.5 rounded border transition-colors cursor-pointer ${
                  manualExpandPreAnalysis
                    ? 'bg-slate-800 text-white border-slate-800'
                    : 'text-slate-700 hover:bg-slate-100 bg-slate-50 border-slate-300'
                }`}
                title="展開聲學體檢面板，可改選其他語者分離方案（A/B/C）並重新轉寫音檔"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                {manualExpandPreAnalysis
                  ? '收起方案選擇面板'
                  : '換方案重新轉寫'}
                {manualExpandPreAnalysis ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            </>
          )}

          {audioUrl && (
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              onTimeUpdate={handleTimeUpdate}
              className="h-8 max-w-[240px]"
            />
          )}
        </div>

        <div className="flex gap-1.5 text-xs">
          <button
            onClick={() => exportAsDocx(segments, speakers)}
            disabled={!segments.length}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-semibold px-3 py-1.5 rounded cursor-pointer"
          >
            📄 匯出 Word (.docx)
          </button>
          <button
            onClick={() => exportAsMarkdown(segments, speakers)}
            disabled={!segments.length}
            className="border px-2.5 py-1.5 rounded hover:bg-slate-50 disabled:opacity-50 cursor-pointer"
          >
            匯出 Markdown
          </button>
          <button
            onClick={() => exportAsSrt(segments, speakers)}
            disabled={!segments.length}
            className="border px-2.5 py-1.5 rounded hover:bg-slate-50 disabled:opacity-50 cursor-pointer"
          >
            匯出 SRT
          </button>
          <button
            onClick={() => exportProjectJson(segments, speakers)}
            disabled={!segments.length}
            className="border px-2.5 py-1.5 rounded hover:bg-slate-50 disabled:opacity-50 cursor-pointer"
          >
            專案 JSON
          </button>
          {segments.length > 0 && (
            <button
              onClick={clearTranscript}
              className="border border-rose-300 text-rose-600 hover:bg-rose-50 px-2.5 py-1.5 rounded cursor-pointer"
            >
              清空紀錄
            </button>
          )}
        </div>
      </div>

      {/* 轉寫中或完成後之「單行精簡聲學資訊與操作提示列」（不佔用畫面高度） */}
      {audioPreAnalysis && !showFullPreAnalysisCards && (
        <div className="bg-slate-50 border-b border-slate-200 px-5 py-1.5 flex items-center justify-between gap-3 flex-wrap text-xs">
          <div className="flex items-center gap-2 text-slate-600 flex-wrap">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 font-semibold border border-indigo-200 text-[11px]">
              <Activity className="w-3 h-3" />
              目前採用：{activePresetOption?.badge || '方案 A'}（預估{' '}
              {config.minSpeakers}~{config.maxSpeakers} 人）
            </span>
            <span className="text-[11px] text-slate-500">
              SNR {audioPreAnalysis.snrDb}dB · 換人頻率{' '}
              {audioPreAnalysis.turnSwitchRatePerMin} 次/分 · 重疊率{' '}
              {(audioPreAnalysis.overlapRatio * 100).toFixed(1)}%
            </span>
            {hasTranscript && !isRunning && (
              <span className="text-[11px] text-emerald-700 font-medium">
                💡 提示：點擊字詞可聽原音或選同音候選詞；想潤飾全文請點上方「✨ AI 全文順稿」，想換參數重跑請點「換方案重新轉寫」。
              </span>
            )}
          </div>

          <button
            type="button"
            onClick={() => setManualExpandPreAnalysis(true)}
            className="inline-flex items-center gap-1 text-[11px] text-indigo-600 hover:text-indigo-800 font-medium cursor-pointer shrink-0"
          >
            檢視完整聲學報告與 3 方案卡片
            <ChevronDown className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {polishFeedback && (
        <div className="bg-emerald-50 border-b border-emerald-200 px-5 py-2 text-xs text-emerald-800 font-medium flex items-center justify-between">
          <span>{polishFeedback}</span>
          <button
            type="button"
            onClick={() => setPolishFeedback(null)}
            className="text-emerald-600 hover:text-emerald-900 text-[11px] ml-4"
          >
            關閉
          </button>
        </div>
      )}

      {/* 【方向一】：3~5 秒全文深度聲學預分析與 3 組白話情境方案選擇面板（僅在啟動前或使用者主動展開時顯示） */}
      {showFullPreAnalysisCards && (
        <div className="bg-slate-50 border-b border-slate-200 px-6 py-3.5 space-y-3 max-h-[54vh] overflow-y-auto">
          {isPreAnalyzingAudio ? (
            <div className="bg-white rounded-xl border border-indigo-200 p-4 shadow-xs space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 font-semibold text-indigo-900">
                  <Loader2 className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
                  <span>
                    正在對《{audioFile?.name}》執行全場多點深度聲學預分析（約 3～5 秒）...
                  </span>
                </div>
                <span className="font-bold text-indigo-700">
                  {preAnalysisProgress.percent}%
                </span>
              </div>
              <div className="w-full h-2 bg-indigo-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-600 transition-all duration-200"
                  style={{ width: `${preAnalysisProgress.percent}%` }}
                />
              </div>
              <p className="text-[11px] text-slate-600">
                {preAnalysisProgress.detailText ||
                  '正在掃描全場信噪比、雙分位底噪、基頻半音跨度、每分鐘話輪轉換率與預估語者人數...'}
              </p>
            </div>
          ) : (
            audioPreAnalysis && (
              <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs space-y-3">
                {/* 頂部摘要與指標徽章 */}
                <div className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 pb-2">
                  <div className="space-y-1 flex-1 min-w-[260px]">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 font-semibold text-xs border border-indigo-200">
                        <Volume2 className="w-3.5 h-3.5" />
                        全場深度聲學體檢完成 (抽樣{' '}
                        {audioPreAnalysis.sampledWindowsCount} 區段 /{' '}
                        {audioPreAnalysis.sampledDurationSec} 秒)
                      </span>
                      <span className="text-[11px] text-slate-400">
                        耗時{' '}
                        {(audioPreAnalysis.analysisDurationMs / 1000).toFixed(
                          1
                        )}{' '}
                        秒
                      </span>
                      {hasTranscript && (
                        <span className="text-[11px] px-2 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 font-medium">
                          ⚠️ 注意：下方卡片用於「更換語者分離參數並從頭重新轉寫」；若只需順稿請點上方「✨ AI 全文順稿」或直接收合此面板。
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-700 leading-relaxed">
                      {audioPreAnalysis.plainSummary}
                    </p>
                  </div>

                  <div className="flex items-center gap-1.5">
                    {audioFile && !isRunning && (
                      <button
                        type="button"
                        onClick={() => preAnalyzeAudioFile(audioFile)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-slate-200 text-[11px] text-slate-600 hover:bg-slate-50 cursor-pointer"
                        title="重新掃描音檔聲學特徵"
                      >
                        <RefreshCw className="w-3 h-3" />
                        重新檢測
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setManualExpandPreAnalysis(false)}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 text-[11px] font-medium text-slate-700 cursor-pointer"
                    >
                      <ChevronUp className="w-3.5 h-3.5" />
                      收合面板（檢視逐字稿）
                    </button>
                  </div>
                </div>

                {/* 關鍵聲學指標列 */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[11px]">
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80">
                    <div className="text-slate-400">環境信噪比 (SNR)</div>
                    <div className="font-bold text-slate-800">
                      {audioPreAnalysis.snrDb} dB{' '}
                      <span className="font-normal text-slate-500">
                        (底噪 {audioPreAnalysis.noiseFloorDb} dBFS)
                      </span>
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80">
                    <div className="text-slate-400">基頻分佈跨度</div>
                    <div className="font-bold text-slate-800">
                      {audioPreAnalysis.pitchSpreadSemitones} 半音{' '}
                      <span className="font-normal text-slate-500">
                        ({audioPreAnalysis.f0P10Hz}~{audioPreAnalysis.f0P90Hz}
                        Hz)
                      </span>
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80">
                    <div className="text-slate-400">每分鐘換人頻率</div>
                    <div className="font-bold text-slate-800">
                      {audioPreAnalysis.turnSwitchRatePerMin} 次/分
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80">
                    <div className="text-slate-400">重疊搶話比例</div>
                    <div className="font-bold text-slate-800">
                      {(audioPreAnalysis.overlapRatio * 100).toFixed(1)}%{' '}
                      <span className="font-normal text-slate-500">
                        (
                        {audioPreAnalysis.overlapRisk === 'HIGH'
                          ? '偏高'
                          : audioPreAnalysis.overlapRisk === 'MEDIUM'
                          ? '中等'
                          : '低'}
                        )
                      </span>
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80">
                    <div className="text-slate-400">預估語者人數</div>
                    <div className="font-bold text-blue-700">
                      {audioPreAnalysis.estimatedMinSpeakers} ～{' '}
                      {audioPreAnalysis.estimatedMaxSpeakers} 人
                    </div>
                  </div>
                </div>

                {/* 3 組白話情境推薦卡片 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {audioPreAnalysis.recommendations.map((opt) => {
                    const isSelected = selectedPresetId === opt.id;
                    return (
                      <div
                        key={opt.id}
                        onClick={() => applyRecommendedPreset(opt)}
                        className={`rounded-xl border p-3 transition-all cursor-pointer flex flex-col justify-between ${
                          isSelected
                            ? 'border-blue-600 bg-blue-50/40 ring-2 ring-blue-500/20 shadow-xs'
                            : 'border-slate-200 bg-white hover:border-blue-300 hover:bg-slate-50/60'
                        }`}
                      >
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between gap-1">
                            <span
                              className={`text-[11px] font-bold px-2 py-0.5 rounded ${
                                isSelected
                                  ? 'bg-blue-600 text-white'
                                  : 'bg-slate-100 text-slate-700'
                              }`}
                            >
                              {opt.badge}
                            </span>
                            {opt.isRecommended && (
                              <span className="inline-flex items-center gap-0.5 text-[10px] font-semibold px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-300">
                                <Sparkles className="w-3 h-3 text-amber-600" />
                                AI 最佳推薦
                              </span>
                            )}
                          </div>

                          <div className="font-bold text-xs text-slate-800 pt-0.5">
                            {opt.title}
                          </div>
                          <p className="text-[11px] text-slate-600 leading-relaxed">
                            {opt.whyRecommend}
                          </p>
                          <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-100 space-y-0.5">
                            <div>
                              <strong className="text-emerald-700">
                                適合：
                              </strong>
                              {opt.bestFor}
                            </div>
                            <div>
                              <strong className="text-slate-600">
                                建議人數：
                              </strong>
                              {opt.recommendedMinSpeakers}～
                              {opt.recommendedMaxSpeakers} 人
                            </div>
                            {opt.diarizationPatch.formulaCalibration && (
                              <div className="pt-0.5 text-[10px] text-indigo-700 font-medium">
                                <strong>底層聲學公式動態閥值：</strong>
                                諧波門檻{" "}
                                {
                                  opt.diarizationPatch.formulaCalibration
                                    .harmonicityGateThreshold
                                }{" "}
                                · 邊界波峰{" "}
                                {
                                  opt.diarizationPatch.formulaCalibration
                                    .scdPeakCutoff
                                }{" "}
                                · 微觀視窗{" "}
                                {
                                  opt.diarizationPatch.formulaCalibration
                                    .maxMicroTurnSec
                                }
                                s · 階躍 ≥
                                {
                                  opt.diarizationPatch.formulaCalibration
                                    .stepDbDiffFloor
                                }
                                dB
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="pt-2.5 mt-2 border-t border-slate-100 flex items-center justify-between">
                          <span className="text-[11px] font-medium text-blue-700 flex items-center gap-1">
                            {isSelected ? (
                              <>
                                <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" />
                                目前已選取此方案
                              </>
                            ) : (
                              '點選切換此方案'
                            )}
                          </span>
                          {!isRunning && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleConfirmAndLaunch(opt);
                              }}
                              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors cursor-pointer ${
                                isSelected
                                  ? 'bg-emerald-600 hover:bg-emerald-700 text-white'
                                  : 'bg-slate-800 hover:bg-slate-900 text-white'
                              }`}
                            >
                              {hasTranscript
                                ? '🔄 改用此方案重新轉寫'
                                : '▶ 以此方案啟動轉寫'}
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* 底部確認啟動列 */}
                {!isRunning && (
                  <div className="flex items-center justify-between pt-1 text-xs text-slate-600 flex-wrap gap-2">
                    <span>
                      目前設定：預估{' '}
                      <strong className="text-slate-800">
                        {config.minSpeakers}～{config.maxSpeakers} 人
                      </strong>{' '}
                      · 檢核管線已啟用{' '}
                      <strong className="text-indigo-700">
                        「Stage 1 先抓句子語意完整性 → Stage 2 單詞槽位精修」
                      </strong>
                    </span>
                    <div className="flex items-center gap-2">
                      {hasTranscript && (
                        <button
                          type="button"
                          onClick={() => setManualExpandPreAnalysis(false)}
                          className="px-3.5 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 font-semibold cursor-pointer"
                        >
                          保留目前結果並收合
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => handleConfirmAndLaunch()}
                        className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-4 py-1.5 rounded-lg shadow-sm transition-colors cursor-pointer"
                      >
                        <Play className="w-3.5 h-3.5 fill-current" />
                        {hasTranscript
                          ? '套用新方案並重新轉寫全檔'
                          : '確認方案並開始轉寫'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          )}
        </div>
      )}

      {/* 逐字稿段落列表 */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-white">
        {segments.length === 0 && !isPreAnalyzingAudio && (
          <div className="h-full flex flex-col items-center justify-center text-center text-slate-400 space-y-2 py-12">
            <p className="text-sm font-medium text-slate-500">
              {audioFile
                ? '請於上方確認聲學預分析推薦方案，並點擊「確認方案並開始轉寫」'
                : '請點擊左上角「載入會議錄音檔」開始 3～5 秒全文深度聲學預分析'}
            </p>
            <p className="text-xs text-slate-400 max-w-md">
              系統將自動完成全場多點聲學體檢、推薦最佳語者分離參數，並以「先句後詞（先修跨段尾字掉落與句子完整性，再修同音單詞）」雙階段檢核產出高品質逐字稿。
            </p>
          </div>
        )}

        {segments.map((seg) => {
          const sp = speakers[seg.speakerId];
          return (
            <div key={seg.id} className="flex gap-4 items-start">
              <div className="w-36 shrink-0 pt-0.5">
                <span
                  className="inline-block text-xs font-bold px-2 py-0.5 rounded text-white"
                  style={{ backgroundColor: sp?.color || '#475569' }}
                >
                  {sp?.name || seg.speakerId}
                </span>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  {Math.floor(seg.start / 60)}:
                  {String(Math.floor(seg.start % 60)).padStart(2, '0')}
                </div>
                {renderSentenceIntegrityBadge(seg)}
              </div>

              <div className="flex-1 text-sm leading-relaxed text-slate-800">
                {seg.words.map((w, idx) => {
                  const isWordActive =
                    currentTime >= w.start && currentTime <= w.end;
                  return (
                    <WordTokenNode
                      key={w.id}
                      segmentId={seg.id}
                      wordIndex={idx}
                      word={w}
                      isActive={isWordActive}
                      onPlaySnippet={seekToWord}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};