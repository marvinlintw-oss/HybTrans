// src/components/editor/TranscriptEditor.tsx
import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardList,
  Copy,
  Download,
  FileText,
  HelpCircle,
  Info,
  ListChecks,
  Loader2,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Trash2,
  Users,
  Volume2,
  Wand2,
  X,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { usePipelineScheduler } from '../../hooks/usePipelineScheduler';
import { useAudioPlayerSync } from '../../hooks/useAudioPlayerSync';
import { WordTokenNode } from './WordTokenNode';
import {
  exportAsDocx,
  exportAsMarkdown,
  exportAsSrt,
  exportMeetingMinutesAsDocx,
  exportMeetingMinutesAsMarkdown,
  exportProjectJson,
  formatMeetingMinutesToMarkdown,
} from '../../utils/exporter';
import { AudioRecommendationOption, Segment } from '../../types';

/**
 * 內建浮出式說明視窗組件（支援電腦滑鼠懸停與手機/平板點擊展開，手機端自動置中防裁切）
 */
const HoverInfoPopover: React.FC<{
  label?: string;
  title: string;
  children: React.ReactNode;
  align?: 'left' | 'right';
}> = ({ label, title, children, align = 'right' }) => {
  const [pinned, setPinned] = useState(false);
  const popRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!pinned) return;
    const handleOutside = (e: MouseEvent | TouchEvent) => {
      if (popRef.current && !popRef.current.contains(e.target as Node)) {
        setPinned(false);
      }
    };
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('touchstart', handleOutside);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('touchstart', handleOutside);
    };
  }, [pinned]);

  return (
    <div
      ref={popRef}
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
        className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-600 focus:text-indigo-600 transition-colors cursor-pointer focus:outline-none p-0.5"
      >
        <HelpCircle className="w-3.5 h-3.5 shrink-0" />
        {label && (
          <span className="underline decoration-dotted underline-offset-2">
            {label}
          </span>
        )}
      </button>

      <div
        className={`max-sm:fixed max-sm:left-3 max-sm:right-3 max-sm:top-16 max-sm:w-auto sm:absolute ${
          align === 'right' ? 'sm:right-0' : 'sm:left-0'
        } sm:top-full sm:mt-1.5 z-50 sm:w-72 p-3 rounded-xl bg-slate-900/95 text-slate-100 text-[11px] leading-relaxed shadow-xl border border-slate-700 transition-all duration-150 ${
          pinned
            ? 'opacity-100 visible pointer-events-auto'
            : 'opacity-0 invisible pointer-events-none sm:group-hover:opacity-100 sm:group-hover:visible sm:group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:visible'
        }`}
      >
        <div className="font-semibold text-white mb-1 flex items-center justify-between gap-1.5 border-b border-slate-700/80 pb-1">
          <span className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span>{title}</span>
          </span>
          {pinned && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setPinned(false);
              }}
              className="text-slate-400 hover:text-white p-0.5 sm:hidden"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
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
  else if (hasSelfQa) badgeLabel = '✨ 已校準問答語用與語者';
  else if (hasSyntaxMerge) badgeLabel = '✨ 已合併換氣斷裂殘句';
  else if (hasHallucination) badgeLabel = '✨ 已剝離摩擦重複幻覺';

  return (
    <div className="inline-flex items-center gap-1 sm:mt-1">
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

export interface TranscriptEditorProps {
  onOpenMobileSettings?: (tab: 'speakers' | 'dictionary') => void;
}

export const TranscriptEditor: React.FC<TranscriptEditorProps> = ({
  onOpenMobileSettings,
}) => {
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
    meetingMinutes,
    isGeneratingMinutes,
  } = useAppStore();

  const {
    isRunning,
    preAnalyzeAudioFile,
    startPipeline,
    triggerRetroactivePass,
    generateMeetingMinutes,
  } = usePipelineScheduler();
  const { audioRef, seekToWord, handleTimeUpdate } = useAudioPlayerSync();

  const [manualExpandPreAnalysis, setManualExpandPreAnalysis] =
    useState<boolean>(false);
  const [isPolishingText, setIsPolishingText] = useState<boolean>(false);
  const [polishFeedback, setPolishFeedback] = useState<string | null>(null);
  const [showMobileExportMenu, setShowMobileExportMenu] =
    useState<boolean>(false);
  const [showMinutesCard, setShowMinutesCard] = useState<boolean>(true);
  const [copiedMinutes, setCopiedMinutes] = useState<boolean>(false);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const minutesSectionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!showMobileExportMenu) return;
    const handleOutside = (e: MouseEvent | TouchEvent) => {
      if (
        exportMenuRef.current &&
        !exportMenuRef.current.contains(e.target as Node)
      ) {
        setShowMobileExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('touchstart', handleOutside);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('touchstart', handleOutside);
    };
  }, [showMobileExportMenu]);

  const hasTranscript = segments.length > 0;

  const handleAudioSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setManualExpandPreAnalysis(true);
    setPolishFeedback(null);
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
    setPolishFeedback(
      '正在執行「Stage 1 跨句結構修復 + Stage 2 全場詞彙統一順稿」（不重跑音訊）...'
    );
    try {
      const fixedCount = await triggerRetroactivePass();
      setPolishFeedback(
        fixedCount > 0
          ? `✅ 全文順稿完成！共自動修復 ${fixedCount} 處跨句結構與同音詞彙（可直接點擊下方綠底/黃底詞彙檢視或微調）。`
          : '✅ 全文順稿檢查完成！目前跨句語意與全場詞彙已全數一致。'
      );
      setTimeout(() => setPolishFeedback(null), 5000);
    } finally {
      setIsPolishingText(false);
    }
  };

  const handleGenerateAndScrollMinutes = async () => {
    if (isGeneratingMinutes || segments.length === 0) return;
    setShowMinutesCard(true);
    await generateMeetingMinutes();
    setTimeout(() => {
      minutesSectionRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    }, 100);
  };

  const handleCopyMinutesMarkdown = async () => {
    if (!meetingMinutes) return;
    try {
      const md = formatMeetingMinutesToMarkdown(meetingMinutes);
      await navigator.clipboard.writeText(md);
      setCopiedMinutes(true);
      setTimeout(() => setCopiedMinutes(false), 2500);
    } catch {
      // ignore clipboard fallback
    }
  };

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
      {/* 頂部工具列 (RWD 自適應：支援手機單手操作、平板與桌機完整並列) */}
      <div className="flex items-center justify-between px-3 sm:px-5 py-2 sm:py-2.5 bg-white border-b border-slate-200 gap-2 flex-wrap shrink-0">
        <div className="flex items-center gap-2 flex-wrap w-full sm:w-auto">
          <label className="bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white text-xs font-semibold px-3 sm:px-3.5 py-1.5 rounded-lg cursor-pointer shadow-xs transition-colors text-center">
            {isRunning
              ? '微批次轉寫中...'
              : isPreAnalyzingAudio
              ? '聲學預分析中...'
              : audioFile
              ? '更換錄音檔'
              : '載入會議錄音檔'}
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
              className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold px-3 sm:px-3.5 py-1.5 rounded-lg shadow-xs transition-colors cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-current shrink-0" />
              <span>套用推薦方案並開始轉寫</span>
            </button>
          )}

          {/* 已有逐字稿結果時：並列「✨ AI 全文順稿」、「📝 生成會議摘要與結論」與「換方案重跑」 */}
          {hasTranscript && (
            <>
              {!isRunning && (
                <button
                  type="button"
                  onClick={handlePolishOnlyWithoutRerun}
                  disabled={isPolishingText}
                  className="inline-flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white text-xs font-semibold px-3 py-1.5 rounded-lg shadow-xs transition-colors cursor-pointer"
                >
                  {isPolishingText ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                  ) : (
                    <Wand2 className="w-3.5 h-3.5 shrink-0" />
                  )}
                  <span>✨ AI 全文順稿</span>
                </button>
              )}

              <button
                type="button"
                onClick={handleGenerateAndScrollMinutes}
                disabled={isGeneratingMinutes}
                className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-400 text-white text-xs font-semibold px-3 py-1.5 rounded-lg shadow-xs transition-colors cursor-pointer"
              >
                {isGeneratingMinutes ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                ) : (
                  <ClipboardList className="w-3.5 h-3.5 shrink-0" />
                )}
                <span>
                  {meetingMinutes
                    ? '📋 會議摘要與結論'
                    : '📝 生成會議摘要與行動事項'}
                </span>
              </button>

              {!isRunning && (
                <button
                  type="button"
                  onClick={() => setManualExpandPreAnalysis((prev) => !prev)}
                  className={`inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1.5 rounded-lg border transition-colors cursor-pointer ${
                    manualExpandPreAnalysis
                      ? 'bg-slate-800 text-white border-slate-800'
                      : 'text-slate-700 hover:bg-slate-100 bg-slate-50 border-slate-300'
                  }`}
                >
                  <RotateCcw className="w-3.5 h-3.5 shrink-0" />
                  <span>
                    {manualExpandPreAnalysis ? '收起方案面板' : '換方案重跑'}
                  </span>
                  {manualExpandPreAnalysis ? (
                    <ChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5" />
                  )}
                </button>
              )}
            </>
          )}

          {/* 右側匯出區 (手機/平板收納於下拉選單，桌機完整展開) */}
          <div className="flex items-center gap-1.5 ml-auto sm:ml-0">
            <button
              type="button"
              onClick={() => exportAsDocx(segments, speakers, meetingMinutes)}
              disabled={!segments.length}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white text-xs font-semibold px-2.5 sm:px-3 py-1.5 rounded-lg cursor-pointer transition-colors"
            >
              📄 <span className="hidden xs:inline">匯出 </span>Word
            </button>

            {/* 桌機大螢幕 (xl+) 直接展開所有匯出按鈕 */}
            <div className="hidden xl:flex items-center gap-1.5 text-xs">
              <button
                type="button"
                onClick={() =>
                  exportAsMarkdown(segments, speakers, meetingMinutes)
                }
                disabled={!segments.length}
                className="border border-slate-200 px-2.5 py-1.5 rounded-lg hover:bg-slate-50 disabled:opacity-50 cursor-pointer"
              >
                匯出 Markdown
              </button>
              <button
                type="button"
                onClick={() => exportAsSrt(segments, speakers)}
                disabled={!segments.length}
                className="border border-slate-200 px-2.5 py-1.5 rounded-lg hover:bg-slate-50 disabled:opacity-50 cursor-pointer"
              >
                匯出 SRT
              </button>
              <button
                type="button"
                onClick={() =>
                  exportProjectJson(segments, speakers, meetingMinutes)
                }
                disabled={!segments.length}
                className="border border-slate-200 px-2.5 py-1.5 rounded-lg hover:bg-slate-50 disabled:opacity-50 cursor-pointer"
              >
                專案 JSON
              </button>
              {segments.length > 0 && (
                <button
                  type="button"
                  onClick={clearTranscript}
                  className="border border-rose-300 text-rose-600 hover:bg-rose-50 px-2.5 py-1.5 rounded-lg cursor-pointer"
                >
                  清空紀錄
                </button>
              )}
            </div>

            {/* 手機與平板 (<xl)：收納為「更多匯出 ▾」選單，節省工具列空間 */}
            <div ref={exportMenuRef} className="relative xl:hidden">
              <button
                type="button"
                disabled={!segments.length}
                onClick={() => setShowMobileExportMenu((prev) => !prev)}
                className="inline-flex items-center gap-1 border border-slate-200 bg-white hover:bg-slate-50 disabled:opacity-40 text-slate-700 text-xs font-medium px-2.5 py-1.5 rounded-lg cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>更多</span>
                <ChevronDown className="w-3 h-3" />
              </button>

              {showMobileExportMenu && segments.length > 0 && (
                <div className="absolute right-0 top-full mt-1.5 w-48 bg-white border border-slate-200 rounded-xl shadow-xl p-1.5 z-40 text-xs space-y-1">
                  <button
                    type="button"
                    onClick={() => {
                      exportAsMarkdown(segments, speakers, meetingMinutes);
                      setShowMobileExportMenu(false);
                    }}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-slate-100 text-slate-700 font-medium cursor-pointer"
                  >
                    📝 匯出 Markdown (.md)
                  </button>
                  {meetingMinutes && (
                    <button
                      type="button"
                      onClick={() => {
                        exportMeetingMinutesAsDocx(meetingMinutes);
                        setShowMobileExportMenu(false);
                      }}
                      className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-emerald-50 text-emerald-700 font-medium cursor-pointer"
                    >
                      📋 單獨匯出會議摘要 (.docx)
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      exportAsSrt(segments, speakers);
                      setShowMobileExportMenu(false);
                    }}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-slate-100 text-slate-700 font-medium cursor-pointer"
                  >
                    🎬 匯出字幕檔 (.srt)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      exportProjectJson(segments, speakers, meetingMinutes);
                      setShowMobileExportMenu(false);
                    }}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-slate-100 text-slate-700 font-medium cursor-pointer"
                  >
                    💾 匯出專案備份 (.json)
                  </button>
                  <div className="border-t border-slate-100 my-1" />
                  <button
                    type="button"
                    onClick={() => {
                      clearTranscript();
                      setShowMobileExportMenu(false);
                    }}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-rose-50 text-rose-600 font-medium flex items-center gap-1.5 cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>清空目前逐字稿</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 音訊播放器：手機端自動滿版寬度方便手指拖曳時間軸，電腦端維持精緻寬度 */}
        {audioUrl && (
          <div className="w-full sm:w-auto pt-0.5 sm:pt-0">
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              onTimeUpdate={handleTimeUpdate}
              className="h-8 w-full sm:w-56 md:w-64"
            />
          </div>
        )}
      </div>

      {/* 轉寫中或完成後之「單行精簡聲學資訊與操作提示列」 */}
      {audioPreAnalysis && !showFullPreAnalysisCards && (
        <div className="bg-slate-50 border-b border-slate-200 px-3 sm:px-5 py-1.5 flex items-center justify-between gap-2 flex-wrap text-xs shrink-0">
          <div className="flex items-center gap-2 text-slate-600 flex-wrap">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 font-semibold border border-indigo-200 text-[11px]">
              <Activity className="w-3 h-3 shrink-0" />
              <span>
                {activePresetOption?.badge || '方案 A'}（{config.minSpeakers}~
                {config.maxSpeakers} 人）
              </span>
            </span>
            <span className="text-[11px] text-slate-500">
              SNR {audioPreAnalysis.snrDb}dB · 換人{' '}
              {audioPreAnalysis.turnSwitchRatePerMin}次/分
            </span>
            {hasTranscript && !isRunning && (
              <span className="text-[11px] text-emerald-700 font-medium hidden md:inline">
                💡 點擊字詞可聽原音或選同音候選詞；點「📝 生成會議摘要與行動事項」可整理全場結論。
              </span>
            )}
          </div>

          <button
            type="button"
            onClick={() => setManualExpandPreAnalysis(true)}
            className="inline-flex items-center gap-1 text-[11px] text-indigo-600 hover:text-indigo-800 font-medium cursor-pointer shrink-0"
          >
            <span>聲學報告與方案</span>
            <ChevronDown className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {polishFeedback && (
        <div className="bg-emerald-50 border-b border-emerald-200 px-3 sm:px-5 py-2 text-xs text-emerald-800 font-medium flex items-center justify-between gap-2 shrink-0">
          <span>{polishFeedback}</span>
          <button
            type="button"
            onClick={() => setPolishFeedback(null)}
            className="text-emerald-600 hover:text-emerald-900 text-[11px] shrink-0"
          >
            關閉
          </button>
        </div>
      )}

      {/* 【方向一】：3~5 秒全文深度聲學預分析與 3 組白話情境方案選擇面板 */}
      {showFullPreAnalysisCards && (
        <div className="bg-slate-50 border-b border-slate-200 px-3 sm:px-6 py-3 space-y-3 max-h-[65dvh] sm:max-h-[56vh] overflow-y-auto shrink-0">
          {isPreAnalyzingAudio ? (
            <div className="bg-white rounded-xl border border-indigo-200 p-3.5 sm:p-4 shadow-xs space-y-2.5">
              <div className="flex items-center justify-between text-xs gap-2">
                <div className="flex items-center gap-2 font-semibold text-indigo-900 min-w-0">
                  <Loader2 className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
                  <span className="truncate">
                    正在對《{audioFile?.name}》執行全場多點聲學預分析（約 3～5 秒）...
                  </span>
                </div>
                <span className="font-bold text-indigo-700 shrink-0">
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
                  '正在掃描全場信噪比、雙分位底噪、YIN 基頻半音跨度、每分鐘話輪轉換率與預估語者人數...'}
              </p>
            </div>
          ) : (
            audioPreAnalysis && (
              <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4 shadow-xs space-y-3">
                {/* 頂部摘要與指標徽章 */}
                <div className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 pb-2">
                  <div className="space-y-1 flex-1 min-w-[220px]">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 font-semibold text-xs border border-indigo-200">
                        <Volume2 className="w-3.5 h-3.5 shrink-0" />
                        <span>
                          全場深度聲學體檢完成 (抽樣{' '}
                          {audioPreAnalysis.sampledWindowsCount} 區段 /{' '}
                          {audioPreAnalysis.sampledDurationSec} 秒)
                        </span>
                      </span>
                      <span className="text-[11px] text-slate-400">
                        耗時{' '}
                        {(audioPreAnalysis.analysisDurationMs / 1000).toFixed(
                          1
                        )}{' '}
                        秒
                      </span>
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
                      >
                        <RefreshCw className="w-3 h-3" />
                        <span>重新檢測</span>
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setManualExpandPreAnalysis(false)}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 text-[11px] font-medium text-slate-700 cursor-pointer"
                    >
                      <ChevronUp className="w-3.5 h-3.5" />
                      <span>收合面板</span>
                    </button>
                  </div>
                </div>

                {/* 關鍵聲學指標列 (RWD: 手機 2 欄、電腦 5 欄) */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[11px]">
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80">
                    <div className="text-slate-400">環境信噪比 (SNR)</div>
                    <div className="font-bold text-slate-800">
                      {audioPreAnalysis.snrDb} dB{' '}
                      <span className="font-normal text-slate-500">
                        ({audioPreAnalysis.noiseFloorDb} dBFS)
                      </span>
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80">
                    <div className="text-slate-400">基頻分佈跨度</div>
                    <div className="font-bold text-slate-800">
                      {audioPreAnalysis.pitchSpreadSemitones} 半音{' '}
                      <span className="font-normal text-slate-500">
                        ({audioPreAnalysis.f0P10Hz}~{audioPreAnalysis.f0P90Hz}Hz)
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
                  <div className="bg-slate-50 rounded-lg px-2.5 py-1.5 border border-slate-200/80 col-span-2 sm:col-span-1">
                    <div className="text-slate-400">預估語者人數</div>
                    <div className="font-bold text-blue-700">
                      {audioPreAnalysis.estimatedMinSpeakers} ～{' '}
                      {audioPreAnalysis.estimatedMaxSpeakers} 人
                    </div>
                  </div>
                </div>

                {/* 3 組白話情境推薦卡片 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 sm:gap-3">
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
                          </div>
                        </div>

                        <div className="pt-2.5 mt-2 border-t border-slate-100 flex items-center justify-between gap-2">
                          <span className="text-[11px] font-medium text-blue-700 flex items-center gap-1">
                            {isSelected ? (
                              <>
                                <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                                <span>目前已選取</span>
                              </>
                            ) : (
                              '點選切換'
                            )}
                          </span>
                          {!isRunning && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleConfirmAndLaunch(opt);
                              }}
                              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-colors cursor-pointer ${
                                isSelected
                                  ? 'bg-emerald-600 hover:bg-emerald-700 text-white'
                                  : 'bg-slate-800 hover:bg-slate-900 text-white'
                              }`}
                            >
                              {hasTranscript
                                ? '🔄 以此方案重跑'
                                : '▶ 以此方案啟動'}
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
                      · 已啟用{' '}
                      <strong className="text-indigo-700">
                        「YIN 基頻防倍頻 + 遠場響度補償 + 早期 AI 語境介入」
                      </strong>
                    </span>
                    <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                      {hasTranscript && (
                        <button
                          type="button"
                          onClick={() => setManualExpandPreAnalysis(false)}
                          className="px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 font-semibold cursor-pointer"
                        >
                          保留結果並收合
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => handleConfirmAndLaunch()}
                        className="inline-flex items-center justify-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-4 py-1.5 rounded-lg shadow-sm transition-colors cursor-pointer flex-1 sm:flex-initial"
                      >
                        <Play className="w-3.5 h-3.5 fill-current" />
                        <span>
                          {hasTranscript
                            ? '套用新方案並重新轉寫'
                            : '確認方案並開始轉寫'}
                        </span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          )}
        </div>
      )}

      {/* 逐字稿段落列表與會議 AI 摘要面板 */}
      <div className="flex-1 overflow-y-auto p-3 sm:p-5 md:p-6 space-y-2.5 sm:space-y-4 bg-white">
        {segments.length === 0 && !isPreAnalyzingAudio && (
          <div className="h-full flex flex-col items-center justify-center text-center text-slate-400 space-y-3 py-10 px-4">
            <p className="text-sm font-medium text-slate-600">
              {audioFile
                ? '請於上方確認聲學預分析推薦方案，並點擊「確認方案並開始轉寫」'
                : '請點擊左上角「載入會議錄音檔」開始 3～5 秒全文深度聲學預分析'}
            </p>
            <p className="text-xs text-slate-400 max-w-md leading-relaxed">
              系統將自動完成全場多點聲學體檢、推薦最佳語者分離參數，並以「先句後詞（先修跨段尾字掉落與句子完整性，再修同音單詞）」雙階段檢核產出高品質逐字稿與會議摘要。
            </p>

            {/* 手機/平板快速設定捷徑按鈕 */}
            {onOpenMobileSettings && (
              <div className="flex items-center gap-2 pt-2 lg:hidden flex-wrap justify-center">
                <button
                  type="button"
                  onClick={() => onOpenMobileSettings('speakers')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 hover:bg-slate-100 text-xs text-slate-700 font-medium cursor-pointer"
                >
                  <Users className="w-3.5 h-3.5 text-blue-600" />
                  <span>設定會議人數與情境</span>
                </button>
                <button
                  type="button"
                  onClick={() => onOpenMobileSettings('dictionary')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 hover:bg-slate-100 text-xs text-slate-700 font-medium cursor-pointer"
                >
                  <BookOpen className="w-3.5 h-3.5 text-indigo-600" />
                  <span>同步字典或上傳簡報</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* 結構化會議 AI 摘要、重要結論與行動事項報告面板 */}
        {meetingMinutes && (
          <div
            ref={minutesSectionRef}
            className="rounded-2xl border border-emerald-200 bg-gradient-to-br from-emerald-50/70 via-white to-indigo-50/40 p-3.5 sm:p-5 shadow-xs space-y-3.5 mb-4"
          >
            <div className="flex items-start justify-between gap-2 flex-wrap border-b border-emerald-200/70 pb-2.5">
              <div className="space-y-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-600 text-white text-xs font-bold shadow-2xs">
                    <ClipboardList className="w-3.5 h-3.5" />
                    <span>會議 AI 摘要、結論與行動事項</span>
                  </span>
                  <span className="text-[11px] text-slate-500">
                    產生時間：
                    {new Date(meetingMinutes.generatedAt).toLocaleTimeString(
                      'zh-TW',
                      { hour: '2-digit', minute: '2-digit', second: '2-digit' }
                    )}
                  </span>
                </div>
                <h3 className="text-sm sm:text-base font-bold text-slate-900">
                  {meetingMinutes.title}
                </h3>
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  type="button"
                  onClick={handleCopyMinutesMarkdown}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 text-xs font-medium cursor-pointer shadow-2xs"
                >
                  {copiedMinutes ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                      <span className="text-emerald-700">已複製 Markdown</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5 text-slate-500" />
                      <span>複製摘要</span>
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => exportMeetingMinutesAsMarkdown(meetingMinutes)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 text-xs font-medium cursor-pointer shadow-2xs"
                >
                  <FileText className="w-3.5 h-3.5 text-indigo-600" />
                  <span>匯出 .md</span>
                </button>
                <button
                  type="button"
                  onClick={() => exportMeetingMinutesAsDocx(meetingMinutes)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold cursor-pointer shadow-2xs"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>匯出摘要 Word</span>
                </button>
                <button
                  type="button"
                  onClick={() => generateMeetingMinutes()}
                  disabled={isGeneratingMinutes}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-white hover:bg-slate-50 text-slate-600 border border-slate-200 text-xs cursor-pointer"
                  title="依據最新修改之逐字稿與語者姓名重新生成摘要"
                >
                  <RefreshCw
                    className={`w-3.5 h-3.5 ${
                      isGeneratingMinutes ? 'animate-spin text-emerald-600' : ''
                    }`}
                  />
                  <span className="hidden sm:inline">重新生成</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowMinutesCard((prev) => !prev)}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium cursor-pointer"
                >
                  {showMinutesCard ? (
                    <>
                      <ChevronUp className="w-3.5 h-3.5" />
                      <span>收合</span>
                    </>
                  ) : (
                    <>
                      <ChevronDown className="w-3.5 h-3.5" />
                      <span>展開報告</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {showMinutesCard && (
              <div className="space-y-4 text-xs sm:text-sm text-slate-800">
                {/* 1. 執行摘要 */}
                <div className="bg-white/90 rounded-xl p-3 border border-slate-200/80 space-y-1">
                  <div className="font-bold text-xs text-indigo-900 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                    <span>一、會議執行摘要 (Executive Summary)</span>
                  </div>
                  <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
                    {meetingMinutes.executiveSummary}
                  </p>
                </div>

                {/* 2. 重要結論與裁示事項 */}
                <div className="bg-white/90 rounded-xl p-3 border border-slate-200/80 space-y-1.5">
                  <div className="font-bold text-xs text-emerald-900 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    <span>二、重要結論與裁示事項</span>
                  </div>
                  <ul className="space-y-1.5 pl-1">
                    {meetingMinutes.keyConclusions.map((item, idx) => (
                      <li
                        key={idx}
                        className="text-xs sm:text-sm text-slate-700 leading-relaxed flex items-start gap-2"
                      >
                        <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-emerald-100 text-emerald-800 font-bold text-[10px] shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* 3. 各議題討論重點與決議 */}
                {meetingMinutes.topicBreakdowns.length > 0 && (
                  <div className="bg-white/90 rounded-xl p-3 border border-slate-200/80 space-y-2">
                    <div className="font-bold text-xs text-slate-900 flex items-center gap-1.5">
                      <FileText className="w-3.5 h-3.5 text-blue-600" />
                      <span>三、各議題討論重點與決議</span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                            <th className="py-1.5 px-2.5 font-semibold w-1/4">
                              議題 / 專案名稱
                            </th>
                            <th className="py-1.5 px-2.5 font-semibold w-5/12">
                              討論現況與重點
                            </th>
                            <th className="py-1.5 px-2.5 font-semibold w-1/3">
                              決議與裁示
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {meetingMinutes.topicBreakdowns.map((t, idx) => (
                            <tr key={idx} className="hover:bg-slate-50/80">
                              <td className="py-2 px-2.5 align-top font-semibold text-slate-800">
                                <div>{t.topic}</div>
                                {t.timeRange && (
                                  <span className="inline-block mt-0.5 text-[10px] font-mono text-slate-400">
                                    [{t.timeRange}]
                                  </span>
                                )}
                              </td>
                              <td className="py-2 px-2.5 align-top text-slate-600 leading-relaxed">
                                {t.discussionPoints}
                              </td>
                              <td className="py-2 px-2.5 align-top font-medium text-emerald-800 leading-relaxed">
                                {t.decision}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* 4. 後續行動事項追蹤表 */}
                {meetingMinutes.actionItems.length > 0 && (
                  <div className="bg-white/90 rounded-xl p-3 border border-slate-200/80 space-y-2">
                    <div className="font-bold text-xs text-amber-900 flex items-center gap-1.5">
                      <ListChecks className="w-3.5 h-3.5 text-amber-600" />
                      <span>四、後續行動事項追蹤表 (Action Items)</span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 bg-amber-50/60 text-slate-700">
                            <th className="py-1.5 px-2 font-semibold w-10">
                              #
                            </th>
                            <th className="py-1.5 px-2.5 font-semibold">
                              具體待辦與行動事項
                            </th>
                            <th className="py-1.5 px-2.5 font-semibold w-36">
                              負責語者 / 單位
                            </th>
                            <th className="py-1.5 px-2.5 font-semibold w-28">
                              目標時程
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {meetingMinutes.actionItems.map((act, idx) => (
                            <tr key={act.id} className="hover:bg-amber-50/20">
                              <td className="py-2 px-2 align-top font-mono text-slate-400">
                                {idx + 1}
                              </td>
                              <td className="py-2 px-2.5 align-top font-medium text-slate-800 leading-relaxed">
                                {act.task}
                              </td>
                              <td className="py-2 px-2.5 align-top text-blue-700 font-semibold">
                                {act.owner}
                              </td>
                              <td className="py-2 px-2.5 align-top text-amber-800 font-medium">
                                {act.deadline}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {segments.map((seg) => {
          const sp = speakers[seg.speakerId];
          const timeStr = `${Math.floor(seg.start / 60)}:${String(
            Math.floor(seg.start % 60)
          ).padStart(2, '0')}`;

          return (
            <div
              key={seg.id}
              className="flex flex-col sm:flex-row gap-1.5 sm:gap-4 items-start p-2.5 sm:p-0 rounded-xl sm:rounded-none border sm:border-0 border-slate-100 bg-slate-50/40 sm:bg-transparent"
            >
              {/* 語者標籤與時間戳：手機端單行橫向排列，平板/電腦端左側固定寬度欄 */}
              <div className="w-full sm:w-32 md:w-36 shrink-0 sm:pt-0.5 flex sm:block items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 sm:block">
                  <span
                    className="inline-block text-xs font-bold px-2 py-0.5 rounded text-white shadow-2xs"
                    style={{ backgroundColor: sp?.color || '#475569' }}
                  >
                    {sp?.name || seg.speakerId}
                  </span>
                  <div className="text-[11px] text-slate-400 sm:mt-0.5 font-mono">
                    {timeStr}
                  </div>
                </div>
                {renderSentenceIntegrityBadge(seg)}
              </div>

              {/* 逐字稿文字區：手機端佔滿 100% 寬度且字體微幅放大 (15px) 利於閱讀與點擊 */}
              <div className="w-full sm:flex-1 text-[15px] sm:text-sm leading-relaxed text-slate-800 break-words">
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
                      onPlaySnippet={(start) => seekToWord(start)}
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
