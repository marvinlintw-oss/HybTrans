// src/components/editor/MacroChunkProgress.tsx
import React, { useState } from 'react';
import {
  Loader2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  Layers,
  Sparkles,
  Square,
  FileText,
  ShieldCheck,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { usePipelineScheduler } from '../../hooks/usePipelineScheduler';
import { ChunkStage } from '../../types';

function formatMinSec(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

const STAGE_LABEL_MAP: Record<ChunkStage, string> = {
  idle: '等待處理',
  preprocessing: '音訊切片與降噪中',
  diarizing: '語者分離中',
  DIARIZING: '語者分離中',
  transcribing: '語音轉文字中',
  TRANSCRIBING_ASR: '語音轉文字中',
  correcting: '先句後詞兩階段語境修正中',
  refining: '先句後詞兩階段語境修正中',
  LLM_REFINING: '先句後詞兩階段語境修正中',
  relistening: '模糊段落重聽校準中',
  completed: '已完成',
  COMPLETED: '已完成',
  error: '發生錯誤',
};

export const MacroChunkProgress: React.FC = () => {
  const { chunks, config, modelDownload, rollingMeetingSummary } =
    useAppStore();
  const { isProcessing, stopPipeline } = usePipelineScheduler();
  const [expandSummary, setExpandSummary] = useState<boolean>(false);

  const hasChunks = chunks.length > 0;
  const isDownloading = modelDownload?.active === true;

  if (!hasChunks && !isDownloading) {
    return null;
  }

  const totalChunks = chunks.length;
  const completedCount = chunks.filter(
    (c) => c.stage === 'completed' || c.stage === 'COMPLETED'
  ).length;
  const activeChunk =
    chunks.find(
      (c) =>
        c.stage !== 'idle' &&
        c.stage !== 'completed' &&
        c.stage !== 'COMPLETED'
    ) ||
    (completedCount === totalChunks && totalChunks > 0
      ? chunks[totalChunks - 1]
      : chunks[0]);

  const isAllCompleted = totalChunks > 0 && completedCount === totalChunks;
  const chunkMin = Math.round((config.chunkDurationSec || 300) / 60);

  const overallProgress =
    totalChunks > 0
      ? Math.min(
          100,
          Math.round(
            ((completedCount * 100 +
              (activeChunk &&
              activeChunk.stage !== 'completed' &&
              activeChunk.stage !== 'COMPLETED'
                ? activeChunk.progress
                : 0)) /
              (totalChunks * 100)) *
              100
          )
        )
      : 0;

  return (
    <div className="bg-white border-b border-slate-200 px-4 py-1.5 shadow-xs space-y-1">
      <div className="w-full flex flex-wrap items-center justify-between gap-2 text-xs">
        {/* 左側：切塊資訊與當前進度 */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-medium">
            <Layers className="w-3.5 h-3.5 text-blue-600" />
            共 <strong>{totalChunks}</strong> 塊（每塊 {chunkMin} 分鐘）
          </span>

          {isProcessing && (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[11px]">
              <ShieldCheck className="w-3.5 h-3.5" />
              防休眠保護已啟用
            </span>
          )}

          {activeChunk && (
            <div className="flex items-center gap-2 text-slate-700">
              {isAllCompleted ? (
                <span className="inline-flex items-center gap-1 text-emerald-600 font-semibold">
                  <CheckCircle2 className="w-4 h-4" />
                  全部處理完成！（可直接檢視與點擊下方逐字稿微調）
                </span>
              ) : (
                <>
                  {isProcessing && (
                    <Loader2 className="w-3.5 h-3.5 text-blue-600 animate-spin shrink-0" />
                  )}
                  <span>
                    第{' '}
                    <strong className="text-blue-700">
                      {activeChunk.index + 1}/{totalChunks}
                    </strong>{' '}
                    塊
                    <span className="text-slate-400 ml-1">
                      ({formatMinSec(activeChunk.startTime)}~
                      {formatMinSec(activeChunk.endTime)})
                    </span>
                  </span>
                  <span className="text-slate-300">|</span>
                  <span className="font-medium text-indigo-700 flex items-center gap-1">
                    <Sparkles className="w-3.5 h-3.5" />
                    {activeChunk.detailText ||
                      STAGE_LABEL_MAP[activeChunk.stage]}{' '}
                    ({activeChunk.progress}%)
                  </span>
                </>
              )}
            </div>
          )}
        </div>

        {/* 右側：總進度條與「停止轉檔」（避免與下方工具列的 AI 全文順稿按鈕重複） */}
        <div className="flex items-center gap-2">
          {totalChunks > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-slate-500 shrink-0">
                {overallProgress}%
              </span>
              <div className="w-24 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    isAllCompleted ? 'bg-emerald-500' : 'bg-blue-600'
                  }`}
                  style={{ width: `${overallProgress}%` }}
                />
              </div>
            </div>
          )}

          {isProcessing && (
            <button
              type="button"
              onClick={stopPipeline}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-red-600 hover:bg-red-700 text-white font-medium text-xs shadow-sm transition-colors cursor-pointer"
            >
              <Square className="w-3 h-3 fill-current" />
              停止
            </button>
          )}
        </div>
      </div>

      {/* 背景下載 / 載入 AI 模型即時進度列 */}
      {isDownloading && (
        <div className="px-3 py-1 bg-blue-50/90 border border-blue-200 rounded flex items-center justify-between gap-3 text-xs text-blue-900">
          <div className="flex items-center gap-1.5 min-w-0">
            <Download className="w-3.5 h-3.5 text-blue-600 animate-bounce shrink-0" />
            <span className="font-medium truncate">
              準備模型 ({modelDownload.modelName})：{modelDownload.fileName}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="font-semibold text-blue-700">
              {modelDownload.progress}%
            </span>
            <div className="w-24 h-1.5 bg-blue-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all duration-200"
                style={{ width: `${modelDownload.progress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* 動態滾動會議摘要（預設單行緊湊顯示，可點擊展開） */}
      {rollingMeetingSummary && (
        <div
          onClick={() => setExpandSummary((prev) => !prev)}
          className="px-2.5 py-1 bg-indigo-50/50 border border-indigo-200/70 rounded flex items-start justify-between gap-2 text-xs text-slate-700 cursor-pointer hover:bg-indigo-50/80 transition-colors"
        >
          <div className="flex items-start gap-1.5 min-w-0 flex-1">
            <FileText className="w-3.5 h-3.5 text-indigo-600 shrink-0 mt-0.5" />
            <div
              className={`text-[11px] text-slate-600 leading-relaxed flex-1 ${
                expandSummary
                  ? 'whitespace-pre-line'
                  : 'truncate whitespace-nowrap overflow-hidden'
              }`}
            >
              {expandSummary
                ? rollingMeetingSummary
                : rollingMeetingSummary.replace(/\n+/g, ' ｜ ')}
            </div>
          </div>
          <span className="text-[10px] text-indigo-600 shrink-0 inline-flex items-center gap-0.5 mt-0.5">
            {expandSummary ? (
              <>
                收合 <ChevronUp className="w-3 h-3" />
              </>
            ) : (
              <>
                展開摘要 <ChevronDown className="w-3 h-3" />
              </>
            )}
          </span>
        </div>
      )}
    </div>
  );
};