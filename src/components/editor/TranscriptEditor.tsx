// src/components/editor/TranscriptEditor.tsx
import React, { useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { usePipelineScheduler } from '../../hooks/usePipelineScheduler';
import { useAudioPlayerSync } from '../../hooks/useAudioPlayerSync';
import { WordTokenNode } from './WordTokenNode';
import {
  exportAsDocx,
  exportAsMarkdown,
  exportAsSrt,
  exportProjectJson
} from '../../utils/exporter';

export const TranscriptEditor: React.FC = () => {
  const { segments, speakers, currentTime, clearTranscript } = useAppStore();
  const { isRunning, startPipeline } = usePipelineScheduler();
  const { audioRef, seekToWord, handleTimeUpdate } = useAudioPlayerSync();
  const [audioUrl, setAudioUrl] = useState<string>('');

  const handleAudioSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudioUrl(URL.createObjectURL(file));
    startPipeline(file);
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-6 py-3 bg-white border-b border-slate-200">
        <div className="flex items-center gap-3">
          <label className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2 rounded cursor-pointer">
            {isRunning ? '微批次流水線轉寫中...' : '載入會議錄音檔 (MP3/WAV/M4A)'}
            <input
              type="file"
              accept="audio/*"
              onChange={handleAudioSelect}
              disabled={isRunning}
              className="hidden"
            />
          </label>
          {audioUrl && (
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              onTimeUpdate={handleTimeUpdate}
              className="h-8"
            />
          )}
        </div>

        <div className="flex gap-2 text-xs">
          <button
            onClick={() => exportAsDocx(segments, speakers)}
            disabled={!segments.length}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-semibold px-3 py-1.5 rounded"
          >
            📄 匯出 Word (.docx)
          </button>
          <button
            onClick={() => exportAsMarkdown(segments, speakers)}
            className="border px-3 py-1.5 rounded hover:bg-slate-50"
          >
            匯出 Markdown
          </button>
          <button
            onClick={() => exportAsSrt(segments, speakers)}
            className="border px-3 py-1.5 rounded hover:bg-slate-50"
          >
            匯出 SRT
          </button>
          <button
            onClick={() => exportProjectJson(segments, speakers)}
            className="border px-3 py-1.5 rounded hover:bg-slate-50"
          >
            專案 JSON
          </button>
          {segments.length > 0 && (
            <button
              onClick={clearTranscript}
              className="border border-rose-300 text-rose-600 hover:bg-rose-50 px-2.5 py-1.5 rounded"
            >
              清空紀錄
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-white">
        {segments.map((seg) => {
          const sp = speakers[seg.speakerId];
          return (
            <div key={seg.id} className="flex gap-4 items-start">
              <div className="w-28 shrink-0 pt-0.5">
                <span
                  className="inline-block text-xs font-bold px-2 py-0.5 rounded text-white"
                  style={{ backgroundColor: sp?.color || '#475569' }}
                >
                  {sp?.name || seg.speakerId}
                </span>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  {Math.floor(seg.start / 60)}:{String(Math.floor(seg.start % 60)).padStart(2, '0')}
                </div>
              </div>

              <div className="flex-1 text-sm leading-relaxed text-slate-800">
                {seg.words.map((w, idx) => {
                  const isWordActive = currentTime >= w.start && currentTime <= w.end;
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