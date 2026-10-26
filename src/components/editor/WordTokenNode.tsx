// src/components/editor/WordTokenNode.tsx
import React, { useEffect, useRef, useState } from 'react';
import { RotateCcw, Volume2, X } from 'lucide-react';
import { WordSpan } from '../../types';
import { useAppStore } from '../../store/useAppStore';

interface Props {
  segmentId: string;
  wordIndex: number;
  word: WordSpan;
  isActive: boolean;
  onPlaySnippet: (start: number, end: number) => void;
}

export const WordTokenNode: React.FC<Props> = ({
  segmentId,
  wordIndex,
  word,
  isActive,
  onPlaySnippet,
}) => {
  const [openMenu, setOpenMenu] = useState(false);
  const nodeRef = useRef<HTMLSpanElement | null>(null);
  const revertWord = useAppStore((s) => s.revertWord);

  const hasInteractiveDetails =
    word.status === 'auto_corrected' ||
    word.status === 'semantic_uncertain' ||
    Boolean(word.candidates && word.candidates.length > 0);

  useEffect(() => {
    if (!openMenu) return;
    const handleOutside = (e: MouseEvent | TouchEvent) => {
      if (nodeRef.current && !nodeRef.current.contains(e.target as Node)) {
        setOpenMenu(false);
      }
    };
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('touchstart', handleOutside);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('touchstart', handleOutside);
    };
  }, [openMenu]);

  const getStyleClass = () => {
    const activeRing = isActive ? 'ring-2 ring-blue-500 rounded px-0.5' : '';
    switch (word.status) {
      case 'auto_corrected':
        return `bg-emerald-100 text-emerald-900 border-b border-emerald-400 rounded-xs px-0.5 cursor-pointer ${activeRing}`;
      case 'acoustic_uncertain':
        return `underline decoration-wavy decoration-orange-500 text-orange-950 cursor-pointer ${activeRing}`;
      case 'semantic_uncertain':
        return `bg-amber-200 text-amber-950 font-medium px-0.5 rounded-xs cursor-pointer ${activeRing}`;
      default:
        return `hover:bg-slate-100 active:bg-slate-200 rounded-xs cursor-pointer ${activeRing}`;
    }
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onPlaySnippet(word.start, word.end + 0.3);
    if (hasInteractiveDetails) {
      setOpenMenu((prev) => !prev);
    }
  };

  const canRevertToOriginal =
    Boolean(word.originalText) && word.originalText !== word.text;

  return (
    <span ref={nodeRef} className="relative inline-block">
      <span
        onClick={handleClick}
        title={
          word.status === 'auto_corrected'
            ? `原始聽寫：「${word.originalText || word.text}」｜原因：${
                word.reason || '語境修正'
              }（點擊聽原音並檢視選項）`
            : `信心度：${Math.round(word.confidence * 100)}%（點擊播放原音）`
        }
        className={getStyleClass()}
      >
        {word.text}
      </span>

      {openMenu && hasInteractiveDetails && (
        <div className="max-sm:fixed max-sm:left-3 max-sm:right-3 max-sm:bottom-16 sm:absolute z-40 sm:left-0 sm:top-full sm:mt-1 sm:w-64 bg-white border border-slate-200 rounded-xl shadow-xl p-2.5 text-xs space-y-2">
          <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
            <span className="font-semibold text-slate-800 flex items-center gap-1">
              <Volume2 className="w-3.5 h-3.5 text-blue-600 shrink-0" />
              <span>詞彙校對與原音檢視</span>
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setOpenMenu(false);
              }}
              className="text-slate-400 hover:text-slate-600 p-0.5 cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {word.originalText && (
            <div className="text-[11px] text-slate-600 bg-slate-50 px-2 py-1 rounded border border-slate-100">
              原始聽寫：<strong className="text-slate-800">{word.originalText}</strong>
            </div>
          )}

          {word.reason && (
            <div className="text-[11px] text-slate-500 leading-relaxed">
              診斷說明：{word.reason}
            </div>
          )}

          {word.candidates && word.candidates.length > 0 && (
            <div className="space-y-1">
              <div className="text-[10px] font-medium text-slate-400">
                點選替換同音候選詞：
              </div>
              <div className="flex flex-wrap gap-1">
                {word.candidates.map((cand) => (
                  <button
                    key={cand}
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      revertWord(segmentId, wordIndex, cand);
                      setOpenMenu(false);
                    }}
                    className={`px-2 py-1 rounded border text-xs font-medium transition-colors cursor-pointer ${
                      cand === word.text
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-amber-50/70 hover:bg-amber-100 text-slate-800 border-amber-200'
                    }`}
                  >
                    {cand}
                  </button>
                ))}
              </div>
            </div>
          )}

          {canRevertToOriginal && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                revertWord(segmentId, wordIndex);
                setOpenMenu(false);
              }}
              className="w-full flex items-center justify-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium text-[11px] transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              還原為原始聽寫「{word.originalText}」
            </button>
          )}
        </div>
      )}
    </span>
  );
};