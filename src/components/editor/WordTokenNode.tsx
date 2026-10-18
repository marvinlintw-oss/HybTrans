import React, { useState } from 'react';
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
  const revertWord = useAppStore((s) => s.revertWord);

  const getStyleClass = () => {
    const activeRing = isActive ? 'ring-2 ring-blue-500 rounded px-0.5' : '';
    switch (word.status) {
      case 'auto_corrected':
        return `bg-emerald-100 text-emerald-900 border-b border-emerald-400 cursor-pointer ${activeRing}`;
      case 'acoustic_uncertain':
        return `underline decoration-wavy decoration-orange-500 text-orange-950 cursor-pointer ${activeRing}`;
      case 'semantic_uncertain':
        return `bg-amber-200 text-amber-950 font-medium px-0.5 rounded cursor-pointer ${activeRing}`;
      default:
        return `hover:bg-slate-100 cursor-pointer ${activeRing}`;
    }
  };

  const handleClick = () => {
    onPlaySnippet(word.start, word.end + 0.3);
    if (word.status === 'auto_corrected') {
      revertWord(segmentId, wordIndex); // 點擊一鍵還原原始聽寫字
    } else if (word.status === 'semantic_uncertain') {
      setOpenMenu((prev) => !prev);
    }
  };

  return (
    <span className="relative inline-block group">
      <span
        onClick={handleClick}
        title={
          word.status === 'auto_corrected'
            ? `原始聽寫：「${word.originalText}」｜原因：${word.reason || '語境修正'}（點擊還原）`
            : `信心度：${Math.round(word.confidence * 100)}%`
        }
        className={getStyleClass()}
      >
        {word.text}
      </span>

      {openMenu && word.candidates && word.candidates.length > 0 && (
        <div className="absolute z-30 left-0 top-full mt-1 w-44 bg-white border border-slate-200 rounded-md shadow-lg p-1 text-xs">
          <div className="px-2 py-1 text-slate-500 border-b">同音候選詞（點擊套用）</div>
          {word.candidates.map((cand) => (
            <button
              key={cand}
              onClick={() => {
                revertWord(segmentId, wordIndex, cand);
                setOpenMenu(false);
              }}
              className="block w-full text-left px-2 py-1.5 hover:bg-amber-50 rounded text-slate-800 font-medium"
            >
              {cand}
            </button>
          ))}
        </div>
      )}
    </span>
  );
};