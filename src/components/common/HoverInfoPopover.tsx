// src/components/common/HoverInfoPopover.tsx
import React from 'react';
import { HelpCircle, Info } from 'lucide-react';

export interface HoverInfoPopoverProps {
  label?: string;
  title: string;
  children: React.ReactNode;
  align?: 'left' | 'right';
}

/**
 * 通用浮出式說明視窗元件 (Hover / Focus Popover)
 * 供 SpeakerSyncPanel、TranscriptEditor 等元件顯示非技術人員友善的「適用情境 / 不適用情境 / 進階原理」說明
 */
export const HoverInfoPopover: React.FC<HoverInfoPopoverProps> = ({
  label,
  title,
  children,
  align = 'right',
}) => {
  return (
    <div className="relative inline-flex items-center group">
      <button
        type="button"
        className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-600 transition-colors cursor-help focus:outline-none"
      >
        <HelpCircle className="w-3.5 h-3.5 shrink-0" />
        {label && (
          <span className="underline decoration-dotted underline-offset-2">
            {label}
          </span>
        )}
      </button>

      {/* 浮出式說明卡片 */}
      <div
        className={`absolute ${
          align === 'right' ? 'right-0' : 'left-0'
        } top-full mt-1.5 z-50 w-72 p-3 rounded-xl bg-slate-900/95 text-slate-100 text-[11px] leading-relaxed shadow-xl border border-slate-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible group-focus-within:opacity-100 group-focus-within:visible transition-all duration-150 pointer-events-none group-hover:pointer-events-auto`}
      >
        <div className="font-semibold text-white mb-1 flex items-center gap-1.5 border-b border-slate-700/80 pb-1">
          <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span>{title}</span>
        </div>
        <div className="text-slate-300 space-y-1">{children}</div>
      </div>
    </div>
  );
};

export default HoverInfoPopover;