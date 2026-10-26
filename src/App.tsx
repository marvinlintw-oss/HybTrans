// src/App.tsx
import React, { useState } from 'react';
import {
  BookOpen,
  FileText,
  PanelLeftClose,
  PanelLeftOpen,
  Users,
} from 'lucide-react';
import { EngineAndKeyBar } from './components/header/EngineAndKeyBar';
import { KnowledgeDictPanel } from './components/sidebar/KnowledgeDictPanel';
import { SpeakerSyncPanel } from './components/sidebar/SpeakerSyncPanel';
import { MacroChunkProgress } from './components/editor/MacroChunkProgress';
import { TranscriptEditor } from './components/editor/TranscriptEditor';
import { useAppStore } from './store/useAppStore';

type MobileWorkspaceTab = 'transcript' | 'speakers' | 'dictionary';

export const App: React.FC = () => {
  const [mobileTab, setMobileTab] = useState<MobileWorkspaceTab>('transcript');
  const [desktopSidebarCollapsed, setDesktopSidebarCollapsed] =
    useState<boolean>(false);

  const segmentsCount = useAppStore((s) => s.segments.length);
  const speakersCount = useAppStore((s) => Object.keys(s.speakers).length);
  const dictCount = useAppStore((s) => s.dictionary.length);
  const phoneticCount = useAppStore((s) => s.phoneticCandidates.length);
  const isRunning = useAppStore((s) => s.isRunning);

  return (
    <div className="flex flex-col h-[100dvh] w-full overflow-hidden bg-slate-50">
      {/* 頂部導覽與引擎設定列 (RWD 自適應) */}
      <EngineAndKeyBar />

      {/* 巨集切塊轉寫進度與全場滾動摘要列 */}
      <MacroChunkProgress />

      {/* 主工作區：桌機 (lg+) 為左右雙欄可收合；手機與平板 (<lg) 為全寬分頁並保持 DOM 常駐不中斷音訊播放 */}
      <div className="flex flex-1 min-h-0 overflow-hidden relative">
        {/* 左側設定欄：桌機常駐顯示 (可折疊)，手機/平板於點選「語者」或「字典」分頁時全寬顯示 */}
        <aside
          className={`bg-slate-50/70 border-r border-slate-200 overflow-y-auto shrink-0 transition-all duration-200 ${
            desktopSidebarCollapsed ? 'lg:hidden' : 'lg:block lg:w-80 xl:w-[340px]'
          } ${
            mobileTab === 'transcript'
              ? 'hidden'
              : 'block w-full h-full p-3 sm:p-4 space-y-3'
          } lg:p-2.5 lg:space-y-3`}
        >
          {/* 手機/平板模式下，可同時選擇單獨檢視對應面板或上下滑動檢視 */}
          <div
            className={
              mobileTab === 'speakers'
                ? 'block'
                : mobileTab === 'dictionary'
                ? 'hidden lg:block'
                : 'block'
            }
          >
            <SpeakerSyncPanel />
          </div>

          <div
            className={
              mobileTab === 'dictionary'
                ? 'block'
                : mobileTab === 'speakers'
                ? 'hidden lg:block'
                : 'block'
            }
          >
            <KnowledgeDictPanel />
          </div>
        </aside>

        {/* 桌機專用：側邊欄快速收合/展開懸浮小按鈕 */}
        <button
          type="button"
          onClick={() => setDesktopSidebarCollapsed((prev) => !prev)}
          title={
            desktopSidebarCollapsed
              ? '展開左側語者與字典面板'
              : '收合左側面板以最大化逐字稿空間'
          }
          className="hidden lg:inline-flex items-center justify-center w-5 h-12 bg-white hover:bg-slate-100 border border-l-0 border-slate-200 rounded-r-md shadow-2xs text-slate-400 hover:text-slate-700 absolute left-0 top-1/2 -translate-y-1/2 z-20 cursor-pointer transition-all"
          style={{
            left: desktopSidebarCollapsed ? '0px' : undefined,
          }}
        >
          {desktopSidebarCollapsed ? (
            <PanelLeftOpen className="w-3.5 h-3.5" />
          ) : (
            <PanelLeftClose className="w-3.5 h-3.5" />
          )}
        </button>

        {/* 右側主編輯區：保持常駐掛載 (避免切換手機分頁時 <audio> 播放中斷) */}
        <main
          className={`flex-1 min-w-0 h-full overflow-hidden flex flex-col ${
            mobileTab === 'transcript' ? 'flex' : 'hidden lg:flex'
          }`}
        >
          <TranscriptEditor
            onOpenMobileSettings={(tab) => setMobileTab(tab)}
          />
        </main>
      </div>

      {/* 手機與平板專屬底部導覽列 (< 1024px 顯示，支援 iOS/Android 安全區域) */}
      <nav
        aria-label="行動裝置工作區切換"
        className="lg:hidden bg-white border-t border-slate-200 px-2 py-1 safe-area-pb shrink-0 z-30 shadow-[0_-2px_10px_rgba(15,23,42,0.04)]"
      >
        <div className="grid grid-cols-3 gap-1 max-w-md mx-auto">
          <button
            type="button"
            onClick={() => setMobileTab('transcript')}
            className={`flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              mobileTab === 'transcript'
                ? 'bg-blue-50 text-blue-700'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            <FileText className="w-4 h-4 shrink-0" />
            <span>逐字稿</span>
            {segmentsCount > 0 ? (
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded-full ${
                  mobileTab === 'transcript'
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-200 text-slate-700'
                }`}
              >
                {segmentsCount}
              </span>
            ) : isRunning ? (
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
            ) : null}
          </button>

          <button
            type="button"
            onClick={() => setMobileTab('speakers')}
            className={`relative flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              mobileTab === 'speakers'
                ? 'bg-blue-50 text-blue-700'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            <Users className="w-4 h-4 shrink-0" />
            <span>語者與情境</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded-full ${
                mobileTab === 'speakers'
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-200 text-slate-700'
              }`}
            >
              {speakersCount}
            </span>
            {phoneticCount > 0 && (
              <span
                title={`發現 ${phoneticCount} 處同音稱謂可修正`}
                className="w-2 h-2 rounded-full bg-amber-500 absolute top-1.5 right-2"
              />
            )}
          </button>

          <button
            type="button"
            onClick={() => setMobileTab('dictionary')}
            className={`flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              mobileTab === 'dictionary'
                ? 'bg-blue-50 text-blue-700'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            <BookOpen className="w-4 h-4 shrink-0" />
            <span>字典與簡報</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded-full ${
                mobileTab === 'dictionary'
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-200 text-slate-700'
              }`}
            >
              {dictCount}
            </span>
          </button>
        </div>
      </nav>
    </div>
  );
};

export default App;