// src/App.tsx
import React from 'react';
import { EngineAndKeyBar } from './components/header/EngineAndKeyBar';
import { KnowledgeDictPanel } from './components/sidebar/KnowledgeDictPanel';
import { SpeakerSyncPanel } from './components/sidebar/SpeakerSyncPanel';
import { MacroChunkProgress } from './components/editor/MacroChunkProgress';
import { TranscriptEditor } from './components/editor/TranscriptEditor';

export const App: React.FC = () => {
  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden">
      <EngineAndKeyBar />
      <MacroChunkProgress />
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 border-r border-slate-200 bg-white overflow-y-auto shrink-0">
          <SpeakerSyncPanel />
          <KnowledgeDictPanel />
        </aside>
        <main className="flex-1 overflow-hidden">
          <TranscriptEditor />
        </main>
      </div>
    </div>
  );
};
export default App;