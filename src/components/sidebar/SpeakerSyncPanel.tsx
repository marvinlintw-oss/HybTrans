// src/components/sidebar/SpeakerSyncPanel.tsx
import React, { useState } from 'react';
import { Users, Merge, CheckCircle2, Sliders, Sparkles, Info } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import * as phoneticModule from '../../utils/phoneticMatcher';

export const SpeakerSyncPanel: React.FC = () => {
  const {
    config,
    updateConfig,
    speakers,
    segments,
    renameSpeaker,
    mergeSpeakers,
    phoneticCandidates,
    setPhoneticCandidates,
    applyPhoneticReplacements,
  } = useAppStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [nameInput, setNameInput] = useState('');
  const [mergeSourceId, setMergeSourceId] = useState<string>('');
  const [mergeTargetId, setMergeTargetId] = useState<string>('');

  const speakerList = Object.values(speakers);
  const hasMultipleSpeakers = speakerList.length >= 2;
  const hasPhoneticCandidates = phoneticCandidates.length > 0;
  const isLockedFiveToSix = config.minSpeakers === 5 && config.maxSpeakers === 6;

  const handleStartEdit = (id: string, currentName: string) => {
    setEditingId(id);
    setNameInput(currentName);
  };

  const handleConfirmRename = (id: string) => {
    const trimmed = nameInput.trim();
    if (trimmed) {
      renameSpeaker(id, trimmed);

      try {
        const matcherFn = (phoneticModule as Record<string, unknown>)
          .scanHomophoneSpeakerMentions as (...args: unknown[]) => unknown;

        if (typeof matcherFn === 'function') {
          const fullText = segments
            .map((seg) => seg.words.map((w) => w.text).join(''))
            .join('\n');

          let result: unknown;
          try {
            result = matcherFn(fullText, trimmed, segments);
          } catch {
            result = matcherFn(segments, trimmed);
          }

          if (Array.isArray(result)) {
            setPhoneticCandidates(result);
          }
        }
      } catch (err) {
        console.warn('掃描同音稱謂略過:', err);
      }
    }
    setEditingId(null);
  };

  const handleMinChange = (val: number) => {
    const safeMin = Math.max(1, Math.min(10, val));
    const safeMax = Math.max(safeMin, config.maxSpeakers);
    updateConfig({ minSpeakers: safeMin, maxSpeakers: safeMax });
  };

  const handleMaxChange = (val: number) => {
    const safeMax = Math.max(1, Math.min(10, val));
    const safeMin = Math.min(config.minSpeakers, safeMax);
    updateConfig({ minSpeakers: safeMin, maxSpeakers: safeMax });
  };

  const handleMergeConfirm = () => {
    if (mergeSourceId && mergeTargetId && mergeSourceId !== mergeTargetId) {
      mergeSpeakers(mergeSourceId, mergeTargetId);
      setMergeSourceId('');
      setMergeTargetId('');
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-sm space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <div className="flex items-center gap-1.5">
          <Users className="w-4 h-4 text-blue-600" />
          <h3 className="font-semibold text-slate-800 text-sm">與會語者管理</h3>
          <span
            title="系統會自動依男女音高與音色區分發言者，並控制在您設定的人數上限內"
            className="text-slate-400 hover:text-slate-600 cursor-help"
          >
            <Info className="w-3.5 h-3.5" />
          </span>
        </div>
        <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full font-medium">
          {speakerList.length} 人
        </span>
      </div>

      {/* 會議人數設定 */}
      <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-200/80 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-700 flex items-center gap-1">
            <Sliders className="w-3.5 h-3.5 text-slate-500" />
            會議人數範圍
          </span>
          <button
            type="button"
            onClick={() => updateConfig({ minSpeakers: 5, maxSpeakers: 6 })}
            className={`text-[11px] px-2 py-0.5 rounded border transition-colors ${
              isLockedFiveToSix
                ? 'bg-blue-600 text-white border-blue-600 font-medium'
                : 'bg-white text-blue-600 border-blue-200 hover:bg-blue-50'
            }`}
          >
            5~6 人會議
          </button>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="bg-white px-2 py-1.5 rounded border border-slate-200 flex items-center justify-between">
            <span className="text-xs text-slate-500">最少</span>
            <input
              type="number"
              min={1}
              max={10}
              value={config.minSpeakers}
              onChange={(e) => handleMinChange(Number(e.target.value))}
              className="w-10 text-right font-semibold text-xs text-slate-800 focus:outline-none border-b border-slate-300"
            />
          </div>
          <div className="bg-white px-2 py-1.5 rounded border border-slate-200 flex items-center justify-between">
            <span className="text-xs text-slate-500">最多</span>
            <input
              type="number"
              min={1}
              max={10}
              value={config.maxSpeakers}
              onChange={(e) => handleMaxChange(Number(e.target.value))}
              className="w-10 text-right font-semibold text-xs text-blue-700 focus:outline-none border-b border-blue-300"
            />
          </div>
        </div>
      </div>

      {/* 語者清單 */}
      <div className="space-y-1.5 max-h-52 overflow-y-auto pr-1">
        {speakerList.length === 0 ? (
          <p className="text-xs text-slate-400 text-center py-2.5">
            上傳音檔後將自動識別發言者
          </p>
        ) : (
          speakerList.map((sp) => {
            const isEditing = editingId === sp.id;
            const f0Val = sp.centroid?.[0] ? Math.round(sp.centroid[0]) : null;
            return (
              <div
                key={sp.id}
                title={f0Val ? `聲紋特徵基準：約 ${f0Val}Hz` : undefined}
                className="flex items-center justify-between p-2 rounded-lg border border-slate-100 hover:border-slate-200 bg-slate-50/50"
              >
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: sp.color }}
                  />
                  {isEditing ? (
                    <input
                      type="text"
                      value={nameInput}
                      onChange={(e) => setNameInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleConfirmRename(sp.id)}
                      placeholder="輸入姓名或職稱"
                      className="text-xs px-2 py-0.5 border border-blue-400 rounded w-full focus:outline-none"
                      autoFocus
                    />
                  ) : (
                    <span className="text-xs font-medium text-slate-800 truncate">
                      {sp.name}
                    </span>
                  )}
                </div>

                {isEditing ? (
                  <button
                    type="button"
                    onClick={() => handleConfirmRename(sp.id)}
                    className="text-xs text-blue-600 font-medium px-2 py-0.5 hover:bg-blue-50 rounded"
                  >
                    確定
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleStartEdit(sp.id, sp.name)}
                    className="text-[11px] text-slate-500 hover:text-blue-600 px-1.5 py-0.5"
                  >
                    更名
                  </button>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* 語者手動合併工具 */}
      {hasMultipleSpeakers && (
        <div className="pt-2 border-t border-slate-100 space-y-1.5">
          <div className="flex items-center gap-1 text-xs font-medium text-slate-600">
            <Merge className="w-3.5 h-3.5" />
            <span>合併同一人之語者標籤</span>
          </div>
          <div className="flex items-center gap-1.5">
            <select
              value={mergeSourceId}
              onChange={(e) => setMergeSourceId(e.target.value)}
              className="text-xs border border-slate-200 rounded px-1.5 py-1 flex-1 bg-white truncate"
            >
              <option value="">選擇來源...</option>
              {speakerList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <span className="text-xs text-slate-400">➔</span>
            <select
              value={mergeTargetId}
              onChange={(e) => setMergeTargetId(e.target.value)}
              className="text-xs border border-slate-200 rounded px-1.5 py-1 flex-1 bg-white truncate"
            >
              <option value="">併入目標...</option>
              {speakerList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleMergeConfirm}
              disabled={!mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId}
              className="text-xs bg-slate-800 text-white px-2 py-1 rounded disabled:opacity-40 shrink-0"
            >
              合併
            </button>
          </div>
        </div>
      )}

      {/* 內文同音稱謂連動修正清單 */}
      {hasPhoneticCandidates && (
        <div className="p-2.5 bg-amber-50/80 border border-amber-200 rounded-lg space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-amber-900 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-600" />
              同音稱謂建議 ({phoneticCandidates.length})
            </span>
            <button
              type="button"
              onClick={applyPhoneticReplacements}
              className="text-[11px] bg-amber-600 text-white px-2 py-0.5 rounded flex items-center gap-1 hover:bg-amber-700"
            >
              <CheckCircle2 className="w-3 h-3" />
              全部替換
            </button>
          </div>
          <div className="max-h-24 overflow-y-auto space-y-1 text-[11px] text-amber-800">
            {phoneticCandidates.map((c, idx) => (
              <div key={`${c.segmentId}-${idx}`} className="truncate">
                「{c.matchedText}」➔ <strong>{c.targetSpeakerName}</strong>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};