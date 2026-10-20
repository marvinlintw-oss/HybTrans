// src/components/sidebar/SpeakerSyncPanel.tsx
import React, { useState } from 'react';
import {
  Users,
  Merge,
  CheckCircle2,
  Sliders,
  Sparkles,
  Info,
  Cpu,
  Globe2,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import {
  CompliantOnnxSpeakerModel,
  DiarizationClusterMethod,
  DiarizationEngineMode,
  MeetingLanguageHint,
} from '../../types';
import * as phoneticModule from '../../utils/phoneticMatcher';

export const SpeakerSyncPanel: React.FC = () => {
  const {
    config,
    updateConfig,
    updateDiarizationConfig,
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

  const diarCfg = config.diarizationConfig || {
    engineMode: 'mode_b_advanced_dsp',
    clusterMethod: 'spectral_cosine',
    executionBackend: 'wasm',
    onnxModelFilename: 'nemo_en_titanet_small.int8.onnx',
    languageHint: 'auto_multilingual',
  };

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
          <h3 className="font-semibold text-slate-800 text-sm">與會語者與分離策略</h3>
          <span
            title="支援三階段可插拔聲紋引擎：可自由切換經典 32維 MFCC、進階 42維去內容耦合 DSP，以及 100% 本地自託管之 NeMo TitaNet / WeSpeaker ONNX 深度聲紋模型"
            className="text-slate-400 hover:text-slate-600 cursor-help"
          >
            <Info className="w-3.5 h-3.5" />
          </span>
        </div>
        <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full font-medium">
          {speakerList.length} 人
        </span>
      </div>

      {/* 可插拔語者分離引擎策略與多語系設定 */}
      <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-200/80 space-y-2 text-xs">
        <div className="flex items-center justify-between gap-1">
          <span className="text-slate-600 font-medium flex items-center gap-1 shrink-0">
            <Cpu className="w-3.5 h-3.5 text-indigo-600" />
            分離策略:
          </span>
          <select
            value={diarCfg.engineMode}
            onChange={(e) =>
              updateDiarizationConfig({
                engineMode: e.target.value as DiarizationEngineMode,
              })
            }
            className="bg-white border border-slate-200 rounded px-1.5 py-1 text-[11px] font-semibold text-slate-800 flex-1 truncate focus:outline-none"
          >
            <option value="mode_b_advanced_dsp">
              Mode B：進階純 DSP (42維 CMVN+F3/F4) ✨
            </option>
            <option value="mode_a_classic">
              Mode A：經典極速 (32維 MFCC+硬性乒乓)
            </option>
            <option value="mode_c_onnx_neural">
              Mode C：本地 ONNX 深度模型 (NeMo/WeSpeaker)
            </option>
          </select>
        </div>

        {/* Mode C 專屬：選擇合規自託管模型與隔離執行後端 */}
        {diarCfg.engineMode === 'mode_c_onnx_neural' && (
          <div className="grid grid-cols-2 gap-1.5 pt-0.5">
            <select
              value={diarCfg.onnxModelFilename}
              onChange={(e) =>
                updateDiarizationConfig({
                  onnxModelFilename: e.target.value as CompliantOnnxSpeakerModel,
                })
              }
              title="請將對應模型放置於 /public/models/ 目錄下"
              className="bg-white border border-indigo-200 rounded px-1.5 py-1 text-[10px] text-indigo-900 truncate"
            >
              <option value="nemo_en_titanet_small.int8.onnx">
                NeMo TitaNet-Small (INT8)
              </option>
              <option value="wespeaker_voxceleb_resnet34_int8.onnx">
                WeSpeaker ResNet34 (INT8)
              </option>
            </select>
            <select
              value={diarCfg.executionBackend}
              onChange={(e) =>
                updateDiarizationConfig({
                  executionBackend: e.target.value as 'wasm' | 'webgpu',
                })
              }
              title="預設使用 WASM SIMD 隔離執行緒，避免與 ASR 搶奪 WebGPU VRAM"
              className="bg-white border border-slate-200 rounded px-1.5 py-1 text-[10px] text-slate-700"
            >
              <option value="wasm">後端：WASM SIMD (不搶顯存)</option>
              <option value="webgpu">後端：WebGPU 加速</option>
            </select>
          </div>
        )}

        <div className="grid grid-cols-2 gap-1.5">
          <div className="flex items-center justify-between bg-white px-2 py-1 rounded border border-slate-200">
            <span className="text-[11px] text-slate-500">分群:</span>
            <select
              value={diarCfg.clusterMethod}
              onChange={(e) =>
                updateDiarizationConfig({
                  clusterMethod: e.target.value as DiarizationClusterMethod,
                })
              }
              className="text-[11px] font-medium text-slate-800 bg-transparent focus:outline-none"
            >
              <option value="spectral_cosine">譜聚類 (Spectral)</option>
              <option value="ahc_cosine">階層凝聚 (AHC)</option>
            </select>
          </div>

          <div className="flex items-center justify-between bg-white px-2 py-1 rounded border border-slate-200">
            <Globe2 className="w-3 h-3 text-slate-400 shrink-0 mr-1" />
            <select
              value={diarCfg.languageHint}
              onChange={(e) =>
                updateDiarizationConfig({
                  languageHint: e.target.value as MeetingLanguageHint,
                })
              }
              title="支援中/英/日跨語系中心化與日語相槌(はい/なるほど)、英語附和詞保護"
              className="text-[11px] font-medium text-slate-800 bg-transparent focus:outline-none w-full"
            >
              <option value="auto_multilingual">中/英/日自動混用</option>
              <option value="zh">華語/台語會議</option>
              <option value="ja">日本語會議 (相槌保護)</option>
              <option value="en">English Meeting</option>
            </select>
          </div>
        </div>
      </div>

      {/* 會議人數範圍設定 */}
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
            const f0Val = sp.f0Hz
              ? Math.round(sp.f0Hz)
              : sp.centroid?.[0]
              ? Math.round(sp.centroid[0])
              : null;
            return (
              <div
                key={sp.id}
                title={
                  f0Val
                    ? `聲紋特徵基準：約 ${f0Val}Hz (${sp.featureMode || 'DSP'})`
                    : undefined
                }
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