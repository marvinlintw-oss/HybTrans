// src/hooks/usePipelineScheduler.ts
import { useCallback, useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import {
  CorrectionPatchItem,
  FineDiarizationTurn,
  Segment,
  Speaker,
  SpeakerTrackBatch,
  TriLevelContextPayload,
  VirtualTrackMapItem,
  WordSpan,
} from '../types';
import {
  acquireKeepAwakeLock,
  encodeFloat32ToWav,
  extractAndPreprocessMacroChunk,
  getAudioDuration,
  releaseKeepAwakeLock,
  TARGET_SAMPLE_RATE,
} from '../utils/audioProcessor';
import {
  applySentenceLevelLocalCorrection,
  buildTriLevelContextPayload,
  evolveRollingMeetingSummary,
  executeBoundedQuotaRelisten,
  mineAdaptiveHighFreqTerms,
  runGlobalRetroactiveContextPass,
  runTriLevelWindowCorrection,
} from '../services/engineRouter';
import {
  transcribeWithGeminiAudio,
  transcribeWithGroq,
} from '../services/byokCloudClient';

let globalAbortFlag = false;
let globalIsProcessing = false;
const listeners = new Set<(val: boolean) => void>();

function setGlobalProcessing(val: boolean) {
  globalIsProcessing = val;
  listeners.forEach((fn) => fn(val));
}

/**
 * 根據語者虛擬音軌映射表 `timeMap`，從當前 5 分鐘巨集區塊的原始無損波形中抽取該語者的所有發聲片段，
 * 並在片段與片段之間插入 `0.35s` 零值靜音隔板（附帶 10ms 線性淡入淡出防爆音），組裝成純淨同語者音訊！
 */
function stitchVirtualSpeakerTrackSamples(
  chunkSamples: Float32Array,
  chunkStartGlobalSec: number,
  batch: SpeakerTrackBatch,
  sampleRate = TARGET_SAMPLE_RATE
): Float32Array {
  const spacerSamplesCount = Math.round(batch.spacerDurationSec * sampleRate);
  const fadeSamples = Math.min(160, Math.floor(spacerSamplesCount / 4)); // 10ms 微淡入淡出

  const slices: Float32Array[] = [];
  let totalLength = 0;

  for (let i = 0; i < batch.timeMap.length; i++) {
    const item = batch.timeMap[i];
    const startIdx = Math.max(
      0,
      Math.floor((item.realStartSec - chunkStartGlobalSec) * sampleRate)
    );
    const endIdx = Math.min(
      chunkSamples.length,
      Math.ceil((item.realEndSec - chunkStartGlobalSec) * sampleRate)
    );
    const segLen = Math.max(0, endIdx - startIdx);
    const sub = new Float32Array(segLen);
    if (segLen > 0) {
      sub.set(chunkSamples.subarray(startIdx, endIdx));
      for (let f = 0; f < Math.min(fadeSamples, Math.floor(segLen / 2)); f++) {
        const gain = f / fadeSamples;
        sub[f] *= gain;
        sub[segLen - 1 - f] *= gain;
      }
    }
    slices.push(sub);
    totalLength += segLen;
    if (i < batch.timeMap.length - 1) {
      totalLength += spacerSamplesCount;
    }
  }

  const stitched = new Float32Array(totalLength);
  let offset = 0;
  for (let i = 0; i < slices.length; i++) {
    stitched.set(slices[i], offset);
    offset += slices[i].length;
    if (i < slices.length - 1) {
      offset += spacerSamplesCount; // 中間維持 0.0 靜音隔板
    }
  }

  return stitched;
}

/**
 * 當雲端 BYOK 模式 (Groq / Gemini) 搭配「語者虛擬音軌拼接」時，將雲端回傳之相對時間區段還原回原始會議時間軸
 */
function unwarpCloudSegmentsFromVirtualTrack(
  cloudSegs: Segment[],
  timeMap: VirtualTrackMapItem[],
  speakerId: string
): Segment[] {
  if (timeMap.length === 0 || cloudSegs.length === 0) return cloudSegs;

  return cloudSegs.map((seg) => {
    const vStart = seg.start;
    const vEnd = seg.end;
    const vCenter = 0.5 * (vStart + vEnd);

    let bestItem = timeMap[0];
    let bestOverlap = -1;
    for (const item of timeMap) {
      const ov =
        Math.min(vEnd, item.virtualEndSec) -
        Math.max(vStart, item.virtualStartSec);
      if (ov > bestOverlap) {
        bestOverlap = ov;
        bestItem = item;
      }
    }
    if (bestOverlap <= 0) {
      let minDist = Infinity;
      for (const item of timeMap) {
        const c = 0.5 * (item.virtualStartSec + item.virtualEndSec);
        if (Math.abs(vCenter - c) < minDist) {
          minDist = Math.abs(vCenter - c);
          bestItem = item;
        }
      }
    }

    const vSpan = Math.max(
      0.15,
      bestItem.virtualEndSec - bestItem.virtualStartSec
    );
    const rSpan = Math.max(0.15, bestItem.realEndSec - bestItem.realStartSec);
    const clampedVStart = Math.max(
      bestItem.virtualStartSec,
      Math.min(bestItem.virtualEndSec - 0.1, vStart)
    );
    const clampedVEnd = Math.max(
      clampedVStart + 0.15,
      Math.min(bestItem.virtualEndSec, vEnd)
    );

    const realStart = Number(
      (
        bestItem.realStartSec +
        ((clampedVStart - bestItem.virtualStartSec) / vSpan) * rSpan
      ).toFixed(2)
    );
    const realEnd = Number(
      Math.max(
        realStart + 0.2,
        bestItem.realStartSec +
          ((clampedVEnd - bestItem.virtualStartSec) / vSpan) * rSpan
      ).toFixed(2)
    );

    const dur = Math.max(0.2, realEnd - realStart);
    const step = dur / Math.max(1, seg.words.length);

    return {
      ...seg,
      speakerId: bestItem.speakerId || speakerId,
      runnerUpSpeakerId:
        bestItem.runnerUpSpeakerId || bestItem.speakerId || speakerId,
      acousticMargin: bestItem.speakerMargin,
      start: realStart,
      end: realEnd,
      words: seg.words.map((w, idx) => ({
        ...w,
        start: Number((realStart + idx * step).toFixed(2)),
        end: Number((realStart + (idx + 1) * step).toFixed(2)),
      })),
    };
  });
}

export function usePipelineScheduler() {
  const [isProcessing, setIsProcessing] = useState(globalIsProcessing);
  const coResidentAsrWorkerRef = useRef<Worker | null>(null);
  const coResidentLlmWorkerRef = useRef<Worker | null>(null);
  const activeDiarWorkerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const handler = (val: boolean) => setIsProcessing(val);
    listeners.add(handler);
    return () => {
      listeners.delete(handler);
    };
  }, []);

  const terminateAllWorkers = useCallback(() => {
    if (activeDiarWorkerRef.current) {
      activeDiarWorkerRef.current.terminate();
      activeDiarWorkerRef.current = null;
    }
    if (coResidentAsrWorkerRef.current) {
      coResidentAsrWorkerRef.current.terminate();
      coResidentAsrWorkerRef.current = null;
    }
    if (coResidentLlmWorkerRef.current) {
      coResidentLlmWorkerRef.current.terminate();
      coResidentLlmWorkerRef.current = null;
    }
    useAppStore.getState().setModelDownload({
      active: false,
      progress: 0,
      fileName: '',
    });
  }, []);

  const ensureLocalLlmWorker = useCallback((): Worker => {
    if (!coResidentLlmWorkerRef.current) {
      coResidentLlmWorkerRef.current = new Worker(
        new URL('../workers/localLlm.worker.ts', import.meta.url),
        { type: 'module' }
      );
    }
    return coResidentLlmWorkerRef.current;
  }, []);

  const stopPipeline = useCallback(() => {
    globalAbortFlag = true;
    terminateAllWorkers();
    releaseKeepAwakeLock();
    setGlobalProcessing(false);

    const chunks = useAppStore.getState().chunks;
    const activeChunk = chunks.find(
      (c) => c.stage !== 'idle' && c.stage !== 'completed'
    );
    if (activeChunk) {
      useAppStore.getState().updateChunkState(activeChunk.index, {
        stage: 'idle',
        detailText: '已手動停止轉檔（保留目前已產出之逐字稿）',
      });
    }
    console.log('🛑 [Pipeline 已由使用者手動停止]');
  }, [terminateAllWorkers]);

  const invokeLocalLlmWorker = useCallback(
    (
      worker: Worker,
      modelId: string,
      payload: TriLevelContextPayload
    ): Promise<CorrectionPatchItem[]> => {
      return new Promise((resolve) => {
        const timeout = setTimeout(() => {
          useAppStore.getState().setModelDownload({ active: false });
          resolve([]);
        }, 45000);

        const handleMsg = (e: MessageEvent) => {
          if (e.data?.type === 'LLM_LOAD_PROGRESS') {
            const pct = Math.round((Number(e.data.progress) || 0) * 100);
            useAppStore.getState().setModelDownload({
              active: pct < 100,
              modelName: modelId,
              fileName: e.data.text || '正在載入本機 AI 語境校正模型...',
              progress: pct,
            });
            return;
          }
          if (e.data?.type === 'LLM_CORRECTION_SUCCESS') {
            clearTimeout(timeout);
            worker.removeEventListener('message', handleMsg);
            useAppStore.getState().setModelDownload({ active: false });
            resolve(e.data.patches || []);
          } else if (e.data?.type === 'LLM_CORRECTION_ERROR') {
            clearTimeout(timeout);
            worker.removeEventListener('message', handleMsg);
            useAppStore.getState().setModelDownload({ active: false });
            resolve([]);
          }
        };
        worker.addEventListener('message', handleMsg);
        worker.postMessage({
          type: 'CORRECT_TRI_LEVEL_WINDOW',
          modelId,
          payload,
        });
      });
    },
    []
  );

  const triggerRetroactivePass = useCallback(async () => {
    const state = useAppStore.getState();
    if (state.segments.length === 0) return 0;

    let tempLlmWorker: Worker | null = null;
    try {
      if (!state.config.geminiApiKey.trim()) {
        tempLlmWorker = new Worker(
          new URL('../workers/localLlm.worker.ts', import.meta.url),
          { type: 'module' }
        );
      }

      const fixedCount = await runGlobalRetroactiveContextPass({
        config: state.config,
        allSegments: state.segments,
        dictionary: state.dictionary,
        docContextSummary: state.docContextSummary,
        rollingMeetingSummary: state.rollingMeetingSummary,
        runLocalLlmCorrection: tempLlmWorker
          ? (pl) =>
              invokeLocalLlmWorker(
                tempLlmWorker!,
                state.config.localLlmModel,
                pl
              )
          : undefined,
      });
      return fixedCount;
    } finally {
      if (tempLlmWorker) {
        tempLlmWorker.postMessage({ type: 'UNLOAD_LLM' });
        tempLlmWorker.terminate();
      }
    }
  }, [invokeLocalLlmWorker]);

  const buildFallbackTurns = (
    chunkStartSec: number,
    chunkEndSec: number
  ): FineDiarizationTurn[] => {
    const turns: FineDiarizationTurn[] = [];
    for (let t = chunkStartSec; t < chunkEndSec; t += 15) {
      turns.push({
        start: t,
        end: Math.min(chunkEndSec, t + 15),
        speakerId: 'Speaker_01',
        f0Hz: 155,
      });
    }
    return turns;
  };

  const runDiarizationStage = (
    samples: Float32Array,
    chunkIndex: number,
    chunkStartSec: number,
    chunkEndSec: number,
    minSpeakers: number,
    maxSpeakers: number,
    existingSpeakers: Speaker[]
  ): Promise<{
    turns: FineDiarizationTurn[];
    asrWindows: Array<{ start: number; end: number }>;
    speakerTrackBatches: SpeakerTrackBatch[];
    updatedSpeakers: Speaker[];
  }> => {
    return new Promise((resolve) => {
      const worker = new Worker(
        new URL('../workers/diarization.worker.ts', import.meta.url),
        { type: 'module' }
      );
      activeDiarWorkerRef.current = worker;

      let settled = false;
      const fallback = (reason: string) => {
        console.warn(`⚠️ [Diarization Fallback 觸發]: ${reason}`);
        const fb = buildFallbackTurns(chunkStartSec, chunkEndSec);
        return {
          turns: fb,
          asrWindows: fb.map((t) => ({ start: t.start, end: t.end })),
          speakerTrackBatches: [],
          updatedSpeakers: existingSpeakers,
        };
      };

      let watchdogId: ReturnType<typeof setTimeout> | null = null;
      const resetWatchdog = () => {
        if (watchdogId) clearTimeout(watchdogId);
        watchdogId = setTimeout(() => {
          if (settled) return;
          settled = true;
          worker.terminate();
          activeDiarWorkerRef.current = null;
          resolve(fallback('Worker 超過 90 秒無進度回應'));
        }, 90000);
      };
      resetWatchdog();

      worker.onmessage = (e) => {
        if (settled) return;
        const msg = e.data;

        if (msg.type === 'DIARIZE_PROGRESS') {
          resetWatchdog();
          const subProgress = 12 + Math.round(((msg.progress || 0) / 100) * 18);
          useAppStore.getState().updateChunkState(chunkIndex, {
            stage: 'diarizing',
            progress: subProgress,
            detailText: msg.detailText || '聲紋特徵提取中...',
          });
          return;
        }

        if (msg.type === 'DIARIZE_SUCCESS' || msg.type === 'RESULT') {
          settled = true;
          if (watchdogId) clearTimeout(watchdogId);
          worker.terminate();
          activeDiarWorkerRef.current = null;
          const turns: FineDiarizationTurn[] =
            msg.turns || msg.data?.turns || [];
          const asrWindows =
            Array.isArray(msg.asrWindows) && msg.asrWindows.length > 0
              ? msg.asrWindows
              : turns.map((t: { start: number; end: number }) => ({
                  start: t.start,
                  end: t.end,
                }));
          const speakerTrackBatches: SpeakerTrackBatch[] = Array.isArray(
            msg.speakerTrackBatches
          )
            ? msg.speakerTrackBatches
            : [];
          const updatedSpeakers =
            msg.updatedSpeakers || msg.data?.speakers || existingSpeakers;
          resolve(
            turns.length > 0
              ? { turns, asrWindows, speakerTrackBatches, updatedSpeakers }
              : fallback('Turns 為空陣列')
          );
        } else if (msg.type === 'DIARIZE_ERROR' || msg.type === 'ERROR') {
          settled = true;
          if (watchdogId) clearTimeout(watchdogId);
          worker.terminate();
          activeDiarWorkerRef.current = null;
          resolve(fallback(msg.error || 'Worker 回報錯誤'));
        }
      };

      worker.onerror = (err) => {
        if (settled) return;
        settled = true;
        if (watchdogId) clearTimeout(watchdogId);
        worker.terminate();
        activeDiarWorkerRef.current = null;
        resolve(fallback(err.message || 'Worker 執行期異常'));
      };

      const cleanSpeakers: Speaker[] = JSON.parse(
        JSON.stringify(existingSpeakers)
      );
      const copySamples = new Float32Array(samples);
      const diarizationConfig =
        useAppStore.getState().config.diarizationConfig;

      worker.postMessage(
        {
          type: 'DIARIZE_CHUNK',
          samples: copySamples,
          sampleRate: TARGET_SAMPLE_RATE,
          chunkIndex,
          chunkStartSec,
          minSpeakers,
          maxSpeakers,
          existingSpeakers: cleanSpeakers,
          diarizationConfig,
        },
        [copySamples.buffer]
      );
    });
  };

  const startPipeline = useCallback(
    async (file: File) => {
      if (globalIsProcessing) return;
      setGlobalProcessing(true);
      globalAbortFlag = false;

      const store = useAppStore.getState();

      try {
        await acquireKeepAwakeLock();

        const totalDurationSec = await getAudioDuration(file);
        store.clearTranscript(false);
        store.initChunks(totalDurationSec);

        const chunks = useAppStore.getState().chunks;

        for (let cIdx = 0; cIdx < chunks.length; cIdx++) {
          if (globalAbortFlag) break;

          const currentConfig = useAppStore.getState().config;
          const chunkMeta = useAppStore.getState().chunks[cIdx];
          const engineMode =
            currentConfig.diarizationConfig?.engineMode ||
            'mode_0_heritage_classic';
          const useVirtualTrackStitching =
            currentConfig.diarizationConfig
              ?.enableVirtualSpeakerTrackStitching ?? true;

          const modeLabel =
            engineMode === 'mode_0_heritage_classic'
              ? 'Mode 0 初代經典穩健引擎'
              : engineMode === 'mode_a_classic'
              ? 'Mode A 純 DSP 極速'
              : engineMode === 'mode_c_onnx_neural'
              ? 'Mode C WeSpeaker ONNX 雙尺度'
              : 'Mode B 雙流形譜聚類';

          useAppStore.getState().updateChunkState(cIdx, {
            stage: 'preprocessing',
            progress: 10,
            detailText: '音訊切片與 85Hz 降噪濾波中...',
          });

          const { samples: chunkSamples } =
            await extractAndPreprocessMacroChunk(
              file,
              chunkMeta.startTime,
              chunkMeta.endTime,
              totalDurationSec
            );

          if (globalAbortFlag) break;

          useAppStore.getState().updateChunkState(cIdx, {
            stage: 'diarizing',
            progress: 15,
            detailText: `${modeLabel} 黃金長句定錨與聲紋分離中 (${currentConfig.minSpeakers}~${currentConfig.maxSpeakers} 人)...`,
          });

          const existingSpeakers = Object.values(
            useAppStore.getState().speakers
          );
          const {
            turns: fineTurns,
            asrWindows,
            speakerTrackBatches,
            updatedSpeakers,
          } = await runDiarizationStage(
            chunkSamples,
            cIdx,
            chunkMeta.startTime,
            chunkMeta.endTime,
            currentConfig.minSpeakers,
            currentConfig.maxSpeakers,
            existingSpeakers
          );

          if (globalAbortFlag) break;

          useAppStore.getState().upsertSpeakers(updatedSpeakers);

          const useCloudAsr =
            currentConfig.mode === 'byok_cloud' &&
            Boolean(
              currentConfig.groqApiKey.trim() ||
                currentConfig.geminiApiKey.trim()
            );

          let asrWorker: Worker | null = null;
          if (!useCloudAsr) {
            if (!coResidentAsrWorkerRef.current) {
              coResidentAsrWorkerRef.current = new Worker(
                new URL('../workers/localAsr.worker.ts', import.meta.url),
                { type: 'module' }
              );
            }
            asrWorker = coResidentAsrWorkerRef.current;
          }

          if (
            currentConfig.keepModelsCoResident &&
            !currentConfig.geminiApiKey.trim()
          ) {
            ensureLocalLlmWorker();
          }

          let pendingWindowSegments: Segment[] = [];
          let pendingWindowChars = 0;
          const flaggedForRelistenIds = new Set<string>();

          const flushPendingWindowCorrection = async (
            activeLlmWorker?: Worker | null
          ) => {
            if (pendingWindowSegments.length === 0 || globalAbortFlag) return;
            const latestState = useAppStore.getState();

            const newRollingSummary = await evolveRollingMeetingSummary({
              config: currentConfig,
              docContextSummary: latestState.docContextSummary,
              currentSummary: latestState.rollingMeetingSummary,
              recentSegments: pendingWindowSegments,
              dictionary: latestState.dictionary,
            });
            useAppStore.getState().setRollingMeetingSummary(newRollingSummary);

            const firstPendingId = pendingWindowSegments[0].id;
            const firstIdx = latestState.segments.findIndex(
              (s) => s.id === firstPendingId
            );
            const priorSegments =
              firstIdx > 0 ? latestState.segments.slice(0, firstIdx) : [];

            const pendingIdSet = new Set(
              pendingWindowSegments.map((s) => s.id)
            );
            const currentTargetSegments = latestState.segments.filter((s) =>
              pendingIdSet.has(s.id)
            );

            const payload = buildTriLevelContextPayload(
              priorSegments,
              currentTargetSegments.length > 0
                ? currentTargetSegments
                : pendingWindowSegments,
              latestState.speakers,
              latestState.dictionary,
              latestState.docContextSummary,
              currentConfig.mediumAttentionChars || 1200,
              newRollingSummary
            );

            const patches = await runTriLevelWindowCorrection({
              config: currentConfig,
              payload,
              dictionary: latestState.dictionary,
              runLocalLlmCorrection: activeLlmWorker
                ? (pl) =>
                    invokeLocalLlmWorker(
                      activeLlmWorker,
                      currentConfig.localLlmModel,
                      pl
                    )
                : undefined,
              onApplyPatches: (appliedPatches, segIds) => {
                useAppStore
                  .getState()
                  .applyCorrectionPatches(appliedPatches, segIds);
              },
            });

            for (const p of patches) {
              if (p.requireRelisten || p.status === 'semantic_uncertain') {
                flaggedForRelistenIds.add(p.segmentId);
              }
            }

            pendingWindowSegments = [];
            pendingWindowChars = 0;
          };

          // ==================================================================
          // 分支 A：【升級版】語者專屬虛擬音軌拼接轉寫 (Virtual Speaker-Track Stitching)
          // ==================================================================
          if (useVirtualTrackStitching && speakerTrackBatches.length > 0) {
            useAppStore.getState().updateChunkState(cIdx, {
              stage: 'transcribing',
              progress: 35,
              detailText: `已建立 ${updatedSpeakers.length} 位語者專屬虛擬音軌 (共 ${speakerTrackBatches.length} 個純淨批次)，開始分軌轉寫...`,
            });

            for (let bIdx = 0; bIdx < speakerTrackBatches.length; bIdx++) {
              if (globalAbortFlag) break;
              const batch = speakerTrackBatches[bIdx];
              const stitchedSamples = stitchVirtualSpeakerTrackSamples(
                chunkSamples,
                chunkMeta.startTime,
                batch,
                TARGET_SAMPLE_RATE
              );

              const spName =
                useAppStore.getState().speakers[batch.speakerId]?.name ||
                batch.speakerId;
              const batchProgress =
                35 +
                Math.round(
                  ((bIdx + 1) / Math.max(1, speakerTrackBatches.length)) * 45
                );

              useAppStore.getState().updateChunkState(cIdx, {
                stage: 'transcribing',
                progress: batchProgress,
                detailText: `專屬音軌轉寫 [${spName}] (${bIdx + 1}/${speakerTrackBatches.length} 批次)...`,
              });

              let batchSegments: Segment[] = [];

              if (useCloudAsr) {
                const stitchedWav = encodeFloat32ToWav(
                  stitchedSamples,
                  TARGET_SAMPLE_RATE
                );
                const hotwords = useAppStore
                  .getState()
                  .dictionary.map((d) => d.term)
                  .slice(0, 40)
                  .join('、');

                const rawCloudSegs = currentConfig.groqApiKey.trim()
                  ? await transcribeWithGroq(
                      stitchedWav,
                      currentConfig.groqApiKey.trim(),
                      hotwords,
                      0,
                      batch.speakerId,
                      cIdx,
                      currentConfig.acousticThreshold
                    )
                  : await transcribeWithGeminiAudio(
                      stitchedWav,
                      currentConfig.geminiApiKey.trim(),
                      hotwords,
                      0,
                      batch.speakerId,
                      cIdx,
                      currentConfig.acousticThreshold
                    );

                batchSegments = unwarpCloudSegmentsFromVirtualTrack(
                  rawCloudSegs,
                  batch.timeMap,
                  batch.speakerId
                );
              } else if (asrWorker) {
                const activeWorker = asrWorker;
                const languageHint =
                  currentConfig.diarizationConfig?.languageHint || 'zh';

                batchSegments = await new Promise<Segment[]>(
                  (resolve, reject) => {
                    const handleBatchMsg = (e: MessageEvent) => {
                      if (globalAbortFlag) {
                        activeWorker.removeEventListener(
                          'message',
                          handleBatchMsg
                        );
                        resolve([]);
                        return;
                      }
                      if (e.data?.type === 'ASR_DOWNLOAD_PROGRESS') {
                        useAppStore.getState().setModelDownload({
                          active: Boolean(e.data.active),
                          modelName:
                            e.data.modelName || currentConfig.localAsrModel,
                          fileName: e.data.fileName || '模型權重檔',
                          progress: Number(e.data.progress) || 0,
                        });
                        return;
                      }
                      if (
                        e.data?.type === 'SPEAKER_BATCH_TRANSCRIBED' &&
                        e.data.batchIndex === bIdx
                      ) {
                        activeWorker.removeEventListener(
                          'message',
                          handleBatchMsg
                        );
                        resolve(e.data.segments || []);
                      } else if (
                        e.data?.type === 'ASR_ERROR' ||
                        e.data?.type === 'ERROR'
                      ) {
                        activeWorker.removeEventListener(
                          'message',
                          handleBatchMsg
                        );
                        reject(new Error(e.data.error));
                      }
                    };
                    activeWorker.addEventListener('message', handleBatchMsg);
                    activeWorker.postMessage({
                      type: 'TRANSCRIBE_SPEAKER_TRACK_BATCH',
                      stitchedSamples,
                      timeMap: batch.timeMap,
                      speakerId: batch.speakerId,
                      chunkIndex: cIdx,
                      batchIndex: bIdx,
                      modelName: currentConfig.localAsrModel,
                      languageHint,
                      acousticThreshold: currentConfig.acousticThreshold,
                    });
                  }
                );
              }

              if (globalAbortFlag) break;

              if (batchSegments.length > 0) {
                const latestDict = useAppStore.getState().dictionary;
                const quickTerms = latestDict.map((d) => d.term);
                const { updatedSegments: instantCorrected } =
                  applySentenceLevelLocalCorrection(
                    batchSegments,
                    quickTerms,
                    latestDict
                  );
                useAppStore.getState().appendSegments(instantCorrected);
                pendingWindowSegments.push(...instantCorrected);

                const addedChars = instantCorrected.reduce(
                  (acc, s) =>
                    acc + s.words.reduce((wAcc, w) => wAcc + w.text.length, 0),
                  0
                );
                pendingWindowChars += addedChars;

                if (
                  pendingWindowChars >= (currentConfig.highAttentionChars || 300)
                ) {
                  useAppStore.getState().updateChunkState(cIdx, {
                    stage: 'correcting',
                    detailText: `三層關注度介入：修正近 ${pendingWindowChars} 字...`,
                  });
                  await flushPendingWindowCorrection(
                    currentConfig.keepModelsCoResident
                      ? coResidentLlmWorkerRef.current
                      : null
                  );
                }
              }
            }
          } else {
            // ================================================================
            // 分支 B：傳統混合音訊視窗轉寫 + 事後 Viterbi 對齊
            // ================================================================
            for (let wIdx = 0; wIdx < asrWindows.length; wIdx++) {
              if (globalAbortFlag) break;
              const win = asrWindows[wIdx];

              const relStartSample = Math.max(
                0,
                Math.floor(
                  (win.start - chunkMeta.startTime) * TARGET_SAMPLE_RATE
                )
              );
              const relEndSample = Math.min(
                chunkSamples.length,
                Math.ceil((win.end - chunkMeta.startTime) * TARGET_SAMPLE_RATE)
              );
              const winSamples = chunkSamples.slice(
                relStartSample,
                relEndSample
              );

              const winProgress =
                35 +
                Math.round(((wIdx + 1) / Math.max(1, asrWindows.length)) * 45);
              useAppStore.getState().updateChunkState(cIdx, {
                stage: 'transcribing',
                progress: winProgress,
                detailText: `語境轉寫與多語系子句聲紋對齊 (${wIdx + 1}/${asrWindows.length})`,
              });

              let windowSegments: Segment[] = [];

              if (useCloudAsr) {
                const turnWav = encodeFloat32ToWav(
                  winSamples,
                  TARGET_SAMPLE_RATE
                );
                const hotwords = useAppStore
                  .getState()
                  .dictionary.map((d) => d.term)
                  .slice(0, 40)
                  .join('、');
                const primarySp =
                  fineTurns.find(
                    (ft) => ft.end > win.start && ft.start < win.end
                  )?.speakerId || 'Speaker_01';

                windowSegments = currentConfig.groqApiKey.trim()
                  ? await transcribeWithGroq(
                      turnWav,
                      currentConfig.groqApiKey.trim(),
                      hotwords,
                      win.start,
                      primarySp,
                      cIdx,
                      currentConfig.acousticThreshold
                    )
                  : await transcribeWithGeminiAudio(
                      turnWav,
                      currentConfig.geminiApiKey.trim(),
                      hotwords,
                      win.start,
                      primarySp,
                      cIdx,
                      currentConfig.acousticThreshold
                    );
              } else if (asrWorker) {
                const activeWorker = asrWorker;
                const languageHint =
                  currentConfig.diarizationConfig?.languageHint || 'zh';

                windowSegments = await new Promise<Segment[]>(
                  (resolve, reject) => {
                    const handleAsrMsg = (e: MessageEvent) => {
                      if (globalAbortFlag) {
                        activeWorker.removeEventListener(
                          'message',
                          handleAsrMsg
                        );
                        resolve([]);
                        return;
                      }
                      if (e.data?.type === 'ASR_DOWNLOAD_PROGRESS') {
                        useAppStore.getState().setModelDownload({
                          active: Boolean(e.data.active),
                          modelName:
                            e.data.modelName || currentConfig.localAsrModel,
                          fileName: e.data.fileName || '模型權重檔',
                          progress: Number(e.data.progress) || 0,
                        });
                        return;
                      }
                      if (
                        e.data?.type === 'WINDOW_TRANSCRIBED' &&
                        e.data.windowIndex === wIdx
                      ) {
                        activeWorker.removeEventListener(
                          'message',
                          handleAsrMsg
                        );
                        resolve(e.data.segments || []);
                      } else if (
                        e.data?.type === 'ASR_ERROR' ||
                        e.data?.type === 'ERROR'
                      ) {
                        activeWorker.removeEventListener(
                          'message',
                          handleAsrMsg
                        );
                        reject(new Error(e.data.error));
                      }
                    };
                    activeWorker.addEventListener('message', handleAsrMsg);
                    activeWorker.postMessage({
                      type: 'TRANSCRIBE_WINDOW_AND_ALIGN',
                      windowSamples: winSamples,
                      windowStartSec: win.start,
                      windowEndSec: win.end,
                      fineTurns,
                      chunkIndex: cIdx,
                      windowIndex: wIdx,
                      modelName: currentConfig.localAsrModel,
                      languageHint,
                      acousticThreshold: currentConfig.acousticThreshold,
                      asrBeamSize: currentConfig.asrBeamSize,
                      backchannelMarginThreshold:
                        currentConfig.diarizationConfig
                          ?.backchannelMarginThreshold ?? 0.12,
                      pingPongMaxDurSec:
                        currentConfig.diarizationConfig?.pingPongMaxDurSec ??
                        1.6,
                      engineMode,
                    });
                  }
                );
              }

              if (globalAbortFlag) break;

              if (windowSegments.length > 0) {
                const latestDict = useAppStore.getState().dictionary;
                const quickTerms = latestDict.map((d) => d.term);
                const { updatedSegments: instantCorrected } =
                  applySentenceLevelLocalCorrection(
                    windowSegments,
                    quickTerms,
                    latestDict
                  );
                useAppStore.getState().appendSegments(instantCorrected);
                pendingWindowSegments.push(...instantCorrected);

                const addedChars = instantCorrected.reduce(
                  (acc, s) =>
                    acc + s.words.reduce((wAcc, w) => wAcc + w.text.length, 0),
                  0
                );
                pendingWindowChars += addedChars;

                if (
                  pendingWindowChars >= (currentConfig.highAttentionChars || 300)
                ) {
                  useAppStore.getState().updateChunkState(cIdx, {
                    stage: 'correcting',
                    detailText: `三層關注度介入：修正近 ${pendingWindowChars} 字...`,
                  });
                  await flushPendingWindowCorrection(
                    currentConfig.keepModelsCoResident
                      ? coResidentLlmWorkerRef.current
                      : null
                  );
                }
              }
            }
          }

          if (globalAbortFlag) break;

          if (pendingWindowSegments.length > 0) {
            useAppStore.getState().updateChunkState(cIdx, {
              stage: 'correcting',
              progress: 84,
              detailText: `語境收尾校正與會議摘要更新中...`,
            });
            await flushPendingWindowCorrection(
              currentConfig.keepModelsCoResident
                ? coResidentLlmWorkerRef.current
                : null
            );
          }

          useAppStore.getState().updateChunkState(cIdx, {
            stage: 'relistening',
            progress: 90,
            detailText: `改變切法截檔重聽校準中 (上限 ${currentConfig.maxRelistenQuota} 段)...`,
          });

          const latestStateAfterAsr = useAppStore.getState();
          const chunkSegments = latestStateAfterAsr.segments.filter(
            (s) => s.chunkIndex === cIdx
          );
          const verifiedTerms = mineAdaptiveHighFreqTerms(
            latestStateAfterAsr.segments,
            latestStateAfterAsr.dictionary,
            latestStateAfterAsr.docContextSummary
          );

          const relistenedUsed = await executeBoundedQuotaRelisten({
            config: currentConfig,
            chunkSamples,
            chunkStartGlobalSec: chunkMeta.startTime,
            chunkSegments,
            flaggedSegmentIds: flaggedForRelistenIds,
            verifiedTerms,
            relistenWithLocalAsr: asrWorker
              ? (subSamples, _paddedStartSec, targetSeg) =>
                  new Promise<WordSpan[] | null>((resolve) => {
                    const activeW = asrWorker!;
                    const timeout = setTimeout(() => resolve(null), 12000);
                    const handleRelisten = (e: MessageEvent) => {
                      if (
                        e.data?.type === 'RELISTEN_SUCCESS' &&
                        e.data.segmentId === targetSeg.id
                      ) {
                        clearTimeout(timeout);
                        activeW.removeEventListener('message', handleRelisten);
                        resolve(e.data.newWords);
                      } else if (e.data?.type === 'RELISTEN_ERROR') {
                        clearTimeout(timeout);
                        activeW.removeEventListener('message', handleRelisten);
                        resolve(null);
                      }
                    };
                    activeW.addEventListener('message', handleRelisten);
                    activeW.postMessage({
                      type: 'RELISTEN_SUB_SEGMENT',
                      subSamples,
                      targetSegment: targetSeg,
                      modelName: currentConfig.localAsrModel,
                      languageHint:
                        currentConfig.diarizationConfig?.languageHint || 'zh',
                    });
                  })
              : undefined,
            onSegmentRelistened: (segId, newWords) => {
              useAppStore
                .getState()
                .replaceSegmentAfterRelisten(segId, newWords);
            },
          });

          if (
            !currentConfig.keepModelsCoResident &&
            asrWorker &&
            !currentConfig.geminiApiKey.trim()
          ) {
            await new Promise<void>((resolve) => {
              const t = setTimeout(() => resolve(), 3000);
              const onUnload = (e: MessageEvent) => {
                if (e.data?.type === 'UNLOAD_DONE') {
                  clearTimeout(t);
                  asrWorker!.removeEventListener('message', onUnload);
                  resolve();
                }
              };
              asrWorker!.addEventListener('message', onUnload);
              asrWorker!.postMessage({ type: 'UNLOAD_MODEL' });
            });
          }

          useAppStore.getState().updateChunkState(cIdx, {
            stage: 'correcting',
            progress: 96,
            detailText: `全局回溯精修與 TPST 語者立場複查...`,
          });

          const activeLlmForRetro = !currentConfig.geminiApiKey.trim()
            ? ensureLocalLlmWorker()
            : null;

          const finalState = useAppStore.getState();
          await runGlobalRetroactiveContextPass({
            config: currentConfig,
            allSegments: finalState.segments,
            dictionary: finalState.dictionary,
            docContextSummary: finalState.docContextSummary,
            rollingMeetingSummary: finalState.rollingMeetingSummary,
            runLocalLlmCorrection: activeLlmForRetro
              ? (pl) =>
                  invokeLocalLlmWorker(
                    activeLlmForRetro,
                    currentConfig.localLlmModel,
                    pl
                  )
              : undefined,
          });

          if (
            !currentConfig.keepModelsCoResident &&
            activeLlmForRetro &&
            cIdx < chunks.length - 1
          ) {
            await new Promise<void>((resolve) => {
              const t = setTimeout(() => resolve(), 3000);
              const onUnload = (e: MessageEvent) => {
                if (e.data?.type === 'LLM_UNLOAD_DONE') {
                  clearTimeout(t);
                  activeLlmForRetro.removeEventListener('message', onUnload);
                  resolve();
                }
              };
              activeLlmForRetro.addEventListener('message', onUnload);
              activeLlmForRetro.postMessage({ type: 'UNLOAD_LLM' });
            });
          }

          useAppStore.getState().updateChunkState(cIdx, {
            stage: 'completed',
            progress: 100,
            detailText: `本塊完成 (重聽 ${relistenedUsed} 段 + 已回溯順稿)`,
            relistenUsed: relistenedUsed,
          });
        }
      } catch (err) {
        console.error('❌ Pipeline 執行發生錯誤:', err);
      } finally {
        await releaseKeepAwakeLock();
        terminateAllWorkers();
        setGlobalProcessing(false);
      }
    },
    [ensureLocalLlmWorker, invokeLocalLlmWorker, terminateAllWorkers]
  );

  return {
    isProcessing,
    isRunning: isProcessing,
    startPipeline,
    runPipeline: startPipeline,
    stopPipeline,
    triggerRetroactivePass,
  };
}