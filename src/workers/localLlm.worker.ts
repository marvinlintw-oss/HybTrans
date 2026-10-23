// src/workers/localLlm.worker.ts
import { CreateMLCEngine, MLCEngine } from '@mlc-ai/web-llm';
import * as OpenCC from 'opencc-js';
import {
  CorrectionPatchItem,
  Segment,
  StructuredSlotGerResponse,
  SuspectSlotItem,
  TriLevelContextPayload,
} from '../types';

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

let llmEngine: MLCEngine | null = null;
let currentModelId: string | null = null;

/**
 * 【問題二 2.3】：針對 WebLLM XGrammar 最佳化之嚴格 JSON Schema
 * 將 brief_reason (15字內微推理 Micro-CoT) 排在 action 與 selected_text 之前，
 * 強迫 2B/3B 小模型先執行語境推理再決定填空答案，並免除 wordIndex 算術！
 */
const HYBTRANS_SLOT_GER_JSON_SCHEMA = {
  type: 'object',
  properties: {
    slot_corrections: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          slot_id: {
            type: 'string',
            enum: ['#S1', '#S2', '#S3', '#S4', '#S5'],
          },
          brief_reason: {
            type: 'string',
            description: '15字以內說明前後文語意搭配理由',
          },
          action: {
            type: 'string',
            enum: ['KEEP_ORIGINAL', 'APPLY_HIGH_CONF', 'MARK_UNCERTAIN'],
          },
          selected_text: {
            type: 'string',
            description:
              '僅填寫該槽位替換後的詞彙本身，嚴禁包含槽位外的上下文',
          },
          uncertain_candidates: {
            type: 'array',
            items: { type: 'string' },
            description:
              '若action為MARK_UNCERTAIN，列出2至3個近音候選詞供UI黃底選單使用',
          },
        },
        required: [
          'slot_id',
          'brief_reason',
          'action',
          'selected_text',
          'uncertain_candidates',
        ],
        additionalProperties: false,
      },
    },
    turn_splits: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          utterance_id: { type: 'string' },
          pragmatic_type: {
            type: 'string',
            enum: [
              'NO_SPLIT',
              'QA_SPEAKER_CHANGE',
              'RHETORICAL_SELF_ANSWER',
              'BACKCHANNEL_INTERJECTION',
            ],
          },
          split_before_anchor: {
            type: 'string',
            description:
              '若為QA_SPEAKER_CHANGE，填寫回答句開頭的精確2至4個字作為切分錨點',
          },
          new_speaker_id: { type: 'string' },
        },
        required: [
          'utterance_id',
          'pragmatic_type',
          'split_before_anchor',
          'new_speaker_id',
        ],
        additionalProperties: false,
      },
    },
  },
  required: ['slot_corrections', 'turn_splits'],
  additionalProperties: false,
};

/**
 * 檢查當前 GPU 是否支援 WebGPU `shader-f16` 擴充功能。
 * 若不支援 `shader-f16`，自動將 `-q4f16_1-MLC` 模型識別碼轉換為 `-q4f32_1-MLC`。
 */
async function resolveFp16CompatibleModelId(
  requestedModelId: string
): Promise<string> {
  const baseId = requestedModelId || 'gemma-2-2b-it-q4f16_1-MLC';
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as unknown as { gpu?: any }).gpu;
    if (!gpu || typeof gpu.requestAdapter !== 'function') {
      return baseId.replace('-q4f16_1-MLC', '-q4f32_1-MLC');
    }
    const adapter = await gpu.requestAdapter();
    const supportsF16 = Boolean(
      adapter?.features &&
        typeof adapter.features.has === 'function' &&
        adapter.features.has('shader-f16')
    );
    if (!supportsF16 && baseId.includes('-q4f16_1-MLC')) {
      const f32Id = baseId.replace('-q4f16_1-MLC', '-q4f32_1-MLC');
      console.info(
        `ℹ️ [WebLLM] 顯卡不支援 shader-f16 (fp16)，自動切換至相容模型: ${f32Id}`
      );
      return f32Id;
    }
    return baseId;
  } catch {
    return baseId.replace('-q4f16_1-MLC', '-q4f32_1-MLC');
  }
}

async function ensureLlmEngine(modelId: string): Promise<MLCEngine> {
  const resolvedId = await resolveFp16CompatibleModelId(modelId);
  if (llmEngine && currentModelId === resolvedId) {
    return llmEngine;
  }

  if (llmEngine) {
    try {
      await llmEngine.unload();
    } catch {
      // ignore
    }
  }

  const progressCb = (report: { progress?: number; text?: string }) => {
    self.postMessage({
      type: 'LLM_LOAD_PROGRESS',
      progress: report.progress ?? 0,
      text: report.text || '正在載入本機語意與槽位校正模型...',
    });
  };

  try {
    llmEngine = await CreateMLCEngine(resolvedId, {
      initProgressCallback: progressCb,
    });
    currentModelId = resolvedId;
    return llmEngine;
  } catch (err) {
    const msg = String((err as Error)?.message || err);
    if (
      (msg.includes('f16') || msg.includes('fp16')) &&
      resolvedId.includes('-q4f16_1-MLC')
    ) {
      const fallbackF32Id = resolvedId.replace('-q4f16_1-MLC', '-q4f32_1-MLC');
      console.warn(
        `⚠️ [WebLLM] 載入 ${resolvedId} 遇到 fp16 限制，自動改載入 ${fallbackF32Id}`
      );
      llmEngine = await CreateMLCEngine(fallbackF32Id, {
        initProgressCallback: progressCb,
      });
      currentModelId = fallbackF32Id;
      return llmEngine;
    }
    throw err;
  }
}

function isSegmentAcousticallyLocked(seg: Segment): boolean {
  const dur = seg.end - seg.start;
  const margin = seg.acousticMargin ?? 0.18;
  return margin >= 0.16 && dur >= 1.2;
}

function fallbackBuildSlottedView(segments: Segment[]): {
  transcript: string;
  slotMap: Map<string, SuspectSlotItem>;
} {
  const slotMap = new Map<string, SuspectSlotItem>();
  let counter = 1;
  const lines = segments.map((seg) => {
    const isLocked = isSegmentAcousticallyLocked(seg);
    const gateTag = isLocked
      ? `[LOCKED | speaker=${seg.speakerId}]`
      : `[REVIEW | top1=${seg.speakerId}, top2=${
          seg.runnerUpSpeakerId || seg.speakerId
        }]`;
    const rendered = seg.words
      .map((w, idx) => {
        if (
          counter <= 5 &&
          (w.status === 'semantic_uncertain' ||
            (w.candidates && w.candidates.length > 0))
        ) {
          const slotId = `#S${counter++}`;
          slotMap.set(slotId, {
            slotId,
            segmentId: seg.id,
            startWordIdx: idx,
            endWordIdx: idx + 1,
            originalText: w.text,
            leftContext: seg.words[idx - 1]?.text || '',
            rightContext: seg.words[idx + 1]?.text || '',
            anomalyScore: 0.7,
            anomalyType: 'phonetic_gravity_tension',
            candidates: w.candidates || [],
          });
          return `⟪${slotId}: ${w.text} | 候選: ${(w.candidates || []).join(
            ', '
          )}⟫`;
        }
        return w.text;
      })
      .join('');
    return `[${seg.id}] ${gateTag} ${rendered}`;
  });
  return { transcript: lines.join('\n'), slotMap };
}

self.onmessage = async (event: MessageEvent) => {
  const data = event.data || {};

  if (data.type === 'UNLOAD_LLM' || data.type === 'DISPOSE') {
    if (llmEngine) {
      try {
        await llmEngine.unload();
      } catch {
        // ignore
      }
      llmEngine = null;
      currentModelId = null;
    }
    self.postMessage({ type: 'LLM_UNLOAD_DONE' });
    return;
  }

  if (data.type === 'CORRECT_TRI_LEVEL_WINDOW') {
    const { modelId, payload } = data as {
      modelId: string;
      payload: TriLevelContextPayload;
    };

    try {
      const engine = await ensureLlmEngine(
        modelId || 'gemma-2-2b-it-q4f16_1-MLC'
      );

      const targetSegments = payload.highAttentionTargetSegments || [];
      if (targetSegments.length === 0) {
        self.postMessage({ type: 'LLM_CORRECTION_SUCCESS', patches: [] });
        return;
      }

      const slotMap = new Map<string, SuspectSlotItem>();
      if (Array.isArray(payload.suspectSlots)) {
        for (const s of payload.suspectSlots) {
          slotMap.set(s.slotId, s);
        }
      }

      let slottedTranscript = payload.slottedPromptTranscript || '';
      if (!slottedTranscript) {
        const fb = fallbackBuildSlottedView(targetSegments);
        slottedTranscript = fb.transcript;
        fb.slotMap.forEach((v, k) => slotMap.set(k, v));
      }

      const hotwords =
        payload.lowAttentionGlobalMemory.verifiedTerms.join('、');

      const systemPrompt = `你是台灣政商與學術會議逐字稿之「免索引槽位校對與 LG-TPST 對話語用引擎」。
請針對目標段落中標記為 ⟪#S1: 原詞 | 候選: ...⟫ 的懷疑槽位與 [REVIEW] 語者段落進行推敲，並嚴格輸出符合 Schema 之 JSON 物件：
1. 【槽位填空校對 (slot_corrections)】：
   - 先在 brief_reason 寫下 15 字以內的上下文搭配理由（微推理 Micro-CoT）。
   - 若候選詞或近音詞能讓整句符合【全局會議背景】與前後文邏輯，將 action 設為 "APPLY_HIGH_CONF"，並在 selected_text 僅填入該槽位替換後的詞彙本身（嚴禁包含槽位外的前後文字！）。
   - 若有 2~3 個近音詞皆合理但無法 100% 肯定，將 action 設為 "MARK_UNCERTAIN"，並在 uncertain_candidates 列出 2~3 個候選詞。
   - 若原詞已完全通順合理，將 action 設為 "KEEP_ORIGINAL"。
2. 【語用拆句與語者移轉 (turn_splits)】：
   - [LOCKED] 段落嚴禁變更語者；[REVIEW | top1=..., top2=...] 段落若出現「前句提問、後句直接回答卻標同一人（QA_SPEAKER_CHANGE）」或「短促附和插話（BACKCHANNEL_INTERJECTION）」，請填入 new_speaker_id（如 top2 語者）與回答開頭 2~4 字錨點 split_before_anchor。`;

      const userPrompt = `【全局會議背景與摘要】：
${payload.lowAttentionGlobalMemory.docSummary || '無'}

【全場高主權實體與領域熱詞】：
${hotwords || '無'}

【前文對話脈絡 (唯讀)】：
${payload.mediumAttentionRollingText.slice(-600) || '無'}

【待執行槽位填空與語用檢驗之目標段落】：
${slottedTranscript}`;

      // 優先啟用 WebLLM 內建 XGrammar 結構化 JSON Schema 約束解碼
      let rawContent = '{}';
      try {
        const reply = await engine.chat.completions.create({
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0.1,
          max_tokens: 650,
          response_format: {
            type: 'json_object',
            schema: JSON.stringify(HYBTRANS_SLOT_GER_JSON_SCHEMA),
          },
        });
        rawContent = reply.choices?.[0]?.message?.content || '{}';
      } catch {
        const fallbackReply = await engine.chat.completions.create({
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0.1,
          max_tokens: 650,
        });
        rawContent = fallbackReply.choices?.[0]?.message?.content || '{}';
      }

      const patches: CorrectionPatchItem[] = [];
      const validSpeakerIds = new Set<string>();
      for (const s of targetSegments) {
        if (s.speakerId) validSpeakerIds.add(s.speakerId);
        if (s.runnerUpSpeakerId) validSpeakerIds.add(s.runnerUpSpeakerId);
      }

      const objMatch = rawContent.match(/\{[\s\S]*\}/);
      if (objMatch) {
        const parsed = JSON.parse(objMatch[0]) as Partial<StructuredSlotGerResponse>;

        // 1. 解析免索引槽位修正 (slot_corrections)
        if (Array.isArray(parsed.slot_corrections)) {
          for (const sc of parsed.slot_corrections) {
            const slot = slotMap.get(sc.slot_id);
            if (!slot) continue;
            if (sc.action === 'KEEP_ORIGINAL') continue;

            const cleanSelected = toTwp((sc.selected_text || '').trim());
            const cleanReason = toTwp(
              sc.brief_reason || '依槽位候選與上下文語意推敲修正'
            );
            const cleanCandidates = Array.isArray(sc.uncertain_candidates)
              ? sc.uncertain_candidates
                  .map((c) => toTwp(String(c).trim()))
                  .filter(Boolean)
              : slot.candidates;

            if (
              sc.action === 'APPLY_HIGH_CONF' &&
              cleanSelected &&
              cleanSelected !== slot.originalText &&
              Math.abs(cleanSelected.length - slot.originalText.length) <= 2
            ) {
              patches.push({
                segmentId: slot.segmentId,
                wordIndex: slot.startWordIdx,
                endWordIndex: slot.endWordIdx,
                originalText: slot.originalText,
                status: 'auto_corrected',
                correctedText: cleanSelected,
                candidates: cleanCandidates,
                reason: `[${sc.slot_id}] ${cleanReason}`,
              });
            } else if (sc.action === 'MARK_UNCERTAIN') {
              patches.push({
                segmentId: slot.segmentId,
                wordIndex: slot.startWordIdx,
                endWordIndex: slot.endWordIdx,
                originalText: slot.originalText,
                status: 'semantic_uncertain',
                candidates:
                  cleanCandidates.length > 0 ? cleanCandidates : slot.candidates,
                reason: `[${sc.slot_id}] ${cleanReason}`,
                requireRelisten: false,
              });
            }
          }
        }

        // 2. 解析 LG-TPST 語用拆句與語者移轉 (turn_splits)
        if (Array.isArray(parsed.turn_splits)) {
          for (const ts of parsed.turn_splits) {
            if (
              !ts.utterance_id ||
              ts.pragmatic_type === 'NO_SPLIT' ||
              ts.pragmatic_type === 'RHETORICAL_SELF_ANSWER'
            ) {
              continue;
            }
            const targetSeg = targetSegments.find(
              (s) => s.id === ts.utterance_id
            );
            if (!targetSeg || isSegmentAcousticallyLocked(targetSeg)) continue;

            const candidateSpk = (ts.new_speaker_id || '').trim();
            const isValidSpk =
              validSpeakerIds.has(candidateSpk) ||
              /^Speaker_\d{2}$/.test(candidateSpk);
            if (!isValidSpk || candidateSpk === targetSeg.speakerId) continue;

            patches.push({
              segmentId: targetSeg.id,
              wordIndex: -1,
              originalText: targetSeg.speakerId,
              status: 'auto_corrected',
              newSpeakerId: candidateSpk,
              splitBeforeAnchor: toTwp((ts.split_before_anchor || '').trim()),
              reason: `LG-TPST 語用校正 (${ts.pragmatic_type}: ${targetSeg.speakerId} → ${candidateSpk})`,
            });
          }
        }
      }

      self.postMessage({
        type: 'LLM_CORRECTION_SUCCESS',
        patches,
      });
    } catch (err) {
      self.postMessage({
        type: 'LLM_CORRECTION_ERROR',
        error: (err as Error).message,
        patches: [],
      });
    }
  }
};