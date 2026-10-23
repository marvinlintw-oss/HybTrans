// src/workers/localLlm.worker.ts
// 本地 WebLLM (Qwen2.5-3B / Gemma-2-2B) 免索引槽位校對與三級聲學信心度閘控 (Tier 1~3) 執行引擎

import { CreateMLCEngine, MLCEngine } from '@mlc-ai/web-llm';
import * as OpenCC from 'opencc-js';
import {
  AcousticSlotTier,
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
 * 針對 WebLLM XGrammar 最佳化之嚴格 JSON Schema
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
              '若為QA_SPEAKER_CHANGE或BACKCHANNEL_INTERJECTION，填寫回答句或附和句開頭的精確2至4個字作為切分錨點',
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
    contextual_fixes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          utterance_id: { type: 'string' },
          brief_reason: {
            type: 'string',
            description: '15字以內說明根據前文脈絡修正詞彙或標點的理由',
          },
          wrong_span: {
            type: 'string',
            description:
              '原句中與上下文不符或漏標點之局部片段(2至8字，須精確出現於原句)',
          },
          corrected_span: {
            type: 'string',
            description: '呼應上下文脈絡或補齊標點後之修正片段',
          },
        },
        required: [
          'utterance_id',
          'brief_reason',
          'wrong_span',
          'corrected_span',
        ],
        additionalProperties: false,
      },
    },
  },
  required: ['slot_corrections', 'contextual_fixes', 'turn_splits'],
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

function getSegmentSlotTier(seg: Segment): AcousticSlotTier {
  if (seg.slotTier) return seg.slotTier;
  const dur = seg.end - seg.start;
  const logOdds =
    seg.logOddsMargin ??
    ((seg.acousticMargin ?? 0.18) >= 0.16 ? 2.0 : 1.0);
  const overlap = seg.overlapRatio ?? 0.05;

  if (overlap >= 0.25 || logOdds < 0.45) {
    return 'tier3_overlap_split';
  }
  if (logOdds > 1.8 && overlap < 0.15 && dur >= 1.0) {
    return 'tier1_locked_anchor';
  }
  return 'tier2_candidate_constrained';
}

function fallbackBuildSlottedView(segments: Segment[]): {
  transcript: string;
  slotMap: Map<string, SuspectSlotItem>;
} {
  const slotMap = new Map<string, SuspectSlotItem>();
  let counter = 1;
  const lines = segments.map((seg) => {
    const tier = getSegmentSlotTier(seg);
    const runnerUp = seg.runnerUpSpeakerId || seg.speakerId;
    const gateTag =
      tier === 'tier1_locked_anchor'
        ? `[LOCKED | speaker=${seg.speakerId}]`
        : tier === 'tier3_overlap_split'
        ? `[OVERLAP | top1=${seg.speakerId}, top2=${runnerUp}]`
        : `[REVIEW | top1=${seg.speakerId}, top2=${runnerUp}]`;

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

      const systemPrompt = `你是台灣政商與學術會議逐字稿之「上下文脈絡語意校對、標點修復與三級聲學信心度閘控引擎」。
請結合【全局會議背景】、【前文對話脈絡】與【目標段落】，嚴格輸出符合 Schema 之 JSON 物件：
1. 【槽位填空校對 (slot_corrections)】：
   - 針對標記為 ⟪#S1: 原詞 | 候選: ...⟫ 的懷疑槽位，先在 brief_reason 寫下 15 字內的上下文呼應理由。
   - 若候選詞、同音詞或補標點版本能讓句子呼應前後文邏輯（例如前文討論送某人東西，後文「神秘如果我們再寄給他」應修正為「神秘禮物」或「神秘！如果」），將 action 設為 "APPLY_HIGH_CONF"，並在 selected_text 填入替換後的片段本身。
   - 若有 2~3 個近音詞皆可能，將 action 設為 "MARK_UNCERTAIN"，並在 uncertain_candidates 列出候選詞。
2. 【全句上下文語意與標點主動校對 (contextual_fixes)】：
   - 除了 #S1 槽位外，請主動檢視目標段落每一句是否與【前文對話脈絡】呼應、語意是否通順、標點符號（如感嘆號「！」、問號「？」、頓號或子句逗號「，」）是否缺漏或誤植。
   - 若發現句中有局部同音誤聽或標點粘連錯誤（2~8 字），請在 contextual_fixes 填入 utterance_id、brief_reason、wrong_span（原句中的精確片段）與 corrected_span（修正後的詞彙或補齊標點片段）。
3. 【三級聲學信心度語用拆句與語者移轉 (turn_splits)】：
   - Tier 1 [LOCKED] 段落：唯讀鎖定，嚴禁變更語者！
   - Tier 2 [REVIEW | top1=..., top2=...] 段落：僅能在 top1 與 top2 間依問答語意決定歸屬。
   - Tier 3 [OVERLAP | top1=..., top2=...] 段落：允許將粘連在同一句內的「主講句 + 附和短句」在斷點處填入 split_before_anchor 並指派給 top2。`;

      const userPrompt = `【全局會議背景與摘要】：
${payload.lowAttentionGlobalMemory.docSummary || '無'}

【全場高主權實體與領域熱詞】：
${hotwords || '無'}

【前文對話脈絡 (唯讀)】：
${payload.mediumAttentionRollingText.slice(-600) || '無'}

【待執行槽位填空與語用檢驗之目標段落】：
${slottedTranscript}`;

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
              Math.abs(cleanSelected.length - slot.originalText.length) <= 3
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

        // 2. 解析全句上下文語意與標點主動校對 (contextual_fixes)
        if (Array.isArray(parsed.contextual_fixes)) {
          for (const cf of parsed.contextual_fixes) {
            if (!cf.utterance_id || !cf.wrong_span || !cf.corrected_span) {
              continue;
            }
            const targetSeg = targetSegments.find(
              (s) => s.id === cf.utterance_id
            );
            if (!targetSeg) continue;

            const wrongSpan = toTwp(cf.wrong_span.trim());
            const correctedSpan = toTwp(cf.corrected_span.trim());
            if (
              !wrongSpan ||
              !correctedSpan ||
              wrongSpan === correctedSpan ||
              wrongSpan.length < 2 ||
              Math.abs(correctedSpan.length - wrongSpan.length) > 3
            ) {
              continue;
            }

            let charPos = 0;
            const wRanges = targetSeg.words.map((w, idx) => {
              const start = charPos;
              charPos += w.text.length;
              return { idx, start, end: charPos, text: w.text };
            });
            const fullText = targetSeg.words.map((w) => w.text).join('');
            const hitCharIdx = fullText.indexOf(wrongSpan);
            if (hitCharIdx === -1) continue;

            const hitEndCharIdx = hitCharIdx + wrongSpan.length;
            const hitWords = wRanges.filter(
              (r) => r.end > hitCharIdx && r.start < hitEndCharIdx
            );
            if (hitWords.length === 0) continue;

            const sWIdx = hitWords[0].idx;
            const eWIdx = hitWords[hitWords.length - 1].idx + 1;
            const origWordsText = targetSeg.words
              .slice(sWIdx, eWIdx)
              .map((w) => w.text)
              .join('');
            const replacedWordsText = origWordsText.replace(
              wrongSpan,
              correctedSpan
            );

            // 避免與已經由 #S1 槽位修正的區段重複
            if (
              patches.some(
                (p) =>
                  p.segmentId === targetSeg.id &&
                  p.wordIndex >= 0 &&
                  sWIdx < (p.endWordIndex ?? p.wordIndex + 1) &&
                  eWIdx > p.wordIndex
              )
            ) {
              continue;
            }

            const cleanReason = toTwp(
              cf.brief_reason || '依前後文脈絡與語意呼應修正'
            );
            patches.push({
              segmentId: targetSeg.id,
              wordIndex: sWIdx,
              endWordIndex: eWIdx,
              originalText: origWordsText,
              status: 'auto_corrected',
              correctedText: replacedWordsText,
              reason: `[AI 脈絡校對] ${cleanReason}`,
            });
          }
        }

        // 3. 解析三級聲學信心度閘控之語用拆句與語者移轉 (turn_splits)
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
            if (!targetSeg) continue;

            const tier = getSegmentSlotTier(targetSeg);
            // Tier 1 唯讀鎖定：嚴禁 LLM 修改語者
            if (tier === 'tier1_locked_anchor') continue;

            let candidateSpk = (ts.new_speaker_id || '').trim();
            // Tier 2 二選一受限候選集：強制約束於 runnerUpSpeakerId
            if (
              tier === 'tier2_candidate_constrained' &&
              targetSeg.runnerUpSpeakerId &&
              targetSeg.runnerUpSpeakerId !== targetSeg.speakerId
            ) {
              candidateSpk = targetSeg.runnerUpSpeakerId;
            }

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
              reason: `LG-TPST 三級閘控語用校正 (${ts.pragmatic_type}: ${targetSeg.speakerId} → ${candidateSpk})`,
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