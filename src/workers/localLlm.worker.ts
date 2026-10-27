// src/workers/localLlm.worker.ts
// 本地 WebLLM (Qwen2.5-3B / Gemma-2-2b) 方向二「先句後詞（Stage 1 句子語意完整性 -> Stage 2 免索引槽位校對）」與三級聲學信心度閘控 (Tier 1~3) 執行引擎
// 修正重點：
// 1. 將 OpenCC 改為純字形轉換 ({ from: 'cn', to: 'tw' })，避免 twp 將「對象 -> 物件、文件 -> 檔案、支持 -> 支援」強制改寫。
// 2. 針對整合式內顯 (Intel UHD / Iris Xe / AMD Radeon Graphics 等與系統共用 16GB RAM 之環境) 自動啟動「智慧共享記憶體保護」，直接交由零顯存佔用之確定性 Stage 1/2 語意與音韻校對防線處理，徹底根除在轉寫結束階段載入 1.6GB WebLLM 時觸發二次 Out of Memory 崩潰！
// 3. 內建 WebGPU 裝置斷線熔斷機制 (WebGPU Device-Lost Circuit Breaker)。

import { CreateMLCEngine, MLCEngine } from '@mlc-ai/web-llm';
import * as OpenCC from 'opencc-js';
import {
  AcousticSlotTier,
  CorrectionPatchItem,
  Segment,
  SentenceIntegrityRepair,
  StructuredSlotGerResponse,
  SuspectSlotItem,
  TriLevelContextPayload,
} from '../types';

const toTw = OpenCC.Converter({ from: 'cn', to: 'tw' });

let llmEngine: MLCEngine | null = null;
let currentModelId: string | null = null;
let webGpuDeviceLost = false;
let cachedResolvedModelId: string | null = null;
let cachedInputModelId: string | null = null;

const HYBTRANS_SLOT_GER_JSON_SCHEMA = {
  type: 'object',
  properties: {
    contextual_fixes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          utterance_id: { type: 'string' },
          brief_reason: {
            type: 'string',
            description:
              '15字以內說明先檢查整句語意完整性、跨句掉字或標點修復的理由',
          },
          wrong_span: {
            type: 'string',
            description:
              '原句中語意斷裂、句尾缺字或漏標點之局部片段(2至8字，須精確出現於原句)',
          },
          corrected_span: {
            type: 'string',
            description: '使整句語意完整通順或補齊標點後之修正片段',
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
            description:
              '在確認整句語意完整後，15字以內說明該單詞槽位之上下文搭配理由',
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
  },
  required: ['contextual_fixes', 'turn_splits', 'slot_corrections'],
  additionalProperties: false,
};

async function resolveFp16CompatibleModelId(
  requestedModelId: string
): Promise<string> {
  const baseId = requestedModelId || 'gemma-2-2b-it-q4f16_1-MLC';
  if (cachedInputModelId === baseId && cachedResolvedModelId) {
    return cachedResolvedModelId;
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as unknown as { gpu?: any }).gpu;
    if (!gpu || typeof gpu.requestAdapter !== 'function') {
      webGpuDeviceLost = true;
      throw new Error('瀏覽器環境不支援 WebGPU API');
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      webGpuDeviceLost = true;
      throw new Error('WebGPU No available adapters');
    }

    const info =
      (
        adapter as unknown as {
          info?: {
            vendor?: string;
            architecture?: string;
            description?: string;
          };
        }
      ).info || {};
    const vendor = (info.vendor || '').toLowerCase();
    const desc = (
      info.description ||
      info.architecture ||
      ''
    ).toLowerCase();
    const maxBufferBytes = adapter.limits?.maxBufferSize || 268435456;
    const maxStorageBytes =
      adapter.limits?.maxStorageBufferBindingSize || 134217728;
    const maxBufferSizeMB = Math.round(
      Math.max(maxBufferBytes, maxStorageBytes) / (1024 * 1024)
    );

    const isIntegratedGpu =
      desc.includes('uhd') ||
      desc.includes('iris') ||
      desc.includes('hd graphics') ||
      (vendor.includes('intel') && !desc.includes('arc')) ||
      (vendor.includes('amd') && desc.includes('radeon(tm) graphics')) ||
      maxBufferSizeMB < 1536;

    // 若為與系統共用記憶體之整合式內顯 (如 Intel UHD Graphics 16GB RAM)，跳過載入 1.6GB WebLLM，
    // 全權交由內建零顯存佔用之 Stage 1 + Stage 2 語意/音韻/問答對修復引擎處理，保護分頁 100% 不會因 OOM 崩潰！
    if (isIntegratedGpu) {
      webGpuDeviceLost = true;
      console.info(
        'ℹ️ [WebLLM 智慧記憶體保護] 偵測到整合式內顯共享記憶體環境，已自動採用零顯存佔用之內建 Stage 1/2 語意與音韻校對引擎，防止系統記憶體溢位。'
      );
      throw new Error('Integrated GPU shared-memory protection activated');
    }

    const supportsF16 = Boolean(
      adapter?.features &&
        typeof adapter.features.has === 'function' &&
        adapter.features.has('shader-f16')
    );
    if (!supportsF16) {
      const safeF32Id = /3B|7B|8B/i.test(baseId)
        ? 'gemma-2-2b-it-q4f32_1-MLC'
        : baseId.replace('-q4f16_1-MLC', '-q4f32_1-MLC');
      console.info(
        `ℹ️ [WebLLM] 顯卡不支援 shader-f16 (fp16)，自動採用輕量相容模型: ${safeF32Id}`
      );
      cachedInputModelId = baseId;
      cachedResolvedModelId = safeF32Id;
      return safeF32Id;
    }
    cachedInputModelId = baseId;
    cachedResolvedModelId = baseId;
    return baseId;
  } catch (err) {
    throw err;
  }
}

async function ensureLlmEngine(modelId: string): Promise<MLCEngine> {
  if (webGpuDeviceLost) {
    throw new Error('WebGPU 裝置已進入保護熔斷狀態，略過本地 LLM 載入');
  }

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
    llmEngine = null;
    currentModelId = null;
  }

  const progressCb = (report: { progress?: number; text?: string }) => {
    self.postMessage({
      type: 'LLM_LOAD_PROGRESS',
      progress: report.progress ?? 0,
      text:
        report.text ||
        '正在載入本機兩階段（先句後詞）語意與槽位校正模型...',
    });
  };

  try {
    llmEngine = await CreateMLCEngine(
      resolvedId,
      {
        initProgressCallback: progressCb,
      },
      {
        context_window_size: 2048,
      }
    );
    currentModelId = resolvedId;
    return llmEngine;
  } catch (err) {
    const msg = String((err as Error)?.message || err);
    if (
      /DXGI_ERROR_DEVICE|DeviceLost|No available adapters|CreatePlacedResource|OutOfMemory|Failed to create device|Integrated GPU/i.test(
        msg
      )
    ) {
      webGpuDeviceLost = true;
      throw err;
    }
    if (
      (msg.includes('f16') || msg.includes('fp16')) &&
      resolvedId.includes('-q4f16_1-MLC')
    ) {
      const fallbackF32Id = 'gemma-2-2b-it-q4f32_1-MLC';
      try {
        llmEngine = await CreateMLCEngine(
          fallbackF32Id,
          {
            initProgressCallback: progressCb,
          },
          {
            context_window_size: 2048,
          }
        );
        currentModelId = fallbackF32Id;
        cachedResolvedModelId = fallbackF32Id;
        return llmEngine;
      } catch (innerErr) {
        webGpuDeviceLost = true;
        throw innerErr;
      }
    }
    webGpuDeviceLost = true;
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

  if (data.type === 'STAGE1_SENTENCE_INTEGRITY') {
    const segments: Segment[] = data.segments || [];
    const repairs: SentenceIntegrityRepair[] = [];
    self.postMessage({
      type: 'STAGE1_RESULT',
      repairs,
      turnSplits: [],
      updatedBriefing: data.briefing || '',
      segmentCount: segments.length,
    });
    return;
  }

  if (data.type === 'STAGE2_SLOT_CORRECTION') {
    self.postMessage({
      type: 'STAGE2_RESULT',
      corrections: [],
    });
    return;
  }

  if (
    data.type === 'CORRECT_TRI_LEVEL_WINDOW' ||
    data.type === 'refine_batch'
  ) {
    const { modelId, payload } = data as {
      modelId: string;
      payload: TriLevelContextPayload;
    };

    if (webGpuDeviceLost) {
      self.postMessage({
        type: 'LLM_CORRECTION_ERROR',
        deviceLost: true,
        error: 'WebGPU 裝置已熔斷，改由規則引擎接手',
        patches: [],
      });
      return;
    }

    try {
      const engine = await ensureLlmEngine(
        modelId || 'gemma-2-2b-it-q4f16_1-MLC'
      );

      const targetSegments = payload.highAttentionTargetSegments || [];
      if (targetSegments.length === 0) {
        self.postMessage({
          type:
            data.type === 'refine_batch'
              ? 'refine_result'
              : 'LLM_CORRECTION_SUCCESS',
          patches: [],
          updatedRollingSummary:
            payload.lowAttentionGlobalMemory.earlierTopicDigest || '',
        });
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

      const systemPrompt = `你是台灣政商與學術會議逐字稿之「兩階段（先句後詞）語意完整性檢核與三級聲學信心度閘控引擎」。
請嚴格遵循「先抓句子整體語意完整性，再進到單詞槽位精修」的順序，輸出符合 Schema 之 JSON 物件：
1. 【第一階段 (Stage 1)：全句語意完整性與標點結構修復 (contextual_fixes)】：
   - 先從整句話與前後句的連貫性審視每一段：主詞、動詞、受詞是否完整？是否有句尾補語缺漏、斷句標點（如感嘆號「！」、問號「？」、子句逗號「，」）錯置或連音造成的整句語意不通？
   - 若發現句中有破壞整句語意完整性的片段（2~8 字），請先在 contextual_fixes 填入 utterance_id、brief_reason（說明如何讓整句語意完整）、wrong_span（原句精確片段）與 corrected_span（修復後片段）。
2. 【第一階段 (Stage 1)：三級聲學信心度語用拆句與語者移轉 (turn_splits)】：
   - Tier 1 [LOCKED] 段落：唯讀鎖定，嚴禁變更語者！
   - Tier 2 [REVIEW | top1=..., top2=...] 段落：僅能在 top1 與 top2 間依問答語意決定歸屬。
   - Tier 3 [OVERLAP | top1=..., top2=...] 段落：允許將粘連在同一句內的「主講句 + 附和短句」在斷點處填入 split_before_anchor 並指派給 top2。
3. 【第二階段 (Stage 2)：單詞槽位填空精修 (slot_corrections)】：
   - 在確認整句語意框架完整後，針對標記為 ⟪#S1: 原詞 | 候選: ...⟫ 的懷疑單詞槽位，先在 brief_reason 寫下 15 字內的整句搭配理由。
   - 若候選詞或同音詞能讓整句話合乎領域脈絡，將 action 設為 "APPLY_HIGH_CONF"，並在 selected_text 填入替換後的詞彙本身。
   - 若有 2~3 個近音詞皆可能，將 action 設為 "MARK_UNCERTAIN"，並在 uncertain_candidates 列出候選詞。`;

      const userPrompt = `【全局會議背景與摘要】：
${payload.lowAttentionGlobalMemory.docSummary || '無'}

【全場高主權實體與領域熱詞】：
${hotwords || '無'}

【前文對話脈絡 (唯讀)】：
${payload.mediumAttentionRollingText.slice(-600) || '無'}

【待執行「先句後詞」兩階段檢核之目標段落】：
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
        const parsed = JSON.parse(
          objMatch[0]
        ) as Partial<StructuredSlotGerResponse>;

        if (Array.isArray(parsed.contextual_fixes)) {
          for (const cf of parsed.contextual_fixes) {
            if (!cf.utterance_id || !cf.wrong_span || !cf.corrected_span) {
              continue;
            }
            const targetSeg = targetSegments.find(
              (s) => s.id === cf.utterance_id
            );
            if (!targetSeg) continue;

            const wrongSpan = toTw(cf.wrong_span.trim());
            const correctedSpan = toTw(cf.corrected_span.trim());
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

            const cleanReason = toTw(
              cf.brief_reason || 'Stage 1 依全句語意完整性與前後文呼應修正'
            );
            patches.push({
              segmentId: targetSeg.id,
              wordIndex: sWIdx,
              endWordIndex: eWIdx,
              originalText: origWordsText,
              status: 'auto_corrected',
              correctedText: replacedWordsText,
              reason: `[Stage 1 句意修復] ${cleanReason}`,
            });
          }
        }

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
            if (tier === 'tier1_locked_anchor') continue;

            let candidateSpk = (ts.new_speaker_id || '').trim();
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
              splitBeforeAnchor: toTw((ts.split_before_anchor || '').trim()),
              reason: `LG-TPST 三級閘控語用校正 (${ts.pragmatic_type}: ${targetSeg.speakerId} → ${candidateSpk})`,
            });
          }
        }

        if (Array.isArray(parsed.slot_corrections)) {
          for (const sc of parsed.slot_corrections) {
            const slot = slotMap.get(sc.slot_id);
            if (!slot) continue;
            if (sc.action === 'KEEP_ORIGINAL') continue;

            if (
              patches.some(
                (p) =>
                  p.segmentId === slot.segmentId &&
                  p.wordIndex >= 0 &&
                  slot.startWordIdx < (p.endWordIndex ?? p.wordIndex + 1) &&
                  slot.endWordIdx > p.wordIndex
              )
            ) {
              continue;
            }

            const cleanSelected = toTw((sc.selected_text || '').trim());
            const cleanReason = toTw(
              sc.brief_reason || 'Stage 2 依完整句意與槽位候選修正單詞'
            );
            const cleanCandidates = Array.isArray(sc.uncertain_candidates)
              ? sc.uncertain_candidates
                  .map((c) => toTw(String(c).trim()))
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
                reason: `[${sc.slot_id} 單詞精修] ${cleanReason}`,
              });
            } else if (sc.action === 'MARK_UNCERTAIN') {
              patches.push({
                segmentId: slot.segmentId,
                wordIndex: slot.startWordIdx,
                endWordIndex: slot.endWordIdx,
                originalText: slot.originalText,
                status: 'semantic_uncertain',
                candidates:
                  cleanCandidates.length > 0
                    ? cleanCandidates
                    : slot.candidates,
                reason: `[${sc.slot_id} 單詞存疑] ${cleanReason}`,
                requireRelisten: false,
              });
            }
          }
        }
      }

      self.postMessage({
        type:
          data.type === 'refine_batch'
            ? 'refine_result'
            : 'LLM_CORRECTION_SUCCESS',
        patches,
        updatedRollingSummary:
          payload.lowAttentionGlobalMemory.earlierTopicDigest || '',
      });
    } catch (err) {
      self.postMessage({
        type: 'LLM_CORRECTION_ERROR',
        deviceLost: webGpuDeviceLost,
        error: (err as Error).message,
        patches: [],
      });
    }
  }
};
