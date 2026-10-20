// src/workers/localLlm.worker.ts
import { CreateMLCEngine, MLCEngine } from '@mlc-ai/web-llm';
import * as OpenCC from 'opencc-js';
import {
  CorrectionPatchItem,
  Segment,
  TriLevelContextPayload,
} from '../types';

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

let llmEngine: MLCEngine | null = null;
let currentModelId: string | null = null;

async function ensureLlmEngine(modelId: string): Promise<MLCEngine> {
  if (llmEngine && currentModelId === modelId) {
    return llmEngine;
  }

  if (llmEngine) {
    try {
      await llmEngine.unload();
    } catch {
      // ignore
    }
  }

  llmEngine = await CreateMLCEngine(modelId, {
    initProgressCallback: (report) => {
      self.postMessage({
        type: 'LLM_LOAD_PROGRESS',
        progress: report.progress,
        text: report.text || '正在載入本機語意與角色校正模型...',
      });
    },
  });

  currentModelId = modelId;
  return llmEngine;
}

/**
 * 將目標視窗段落轉換為帶 [LOCKED] / [REVIEW | top1=..., top2=...] 聲學信心度閘門標記之結構化文本
 * 依據報告問題五規格：當 M(u_m) >= 0.16 且時長 >= 1.2s 時標記為 [LOCKED]，否則標記為 [REVIEW]
 */
function formatSegmentsWithAcousticConfidenceGate(segments: Segment[]): string {
  return segments
    .map((seg) => {
      const dur = seg.end - seg.start;
      const margin = seg.acousticMargin ?? 0.18;
      const isLocked = margin >= 0.16 && dur >= 1.2;
      const top1 = seg.speakerId;
      const top2 = seg.runnerUpSpeakerId || seg.speakerId;
      const gateTag = isLocked
        ? `[LOCKED | speaker=${top1}]`
        : `[REVIEW | top1=${top1}, top2=${top2}, margin=${margin.toFixed(2)}]`;
      const wordsStr = seg.words
        .map((w, idx) => `(${idx})${w.text}`)
        .join(' ');
      return `[${seg.id}] ${gateTag} ${wordsStr}`;
    })
    .join('\n');
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

      const gatedTranscript = formatSegmentsWithAcousticConfidenceGate(targetSegments);
      const hotwords = payload.lowAttentionGlobalMemory.verifiedTerms.join('、');

      const systemPrompt = `系統角色：多語系會議對話語用學（Discourse Pragmatics）與專有名詞同音校正引擎。
輸入格式：帶有段落索引 [seg_id]、聲學信心度閘門標記（[LOCKED] 或 [REVIEW]）與詞彙索引 (wordIndex) 的會議逐字稿。
請嚴格遵守以下零幻覺準則（TPST 原則）：
1. [LOCKED] 段落代表聲紋確鑿無疑，用來推論每位語者的「對話角色與知識立場」（如報告者 vs. 審查提問者）。
2. 僅針對明顯的「同音錯字（對照專有名詞庫與會議摘要）」產出詞彙修正 JSON，嚴禁改寫語氣、增刪字數或捏造原文沒有的句子。
3. 僅輸出合法的 JSON 陣列，格式為：
[{"segmentId":"seg_...","wordIndex":3,"correctedText":"正確詞","reason":"依會議語境與專有名詞庫修正同音字"}]
若無須修正，請直接輸出 []。`;

      const userPrompt = `【全局會議背景與摘要】：
${payload.lowAttentionGlobalMemory.docSummary || '無'}

【已確認專有名詞庫】：
${hotwords || '無'}

【前文對話脈絡】：
${payload.mediumAttentionRollingText.slice(-600) || '無'}

【待檢視之目標段落】：
${gatedTranscript}`;

      const reply = await engine.chat.completions.create({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.1,
        max_tokens: 512,
      });

      const rawContent = reply.choices?.[0]?.message?.content || '[]';
      const jsonMatch = rawContent.match(/\[[\s\S]*\]/);
      if (!jsonMatch) {
        self.postMessage({ type: 'LLM_CORRECTION_SUCCESS', patches: [] });
        return;
      }

      const parsed = JSON.parse(jsonMatch[0]) as Array<{
        segmentId?: string;
        wordIndex?: number;
        correctedText?: string;
        reason?: string;
      }>;

      const patches: CorrectionPatchItem[] = [];
      for (const item of parsed) {
        if (
          !item.segmentId ||
          typeof item.wordIndex !== 'number' ||
          !item.correctedText
        ) {
          continue;
        }
        const targetSeg = targetSegments.find((s) => s.id === item.segmentId);
        const targetWord = targetSeg?.words[item.wordIndex];
        if (!targetSeg || !targetWord) continue;

        const cleanRight = toTwp(item.correctedText.trim());
        // 零幻覺防護：修正詞長度必須與原詞相近，防止小模型整句重寫塞入單一 WordSpan
        if (
          cleanRight &&
          cleanRight !== targetWord.text &&
          Math.abs(cleanRight.length - targetWord.text.length) <= 2
        ) {
          patches.push({
            segmentId: item.segmentId,
            wordIndex: item.wordIndex,
            originalText: targetWord.text,
            status: 'auto_corrected',
            correctedText: cleanRight,
            reason: toTwp(item.reason || '本機 LLM 語境同音修正'),
          });
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