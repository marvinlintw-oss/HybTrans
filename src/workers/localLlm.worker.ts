// src/workers/localLlm.worker.ts
import { CreateMLCEngine, MLCEngine } from '@mlc-ai/web-llm';
import * as OpenCC from 'opencc-js';
import {
  CorrectionPatchItem,
  LocalLlmModel,
  TriLevelContextPayload,
} from '../types';

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

let engine: MLCEngine | null = null;
let currentModelId: LocalLlmModel | null = null;

async function ensureLlmEngine(modelId: LocalLlmModel): Promise<MLCEngine> {
  if (engine && currentModelId === modelId) return engine;
  if (engine) {
    try {
      await engine.unload();
    } catch {
      // ignore unload error
    }
  }
  engine = await CreateMLCEngine(modelId, {
    initProgressCallback: (report) => {
      self.postMessage({
        type: 'LLM_LOAD_PROGRESS',
        progress: report.progress,
        text: report.text,
      });
    },
  });
  currentModelId = modelId;
  return engine;
}

self.onmessage = async (event: MessageEvent) => {
  const data = event.data;

  if (data.type === 'UNLOAD_LLM') {
    if (engine) {
      try {
        await engine.unload();
      } catch {
        // ignore
      }
      engine = null;
      currentModelId = null;
    }
    self.postMessage({ type: 'UNLOAD_LLM_DONE' });
    return;
  }

  if (data.type === 'CORRECT_TRI_LEVEL_WINDOW') {
    const { modelId, payload } = data as {
      modelId: LocalLlmModel;
      payload: TriLevelContextPayload;
    };

    try {
      const llm = await ensureLlmEngine(modelId);
      const compactTargets = payload.highAttentionTargetSegments.map((seg) => ({
        segmentId: seg.id,
        words: seg.words.map((w, idx) => ({ i: idx, t: w.text, c: w.confidence })),
      }));

      const systemPrompt =
        '你是台灣正體中文會議逐字稿校對專家。請根據三層語境，只針對【第3層：高關注度目標區】內的同音錯字輸出 JSON Array 格式的修正補丁。若無錯字請輸出 []。';

      const userPrompt = `【第1層：低關注度背景與熱詞】
摘要：${payload.lowAttentionGlobalMemory.docSummary || '無'}
熱詞表：${payload.lowAttentionGlobalMemory.verifiedTerms.slice(0, 45).join('、') || '無'}
前情提要：${payload.lowAttentionGlobalMemory.earlierTopicDigest || '無'}

【第2層：中關注度前文 (唯讀參考，禁止修改此區)】
${payload.mediumAttentionRollingText.slice(-1200) || '(會議開頭)'}

【第3層：高關注度目標區 (唯一可修改區)】
${JSON.stringify(compactTargets)}

請只輸出 JSON Array，格式為：
[{"segmentId":"...","wordIndex":0,"originalText":"...","status":"auto_corrected","correctedText":"...","reason":"...","requireRelisten":false}]`;

      const reply = await llm.chat.completions.create({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.1,
        max_tokens: 512,
      });

      const content = reply.choices[0]?.message?.content || '[]';
      const jsonMatch = content.match(/\[[\s\S]*\]/);
      const rawJson = jsonMatch ? jsonMatch[0] : '[]';
      const parsed = JSON.parse(rawJson) as CorrectionPatchItem[];

      const patches: CorrectionPatchItem[] = parsed.map((p) => ({
        ...p,
        correctedText: p.correctedText ? toTwp(p.correctedText) : undefined,
        candidates: p.candidates?.map((c) => toTwp(c)),
        reason: toTwp(p.reason || '本機語境推論修正'),
      }));

      self.postMessage({
        type: 'LLM_CORRECTION_SUCCESS',
        patches,
      });
    } catch (err) {
      self.postMessage({
        type: 'LLM_CORRECTION_ERROR',
        error: (err as Error).message || '本機 WebLLM 校正失敗',
      });
    }
  }
};