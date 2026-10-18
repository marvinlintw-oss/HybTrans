// src/services/byokCloudClient.ts
import * as OpenCC from 'opencc-js';
import {
  CorrectionPatchItem,
  Segment,
  TriLevelContextPayload,
  WordSpan,
} from '../types';

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

/**
 * 將純文字句子切分為 WordSpan[] 並線性插值對齊段落時間區間
 */
function buildCloudWordSpans(
  cleanText: string,
  segId: string,
  startSec: number,
  endSec: number,
  baseConfidence: number,
  acousticThreshold = 0.72
): WordSpan[] {
  const tokens =
    cleanText.match(/[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+|[^\s\u4e00-\u9fa5a-zA-Z0-9]/g) || [
      cleanText,
    ];
  const dur = Math.max(0.2, endSec - startSec);
  const step = dur / Math.max(1, tokens.length);

  return tokens.map((tok, idx) => {
    const isPunct = /^[^\u4e00-\u9fa5a-zA-Z0-9]+$/.test(tok);
    const conf = isPunct ? 0.99 : Number(baseConfidence.toFixed(2));
    return {
      id: `${segId}_w_${idx}`,
      text: tok,
      start: Number((startSec + idx * step).toFixed(2)),
      end: Number((startSec + (idx + 1) * step).toFixed(2)),
      confidence: conf,
      status: conf < acousticThreshold ? 'acoustic_uncertain' : 'normal',
    };
  });
}

/**
 * 【向下相容】使用 Groq Whisper API (whisper-large-v3-turbo) 進行高速音訊轉寫
 */
export async function transcribeWithGroq(
  wavBuffer: ArrayBuffer,
  apiKey: string,
  promptHint: string,
  timeOffsetSec: number,
  speakerId: string,
  chunkIndex: number,
  acousticThreshold = 0.72
): Promise<Segment[]> {
  if (!apiKey.trim()) return [];

  const formData = new FormData();
  const blob = new Blob([wavBuffer], { type: 'audio/wav' });
  formData.append('file', blob, `chunk_${chunkIndex}_${speakerId}.wav`);
  formData.append('model', 'whisper-large-v3-turbo');
  formData.append('language', 'zh');
  formData.append('response_format', 'verbose_json');
  if (promptHint) {
    formData.append('prompt', promptHint.slice(0, 220));
  }

  const response = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey.trim()}`,
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Groq Whisper API 請求失敗 (${response.status})`);
  }

  const data = await response.json();
  const rawSegments = Array.isArray(data?.segments) ? data.segments : [];

  if (rawSegments.length === 0 && typeof data?.text === 'string' && data.text.trim()) {
    const cleanText = toTwp(data.text.trim());
    const segId = `groq_c${chunkIndex}_${Date.now()}`;
    const startSec = Number(timeOffsetSec.toFixed(2));
    const endSec = Number((timeOffsetSec + (data.duration || 4.0)).toFixed(2));
    return [
      {
        id: segId,
        chunkIndex,
        speakerId,
        start: startSec,
        end: endSec,
        words: buildCloudWordSpans(cleanText, segId, startSec, endSec, 0.9, acousticThreshold),
        isCorrected: false,
        relistenCount: 0,
      },
    ];
  }

  return rawSegments
    .map((s: Record<string, unknown>, idx: number) => {
      const rawText = typeof s.text === 'string' ? s.text.trim() : '';
      if (!rawText) return null;
      const cleanText = toTwp(rawText);
      const segId = `groq_c${chunkIndex}_${Date.now()}_${idx}`;
      const startSec = Number((timeOffsetSec + Number(s.start || 0)).toFixed(2));
      const endSec = Number((timeOffsetSec + Number(s.end || Number(s.start || 0) + 2)).toFixed(2));
      const avgLogprob = typeof s.avg_logprob === 'number' ? s.avg_logprob : -0.25;
      const conf = Math.min(0.98, Math.max(0.5, Math.exp(avgLogprob)));

      return {
        id: segId,
        chunkIndex,
        speakerId,
        start: startSec,
        end: endSec,
        words: buildCloudWordSpans(cleanText, segId, startSec, endSec, conf, acousticThreshold),
        isCorrected: false,
        relistenCount: 0,
      } as Segment;
    })
    .filter((item: Segment | null): item is Segment => item !== null);
}

/**
 * 【向下相容】使用 Google Gemini 2.5 Flash 進行雲端音訊轉寫
 */
export async function transcribeWithGeminiAudio(
  wavBuffer: ArrayBuffer,
  apiKey: string,
  promptHint: string,
  timeOffsetSec: number,
  speakerId: string,
  chunkIndex: number,
  acousticThreshold = 0.72
): Promise<Segment[]> {
  if (!apiKey.trim()) return [];

  const base64Audio = arrayBufferToBase64(wavBuffer);
  // 16kHz 16-bit Mono WAV 每秒為 32000 bytes
  const estDurationSec = Math.max(1.0, (wavBuffer.byteLength - 44) / 32000);

  const prompt = `請將這段台灣會議語音精準轉寫為台灣正體中文逐字稿。
提示詞與前文背景：${promptHint || '一般公務與專案會議'}
規則：只輸出語音轉寫後的純文字句子（含自然標點符號），若為靜音或無意義雜音請輸出空字串，禁止輸出任何解釋。`;

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(
      apiKey.trim()
    )}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { inlineData: { mimeType: 'audio/wav', data: base64Audio } },
              { text: prompt },
            ],
          },
        ],
        generationConfig: { temperature: 0.05 },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Gemini 音訊轉寫請求失敗 (${response.status})`);
  }

  const data = await response.json();
  const rawText = (data?.candidates?.[0]?.content?.parts?.[0]?.text || '').trim();
  const cleanText = toTwp(rawText);
  if (!cleanText) return [];

  const segId = `gem_c${chunkIndex}_${Date.now()}`;
  const startSec = Number(timeOffsetSec.toFixed(2));
  const endSec = Number((timeOffsetSec + estDurationSec).toFixed(2));

  return [
    {
      id: segId,
      chunkIndex,
      speakerId,
      start: startSec,
      end: endSec,
      words: buildCloudWordSpans(cleanText, segId, startSec, endSec, 0.92, acousticThreshold),
      isCorrected: false,
      relistenCount: 0,
    },
  ];
}

/**
 * 【新版核心】使用 Gemini 2.5 Flash 進行三層關注度 (300字高 / 1200字中 / 前文低) 結構化校正
 */
export async function correctWithGeminiTriLevel(
  apiKey: string,
  payload: TriLevelContextPayload
): Promise<CorrectionPatchItem[]> {
  if (!apiKey || payload.highAttentionTargetSegments.length === 0) return [];

  const targetJson = payload.highAttentionTargetSegments.map((seg) => ({
    segmentId: seg.id,
    speakerId: seg.speakerId,
    words: seg.words.map((w, idx) => ({
      i: idx,
      t: w.text,
      c: Number(w.confidence.toFixed(2)),
    })),
  }));

  const prompt = `你是一位專精於台灣政府與學術會議的逐字稿校對專家。請根據以下「三層關注度語境」修正【高關注度目標區】內的同音錯字、專有名詞誤植與語意模糊詞彙：

【第 1 層：低關注度全域記憶 (唯讀背景與熱詞庫)】
- 會議簡報背景摘要：${payload.lowAttentionGlobalMemory.docSummary || '無'}
- 累積實證熱詞與人名表：${payload.lowAttentionGlobalMemory.verifiedTerms.join('、') || '無'}
- 更早之前情提要：${payload.lowAttentionGlobalMemory.earlierTopicDigest || '無'}

【第 2 層：中關注度前文語境 (前 ~1200 字唯讀對話錨點，禁止對此區輸出修改)】
${payload.mediumAttentionRollingText || '(此為會議開頭)'}

【第 3 層：高關注度目標區 (近 ~300 字唯一可讀寫修正區，含詞索引 i 與聲學信心度 c)】
${JSON.stringify(targetJson)}

請嚴格遵守以下規則並只輸出 JSON Array：
1. 僅針對【第 3 層：高關注度目標區】中有錯字或同音誤植的詞輸出 Patch。
2. 若能從上下文或熱詞庫確定正確詞彙，設定 status 為 "auto_corrected" 並填入 correctedText。
3. 若該處發音模糊、有多種可能且無法單靠文字肯定，設定 status 為 "semantic_uncertain"、填入 2~3 個 candidates，並將 requireRelisten 設為 true 以便系統回頭截檔重聽。
4. 全面使用台灣正體中文慣用詞。`;

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(
      apiKey
    )}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.1,
          responseMimeType: 'application/json',
        },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Gemini 校正請求失敗 (${response.status})`);
  }

  const data = await response.json();
  const rawText = data?.candidates?.[0]?.content?.parts?.[0]?.text || '[]';
  const parsed = JSON.parse(rawText) as CorrectionPatchItem[];

  return parsed.map((p) => ({
    ...p,
    correctedText: p.correctedText ? toTwp(p.correctedText) : undefined,
    candidates: p.candidates?.map((c) => toTwp(c)),
    reason: toTwp(p.reason || '語境推論修正'),
  }));
}

/**
 * 【新版核心：配額制截檔重聽】
 * 將擴充 ±padding 秒後的短音訊 WAV 連同前後文與候選詞直送 Gemini 2.5 Flash 重聽確認
 */
export async function relistenSubSegmentWithGemini(
  apiKey: string,
  subWavBuffer: ArrayBuffer,
  targetSegment: Segment,
  surroundingContextText: string,
  verifiedTerms: string[]
): Promise<WordSpan[] | null> {
  if (!apiKey) return null;

  const base64Audio = arrayBufferToBase64(subWavBuffer);
  const originalSentence = targetSegment.words.map((w) => w.text).join('');
  const uncertainHints = targetSegment.words
    .filter((w) => w.status !== 'normal' || w.confidence < 0.75)
    .map((w) => `「${w.text}」(候選: ${(w.candidates || []).join('/') || '未知'})`)
    .join('、');

  const prompt = `請仔細聆聽這段截取自會議的短音訊（前後已包含緩衝邊界，請只針對核心語句進行重聽校準）。
- 原初次辨識文字：「${originalSentence}」
- 高度存疑或模糊處：${uncertainHints || '整句低信心度'}
- 前後 300 字會議語境：${surroundingContextText}
- 會議專有名詞與人名熱詞：${verifiedTerms.slice(0, 40).join('、')}

請結合「真實人聲發音」與「會議語境」，輸出重聽校準後的完整正確句子（使用台灣正體中文，只輸出純文字句子本身，不要包含時間軸或解釋）。`;

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(
      apiKey
    )}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { inlineData: { mimeType: 'audio/wav', data: base64Audio } },
              { text: prompt },
            ],
          },
        ],
        generationConfig: { temperature: 0.05 },
      }),
    }
  );

  if (!response.ok) return null;
  const data = await response.json();
  const cleanText = toTwp((data?.candidates?.[0]?.content?.parts?.[0]?.text || '').trim());
  if (!cleanText) return null;

  const tokens =
    cleanText.match(/[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+|[^\s\u4e00-\u9fa5a-zA-Z0-9]/g) || [
      cleanText,
    ];
  const dur = Math.max(0.2, targetSegment.end - targetSegment.start);
  const step = dur / Math.max(1, tokens.length);

  return tokens.map((tok, idx) => ({
    id: `${targetSegment.id}_re_${idx}`,
    text: tok,
    start: Number((targetSegment.start + idx * step).toFixed(2)),
    end: Number((targetSegment.start + (idx + 1) * step).toFixed(2)),
    confidence: 0.96,
    status: 'auto_corrected' as const,
    reason: '經截檔重聽校準確認',
  }));
}