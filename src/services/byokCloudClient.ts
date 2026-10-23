// src/services/byokCloudClient.ts
import * as OpenCC from 'opencc-js';
import {
  CorrectionPatchItem,
  Segment,
  SlotCorrectionOutputItem,
  StructuredSlotGerResponse,
  SuspectSlotItem,
  TriLevelContextPayload,
  TurnSplitOutputItem,
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

function buildCloudWordSpans(
  cleanText: string,
  segId: string,
  startSec: number,
  endSec: number,
  baseConfidence: number,
  acousticThreshold = 0.72
): WordSpan[] {
  const tokens =
    cleanText.match(
      /[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+|[^\s\u4e00-\u9fa5a-zA-Z0-9]/g
    ) || [cleanText];
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

  const response = await fetch(
    'https://api.groq.com/openai/v1/audio/transcriptions',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey.trim()}`,
      },
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(`Groq Whisper API 請求失敗 (${response.status})`);
  }

  const data = await response.json();
  const rawSegments = Array.isArray(data?.segments) ? data.segments : [];

  if (
    rawSegments.length === 0 &&
    typeof data?.text === 'string' &&
    data.text.trim()
  ) {
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
        words: buildCloudWordSpans(
          cleanText,
          segId,
          startSec,
          endSec,
          0.9,
          acousticThreshold
        ),
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
      const startSec = Number(
        (timeOffsetSec + Number(s.start || 0)).toFixed(2)
      );
      const endSec = Number(
        (
          timeOffsetSec + Number(s.end || Number(s.start || 0) + 2)
        ).toFixed(2)
      );
      const avgLogprob =
        typeof s.avg_logprob === 'number' ? s.avg_logprob : -0.25;
      const conf = Math.min(0.98, Math.max(0.5, Math.exp(avgLogprob)));

      return {
        id: segId,
        chunkIndex,
        speakerId,
        start: startSec,
        end: endSec,
        words: buildCloudWordSpans(
          cleanText,
          segId,
          startSec,
          endSec,
          conf,
          acousticThreshold
        ),
        isCorrected: false,
        relistenCount: 0,
      } as Segment;
    })
    .filter((item: Segment | null): item is Segment => item !== null);
}

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
  const rawText = (
    data?.candidates?.[0]?.content?.parts?.[0]?.text || ''
  ).trim();
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
      words: buildCloudWordSpans(
        cleanText,
        segId,
        startSec,
        endSec,
        0.92,
        acousticThreshold
      ),
      isCorrected: false,
      relistenCount: 0,
    },
  ];
}

/**
 * 【問題二 2.3 核心升級】：使用 Gemini 2.5 Flash 執行免索引槽位填空協定 (Slot-Anchored GER) 與 LG-TPST 語用拆句
 */
export async function correctWithGeminiTriLevel(
  apiKey: string,
  payload: TriLevelContextPayload
): Promise<CorrectionPatchItem[]> {
  if (!apiKey || payload.highAttentionTargetSegments.length === 0) return [];

  const targetSegments = payload.highAttentionTargetSegments;
  const slotMap = new Map<string, SuspectSlotItem>();
  if (Array.isArray(payload.suspectSlots)) {
    for (const s of payload.suspectSlots) {
      slotMap.set(s.slotId, s);
    }
  }

  const slottedTranscript =
    payload.slottedPromptTranscript ||
    targetSegments
      .map((seg) => {
        const dur = seg.end - seg.start;
        const margin = seg.acousticMargin ?? 0.18;
        const isLocked = margin >= 0.16 && dur >= 1.2;
        const gateTag = isLocked
          ? `[LOCKED | speaker=${seg.speakerId}]`
          : `[REVIEW | top1=${seg.speakerId}, top2=${
              seg.runnerUpSpeakerId || seg.speakerId
            }]`;
        return `[${seg.id}] ${gateTag} ${seg.words.map((w) => w.text).join('')}`;
      })
      .join('\n');

  const prompt = `你是一位專精於台灣政商與學術會議的「免索引槽位校對與 LG-TPST 對話語用專家」。請根據以下「三層關注度語境」，針對【第 3 層：高關注度目標區】內的懷疑槽位 ⟪#S1: 原詞 | 候選: ...⟫ 與 [REVIEW] 段落執行校對：

【第 1 層：低關注度全域記憶 (唯讀背景與熱詞庫)】
- 會議簡報背景摘要：${payload.lowAttentionGlobalMemory.docSummary || '無'}
- 全場高主權實體與熱詞表：${payload.lowAttentionGlobalMemory.verifiedTerms.join('、') || '無'}
- 更早之前情提要：${payload.lowAttentionGlobalMemory.earlierTopicDigest || '無'}

【第 2 層：中關注度前文語境 (前 ~1200 字唯讀對話錨點，禁止對此區輸出修改)】
${payload.mediumAttentionRollingText || '(此為會議開頭)'}

【第 3 層：高關注度目標區 (含 [LOCKED]/[REVIEW] 閘門與 ⟪#S1...⟫ 槽位標籤)】
${slottedTranscript}

請嚴格輸出符合以下結構之 JSON Object（包含 slot_corrections 與 turn_splits 兩個陣列）：
1. slot_corrections：針對每個 #S1~#S5 槽位，先填寫 brief_reason（15字內語境搭配理由），再決定 action ("KEEP_ORIGINAL" | "APPLY_HIGH_CONF" | "MARK_UNCERTAIN")、selected_text（僅填寫槽位替換後的詞彙本身，嚴禁包含槽位外的上下文！）與 uncertain_candidates。
2. turn_splits：[LOCKED] 段落嚴禁變更語者；[REVIEW] 段落若出現「問答對卻同語者 (QA_SPEAKER_CHANGE)」或「短促附和插話 (BACKCHANNEL_INTERJECTION)」，請填寫 utterance_id、pragmatic_type、split_before_anchor（回答句開頭精確 2~4 個字）與 new_speaker_id。`;

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
  const rawText = data?.candidates?.[0]?.content?.parts?.[0]?.text || '{}';
  const parsed: Partial<StructuredSlotGerResponse> | CorrectionPatchItem[] =
    JSON.parse(rawText);

  const results: CorrectionPatchItem[] = [];

  // 相容處理：若回傳為新版 { slot_corrections, turn_splits } 物件
  if (!Array.isArray(parsed) && typeof parsed === 'object' && parsed !== null) {
    const structuredObj = parsed as Partial<StructuredSlotGerResponse>;

    if (Array.isArray(structuredObj.slot_corrections)) {
      for (const sc of structuredObj.slot_corrections as SlotCorrectionOutputItem[]) {
        const slot = slotMap.get(sc.slot_id);
        if (!slot || sc.action === 'KEEP_ORIGINAL') continue;

        const cleanSelected = toTwp((sc.selected_text || '').trim());
        const cleanReason = toTwp(
          sc.brief_reason || '雲端 Gemini 槽位語意推敲修正'
        );
        const cleanCands = Array.isArray(sc.uncertain_candidates)
          ? sc.uncertain_candidates
              .map((c: unknown) => toTwp(String(c).trim()))
              .filter(Boolean)
          : slot.candidates;

        if (
          sc.action === 'APPLY_HIGH_CONF' &&
          cleanSelected &&
          cleanSelected !== slot.originalText
        ) {
          results.push({
            segmentId: slot.segmentId,
            wordIndex: slot.startWordIdx,
            endWordIndex: slot.endWordIdx,
            originalText: slot.originalText,
            status: 'auto_corrected',
            correctedText: cleanSelected,
            candidates: cleanCands,
            reason: `[${sc.slot_id}] ${cleanReason}`,
          });
        } else if (sc.action === 'MARK_UNCERTAIN') {
          results.push({
            segmentId: slot.segmentId,
            wordIndex: slot.startWordIdx,
            endWordIndex: slot.endWordIdx,
            originalText: slot.originalText,
            status: 'semantic_uncertain',
            candidates: cleanCands.length > 0 ? cleanCands : slot.candidates,
            reason: `[${sc.slot_id}] ${cleanReason}`,
          });
        }
      }
    }

    if (Array.isArray(structuredObj.turn_splits)) {
      for (const ts of structuredObj.turn_splits as TurnSplitOutputItem[]) {
        if (
          !ts.utterance_id ||
          ts.pragmatic_type === 'NO_SPLIT' ||
          ts.pragmatic_type === 'RHETORICAL_SELF_ANSWER'
        ) {
          continue;
        }
        const targetSeg = targetSegments.find((s) => s.id === ts.utterance_id);
        if (!targetSeg) continue;
        const dur = targetSeg.end - targetSeg.start;
        const margin = targetSeg.acousticMargin ?? 0.18;
        const isLocked = margin >= 0.16 && dur >= 1.2;
        if (isLocked) continue;

        if (ts.new_speaker_id && ts.new_speaker_id !== targetSeg.speakerId) {
          results.push({
            segmentId: targetSeg.id,
            wordIndex: -1,
            originalText: targetSeg.speakerId,
            status: 'auto_corrected',
            newSpeakerId: ts.new_speaker_id.trim(),
            splitBeforeAnchor: toTwp((ts.split_before_anchor || '').trim()),
            reason: `LG-TPST 對話語用學語者轉移 (${ts.pragmatic_type})`,
          });
        }
      }
    }

    return results;
  }

  // 向下相容舊版 Array 格式
  if (Array.isArray(parsed)) {
    for (const p of parsed) {
      const targetSeg = targetSegments.find((s) => s.id === p.segmentId);
      if (!targetSeg) continue;
      if (p.newSpeakerId && p.newSpeakerId !== targetSeg.speakerId) {
        const dur = targetSeg.end - targetSeg.start;
        const margin = targetSeg.acousticMargin ?? 0.18;
        if (!(margin >= 0.16 && dur >= 1.2)) {
          results.push({
            segmentId: p.segmentId,
            wordIndex: -1,
            originalText: targetSeg.speakerId,
            status: 'auto_corrected',
            newSpeakerId: p.newSpeakerId,
            reason: toTwp(p.reason || 'LG-TPST 對話語用學語者轉移'),
          });
        }
        continue;
      }
      if (typeof p.wordIndex === 'number' && p.wordIndex >= 0) {
        results.push({
          ...p,
          correctedText: p.correctedText ? toTwp(p.correctedText) : undefined,
          candidates: p.candidates?.map((c: string) => toTwp(c)),
          reason: toTwp(p.reason || '語境推論修正'),
        });
      }
    }
  }

  return results;
}

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
    .map(
      (w) => `「${w.text}」(候選: ${(w.candidates || []).join('/') || '未知'})`
    )
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
  const cleanText = toTwp(
    (data?.candidates?.[0]?.content?.parts?.[0]?.text || '').trim()
  );
  if (!cleanText) return null;

  const tokens =
    cleanText.match(
      /[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+|[^\s\u4e00-\u9fa5a-zA-Z0-9]/g
    ) || [cleanText];
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