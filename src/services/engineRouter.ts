// src/services/engineRouter.ts
import * as OpenCC from 'opencc-js';
import {
  CorrectionPatchItem,
  DictEntry,
  EngineConfig,
  Segment,
  Speaker,
  TriLevelContextPayload,
  WordSpan,
} from '../types';
import * as cloudClient from './byokCloudClient';
import { encodeFloat32ToWav, sliceSubSegmentAudio } from '../utils/audioProcessor';
import { toPhoneticSignature } from '../utils/phoneticMatcher';
import { useAppStore } from '../store/useAppStore';

const converter = OpenCC.Converter({ from: 'cn', to: 'twp' });

export const UNIVERSAL_COLLOQUIAL_RULES: Array<{
  pattern: RegExp;
  wrongDesc: string;
  right: string;
  reason: string;
}> = [
  { pattern: /店+面+[凡煩繁販翻]+修+/g, wrongDesc: '店店面販修修', right: '店面翻修', reason: '語境同音與疊字修正：店面翻修' },
  { pattern: /(限定開新店)[，、\s]*\1/g, wrongDesc: '重複子句', right: '$1', reason: '口語重複消除' },
  { pattern: /新的店的裝置/g, wrongDesc: '新的店的裝置', right: '新的店的裝修', reason: '語境同音修正：新的店的裝修' },
  { pattern: /可是如果新的店有充足[，。]?(\s*算)?/g, wrongDesc: '有充足', right: '可是如果新的店的裝修怎麼辦？', reason: '全局語境修正：新的店的裝修怎麼辦' },
  { pattern: /^算[。]?$/g, wrongDesc: '算', right: '不算。', reason: '前後文對話語境修正：不算' },
  { pattern: /^不算對[，。]?$/g, wrongDesc: '不算對', right: '不算。', reason: '前後文對話語境修正：不算' },
  { pattern: /如果是?你的店/g, wrongDesc: '如果是你的店', right: '如果新的店', reason: '語境同音修正：如果新的店' },
  { pattern: /開[心欣辛][店電]/g, wrongDesc: '開心店/開心電', right: '開新店', reason: '語境同音修正：開新店' },
  { pattern: /(反對|要有|限定)開電/g, wrongDesc: '開電', right: '$1開店', reason: '語境同音修正：開店' },
  { pattern: /算不算要有電/g, wrongDesc: '要有電', right: '算不算要有店', reason: '語境同音修正：要有店' },
  { pattern: /加[盲忙茫]的/g, wrongDesc: '加盲的', right: '加盟的', reason: '商業語境修正：加盟的' },
  { pattern: /[狼蘭藍郎][帶代單袋蛋]/g, wrongDesc: '狼帶/郎蛋', right: '廊帶', reason: '區域規劃專有名詞修正：廊帶' },
  { pattern: /走路再[闊擴]/g, wrongDesc: '走路再闊', right: '走路再擴', reason: '語境同音修正：再擴' },
  { pattern: /(輔導|開店|認識|找當地的|找那|那一群|每個|商業師的那個)([^，。]{0,6}?)(夜市|業勢|夜視|夜師|夜食)/g, wrongDesc: '夜師/夜食/夜市', right: '$1$2業師', reason: '輔導專家稱謂修正：業師' },
  { pattern: /夜食(?=只是我在裡面找)/g, wrongDesc: '夜食', right: '業師，', reason: '輔導專家稱謂修正：業師' },
  { pattern: /(夜市|業勢|夜師|夜食)群/g, wrongDesc: '夜師群', right: '業師群', reason: '輔導專家稱謂修正：業師群' },
  { pattern: /(我們|你們)(夜師|夜食|夜市)(自己找|要自己找)/g, wrongDesc: '夜師自己找', right: '$1業師$3', reason: '輔導專家稱謂修正：業師' },
  { pattern: /中央企業師/g, wrongDesc: '中央企業師', right: '中小企業署', reason: '機關名稱修正：中小企業署' },
  { pattern: /[講獎][歷力立][經金劍]/g, wrongDesc: '講歷經/獎勵劍', right: '獎勵金', reason: '補助經費專有名詞修正：獎勵金' },
  { pattern: /(下一期就開始要|開始要寫|用這個當|廊帶)[成針真][見件]/g, wrongDesc: '成見/針見/真見', right: '$1徵件', reason: '標案計畫專有名詞修正：徵件' },
  { pattern: /重新改一個新表/g, wrongDesc: '新表', right: '重新開一個新標', reason: '採購標案語境修正：開新標' },
  { pattern: /星期一個案/g, wrongDesc: '星期一個案', right: '新起一個案', reason: '專案語境修正：新起一個案' },
  { pattern: /開新標跟大家救的路/g, wrongDesc: '大家救的路', right: '開新標跟搭舊的路', reason: '語境同音修正：搭舊的路' },
  { pattern: /出內希望/g, wrongDesc: '出內希望', right: '處內希望', reason: '機關內部稱謂修正：處內' },
  { pattern: /要多[精經]快/g, wrongDesc: '要多精快', right: '要多儘快', reason: '語境同音修正：儘快' },
  { pattern: /開心不要/g, wrongDesc: '開心不要', right: '開新標', reason: '採購標案語境修正：開新標' },
  { pattern: /一個[星心欣]標/g, wrongDesc: '一個星標', right: '一個新標', reason: '採購標案語境修正：新標' },
  { pattern: /一般[族組]/g, wrongDesc: '一般族', right: '一般組', reason: '標案分組稱謂修正：一般組' },
  { pattern: /再去[過擴]充/g, wrongDesc: '再去過充', right: '再去擴充', reason: '採購契約用語修正：擴充' },
  { pattern: /擴充套件/g, wrongDesc: '擴充套件', right: '後續擴充', reason: '政府採購專有名詞修正：後續擴充' },
  { pattern: /先簽約一架/g, wrongDesc: '先簽約一架', right: '先簽約議價', reason: '政府採購專有名詞修正：議價' },
  { pattern: /省的是那個平選/g, wrongDesc: '平選', right: '省的是那個評選', reason: '政府採購專有名詞修正：評選' },
  { pattern: /在施程|加10層|加十層|對，十層/g, wrongDesc: '十層/施程', right: '時程', reason: '專案管理用語修正：時程' },
  { pattern: /地方創[身聲深]/g, wrongDesc: '地方創身', right: '地方創生', reason: '專有名詞修正：地方創生' },
  { pattern: /國法會|過發會/g, wrongDesc: '國法會', right: '國發會', reason: '機關簡稱修正：國發會' },
  { pattern: /青年[賠陪]力/g, wrongDesc: '青年賠力', right: '青年培力', reason: '專有名詞修正：青年培力' },
];

export function mineAdaptiveHighFreqTerms(
  allSegments: Segment[],
  dictionary: DictEntry[],
  docSummary: string
): string[] {
  const termCounts = new Map<string, number>();

  for (const d of dictionary) {
    if (d.term.length >= 2) termCounts.set(d.term, 12);
  }
  const summaryWords = docSummary.match(/[\u4e00-\u9fa5]{2,4}/g) || [];
  for (const sw of summaryWords) {
    termCounts.set(sw, (termCounts.get(sw) || 0) + 4);
  }

  const stopWords = new Set([
    '我們', '你們', '他們', '這個', '那個', '因為', '所以', '但是', '可是',
    '如果', '就是', '可以', '沒有', '不是', '一個', '現在', '時候', '這樣',
    '覺得', '應該', '知道', '部分', '問題', '意思', '可能', '已經', '還是',
    '然後', '我現', '開新', '是說', '上次', '不會', '是要', '那我', '對對',
    '不用', '他要', '他的', '我的', '你的', '比較', '其實', '基本', '後面',
    '裡面', '怎麼', '什麼', '那邊', '這邊', '大家', '他們', '有些', '確定',
    '建議', '發現', '開始', '出來', '起來', '過去', '進來', '東西', '事情',
  ]);

  const badEdgeChars =
    /^[的我你他她它這那在有是就也都還又再很太更不沒別好對啊喔嗎呢吧啦]|(的|了|嗎|呢|吧|啊|喔|啦|們|說|要|會|是|有|在)$/;

  for (const seg of allSegments) {
    for (const w of seg.words) {
      const txt = converter(w.text);
      if (
        txt.length >= 2 &&
        txt.length <= 4 &&
        /^[\u4e00-\u9fa5]+$/.test(txt) &&
        !stopWords.has(txt) &&
        !badEdgeChars.test(txt) &&
        (w.confidence >= 0.86 || w.status === 'auto_corrected')
      ) {
        termCounts.set(txt, (termCounts.get(txt) || 0) + 1);
      }
    }
  }

  return Array.from(termCounts.entries())
    .filter(([, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .map(([term]) => term)
    .slice(0, 75);
}

export async function evolveRollingMeetingSummary(params: {
  config: EngineConfig;
  docContextSummary: string;
  currentSummary: string;
  recentSegments: Segment[];
  dictionary: DictEntry[];
}): Promise<string> {
  const { config, docContextSummary, currentSummary, recentSegments, dictionary } = params;
  const recentText = recentSegments
    .map((s) => s.words.map((w) => w.text).join(''))
    .join(' ');

  if (!recentText.trim()) return currentSummary;

  if (config.geminiApiKey.trim()) {
    try {
      const prompt = `請根據「背景資料」、「目前累積摘要」與「最新一段會議對話」，更新這場會議的【全局核心摘要與關鍵概念】（限 180 字內）：
- 初始簡報背景：${docContextSummary || '無'}
- 目前累積摘要：${currentSummary || '尚無'}
- 最新會議對話：${recentText.slice(-800)}`;

      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(
          config.geminiApiKey.trim()
        )}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.1, maxOutputTokens: 250 },
          }),
        }
      );
      if (res.ok) {
        const data = await res.json();
        const summary = converter((data?.candidates?.[0]?.content?.parts?.[0]?.text || '').trim());
        if (summary) return summary;
      }
    } catch {
      // fallback below
    }
  }

  const allSegs = useAppStore.getState().segments;
  const minedTerms = mineAdaptiveHighFreqTerms(allSegs, dictionary, docContextSummary).slice(0, 14);

  const sentences = recentText
    .split(/[。？！]/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 10 && s.length <= 45);
  const decisionSentences = sentences.filter((s) =>
    /(建議|決定|限定|範圍|萬|分鐘|個月|新標|擴充|獎勵金|輔導|預算|議價|徵件|業師|廊帶)/.test(s)
  );

  const existingConsensusLine = currentSummary
    .split('\n')
    .find((l) => l.startsWith('【近期共識】'));
  const oldItems = existingConsensusLine
    ? existingConsensusLine
        .replace(/^(\s*【近期共識】\s*)+/, '')
        .split('；')
        .filter(Boolean)
    : [];

  const mergedConsensus = Array.from(
    new Set([...oldItems, ...decisionSentences.slice(-2)])
  ).slice(-4);

  const summaryParts = [
    docContextSummary ? `【背景】${docContextSummary.slice(0, 80)}` : '',
    `【全場高頻核心詞】${minedTerms.join('、')}`,
    mergedConsensus.length > 0 ? `【近期共識】${mergedConsensus.join('；')}` : '',
  ].filter(Boolean);

  return summaryParts.join('\n');
}

export function buildTriLevelContextPayload(
  allPriorSegments: Segment[],
  targetWindowSegments: Segment[],
  speakers: Record<string, Speaker>,
  dictionary: DictEntry[],
  docContextSummary: string,
  mediumCharLimit = 1200,
  rollingMeetingSummary = ''
): TriLevelContextPayload {
  let mediumChars = 0;
  const mediumLines: string[] = [];
  const earlierSegments: Segment[] = [];

  for (let i = allPriorSegments.length - 1; i >= 0; i--) {
    const seg = allPriorSegments[i];
    const spName = speakers[seg.speakerId]?.name || seg.speakerId;
    const text = seg.words.map((w) => w.text).join('');
    const line = `[${spName}]: ${text}`;

    if (mediumChars + text.length <= mediumCharLimit) {
      mediumLines.unshift(line);
      mediumChars += text.length;
    } else {
      earlierSegments.unshift(seg);
    }
  }

  const allSegs = useAppStore.getState().segments;
  const adaptiveTerms = mineAdaptiveHighFreqTerms(allSegs, dictionary, docContextSummary);

  const digestSentences: string[] = [];
  let digestLen = 0;
  for (let i = Math.max(0, earlierSegments.length - 10); i < earlierSegments.length; i++) {
    const s = earlierSegments[i];
    const txt = s.words.map((w) => w.text).join('');
    if (txt.length >= 10 && digestLen + txt.length <= 200) {
      digestSentences.push(txt);
      digestLen += txt.length;
    }
  }

  return {
    lowAttentionGlobalMemory: {
      docSummary: [docContextSummary.slice(0, 200), rollingMeetingSummary]
        .filter(Boolean)
        .join('\n'),
      verifiedTerms: adaptiveTerms,
      earlierTopicDigest: rollingMeetingSummary || digestSentences.join('；'),
    },
    mediumAttentionRollingText: mediumLines.join('\n'),
    highAttentionTargetSegments: targetWindowSegments,
  };
}

export function applySentenceLevelLocalCorrection(
  targetSegments: Segment[],
  verifiedTerms: string[],
  dictionary: DictEntry[] = []
): { updatedSegments: Segment[]; patches: CorrectionPatchItem[] } {
  const patches: CorrectionPatchItem[] = [];

  const dynamicRules: Array<{ wrong: string; right: string; reason: string }> = [];
  for (const entry of dictionary) {
    if (entry.aliases && entry.aliases.length > 0) {
      const scopeLabel =
        entry.scope === 'public' || entry.scope === 'global'
          ? '公共字典'
          : entry.scope === 'personal'
          ? '個人字典'
          : '會議字典';
      for (const alias of entry.aliases) {
        if (alias && alias !== entry.term) {
          dynamicRules.push({
            wrong: alias,
            right: entry.term,
            reason: `依${scopeLabel}同音別名自動修正`,
          });
        }
      }
    }
  }

  const phoneticMap = new Map<string, string>();
  for (const hw of verifiedTerms) {
    if (hw.length >= 2 && hw.length <= 4) {
      phoneticMap.set(toPhoneticSignature(hw), hw);
    }
  }

  const updatedSegments = targetSegments.map((seg) => {
    const rawSentence = converter(seg.words.map((w) => w.text).join(''));
    let refinedSentence = rawSentence
      .replace(/店店面/g, '店面')
      .replace(/[凡煩繁販翻]修修/g, '翻修')
      .replace(/？[，。、]+/g, '？');

    const hitReasons: Array<{ wrong: string; right: string; reason: string }> = [];

    for (const rule of UNIVERSAL_COLLOQUIAL_RULES) {
      if (rule.pattern.test(refinedSentence)) {
        refinedSentence = refinedSentence.replace(rule.pattern, (matched, ...args) => {
          let replacement = rule.right;
          if (replacement.includes('$1') && typeof args[0] === 'string') {
            replacement = replacement
              .replace('$1', args[0] || '')
              .replace('$2', typeof args[1] === 'string' ? args[1] : '')
              .replace('$3', typeof args[2] === 'string' ? args[2] : '');
          }
          if (matched !== replacement) {
            hitReasons.push({ wrong: matched, right: replacement, reason: rule.reason });
          }
          return replacement;
        });
      }
    }

    for (const dRule of dynamicRules) {
      if (refinedSentence.includes(dRule.wrong)) {
        refinedSentence = refinedSentence.replaceAll(dRule.wrong, dRule.right);
        hitReasons.push(dRule);
      }
    }

    for (const [sig, correctTerm] of phoneticMap.entries()) {
      if (sig.includes('-') && correctTerm.length >= 2 && correctTerm.length <= 4) {
        const len = correctTerm.length;
        for (let i = 0; i <= refinedSentence.length - len; i++) {
          const sub = refinedSentence.slice(i, i + len);
          if (sub !== correctTerm && /^[\u4e00-\u9fa5]+$/.test(sub)) {
            if (toPhoneticSignature(sub) === sig) {
              hitReasons.push({
                wrong: sub,
                right: correctTerm,
                reason: `依全場高頻詞聲韻母自動收斂為「${correctTerm}」`,
              });
              refinedSentence =
                refinedSentence.slice(0, i) + correctTerm + refinedSentence.slice(i + len);
            }
          }
        }
      }
    }

    if (refinedSentence === rawSentence) {
      return { ...seg, isCorrected: true };
    }

    const tokens =
      refinedSentence.match(
        /[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+|[^\s\u4e00-\u9fa5a-zA-Z0-9]/g
      ) || [refinedSentence];
    const dur = Math.max(0.2, seg.end - seg.start);
    const step = dur / Math.max(1, tokens.length);

    const newWords: WordSpan[] = tokens.map((tok, idx) => {
      const matchedRule = hitReasons.find(
        (r) => r.right.includes(tok) || tok.includes(r.right)
      );
      if (matchedRule && /^[\u4e00-\u9fa5]+$/.test(tok)) {
        patches.push({
          segmentId: seg.id,
          wordIndex: idx,
          originalText: matchedRule.wrong,
          status: 'auto_corrected',
          correctedText: tok,
          reason: matchedRule.reason,
        });
        return {
          id: `${seg.id}_w_${idx}`,
          text: tok,
          originalText: matchedRule.wrong,
          start: Number((seg.start + idx * step).toFixed(2)),
          end: Number((seg.start + (idx + 1) * step).toFixed(2)),
          confidence: 0.96,
          status: 'auto_corrected',
          reason: matchedRule.reason,
        };
      }
      return {
        id: `${seg.id}_w_${idx}`,
        text: tok,
        start: Number((seg.start + idx * step).toFixed(2)),
        end: Number((seg.start + (idx + 1) * step).toFixed(2)),
        confidence: 0.88,
        status: 'normal',
      };
    });

    return {
      ...seg,
      words: newWords,
      isCorrected: true,
    };
  });

  return { updatedSegments, patches };
}

export async function runTriLevelWindowCorrection(params: {
  config: EngineConfig;
  payload: TriLevelContextPayload;
  dictionary: DictEntry[];
  runLocalLlmCorrection?: (payload: TriLevelContextPayload) => Promise<CorrectionPatchItem[]>;
  onApplyPatches: (patches: CorrectionPatchItem[], correctedSegIds: string[]) => void;
}): Promise<CorrectionPatchItem[]> {
  const { config, payload, dictionary, runLocalLlmCorrection, onApplyPatches } = params;
  const targetIds = payload.highAttentionTargetSegments.map((s) => s.id);

  const { updatedSegments, patches: localPatches } = applySentenceLevelLocalCorrection(
    payload.highAttentionTargetSegments,
    payload.lowAttentionGlobalMemory.verifiedTerms,
    dictionary
  );
  useAppStore.getState().replaceBatchSegments(updatedSegments);

  let modelPatches: CorrectionPatchItem[] = [];
  const updatedPayload: TriLevelContextPayload = {
    ...payload,
    highAttentionTargetSegments: updatedSegments,
  };

  if (config.geminiApiKey.trim()) {
    try {
      modelPatches = await cloudClient.correctWithGeminiTriLevel(
        config.geminiApiKey.trim(),
        updatedPayload
      );
    } catch (err) {
      console.warn('雲端三層關注度校正略過，維持本機自洽結果:', err);
    }
  } else if (runLocalLlmCorrection) {
    try {
      modelPatches = await runLocalLlmCorrection(updatedPayload);
    } catch (err) {
      console.warn('本機 WebLLM 三層關注度校正略過:', err);
    }
  }

  if (modelPatches.length > 0) {
    onApplyPatches(modelPatches, targetIds);
  }
  return [...localPatches, ...modelPatches];
}

export async function runGlobalRetroactiveContextPass(params: {
  config: EngineConfig;
  allSegments: Segment[];
  dictionary: DictEntry[];
  docContextSummary: string;
  rollingMeetingSummary: string;
  runLocalLlmCorrection?: (payload: TriLevelContextPayload) => Promise<CorrectionPatchItem[]>;
}): Promise<number> {
  const {
    config,
    allSegments,
    dictionary,
    docContextSummary,
    rollingMeetingSummary,
    runLocalLlmCorrection,
  } = params;
  if (allSegments.length === 0) return 0;

  const globalVerifiedTerms = mineAdaptiveHighFreqTerms(
    allSegments,
    dictionary,
    `${docContextSummary}\n${rollingMeetingSummary}`
  );

  const { updatedSegments, patches } = applySentenceLevelLocalCorrection(
    allSegments,
    globalVerifiedTerms,
    dictionary
  );

  useAppStore.getState().replaceBatchSegments(updatedSegments);

  const uncertainEarlier = updatedSegments
    .filter((s) =>
      s.words.some((w) => w.confidence < 0.78 || w.status === 'semantic_uncertain')
    )
    .slice(0, 10);

  let aiFixedCount = 0;
  if (uncertainEarlier.length > 0) {
    const retroPayload = buildTriLevelContextPayload(
      updatedSegments.slice(-25),
      uncertainEarlier,
      useAppStore.getState().speakers,
      dictionary,
      docContextSummary,
      1500,
      rollingMeetingSummary
    );

    if (config.geminiApiKey.trim()) {
      try {
        const cloudPatches = await cloudClient.correctWithGeminiTriLevel(
          config.geminiApiKey.trim(),
          retroPayload
        );
        if (cloudPatches.length > 0) {
          useAppStore
            .getState()
            .applyCorrectionPatches(
              cloudPatches,
              uncertainEarlier.map((s) => s.id)
            );
          aiFixedCount += cloudPatches.length;
        }
      } catch {
        // ignore
      }
    } else if (runLocalLlmCorrection) {
      try {
        const llmPatches = await runLocalLlmCorrection(retroPayload);
        if (llmPatches.length > 0) {
          useAppStore
            .getState()
            .applyCorrectionPatches(
              llmPatches,
              uncertainEarlier.map((s) => s.id)
            );
          aiFixedCount += llmPatches.length;
        }
      } catch {
        // ignore
      }
    }
  }

  return patches.length + aiFixedCount;
}

export async function executeBoundedQuotaRelisten(params: {
  config: EngineConfig;
  chunkSamples: Float32Array;
  chunkStartGlobalSec: number;
  chunkSegments: Segment[];
  flaggedSegmentIds: Set<string>;
  verifiedTerms: string[];
  relistenWithLocalAsr?: (
    subSamples: Float32Array,
    paddedStartSec: number,
    targetSegment: Segment,
    promptHint: string
  ) => Promise<WordSpan[] | null>;
  onSegmentRelistened: (segmentId: string, newWords: WordSpan[]) => void;
}): Promise<number> {
  const {
    config,
    chunkSamples,
    chunkStartGlobalSec,
    chunkSegments,
    flaggedSegmentIds,
    verifiedTerms,
    relistenWithLocalAsr,
    onSegmentRelistened,
  } = params;

  const quota = Math.max(0, config.maxRelistenQuota || 3);
  if (quota === 0) return 0;

  const candidates = chunkSegments
    .filter((seg) => (seg.relistenCount || 0) < 1 && seg.end - seg.start >= 1.0)
    .map((seg) => {
      const avgConf =
        seg.words.reduce((acc, w) => acc + w.confidence, 0) / Math.max(1, seg.words.length);
      const hasSemanticFlag =
        flaggedSegmentIds.has(seg.id) ||
        seg.words.some(
          (w) => w.status === 'semantic_uncertain' || w.status === 'acoustic_uncertain'
        );
      const priorityScore = avgConf - (hasSemanticFlag ? 0.25 : 0);
      return { seg, avgConf, hasSemanticFlag, priorityScore };
    })
    .filter((item) => item.hasSemanticFlag || item.avgConf < config.acousticThreshold)
    .sort((a, b) => a.priorityScore - b.priorityScore)
    .slice(0, quota);

  let relistenedCount = 0;

  for (const { seg } of candidates) {
    const { subSamples, subWavBuffer, paddedStartGlobalSec } = sliceSubSegmentAudio(
      chunkSamples,
      chunkStartGlobalSec,
      seg.start,
      seg.end,
      Math.max(1.0, config.relistenPaddingSec || 1.0)
    );

    const segIdx = chunkSegments.findIndex((s) => s.id === seg.id);
    const contextSlice = chunkSegments
      .slice(Math.max(0, segIdx - 3), Math.min(chunkSegments.length, segIdx + 4))
      .map((s) => s.words.map((w) => w.text).join(''))
      .join(' / ');

    let newWords: WordSpan[] | null = null;

    if (config.geminiApiKey.trim()) {
      newWords = await cloudClient.relistenSubSegmentWithGemini(
        config.geminiApiKey.trim(),
        subWavBuffer,
        seg,
        contextSlice,
        verifiedTerms
      );
    } else if (relistenWithLocalAsr) {
      const promptHint = [...verifiedTerms.slice(0, 20), contextSlice.slice(0, 100)].join('，');
      const rawRelistenWords = await relistenWithLocalAsr(
        subSamples,
        paddedStartGlobalSec,
        seg,
        promptHint
      );

      if (rawRelistenWords && rawRelistenWords.length > 0) {
        const tempSeg: Segment = { ...seg, words: rawRelistenWords };
        const { updatedSegments } = applySentenceLevelLocalCorrection(
          [tempSeg],
          verifiedTerms,
          useAppStore.getState().dictionary
        );
        const origLen = seg.words.map((w) => w.text).join('').length;
        const newLen = updatedSegments[0].words.map((w) => w.text).join('').length;
        if (newLen >= origLen * 0.7 && newLen <= origLen * 1.45 + 6) {
          newWords = updatedSegments[0].words;
        }
      }
    }

    if (newWords && newWords.length > 0) {
      onSegmentRelistened(seg.id, newWords);
      relistenedCount++;
    }
  }

  return relistenedCount;
}

export class HybridEngineRouter {
  public encodeTurnWav(turnPcm: Float32Array, sampleRate = 16000): ArrayBuffer {
    return encodeFloat32ToWav(turnPcm, sampleRate);
  }
  public getStoreSnapshot() {
    return useAppStore.getState();
  }
}

export const engineRouter = new HybridEngineRouter();