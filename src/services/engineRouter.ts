// src/services/engineRouter.ts
import * as OpenCC from 'opencc-js';
import {
  CorrectionPatchItem,
  DictEntry,
  EngineConfig,
  Segment,
  Speaker,
  SuspectSlotItem,
  TriLevelContextPayload,
  WordSpan,
} from '../types';
import * as cloudClient from './byokCloudClient';
import {
  encodeFloat32ToWav,
  enhanceSubSegmentAudioForRelisten,
} from '../utils/audioProcessor';
import {
  computeTaiwanPhoneticDistance,
  toPhoneticSignature,
  toTaiwanFuzzyPhoneticSyllables,
} from '../utils/phoneticMatcher';
import { useAppStore } from '../store/useAppStore';

const converter = OpenCC.Converter({ from: 'cn', to: 'twp' });

/**
 * Pass 0：跨領域通用之口語異常疊字與跳針消除規則
 */
export const UNIVERSAL_COLLOQUIAL_RULES: Array<{
  pattern: RegExp;
  wrongDesc: string;
  right: string;
  reason: string;
}> = [
  {
    pattern: /([\u4e00-\u9fa5])\1{2,}/g,
    wrongDesc: '單字異常連續疊字',
    right: '$1',
    reason: 'Pass 0 口語異常連續疊字消除',
  },
  {
    pattern: /([\u4e00-\u9fa5]{2,4})(?:[，、\s]*\1)+/g,
    wrongDesc: '雙字以上異常跳針重複',
    right: '$1',
    reason: 'Pass 0 口語跳針重複詞組消除',
  },
  {
    pattern: /([店翻裝改修開標案處署師])(\1)(?=[\u4e00-\u9fa5])/g,
    wrongDesc: '複合詞首字口吃疊字',
    right: '$1',
    reason: 'Pass 0 口語詞首異常疊字消除',
  },
  {
    pattern: /(?<=[\u4e00-\u9fa5])([修店標案處署師])(\1)/g,
    wrongDesc: '複合詞尾字口吃疊字',
    right: '$1',
    reason: 'Pass 0 口語詞尾異常疊字消除',
  },
];

const COMMON_STOP_WORDS = new Set([
  '我們', '你們', '他們', '這個', '那個', '因為', '所以', '但是', '可是',
  '如果', '就是', '可以', '沒有', '不是', '一個', '現在', '時候', '這樣',
  '覺得', '應該', '知道', '部分', '問題', '意思', '可能', '已經', '還是',
  '然後', '是說', '上次', '不會', '是要', '那我', '對對', '不用', '他要',
  '他的', '我的', '你的', '比較', '其實', '基本', '後面', '裡面', '怎麼',
  '什麼', '那邊', '這邊', '大家', '有些', '確定', '建議', '發現', '開始',
  '出來', '起來', '過去', '進來', '東西', '事情', '的話', '來說', '這件',
]);

const BAD_GRAMMATICAL_EDGE_REGEX =
  /^[的我你他她它這那在有是就也都還又再很太更不沒別好對啊喔嗎呢吧啦]|(的|了|嗎|呢|吧|啊|喔|啦|們|說|要|會|是|有|在|過|著)$/;

export function toAccentInsensitivePhoneticKey(text: string): string {
  return toTaiwanFuzzyPhoneticSyllables(text).join('-');
}

/**
 * 【問題一 1.1 第二維度】：高頻語法虛詞與指代詞集合 \mathcal{F}_{gram}
 * 在計算孤立單字鏈懲罰 L_{eff} 時賦予低孤立權重 0.15，避免把正常口語虛詞誤判為碎字鏈
 */
const GRAMMATICAL_FUNCTION_CHARS = new Set(
  '的地得了著過在從跟和與及或把被對為以比連越再又也還都只就才剛正曾已經常會能可要得著過個位名次件種些點處裡上中下內外前後左右這那哪誰什怎幾多少一二三四五六七八九十百千萬億零我你他她它人嗎呢吧啊喔啦哦欸恩嗯唔哈嘿是就有但而卻讓給向往到不沒很太更別好對算'.split(
    ''
  )
);

const QUESTION_CLAUSE_REGEX =
  /([？?]|怎麼辦|對不對|是不是|好不好|可以嗎|是嗎|對嗎|有沒有|算不算|對吧|嗎[。？?]?|呢[。？?]?|か[。？?]?|でしょうか[。？?]?|ますか[。？?]?|ですか[。？?]?)$/i;

const DIRECT_SHORT_ANSWER_REGEX =
  /^(不算[。，]?|算[。，]?|對[，。啊呀]?|對對對[，。]?|沒錯[，。]?|可以[，。啊]?|沒有[，。啊]?|不用[，。]?|不用不用[，。]?|好[，。的啊]?|是[，。啊的]?|不是[，。啊]?|不行[，。]?|有[，。啊]?|はい|ええ|うん|いいえ|なるほど|そうですね|yeah|yes|no|nope|right|sure|okay|ok)([，。！？!?,.\s]|$)/i;

const RHETORICAL_SELF_ANSWER_BRIDGE_REGEX =
  /^(因為|主要是|其實|也就是說|意思是|換句話說|白話講|我意思是|比如說|比方說)/;

export interface SentenceIntelligibilityReport {
  segmentId: string;
  /** 句子內在語意可理解度分數 I(S) \in [0.0, 1.0] (門檻 \tau_{intel} = 0.68) */
  intelligibilityScore: number;
  /** 有效孤立單字鏈最大長度 L_{eff} */
  maxEffectiveOrphanLen: number;
  /** 有效音節密度 \rho_{syll} (字/秒) */
  syllableDensityPerSec: number;
  /** 最大音韻主題引力張力 G_{phon} */
  maxPhoneticGravityTension: number;
  /** 偵測到之異常懷疑區段 */
  anomalousSpans: Array<{
    text: string;
    startCharIdx: number;
    endCharIdx: number;
    phoneticHypotheses: string[];
    anomalyScore: number;
    anomalyType:
      | 'orphan_char_chain'
      | 'phonetic_gravity_tension'
      | 'collocation_surprisal'
      | 'contraction_split';
  }>;
  /** 智慧重聽決策器：僅在物理有效音節密度嚴重不足 (\rho_{syll} < 2.0) 或無近音候選之嚴重缺損時為 true */
  requireAmplifiedRelisten: boolean;
}

export interface StatisticalTermEvidence {
  term: string;
  exactSig: string;
  fuzzySig: string;
  /** 基礎統計證據質量 E(w) */
  evidenceScore: number;
  /** 作為 3~6 字更長複合詞之子詞幹的種類數 (Compound Witness Productivity，如「業師群」見證「業師」) */
  compoundProductivity: number;
  /** 【問題三 Pass 1】：實體主權分數 \mathcal{S}_{sov}(w)，結合複合詞見證、主題共鳴度與字典先驗 */
  sovereigntyScore: number;
  /** 是否來自字典或背景簡報摘要 */
  isPriorVerified: boolean;
}

/**
 * 【問題三 Pass 1 核心：統計構詞包絡、字典誤聽防污染與實體主權分數 \mathcal{S}_{sov}(w) 探勘引擎】
 */
export function analyzeCorpusStatisticalEvidence(
  allSegments: Segment[],
  dictionary: DictEntry[],
  docSummary: string
): Map<string, StatisticalTermEvidence> {
  const evidenceMap = new Map<string, StatisticalTermEvidence>();

  // 【關鍵防污染機制】：收集三層字典中已明確定義為「常見誤聽別名 (aliases)」的所有詞（例如：夜市、業室、商業室、郎貸、證件）
  // 嚴禁這些誤聽別名因為 Whisper 重複聽錯多次而反客為主進入高頻核心詞庫！
  const aliasToCanonicalMap = new Map<string, string>();
  for (const d of dictionary) {
    const cleanTarget = converter(d.term.trim());
    if (Array.isArray(d.aliases)) {
      for (const al of d.aliases) {
        const cleanAl = converter(al.trim());
        if (cleanAl && cleanAl !== cleanTarget) {
          aliasToCanonicalMap.set(cleanAl, cleanTarget);
        }
      }
    }
  }

  const registerOrAdd = (
    rawTerm: string,
    weightDelta: number,
    isPrior = false
  ) => {
    let term = converter(rawTerm.trim());
    if (aliasToCanonicalMap.has(term)) {
      term = aliasToCanonicalMap.get(term)!;
    }

    if (
      term.length < 2 ||
      term.length > 4 ||
      !/^[\u4e00-\u9fa5]+$/.test(term) ||
      COMMON_STOP_WORDS.has(term) ||
      BAD_GRAMMATICAL_EDGE_REGEX.test(term)
    ) {
      return;
    }
    const existing = evidenceMap.get(term);
    if (existing) {
      existing.evidenceScore += weightDelta;
      if (isPrior) existing.isPriorVerified = true;
    } else {
      evidenceMap.set(term, {
        term,
        exactSig: toPhoneticSignature(term),
        fuzzySig: toAccentInsensitivePhoneticKey(term),
        evidenceScore: weightDelta,
        compoundProductivity: 0,
        sovereigntyScore: weightDelta,
        isPriorVerified: isPrior,
      });
    }
  };

  for (const d of dictionary) {
    if (d.term.length >= 2 && d.term.length <= 4) {
      registerOrAdd(d.term, 18.0, true);
    }
  }
  const summaryWords = docSummary.match(/[\u4e00-\u9fa5]{2,4}/g) || [];
  for (const sw of summaryWords) {
    registerOrAdd(sw, 6.5, true);
  }

  const clauses: string[] = [];
  for (const seg of allSegments) {
    const segText = converter(seg.words.map((w) => w.text).join(''));
    const subClauses = segText
      .split(/[，。？！、；：\s,.?!]+/)
      .filter((c) => c.length >= 2);
    clauses.push(...subClauses);

    for (const w of seg.words) {
      const txt = converter(w.text);
      if (txt.length >= 2 && txt.length <= 4) {
        const confWeight =
          w.status === 'auto_corrected'
            ? 2.4
            : w.confidence >= 0.85
            ? w.confidence * w.confidence * 1.8
            : w.confidence >= 0.72
            ? 0.6
            : 0.15;
        registerOrAdd(txt, confWeight, false);
      }
    }
  }

  const ngramFreq = new Map<string, number>();
  for (const cl of clauses) {
    for (let len = 2; len <= 4; len++) {
      for (let i = 0; i <= cl.length - len; i++) {
        const sub = cl.slice(i, i + len);
        if (
          /^[\u4e00-\u9fa5]+$/.test(sub) &&
          !COMMON_STOP_WORDS.has(sub) &&
          !BAD_GRAMMATICAL_EDGE_REGEX.test(sub)
        ) {
          ngramFreq.set(sub, (ngramFreq.get(sub) || 0) + 1);
        }
      }
    }
  }

  for (const [ng, count] of ngramFreq.entries()) {
    if (count >= 2) {
      registerOrAdd(ng, count * 0.95, false);
    }
  }

  // 【問題三 Pass 1 核心】：三字以上高凝聚度複合詞見證 (Compound Witness，例如「業師群」見證「業師」)
  // 與實體主權分數 \mathcal{S}_{sov}(w) 計算
  const allCandidates = Array.from(evidenceMap.values());
  for (const cand of allCandidates) {
    const term = cand.term;
    const distinctContexts = new Set<string>();
    let compoundWitnessBonus = 0.0;

    for (const cl of clauses) {
      const idx = cl.indexOf(term);
      if (idx !== -1 && cl.length > term.length) {
        const leftChar = idx > 0 ? cl[idx - 1] : '^';
        const rightChar =
          idx + term.length < cl.length ? cl[idx + term.length] : '$';
        if (
          leftChar !== '^' &&
          !GRAMMATICAL_FUNCTION_CHARS.has(leftChar)
        ) {
          distinctContexts.add(`L:${leftChar}`);
          compoundWitnessBonus += 2.2;
        }
        if (
          rightChar !== '$' &&
          !GRAMMATICAL_FUNCTION_CHARS.has(rightChar)
        ) {
          distinctContexts.add(`R:${rightChar}`);
          compoundWitnessBonus += 2.6;
        }
      }
    }

    cand.compoundProductivity = distinctContexts.size;
    cand.evidenceScore += distinctContexts.size * 3.2;
    cand.sovereigntyScore = Number(
      (
        cand.evidenceScore +
        compoundWitnessBonus +
        (cand.isPriorVerified ? 10.0 : 0.0)
      ).toFixed(2)
    );
  }

  return evidenceMap;
}

/**
 * 【問題一 1.1 & 1.2 核心：ISIAL 四維正交語意可理解性與懷疑區段定位器】
 * 四維特徵：
 * 1. 內部構詞凝聚度 (NPMI Cohesion)
 * 2. 扣除語法虛詞之有效孤立單字鏈懲罰 (Weighted Orphan Character Chain Penalty L_{eff})
 * 3. 跨虛詞之實詞局部搭配驚訝度 (Collocational Surprisal)
 * 4. 音韻突變度與全場主題引力張力 (Phonetic-Topic Gravity Tension G_{phon}，支援連音縮讀比對)
 */
export function evaluateSentenceIntelligibility(
  seg: Segment,
  evidenceMap: Map<string, StatisticalTermEvidence>,
  dictionary: DictEntry[],
  rollingSummary = ''
): SentenceIntelligibilityReport {
  const fullText = converter(seg.words.map((w) => w.text).join(''));
  const cleanChars = fullText.replace(/[^\u4e00-\u9fa5]/g, '');
  const durSec = Math.max(0.25, seg.end - seg.start);
  const syllableDensityPerSec = Number((cleanChars.length / durSec).toFixed(2));

  const anomalousSpans: SentenceIntelligibilityReport['anomalousSpans'] = [];

  const aliasSet = new Set<string>();
  const highSovereigntyEntities: StatisticalTermEvidence[] = [];

  for (const d of dictionary) {
    const cleanT = converter(d.term.trim());
    if (d.aliases) {
      for (const al of d.aliases) {
        const cleanAl = converter(al.trim());
        if (cleanAl && cleanAl !== cleanT) aliasSet.add(cleanAl);
      }
    }
    if (!evidenceMap.has(cleanT) && cleanT.length >= 2 && cleanT.length <= 4) {
      highSovereigntyEntities.push({
        term: cleanT,
        exactSig: toPhoneticSignature(cleanT),
        fuzzySig: toAccentInsensitivePhoneticKey(cleanT),
        evidenceScore: 20.0,
        compoundProductivity: 3,
        sovereigntyScore: 30.0,
        isPriorVerified: true,
      });
    }
  }

  for (const ev of evidenceMap.values()) {
    if (ev.isPriorVerified || ev.sovereigntyScore >= 5.0) {
      highSovereigntyEntities.push(ev);
    }
  }

  // 【三源混淆集候選網格 (Confusion Set Lattice) 檢索器】
  const retrieveConfusionLatticeAndGravity = (
    sub: string
  ): { candidates: string[]; gravityTension: number } => {
    const scored: Array<{ term: string; rankScore: number; tension: number }> =
      [];
    const sigmaP = 0.22;

    for (const d of dictionary) {
      const canon = converter(d.term.trim());
      if (canon === sub) continue;
      if (
        d.aliases?.some(
          (al) =>
            converter(al.trim()) === sub ||
            (sub.length >= 3 && sub.includes(converter(al.trim())))
        )
      ) {
        scored.push({ term: canon, rankScore: 2.5, tension: 0.92 });
      }
    }

    for (const entity of highSovereigntyEntities) {
      if (entity.term === sub) continue;
      if (Math.abs(entity.term.length - sub.length) > 1) continue;

      const dPhon = computeTaiwanPhoneticDistance(sub, entity.term);
      if (dPhon <= 0.34) {
        const gaussianKernel = Math.exp(
          -(dPhon * dPhon) / (2 * sigmaP * sigmaP)
        );
        const subSov = aliasSet.has(sub)
          ? 0.1
          : evidenceMap.get(sub)?.sovereigntyScore || 0.5;
        const topicResonanceDiff = Math.min(
          1.0,
          Math.max(0.0, (entity.sovereigntyScore - subSov) / 22.0) +
            (rollingSummary.includes(entity.term) ? 0.25 : 0.0)
        );
        const tension = Number(
          (gaussianKernel * Math.max(0.35, topicResonanceDiff)).toFixed(3)
        );
        const sourceLambda = entity.isPriorVerified ? 1.25 : 1.35;
        const rankScore = sourceLambda * tension;

        if (tension >= 0.32) {
          scored.push({ term: entity.term, rankScore, tension });
        }
      }
    }

    scored.sort((a, b) => b.rankScore - a.rankScore);
    const uniqueTerms: string[] = [];
    let maxTension = 0.0;
    for (const item of scored) {
      if (!uniqueTerms.includes(item.term)) {
        uniqueTerms.push(item.term);
        if (item.tension > maxTension) maxTension = item.tension;
      }
      if (uniqueTerms.length >= 4) break;
    }

    return { candidates: uniqueTerms, gravityTension: maxTension };
  };

  // ── 維度 2：扣除語法虛詞之有效孤立單字鏈懲罰 L_{eff} ──
  let maxEffectiveOrphanLen = 0.0;
  let currentOrphanEffLen = 0.0;
  let orphanRunText = '';
  let orphanStartCharIdx = 0;
  let charCursor = 0;

  for (let wIdx = 0; wIdx < seg.words.length; wIdx++) {
    const wText = converter(seg.words[wIdx].text);
    const isCjkWord = /^[\u4e00-\u9fa5]+$/.test(wText);

    if (isCjkWord && wText.length === 1) {
      const wOrphan = GRAMMATICAL_FUNCTION_CHARS.has(wText) ? 0.15 : 1.0;
      if (currentOrphanEffLen === 0 && wOrphan >= 0.8) {
        orphanStartCharIdx = charCursor;
      }
      if (currentOrphanEffLen > 0 || wOrphan >= 0.8) {
        currentOrphanEffLen += wOrphan;
        orphanRunText += wText;
      }
    } else {
      if (currentOrphanEffLen >= 1.8 && orphanRunText.length >= 2) {
        maxEffectiveOrphanLen = Math.max(
          maxEffectiveOrphanLen,
          currentOrphanEffLen
        );
        const { candidates, gravityTension } =
          retrieveConfusionLatticeAndGravity(orphanRunText);
        anomalousSpans.push({
          text: orphanRunText,
          startCharIdx: orphanStartCharIdx,
          endCharIdx: orphanStartCharIdx + orphanRunText.length,
          phoneticHypotheses: candidates,
          anomalyScore: Number(
            Math.min(
              0.98,
              0.45 + currentOrphanEffLen * 0.15 + gravityTension * 0.3
            ).toFixed(2)
          ),
          anomalyType: 'orphan_char_chain',
        });
      }
      currentOrphanEffLen = 0.0;
      orphanRunText = '';
    }
    charCursor += wText.length;
  }

  if (currentOrphanEffLen >= 1.8 && orphanRunText.length >= 2) {
    maxEffectiveOrphanLen = Math.max(
      maxEffectiveOrphanLen,
      currentOrphanEffLen
    );
    const { candidates, gravityTension } =
      retrieveConfusionLatticeAndGravity(orphanRunText);
    anomalousSpans.push({
      text: orphanRunText,
      startCharIdx: orphanStartCharIdx,
      endCharIdx: orphanStartCharIdx + orphanRunText.length,
      phoneticHypotheses: candidates,
      anomalyScore: Number(
        Math.min(
          0.98,
          0.45 + currentOrphanEffLen * 0.15 + gravityTension * 0.3
        ).toFixed(2)
      ),
      anomalyType: 'orphan_char_chain',
    });
  }

  // ── 維度 1、3、4：掃描 2~4 字子字串之構詞凝聚度、跨音節連音縮讀與音韻引力張力 G_{phon} ──
  let maxPhoneticGravityTension = 0.0;

  for (let len = 4; len >= 2; len--) {
    for (let i = 0; i <= fullText.length - len; i++) {
      const sub = fullText.slice(i, i + len);
      if (
        !/^[\u4e00-\u9fa5]+$/.test(sub) ||
        COMMON_STOP_WORDS.has(sub) ||
        BAD_GRAMMATICAL_EDGE_REGEX.test(sub)
      ) {
        continue;
      }

      if (
        anomalousSpans.some(
          (a) => i >= a.startCharIdx && i + len <= a.endCharIdx
        )
      ) {
        continue;
      }

      const subEv = evidenceMap.get(sub);
      const isKnownMisheardAlias = aliasSet.has(sub);
      const isLowPlausibility =
        isKnownMisheardAlias ||
        !subEv ||
        (!subEv.isPriorVerified &&
          subEv.compoundProductivity === 0 &&
          subEv.sovereigntyScore < 3.5);

      if (isLowPlausibility) {
        const { candidates, gravityTension } =
          retrieveConfusionLatticeAndGravity(sub);
        if (gravityTension > maxPhoneticGravityTension) {
          maxPhoneticGravityTension = gravityTension;
        }
        if (candidates.length > 0 && gravityTension >= 0.42) {
          const isContraction = candidates.some(
            (c) => c.length !== sub.length
          );
          anomalousSpans.push({
            text: sub,
            startCharIdx: i,
            endCharIdx: i + len,
            phoneticHypotheses: candidates,
            anomalyScore: Number(
              Math.min(0.98, 0.35 + gravityTension * 0.65).toFixed(2)
            ),
            anomalyType: isContraction
              ? 'contraction_split'
              : 'phonetic_gravity_tension',
          });
        }
      }
    }
  }

  // 形態學膨脹 (Morphological Dilation)：若兩個異常區段距離 <= 1 字（如「郎貸」與「證件」），合併為單一複合槽位
  anomalousSpans.sort((a, b) => a.startCharIdx - b.startCharIdx);
  const dilatedSpans: typeof anomalousSpans = [];
  for (const sp of anomalousSpans) {
    const prev = dilatedSpans[dilatedSpans.length - 1];
    if (
      prev &&
      sp.startCharIdx - prev.endCharIdx <= 1 &&
      sp.endCharIdx - prev.startCharIdx <= 6
    ) {
      const mergedText = fullText.slice(prev.startCharIdx, sp.endCharIdx);
      const combinedHyps = Array.from(
        new Set([
          ...prev.phoneticHypotheses.flatMap((h1) =>
            sp.phoneticHypotheses.map((h2) => `${h1}${h2}`)
          ),
          ...prev.phoneticHypotheses,
          ...sp.phoneticHypotheses,
        ])
      ).slice(0, 4);
      prev.text = mergedText;
      prev.endCharIdx = sp.endCharIdx;
      prev.phoneticHypotheses = combinedHyps;
      prev.anomalyScore = Math.max(prev.anomalyScore, sp.anomalyScore);
    } else {
      dilatedSpans.push({ ...sp });
    }
  }

  // ── 計算句子內在語意可理解度分數 I(S) = 1 - (\alpha \bar{A}(S) + \beta \max A(c_t)) ──
  const avgWordConf =
    seg.words.reduce((acc, w) => acc + w.confidence, 0) /
    Math.max(1, seg.words.length);

  const orphanPenalty =
    maxEffectiveOrphanLen > 1.2
      ? 1.0 - Math.exp(-0.35 * Math.pow(maxEffectiveOrphanLen - 1.2, 2))
      : 0.0;

  const maxSpanAnomaly =
    dilatedSpans.length > 0
      ? Math.max(...dilatedSpans.map((s) => s.anomalyScore))
      : 0.0;

  const meanAnomaly =
    (1.0 - avgWordConf) * 0.25 +
    orphanPenalty * 0.4 +
    maxPhoneticGravityTension * 0.35;

  const combinedPenalty = 0.35 * meanAnomaly + 0.65 * maxSpanAnomaly;

  // 【問題四：智慧重聽決策器】
  // 嚴格限制僅在「有效音節密度嚴重不足 (\rho_{syll} < 2.0 字/秒)」或「短促殘缺且無明確同音候選可由文字修補」時才觸發重聽！
  const hasPhysicalSyllableDeficit =
    (durSec >= 2.2 && syllableDensityPerSec < 2.0) ||
    (durSec <= 1.5 && cleanChars.length <= 2 && avgWordConf < 0.8) ||
    syllableDensityPerSec > 6.4;

  const hasUnresolvedAcousticAnomaly =
    dilatedSpans.some((s) => s.phoneticHypotheses.length === 0) &&
    avgWordConf < 0.74;

  const intelligibilityScore = Number(
    Math.max(
      0.12,
      Math.min(
        0.99,
        1.0 - combinedPenalty - (hasPhysicalSyllableDeficit ? 0.16 : 0.0)
      )
    ).toFixed(2)
  );

  return {
    segmentId: seg.id,
    intelligibilityScore,
    maxEffectiveOrphanLen: Number(maxEffectiveOrphanLen.toFixed(2)),
    syllableDensityPerSec,
    maxPhoneticGravityTension: Number(maxPhoneticGravityTension.toFixed(2)),
    anomalousSpans: dilatedSpans.slice(0, 3),
    requireAmplifiedRelisten:
      hasPhysicalSyllableDeficit || hasUnresolvedAcousticAnomaly,
  };
}

/**
 * 【問題四：四重硬性防退化驗收閘門 (Non-Regression Acceptance Gate)】
 * 任何重聽結果或整句改寫必須同時通過以下 4 項指標才准許覆蓋原句：
 * 1. 標點破碎度 F_{frag}：句中異常句號/問號切割數不得增加（防範「我覺得阿如果。算，我」劣化）
 * 2. 字數合理比 \eta_{len} \in [0.78, 1.45]
 * 3. 音韻保真度 Sim_{phon} = 1 - d_{phon}(S_{orig}, S_{new}) >= 0.58
 * 4. 可理解度淨增量 \Delta U = 100 * (I(S_{new}) - I(S_{orig})) >= 3.5 (即 \Delta I(S) >= 0.035)
 */
export function verifyNonRegressionAcceptanceGate(params: {
  origSegment: Segment;
  candidateSegment: Segment;
  evidenceMap: Map<string, StatisticalTermEvidence>;
  dictionary: DictEntry[];
  rollingSummary: string;
}): {
  accepted: boolean;
  deltaU: number;
  phoneticSimilarity: number;
  lengthRatio: number;
  hasBrokenFragmentation: boolean;
} {
  const {
    origSegment,
    candidateSegment,
    evidenceMap,
    dictionary,
    rollingSummary,
  } = params;

  const origText = converter(origSegment.words.map((w) => w.text).join(''));
  const newText = converter(
    candidateSegment.words.map((w) => w.text).join('')
  );

  const origClean = origText.replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, '');
  const newClean = newText.replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, '');

  if (!newClean) {
    return {
      accepted: false,
      deltaU: -100,
      phoneticSimilarity: 0,
      lengthRatio: 0,
      hasBrokenFragmentation: true,
    };
  }

  const origIntel = evaluateSentenceIntelligibility(
    origSegment,
    evidenceMap,
    dictionary,
    rollingSummary
  );
  const newIntel = evaluateSentenceIntelligibility(
    candidateSegment,
    evidenceMap,
    dictionary,
    rollingSummary
  );

  // 1. 標點破碎度檢查 F_{frag}
  const origMidPeriodCount = (origText.slice(0, -1).match(/[。？！]/g) || [])
    .length;
  const newMidPeriodCount = (newText.slice(0, -1).match(/[。？！]/g) || [])
    .length;
  const hasBrokenFragmentation =
    newMidPeriodCount > origMidPeriodCount && newClean.length < 24;

  // 2. 字數合理比 \eta_{len}
  const lengthRatio = newClean.length / Math.max(1, origClean.length);
  const isLengthReasonable =
    newClean.length >= Math.max(2, Math.floor(origClean.length * 0.78)) &&
    newClean.length <= Math.ceil(origClean.length * 1.45) + 6;

  // 3. 音韻保真度 Sim_{phon}
  const phoneticDistance = computeTaiwanPhoneticDistance(origClean, newClean);
  const phoneticSimilarity = Number((1.0 - phoneticDistance).toFixed(3));
  const isPhoneticFaithful =
    phoneticSimilarity >= 0.58 || origIntel.syllableDensityPerSec < 2.0;

  // 4. 可理解度淨增量 \Delta U >= 3.5 (對應 \Delta I(S) >= 0.035)
  const deltaU = Number(
    ((newIntel.intelligibilityScore - origIntel.intelligibilityScore) * 100).toFixed(
      2
    )
  );

  const accepted =
    !hasBrokenFragmentation &&
    isLengthReasonable &&
    isPhoneticFaithful &&
    deltaU >= 3.5;

  return {
    accepted,
    deltaU,
    phoneticSimilarity,
    lengthRatio: Number(lengthRatio.toFixed(2)),
    hasBrokenFragmentation,
  };
}

/**
 * 【問題二 2.3 核心】：為目標視窗段落建構免 wordIndex 之「內嵌混淆槽位標籤」
 * 將懷疑區段替換為 ⟪#S1: 原詞 | 候選: 候選1, 候選2⟫ 並建立 SlotMap
 */
export function buildSlottedPromptAndMap(
  targetSegments: Segment[],
  evidenceMap: Map<string, StatisticalTermEvidence>,
  dictionary: DictEntry[],
  rollingSummary = ''
): {
  suspectSlots: SuspectSlotItem[];
  slottedPromptTranscript: string;
} {
  const suspectSlots: SuspectSlotItem[] = [];
  let slotCounter = 1;

  const lines: string[] = [];

  for (const seg of targetSegments) {
    const dur = seg.end - seg.start;
    const margin = seg.acousticMargin ?? 0.18;
    const isLocked = margin >= 0.16 && dur >= 1.2;
    const gateTag = isLocked
      ? `[LOCKED | speaker=${seg.speakerId}]`
      : `[REVIEW | top1=${seg.speakerId}, top2=${
          seg.runnerUpSpeakerId || seg.speakerId
        }, margin=${margin.toFixed(2)}]`;

    const intel = evaluateSentenceIntelligibility(
      seg,
      evidenceMap,
      dictionary,
      rollingSummary
    );

    let charPos = 0;
    const wordRanges = seg.words.map((w, idx) => {
      const start = charPos;
      charPos += w.text.length;
      return { idx, start, end: charPos, word: w };
    });

    const segSlots: Array<{
      slotId: string;
      startWordIdx: number;
      endWordIdx: number;
      originalText: string;
      candidates: string[];
    }> = [];

    for (const anom of intel.anomalousSpans) {
      if (slotCounter > 5) break;
      const hitWords = wordRanges.filter(
        (r) => r.end > anom.startCharIdx && r.start < anom.endCharIdx
      );
      if (hitWords.length === 0) continue;

      const sWIdx = hitWords[0].idx;
      const eWIdx = hitWords[hitWords.length - 1].idx + 1;
      if (segSlots.some((s) => sWIdx < s.endWordIdx && eWIdx > s.startWordIdx)) {
        continue;
      }

      const slotId = `#S${slotCounter++}`;
      const origSpanText = seg.words
        .slice(sWIdx, eWIdx)
        .map((w) => w.text)
        .join('');
      const leftContext = seg.words
        .slice(Math.max(0, sWIdx - 2), sWIdx)
        .map((w) => w.text)
        .join('');
      const rightContext = seg.words
        .slice(eWIdx, Math.min(seg.words.length, eWIdx + 2))
        .map((w) => w.text)
        .join('');

      const cands = Array.from(
        new Set([
          ...anom.phoneticHypotheses,
          ...hitWords.flatMap((hw) => hw.word.candidates || []),
        ])
      ).slice(0, 4);

      suspectSlots.push({
        slotId,
        segmentId: seg.id,
        startWordIdx: sWIdx,
        endWordIdx: eWIdx,
        originalText: origSpanText,
        leftContext,
        rightContext,
        anomalyScore: anom.anomalyScore,
        anomalyType: anom.anomalyType,
        candidates: cands,
      });

      segSlots.push({
        slotId,
        startWordIdx: sWIdx,
        endWordIdx: eWIdx,
        originalText: origSpanText,
        candidates: cands,
      });
    }

    let renderedWords = '';
    let wCursor = 0;
    while (wCursor < seg.words.length) {
      const matchedSlot = segSlots.find((s) => s.startWordIdx === wCursor);
      if (matchedSlot) {
        const candStr =
          matchedSlot.candidates.length > 0
            ? matchedSlot.candidates.join(', ')
            : '依上下文推敲';
        renderedWords += `⟪${matchedSlot.slotId}: ${matchedSlot.originalText} | 候選: ${candStr}⟫`;
        wCursor = matchedSlot.endWordIdx;
      } else {
        renderedWords += seg.words[wCursor].text;
        wCursor++;
      }
    }

    lines.push(`[${seg.id}] ${gateTag} ${renderedWords}`);
  }

  return {
    suspectSlots,
    slottedPromptTranscript: lines.join('\n'),
  };
}

/**
 * 【問題三 LG-TPST：偵測不合理「自問自答」並自動拆句移轉給前、後語者】
 */
export function resolveUnreasonableSelfQAAndTransferSpeaker(
  segments: Segment[],
  speakers: Record<string, Speaker>
): {
  resolvedSegments: Segment[];
  transferredSegmentIds: string[];
  flaggedForRelistenIds: string[];
} {
  if (segments.length === 0) {
    return {
      resolvedSegments: [],
      transferredSegmentIds: [],
      flaggedForRelistenIds: [],
    };
  }

  const allSpeakerIds = Object.keys(speakers);
  const transferredSegmentIds: string[] = [];
  const flaggedForRelistenIds: string[] = [];
  const expanded: Segment[] = [];

  const findInterlocutorSpeakerId = (
    currentSpk: string,
    runnerUpSpk?: string,
    neighborSpk?: string
  ): string => {
    if (neighborSpk && neighborSpk !== currentSpk) return neighborSpk;
    if (runnerUpSpk && runnerUpSpk !== currentSpk) return runnerUpSpk;
    const alt = allSpeakerIds.find((id) => id !== currentSpk);
    return alt || currentSpk;
  };

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const prevSeg = expanded.length > 0 ? expanded[expanded.length - 1] : null;
    const nextSeg = i + 1 < segments.length ? segments[i + 1] : null;

    const qWordIdx = seg.words.findIndex((w) => /[？?]/.test(w.text));

    if (qWordIdx >= 0 && qWordIdx < seg.words.length - 1) {
      const beforeWords = seg.words.slice(0, qWordIdx + 1);
      const afterWords = seg.words.slice(qWordIdx + 1);
      const beforeText = beforeWords.map((w) => w.text).join('').trim();
      const afterText = afterWords.map((w) => w.text).join('').trim();

      const isRhetoricalSelfExplain =
        RHETORICAL_SELF_ANSWER_BRIDGE_REGEX.test(afterText);

      if (
        !isRhetoricalSelfExplain &&
        beforeText.length >= 3 &&
        (DIRECT_SHORT_ANSWER_REGEX.test(afterText) || afterText.length <= 10)
      ) {
        const splitTime = Number(
          beforeWords[beforeWords.length - 1].end.toFixed(2)
        );
        const targetAnswerSpeaker = findInterlocutorSpeakerId(
          seg.speakerId,
          seg.runnerUpSpeakerId,
          nextSeg?.speakerId !== seg.speakerId
            ? nextSeg?.speakerId
            : prevSeg?.speakerId
        );

        if (targetAnswerSpeaker !== seg.speakerId) {
          const questionSeg: Segment = {
            ...seg,
            end: splitTime,
            words: beforeWords,
          };
          const answerSegId = `${seg.id}_qa_ans`;
          const answerSeg: Segment = {
            ...seg,
            id: answerSegId,
            speakerId: targetAnswerSpeaker,
            runnerUpSpeakerId: seg.speakerId,
            start: splitTime,
            end: Math.max(splitTime + 0.25, seg.end),
            words: afterWords.map((w, idx) => ({
              ...w,
              id: `${answerSegId}_w_${idx}`,
              status: 'auto_corrected' as const,
              reason: `LG-TPST 消除段內自問自答，將回答移轉至互動語者 (${targetAnswerSpeaker})`,
            })),
            isCorrected: true,
          };

          expanded.push(questionSeg, answerSeg);
          transferredSegmentIds.push(answerSegId);
          continue;
        }
      }

      if (
        !isRhetoricalSelfExplain &&
        beforeText.length <= 9 &&
        afterText.length >= 14 &&
        prevSeg
      ) {
        const questionOwner = findInterlocutorSpeakerId(
          seg.speakerId,
          seg.runnerUpSpeakerId,
          prevSeg.speakerId
        );

        if (questionOwner !== seg.speakerId) {
          const splitTime = Number(
            beforeWords[beforeWords.length - 1].end.toFixed(2)
          );
          const qSegId = `${seg.id}_qa_q`;
          const questionSeg: Segment = {
            ...seg,
            id: qSegId,
            speakerId: questionOwner,
            runnerUpSpeakerId: seg.speakerId,
            start: seg.start,
            end: splitTime,
            words: beforeWords.map((w, idx) => ({
              ...w,
              id: `${qSegId}_w_${idx}`,
              status: 'auto_corrected' as const,
              reason: `LG-TPST 句首短問與長答分離，將提問歸還前位語者 (${questionOwner})`,
            })),
            isCorrected: true,
          };
          const answerSeg: Segment = {
            ...seg,
            start: splitTime,
            words: afterWords,
          };

          expanded.push(questionSeg, answerSeg);
          transferredSegmentIds.push(qSegId);
          continue;
        }
      }
    }

    expanded.push(seg);
  }

  for (let i = 1; i < expanded.length; i++) {
    const prev = expanded[i - 1];
    const curr = expanded[i];
    if (prev.speakerId !== curr.speakerId) continue;

    const prevText = prev.words.map((w) => w.text).join('').trim();
    const currText = curr.words.map((w) => w.text).join('').trim();

    if (
      QUESTION_CLAUSE_REGEX.test(prevText) &&
      !RHETORICAL_SELF_ANSWER_BRIDGE_REGEX.test(currText) &&
      (DIRECT_SHORT_ANSWER_REGEX.test(currText) ||
        (curr.end - curr.start <= 2.2 && (curr.acousticMargin ?? 0.15) < 0.2))
    ) {
      const nextSeg = i + 1 < expanded.length ? expanded[i + 1] : null;
      const prevPrevSeg = i >= 2 ? expanded[i - 2] : null;
      const targetSpk = findInterlocutorSpeakerId(
        curr.speakerId,
        curr.runnerUpSpeakerId,
        nextSeg?.speakerId !== curr.speakerId
          ? nextSeg?.speakerId
          : prevPrevSeg?.speakerId
      );

      if (targetSpk !== curr.speakerId) {
        curr.speakerId = targetSpk;
        curr.isCorrected = true;
        if (curr.words.length > 0) {
          curr.words[0] = {
            ...curr.words[0],
            status: 'auto_corrected',
            reason: `LG-TPST 消除跨段自問自答衝突，移轉至應答語者 (${targetSpk})`,
          };
        }
        transferredSegmentIds.push(curr.id);
      }
    }
  }

  return {
    resolvedSegments: expanded,
    transferredSegmentIds,
    flaggedForRelistenIds,
  };
}

export function mineAdaptiveHighFreqTerms(
  allSegments: Segment[],
  dictionary: DictEntry[],
  docSummary: string
): string[] {
  const evidenceMap = analyzeCorpusStatisticalEvidence(
    allSegments,
    dictionary,
    docSummary
  );

  return Array.from(evidenceMap.values())
    .filter((item) => item.isPriorVerified || item.sovereigntyScore >= 3.8)
    .sort((a, b) => b.sovereigntyScore - a.sovereigntyScore)
    .map((item) => item.term)
    .slice(0, 85);
}

export async function evolveRollingMeetingSummary(params: {
  config: EngineConfig;
  docContextSummary: string;
  currentSummary: string;
  recentSegments: Segment[];
  dictionary: DictEntry[];
}): Promise<string> {
  const {
    config,
    docContextSummary,
    currentSummary,
    recentSegments,
    dictionary,
  } = params;
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
        const summary = converter(
          (data?.candidates?.[0]?.content?.parts?.[0]?.text || '').trim()
        );
        if (summary) return summary;
      }
    } catch {
      // fallback below
    }
  }

  const allSegs = useAppStore.getState().segments;
  const minedTerms = mineAdaptiveHighFreqTerms(
    allSegs,
    dictionary,
    docContextSummary
  ).slice(0, 14);

  const sentences = recentText
    .split(/[。？！]/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 10 && s.length <= 48);

  const genericDecisionRegex =
    /(建議|決定|決議|確認|共識|結論|原則上|重點|目標|進度|預算|時程|方案|規劃|調整|負責|下次|後續)/;
  const decisionSentences = sentences.filter(
    (s) =>
      genericDecisionRegex.test(s) ||
      minedTerms.slice(0, 5).some((term) => term.length >= 2 && s.includes(term))
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
    minedTerms.length > 0 ? `【全場高頻核心詞】${minedTerms.join('、')}` : '',
    mergedConsensus.length > 0
      ? `【近期共識】${mergedConsensus.join('；')}`
      : '',
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
  const evidenceMap = analyzeCorpusStatisticalEvidence(
    allSegs.length > 0 ? allSegs : targetWindowSegments,
    dictionary,
    `${docContextSummary}\n${rollingMeetingSummary}`
  );
  const adaptiveTerms = Array.from(evidenceMap.values())
    .filter((item) => item.isPriorVerified || item.sovereigntyScore >= 3.8)
    .sort((a, b) => b.sovereigntyScore - a.sovereigntyScore)
    .map((item) => item.term)
    .slice(0, 85);

  const digestSentences: string[] = [];
  let digestLen = 0;
  for (
    let i = Math.max(0, earlierSegments.length - 10);
    i < earlierSegments.length;
    i++
  ) {
    const s = earlierSegments[i];
    const txt = s.words.map((w) => w.text).join('');
    if (txt.length >= 10 && digestLen + txt.length <= 200) {
      digestSentences.push(txt);
      digestLen += txt.length;
    }
  }

  // 建立免索引槽位映射與內嵌槽位標籤文本
  const { suspectSlots, slottedPromptTranscript } = buildSlottedPromptAndMap(
    targetWindowSegments,
    evidenceMap,
    dictionary,
    rollingMeetingSummary
  );

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
    suspectSlots,
    slottedPromptTranscript,
  };
}

/**
 * 【Pass 0 + Pass 1 整合】：即時確定性清理 + 跨句實體主權分數 \mathcal{S}_{sov}(w) 音韻自舉收斂 + ISIAL 懷疑區段標記
 */
export function applySentenceLevelLocalCorrection(
  targetSegments: Segment[],
  verifiedTerms: string[],
  dictionary: DictEntry[] = []
): { updatedSegments: Segment[]; patches: CorrectionPatchItem[] } {
  const patches: CorrectionPatchItem[] = [];

  const state = useAppStore.getState();
  const allSegs = state.segments.length > 0 ? state.segments : targetSegments;
  const evidenceMap = analyzeCorpusStatisticalEvidence(
    allSegs,
    dictionary,
    state.docContextSummary
  );

  for (const vt of verifiedTerms) {
    if (!evidenceMap.has(vt) && vt.length >= 2 && vt.length <= 4) {
      evidenceMap.set(vt, {
        term: vt,
        exactSig: toPhoneticSignature(vt),
        fuzzySig: toAccentInsensitivePhoneticKey(vt),
        evidenceScore: 8.5,
        compoundProductivity: 2,
        sovereigntyScore: 14.0,
        isPriorVerified: false,
      });
    }
  }

  const dynamicRules: Array<{ wrong: string; right: string; reason: string }> =
    [];
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
            reason: `Pass 0 依${scopeLabel}同音別名自動修正`,
          });
        }
      }
    }
  }

  // Pass 1：依據實體主權分數 \mathcal{S}_{sov}(w) 建立「精確聲韻主導表」與「台式模糊音韻主導表」
  const exactDominantMap = new Map<string, StatisticalTermEvidence>();
  const fuzzyDominantMap = new Map<string, StatisticalTermEvidence>();

  for (const item of evidenceMap.values()) {
    if (item.sovereigntyScore < 5.0) continue;

    const prevExact = exactDominantMap.get(item.exactSig);
    if (!prevExact || item.sovereigntyScore > prevExact.sovereigntyScore) {
      exactDominantMap.set(item.exactSig, item);
    }

    // 模糊音韻主導錨點必須具備字典先驗或三字以上複合詞見證 (compoundProductivity >= 2)
    if (item.isPriorVerified || item.compoundProductivity >= 2) {
      const prevFuzzy = fuzzyDominantMap.get(item.fuzzySig);
      if (!prevFuzzy || item.sovereigntyScore > prevFuzzy.sovereigntyScore) {
        fuzzyDominantMap.set(item.fuzzySig, item);
      }
    }
  }

  const exactEntries = Array.from(exactDominantMap.entries());
  const fuzzyEntries = Array.from(fuzzyDominantMap.entries());

  const updatedSegments = targetSegments.map((seg) => {
    const rawSentence = converter(seg.words.map((w) => w.text).join(''));
    let refinedSentence = rawSentence.replace(/？[，。、]+/g, '？');

    const hitReasons: Array<{ wrong: string; right: string; reason: string }> =
      [];

    for (const rule of UNIVERSAL_COLLOQUIAL_RULES) {
      if (rule.pattern.test(refinedSentence)) {
        refinedSentence = refinedSentence.replace(
          rule.pattern,
          (matched, ...args) => {
            let replacement = rule.right;
            if (replacement.includes('$1') && typeof args[0] === 'string') {
              replacement = replacement
                .replace('$1', args[0] || '')
                .replace('$2', typeof args[1] === 'string' ? args[1] : '')
                .replace('$3', typeof args[2] === 'string' ? args[2] : '');
            }
            if (matched !== replacement) {
              hitReasons.push({
                wrong: matched,
                right: replacement,
                reason: rule.reason,
              });
            }
            return replacement;
          }
        );
      }
    }

    for (const dRule of dynamicRules) {
      if (refinedSentence.includes(dRule.wrong)) {
        refinedSentence = refinedSentence.split(dRule.wrong).join(dRule.right);
        hitReasons.push(dRule);
      }
    }

    // 消除替換後可能產生的相鄰重複詞組（防禦「店店面翻修翻修」）
    refinedSentence = refinedSentence
      .replace(/([\u4e00-\u9fa5]{2,3})\1+/g, '$1')
      .replace(/([\u4e00-\u9fa5])\1(?=[\u4e00-\u9fa5]{2})/g, '$1')
      .replace(/店店面/g, '店面')
      .replace(/翻修修/g, '翻修');

    // Pass 1 第一層：精確聲韻母實體主權收斂
    for (const [sig, anchor] of exactEntries) {
      const correctTerm = anchor.term;
      if (
        sig.includes('-') &&
        correctTerm.length >= 2 &&
        correctTerm.length <= 4
      ) {
        const len = correctTerm.length;
        for (let i = 0; i <= refinedSentence.length - len; i++) {
          const sub = refinedSentence.slice(i, i + len);
          if (
            sub !== correctTerm &&
            /^[\u4e00-\u9fa5]+$/.test(sub) &&
            !COMMON_STOP_WORDS.has(sub) &&
            !BAD_GRAMMATICAL_EDGE_REGEX.test(sub)
          ) {
            if (toPhoneticSignature(sub) === sig) {
              const subSov = evidenceMap.get(sub)?.sovereigntyScore || 0.5;
              if (anchor.sovereigntyScore >= subSov * 1.8) {
                hitReasons.push({
                  wrong: sub,
                  right: correctTerm,
                  reason: `Pass 1 依實體主權分數 (${anchor.sovereigntyScore}) 自動收斂為「${correctTerm}」`,
                });
                refinedSentence =
                  refinedSentence.slice(0, i) +
                  correctTerm +
                  refinedSentence.slice(i + len);
              }
            }
          }
        }
      }
    }

    // Pass 1 第二層：台式模糊音韻與複合詞見證收斂（如後文「業師群」見證回頭修正前文「夜市」）
    for (const [fKey, anchor] of fuzzyEntries) {
      const correctTerm = anchor.term;
      if (
        fKey.includes('-') &&
        correctTerm.length >= 2 &&
        correctTerm.length <= 4 &&
        anchor.sovereigntyScore >= 8.0
      ) {
        const len = correctTerm.length;
        for (let i = 0; i <= refinedSentence.length - len; i++) {
          const sub = refinedSentence.slice(i, i + len);
          if (
            sub !== correctTerm &&
            /^[\u4e00-\u9fa5]+$/.test(sub) &&
            !COMMON_STOP_WORDS.has(sub) &&
            !BAD_GRAMMATICAL_EDGE_REGEX.test(sub)
          ) {
            if (toAccentInsensitivePhoneticKey(sub) === fKey) {
              const subItem = evidenceMap.get(sub);
              const subSov = subItem?.sovereigntyScore || 0.5;
              const subProd = subItem?.compoundProductivity || 0;
              if (
                anchor.sovereigntyScore >= subSov * 2.0 &&
                (anchor.compoundProductivity > subProd || anchor.isPriorVerified)
              ) {
                hitReasons.push({
                  wrong: sub,
                  right: correctTerm,
                  reason: `Pass 1 依複合詞見證與模糊音韻主權自動收斂為「${correctTerm}」`,
                });
                refinedSentence =
                  refinedSentence.slice(0, i) +
                  correctTerm +
                  refinedSentence.slice(i + len);
              }
            }
          }
        }
      }
    }

    const intelReport = evaluateSentenceIntelligibility(
      { ...seg, words: seg.words },
      evidenceMap,
      dictionary,
      state.rollingMeetingSummary
    );

    if (
      refinedSentence === rawSentence &&
      intelReport.anomalousSpans.length === 0
    ) {
      return { ...seg, isCorrected: true };
    }

    const origAvgConf =
      seg.words.reduce((acc, w) => acc + w.confidence, 0) /
      Math.max(1, seg.words.length);

    const tokens =
      refinedSentence.match(
        /[\u4e00-\u9fa5]{1,2}|[\u3040-\u30ff]{1,3}|[a-zA-Z0-9']+\s*|[^\s\u4e00-\u9fa5\u3040-\u30ffa-zA-Z0-9]/g
      ) || [refinedSentence];
    const dur = Math.max(0.2, seg.end - seg.start);
    const step = dur / Math.max(1, tokens.length);

    const newWords: WordSpan[] = tokens.map((tok, idx) => {
      const matchedRule = hitReasons.find(
        (r) => r.right === tok || (r.right.length > 2 && r.right.includes(tok))
      );
      if (matchedRule && /^[\u4e00-\u9fa5]+$/.test(tok)) {
        const safeOriginalText =
          matchedRule.wrong.length === tok.length ? matchedRule.wrong : tok;
        patches.push({
          segmentId: seg.id,
          wordIndex: idx,
          originalText: safeOriginalText,
          status: 'auto_corrected',
          correctedText: tok,
          reason: matchedRule.reason,
        });
        return {
          id: `${seg.id}_w_${idx}`,
          text: tok,
          originalText: safeOriginalText,
          start: Number((seg.start + idx * step).toFixed(2)),
          end: Number((seg.start + (idx + 1) * step).toFixed(2)),
          confidence: 0.96,
          status: 'auto_corrected',
          reason: matchedRule.reason,
        };
      }

      const matchedAnomaly = intelReport.anomalousSpans.find(
        (a) => a.text === tok || (a.text.length <= 4 && a.text.includes(tok))
      );
      if (matchedAnomaly && /^[\u4e00-\u9fa5]+$/.test(tok)) {
        const candidates = matchedAnomaly.phoneticHypotheses.slice(0, 4);
        patches.push({
          segmentId: seg.id,
          wordIndex: idx,
          originalText: tok,
          status: 'semantic_uncertain',
          candidates,
          reason:
            candidates.length > 0
              ? `ISIAL 音韻引力異常，候選：${candidates.join(' / ')}`
              : '局部構詞斷裂（孤立碎字），待上下文推敲或重聽',
          requireRelisten: intelReport.requireAmplifiedRelisten,
        });
        return {
          id: `${seg.id}_w_${idx}`,
          text: tok,
          start: Number((seg.start + idx * step).toFixed(2)),
          end: Number((seg.start + (idx + 1) * step).toFixed(2)),
          confidence: Math.min(0.68, origAvgConf),
          status: 'semantic_uncertain',
          candidates,
          reason:
            candidates.length > 0
              ? `近音候選：${candidates.join(' / ')}`
              : '語意存疑待重聽',
        };
      }

      const prevWordSameIdx = seg.words[idx];
      const preservedConf = prevWordSameIdx
        ? prevWordSameIdx.confidence
        : Number(origAvgConf.toFixed(2));
      const preservedStatus =
        prevWordSameIdx && prevWordSameIdx.status !== 'auto_corrected'
          ? prevWordSameIdx.status
          : 'normal';

      return {
        id: `${seg.id}_w_${idx}`,
        text: tok,
        start: Number((seg.start + idx * step).toFixed(2)),
        end: Number((seg.start + (idx + 1) * step).toFixed(2)),
        confidence: preservedConf,
        status: preservedStatus,
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

/**
 * 【問題三 Pass 2：滾動視窗之免索引槽位 LLM 深度語意推敲與 LG-TPST 自問自答移轉】
 */
export async function runTriLevelWindowCorrection(params: {
  config: EngineConfig;
  payload: TriLevelContextPayload;
  dictionary: DictEntry[];
  runLocalLlmCorrection?: (
    payload: TriLevelContextPayload
  ) => Promise<CorrectionPatchItem[]>;
  onApplyPatches: (
    patches: CorrectionPatchItem[],
    correctedSegIds: string[]
  ) => void;
}): Promise<CorrectionPatchItem[]> {
  const {
    config,
    payload,
    dictionary,
    runLocalLlmCorrection,
    onApplyPatches,
  } = params;

  // 1. 先執行 Pass 0 + Pass 1 跨句實體主權收斂與 ISIAL 診斷
  const { updatedSegments: step1Segments, patches: localPatches } =
    applySentenceLevelLocalCorrection(
      payload.highAttentionTargetSegments,
      payload.lowAttentionGlobalMemory.verifiedTerms,
      dictionary
    );

  // 2. 執行 LG-TPST 不合理自問自答偵測與前後語者自動拆句移轉
  const state = useAppStore.getState();
  const { resolvedSegments, flaggedForRelistenIds } =
    resolveUnreasonableSelfQAAndTransferSpeaker(step1Segments, state.speakers);

  useAppStore.getState().replaceBatchSegments(resolvedSegments);

  for (const rId of flaggedForRelistenIds) {
    localPatches.push({
      segmentId: rId,
      wordIndex: -1,
      originalText: '',
      status: 'semantic_uncertain',
      reason: '自問自答交界處標記複查',
      requireRelisten: false,
    });
  }

  const targetIds = resolvedSegments.map((s) => s.id);

  // 3. 重新為 resolvedSegments 建立免索引槽位映射表 (#S1 ~ #S5) 與內嵌槽位標籤文本
  const evidenceMap = analyzeCorpusStatisticalEvidence(
    state.segments.length > 0 ? state.segments : resolvedSegments,
    dictionary,
    `${state.docContextSummary}\n${state.rollingMeetingSummary}`
  );
  const { suspectSlots, slottedPromptTranscript } = buildSlottedPromptAndMap(
    resolvedSegments,
    evidenceMap,
    dictionary,
    state.rollingMeetingSummary
  );

  // 若整批句子在 Pass 1 後皆已完全通順 (無任何懷疑槽位且皆為 LOCKED)，直接免呼叫 LLM 節省算力！
  const hasReviewSegments = resolvedSegments.some(
    (s) => (s.acousticMargin ?? 0.18) < 0.16 || s.end - s.start < 1.2
  );
  if (suspectSlots.length === 0 && !hasReviewSegments) {
    return localPatches;
  }

  let modelPatches: CorrectionPatchItem[] = [];
  const updatedPayload: TriLevelContextPayload = {
    ...payload,
    highAttentionTargetSegments: resolvedSegments,
    suspectSlots,
    slottedPromptTranscript,
  };

  if (config.geminiApiKey.trim()) {
    try {
      modelPatches = await cloudClient.correctWithGeminiTriLevel(
        config.geminiApiKey.trim(),
        updatedPayload
      );
    } catch (err) {
      console.warn('雲端三層關注度槽位校正略過，維持本機自洽結果:', err);
    }
  } else if (runLocalLlmCorrection) {
    try {
      modelPatches = await runLocalLlmCorrection(updatedPayload);
    } catch (err) {
      console.warn('本機 WebLLM 槽位校正略過:', err);
    }
  }

  if (modelPatches.length > 0) {
    onApplyPatches(modelPatches, targetIds);
  }
  return [...localPatches, ...modelPatches];
}

/**
 * 【問題三 Pass 3：全局回溯精修 (Retroactive Pass)】
 * 利用全場完整知識庫（如後文出現的「業師群」見證）回頭掃描並分批修復整份逐字稿
 */
export async function runGlobalRetroactiveContextPass(params: {
  config: EngineConfig;
  allSegments: Segment[];
  dictionary: DictEntry[];
  docContextSummary: string;
  rollingMeetingSummary: string;
  runLocalLlmCorrection?: (
    payload: TriLevelContextPayload
  ) => Promise<CorrectionPatchItem[]>;
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

  const { updatedSegments: localUpdated, patches } =
    applySentenceLevelLocalCorrection(
      allSegments,
      globalVerifiedTerms,
      dictionary
    );

  const { resolvedSegments: updatedSegments, transferredSegmentIds } =
    resolveUnreasonableSelfQAAndTransferSpeaker(
      localUpdated,
      useAppStore.getState().speakers
    );

  useAppStore.getState().replaceBatchSegments(updatedSegments);

  const evidenceMap = analyzeCorpusStatisticalEvidence(
    updatedSegments,
    dictionary,
    `${docContextSummary}\n${rollingMeetingSummary}`
  );

  const allUncertainSegments = updatedSegments.filter((s) => {
    const intel = evaluateSentenceIntelligibility(
      s,
      evidenceMap,
      dictionary,
      rollingMeetingSummary
    );
    return (
      intel.intelligibilityScore < 0.72 ||
      intel.anomalousSpans.length > 0 ||
      (s.acousticMargin !== undefined && s.acousticMargin < 0.16) ||
      s.words.some(
        (w) => w.confidence < 0.8 || w.status === 'semantic_uncertain'
      )
    );
  });

  let aiFixedCount = transferredSegmentIds.length;
  const BATCH_SIZE = 10;

  for (
    let offset = 0;
    offset < Math.min(allUncertainSegments.length, 40);
    offset += BATCH_SIZE
  ) {
    const batchTarget = allUncertainSegments.slice(offset, offset + BATCH_SIZE);
    if (batchTarget.length === 0) break;

    const firstTargetIdx = updatedSegments.findIndex(
      (s) => s.id === batchTarget[0].id
    );
    const contextPrior =
      firstTargetIdx > 0
        ? updatedSegments.slice(Math.max(0, firstTargetIdx - 15), firstTargetIdx)
        : updatedSegments.slice(-15);

    const retroPayload = buildTriLevelContextPayload(
      contextPrior,
      batchTarget,
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
              batchTarget.map((s) => s.id)
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
              batchTarget.map((s) => s.id)
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

/**
 * 【問題四：智慧重聽決策器 + 常數線性增益預處理 + 四重硬性防退化驗收閘門】
 */
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

  const state = useAppStore.getState();
  const evidenceMap = analyzeCorpusStatisticalEvidence(
    state.segments.length > 0 ? state.segments : chunkSegments,
    state.dictionary,
    state.docContextSummary
  );

  // 智慧重聽決策器：僅篩選「物理音節密度嚴重不足 (\rho_{syll} < 2.0 字/秒)」或「無同音候選且聲學低信心度」的真實物理缺損段落！
  const candidates = chunkSegments
    .filter((seg) => (seg.relistenCount || 0) < 1 && seg.end - seg.start >= 0.8)
    .map((seg) => {
      const avgConf =
        seg.words.reduce((acc, w) => acc + w.confidence, 0) /
        Math.max(1, seg.words.length);
      const intel = evaluateSentenceIntelligibility(
        seg,
        evidenceMap,
        state.dictionary,
        state.rollingMeetingSummary
      );
      const isPureHomophoneIssue =
        intel.syllableDensityPerSec >= 2.0 &&
        intel.anomalousSpans.length > 0 &&
        intel.anomalousSpans.every((a) => a.phoneticHypotheses.length > 0) &&
        avgConf >= 0.72;

      const needsPhysicalRelisten =
        !isPureHomophoneIssue &&
        (intel.requireAmplifiedRelisten ||
          flaggedSegmentIds.has(seg.id) ||
          avgConf < config.acousticThreshold - 0.04);

      const priorityScore =
        intel.intelligibilityScore * 0.55 +
        avgConf * 0.45 -
        (intel.syllableDensityPerSec < 2.0 ? 0.3 : 0);
      return { seg, avgConf, intel, needsPhysicalRelisten, priorityScore };
    })
    .filter((item) => item.needsPhysicalRelisten)
    .sort((a, b) => a.priorityScore - b.priorityScore)
    .slice(0, quota);

  let relistenedCount = 0;

  for (const { seg, intel } of candidates) {
    // 嚴格停用會破壞 Whisper 相位與造成幻覺的 WSOLA 變速 (enableTimeStretch: false)
    const {
      subSamples,
      subWavBuffer,
      paddedStartGlobalSec,
      hasQuietDropCompensated,
    } = enhanceSubSegmentAudioForRelisten(
      chunkSamples,
      chunkStartGlobalSec,
      seg.start,
      seg.end,
      {
        paddingSec: Math.max(1.0, config.relistenPaddingSec || 1.0),
        enableTimeStretch: false,
      }
    );

    const segIdx = chunkSegments.findIndex((s) => s.id === seg.id);
    const contextSlice = chunkSegments
      .slice(
        Math.max(0, segIdx - 3),
        Math.min(chunkSegments.length, segIdx + 4)
      )
      .map((s) => s.words.map((w) => w.text).join(''))
      .join(' / ');

    const hypothesisHints = intel.anomalousSpans
      .flatMap((a) => a.phoneticHypotheses)
      .slice(0, 10);

    let candidateWords: WordSpan[] | null = null;

    if (config.geminiApiKey.trim()) {
      candidateWords = await cloudClient.relistenSubSegmentWithGemini(
        config.geminiApiKey.trim(),
        subWavBuffer,
        seg,
        contextSlice,
        Array.from(new Set([...hypothesisHints, ...verifiedTerms]))
      );
    } else if (relistenWithLocalAsr) {
      const promptHint = [
        ...hypothesisHints,
        ...verifiedTerms.slice(0, 18),
        contextSlice.slice(0, 100),
      ].join('，');
      candidateWords = await relistenWithLocalAsr(
        subSamples,
        paddedStartGlobalSec,
        seg,
        promptHint
      );
    }

    if (candidateWords && candidateWords.length > 0) {
      const tempSeg: Segment = { ...seg, words: candidateWords };
      const { updatedSegments } = applySentenceLevelLocalCorrection(
        [tempSeg],
        Array.from(new Set([...hypothesisHints, ...verifiedTerms])),
        useAppStore.getState().dictionary
      );
      const refinedSeg = updatedSegments[0];

      // 【問題四：四重硬性防退化驗收閘門】
      const gateResult = verifyNonRegressionAcceptanceGate({
        origSegment: seg,
        candidateSegment: refinedSeg,
        evidenceMap,
        dictionary: useAppStore.getState().dictionary,
        rollingSummary: useAppStore.getState().rollingMeetingSummary,
      });

      if (gateResult.accepted) {
        const enhanceTag = hasQuietDropCompensated
          ? `經無失真線性增益重聽與四重閘門驗收 (+${gateResult.deltaU} 分)`
          : `經聲學重聽與四重防退化閘門驗收 (+${gateResult.deltaU} 分)`;

        const verifiedWords = refinedSeg.words.map((w) => ({
          ...w,
          status: 'auto_corrected' as const,
          reason: w.reason || enhanceTag,
        }));
        onSegmentRelistened(seg.id, verifiedWords);
        relistenedCount++;
      }
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