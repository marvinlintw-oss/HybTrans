// src/utils/viterbiWordAligner.ts
// 句法與語用感知之字級受限 Viterbi 語者解碼器 (Syntax- & Pragmatics-Aware Word-Level Viterbi Decoder)
// 通用性核心升級：
// 1. 導入瀏覽器原生 ICU 自然語言斷詞 (`Intl.Segmenter('zh-TW', { granularity: 'word' })`)，徹底消滅機械式 2 字正則將「我現在是」、「你明年是」、「一般組」從詞中劈開的根因！
// 2. 導入「物理音節與時長質量加權發射機率 (`w_dur`)」：解決 `Intl.Segmenter` 將短促應答句（如「不算。」、「算不算。」、「24號。」、「禮拜三早上。」、「對，」、「好，再來。」）合為單一或雙 Token 後，因進出一詞需承受兩次語者轉移懲罰而被前後長句吞併之根因！
// 3. 聲學通量解鎖機制 (`flux >= 0.46`)：當相鄰字詞間之幀級聲紋後驗分佈發生劇烈翻轉（如男主席 <-> 女報告人快速問答），即使 Whisper 未標出句號，只要非強語法價鍵綁定對，即允許於自然詞界處換人！

import {
  AcousticSlotTier,
  FineDiarizationTurn,
  FrameSpeakerPosteriorMatrix,
  Segment,
  WordSpan,
} from '../types';

export interface RawWordTimestampItem {
  text: string;
  start: number; // 絕對秒數 (毫秒級精度)
  end: number;
  confidence?: number;
  /** 若該字詞位於聲學話輪切點或語用子句邊界，標記為強制開放換人 */
  isAcousticTurnBoundary?: boolean;
}

export interface AlignedWordSlot {
  word: string;
  start: number;
  end: number;
  confidence: number;
  assignedSpeakerIdx: number;
  assignedSpeakerId: string;
  top1SpeakerIdx: number;
  top1SpeakerId: string;
  top1Prob: number;
  top2SpeakerIdx: number;
  top2SpeakerId: string;
  top2Prob: number;
  /** 聲學對數勝算邊界 \Delta L_i = \ln(P_1 + \epsilon) - \ln(P_2 + \epsilon) */
  logOddsMargin: number;
  /** 傳統線性邊際 M = P_1 - P_2 (供既有 UI 與 Segment.acousticMargin 相容) */
  linearMargin: number;
  /** 字級重疊語音指標 \Omega_i \in [0, 1] */
  overlapRatio: number;
}

const QUESTION_MARKERS =
  /[？?]|(嗎|呢|吧|對不對|是不是|好不好|可以嗎|是嗎|對嗎|誰要|誰會|怎麼辦|怎麼樣|怎樣|為什麼|什麼時候|甚麼時候|哪裡|什麼意思|甚麼意思|算不算|可不可以|有沒有|接觸了沒|か|でしょうか|ますか|ですか)[。，？?!\s]*$/i;
const TERMINAL_PUNCT = /[。！!？?\.]+$/;
const CLAUSE_PUNCT = /[，,、；;：:]+$/;
const BACKCHANNEL_SHORT_TOKEN =
  /^(不算[啦]?|算不算|算[啊]?|對[啊呀喔]?|對對對|沒錯|可以[啊呀]?|沒有[啊啦]?|不用|不用不用|好[的啊喔]?|好再來|是[啊的]?|不是[啊]?|了解|有[啊]?|啥|什麼意思|甚麼意思|[0-9一二三四五六七八九十]+號|禮拜[一二三四五六日天](?:早上|下午|晚上)?|はい|ええ|うん|なるほど|yeah|yes|right|sure|ok|okay)[，。！？!?,.\s]*$/i;
const UNIVERSAL_DISCOURSE_STARTER =
  /^(不是|可是|不過|但是|然而|因為|所以|如果|要是|那如果|就只要|沒錯|可以|算|不算|算不算|對|我覺得|所以我|報告主席|好報告|目前的話|現在只有|現在總共|第一個是|第二個是|另外一個|這應該|好再來|所以要|這個也是|這個他們|也是|那原民會|有接觸|接觸哪|什麼意思|甚麼意思|喂|請問|はい|ええ|いや|しかし|yeah|yes|no|but|right|okay|ok)/i;

// 瀏覽器原生 ICU 自然語言斷詞器實例
const zhWordSegmenter: {
  segment: (input: string) => Iterable<{ segment: string; isWordLike?: boolean }>;
} | null =
  typeof Intl !== 'undefined' &&
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  typeof (Intl as any).Segmenter === 'function'
    ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
      new (Intl as any).Segmenter('zh-TW', { granularity: 'word' })
    : null;

/**
 * 【通用自然語言斷詞引擎 (`segmentNaturalLanguageTokens`)】：
 * 使用 `Intl.Segmenter('zh-TW', { granularity: 'word' })` 進行詞彙級切分，並將標點符號吸附於前一個詞尾，
 * 同時將過長詞組控制在 1~3 字、把孤立單字語氣詞/尾字併入前詞，徹底杜絕機械式 `{1,2}` 把「現在」、「明年」、「一般組」劈成兩半！
 */
export function segmentNaturalLanguageTokens(text: string): string[] {
  const clean = (text || '').trim();
  if (!clean) return [];

  const rawWords: string[] = [];

  if (zhWordSegmenter) {
    for (const item of zhWordSegmenter.segment(clean)) {
      const s = item.segment;
      if (!s || /^\s+$/.test(s)) continue;
      if (/^[，。？！、；：,.?!]+$/.test(s)) {
        if (rawWords.length > 0) {
          rawWords[rawWords.length - 1] += s[0];
        }
        continue;
      }
      if (/^[\u4e00-\u9fa5]{5,}$/.test(s)) {
        for (let i = 0; i < s.length; i += 2) {
          rawWords.push(s.slice(i, i + 2));
        }
      } else {
        rawWords.push(s);
      }
    }
  } else {
    const fallback =
      clean.match(
        /[\u4e00-\u9fa5]{1,2}[，。？！、；：,.?!]*|[\u3040-\u30ff]{1,3}[，。？！、；：,.?!]*|[a-zA-Z0-9']+[，。？！、；：,.?!]*|[，。？！、；：,.?!]+/g
      ) || [clean];
    for (const tok of fallback) {
      const st = tok.trim();
      if (!st) continue;
      if (/^[，。？！、；：,.?!]+$/.test(st) && rawWords.length > 0) {
        rawWords[rawWords.length - 1] += st[0];
        continue;
      }
      rawWords.push(st);
    }
  }

  const merged: string[] = [];
  for (let i = 0; i < rawWords.length; i++) {
    const cur = rawWords[i];
    const curPure = cur.replace(/[，。？！、；：,.?!\s]/g, '');
    const prev = merged.length > 0 ? merged[merged.length - 1] : '';
    const prevPure = prev.replace(/[，。？！、；：,.?!\s]/g, '');

    if (
      merged.length > 0 &&
      !/[，。？！、；：,.?!]$/.test(prev) &&
      curPure.length === 1 &&
      /^(的|得|地|了|嗎|呢|吧|啊|呀|喔|哦|啦|囉|嘛|過|著|們|組|處|署|師|店|標|案|錢|人|事|號)$/.test(
        curPure
      ) &&
      prevPure.length <= 2
    ) {
      merged[merged.length - 1] = prev + cur;
    } else {
      merged.push(cur);
    }
  }

  return merged.length > 0 ? merged : [clean];
}

/**
 * 檢查兩相鄰無標點詞是否屬於不可切斷之「強語法價鍵結構 (Syntactic Valence Bound Pair)」：
 * 嚴格排除獨立搶話起手詞（可是、不是、不過、算、不算、算不算、對、好再來、這應該）！
 */
export function isSyntacticallyBoundPair(
  prevRaw: string,
  currRaw: string
): boolean {
  const prevTrim = prevRaw.trim();
  const currTrim = currRaw.trim();
  if (!prevTrim || !currTrim) return false;
  if (/[，。？！、；：,.?!]$/.test(prevTrim)) return false;

  const prevClean = prevTrim.replace(/[，。？！、；：,.?!\s]+$/g, '');
  const currClean = currTrim.replace(/[，。？！、；：,.?!\s]+$/g, '');
  if (!prevClean || !currClean) return false;

  if (
    BACKCHANNEL_SHORT_TOKEN.test(currTrim) ||
    UNIVERSAL_DISCOURSE_STARTER.test(currClean) ||
    /^(可是|不是|不過|但是|然而|那如果|算不算|不算|算|好再來|這應該|誰會)$/.test(
      currClean
    )
  ) {
    return false;
  }

  // 1. 動詞/介詞 + 代名詞受詞 (如：寄給+他、給+他、告訴+我們)
  if (
    /[給寄送交找問請跟對向讓叫幫替被把為愛看聽說告訴通知聯絡協助感謝謝謝陪帶]$/.test(
      prevClean
    ) &&
    /^(我|你|您|他|她|它|我們|你們|他們|她們|大家|誰|自己)/.test(currClean)
  ) {
    return true;
  }

  // 2. 動詞 + 介詞補語/結果補語/趨向補語 (如：要花+在哪裡、寫+在、走+到、開+新店)
  if (
    /[花寫講說做開看找放用拿給走跑搬住留掉完好到在要會能可想敢去來有是讓把被將]$/.test(
      prevClean
    ) &&
    /^(在|到|給|向|往|自|從|出|進|回|過|起|上|下|來|去|哪|那|這|新|舊|好|完|成|為)/.test(
      currClean
    )
  ) {
    return true;
  }

  // 3. 定語「的」結尾 + 名詞中心語，或主語名詞 + 緊接助動詞謂語
  if (
    (/的$/.test(prevClean) &&
      prevClean.length <= 4 &&
      !/^(沒有要|是要|不是要|支持|翻修|開店|不同的|可以的)$/.test(prevClean) &&
      /^[\u4e00-\u9fa5]{1,3}$/.test(currClean)) ||
    (/(的錢|的人|的事|的店|的話|的時候|的問題|的部分|的對象|的時程|的規劃|的案子)$/.test(
      prevClean
    ) &&
      /^(就是|就是要|是|要|會|能|可以|應該|必須|沒有|不會|不能|不用|都|也|還|已經|正在)/.test(
        currClean
      ))
  ) {
    return true;
  }

  // 4. 數詞/範圍/時間量詞結構
  if (
    /([0-9一二三四五六七八九十百千萬兩半點到至~～]|分|秒|時|天|月|年|週)$/.test(
      prevClean
    ) &&
    /^([0-9一二三四五六七八九十百千萬兩半]|到|至|分|秒|點|時|天|日|月|年|週|禮拜|個|位|名|次|件|種|家|間|處|案|萬|元|塊|折|成|倍|差|號)/.test(
      currClean
    )
  ) {
    return true;
  }

  // 5. 代名詞/時間詞主語 + 緊接副詞/時間詞/繫詞
  if (
    /^(我|你|您|他|她|它|我們|你們|他們|現在|目前|今年|明年|下半年|上半年)$/.test(
      prevClean
    ) &&
    /^(現在|目前|今年|明年|下半年|上半年|是|要|會|在|有|都|也|已經|原本|覺得|建議|希望|知道|還有)/.test(
      currClean
    )
  ) {
    return true;
  }

  // 6. 程度/範圍/否定副詞 + 形容詞/助動詞補語
  if (
    /(很|太|更|最|超|蠻|挺|都|還|不|沒|別|真|非常|相當|特別|十分|比較|稍微|有點|已經)$/.test(
      prevClean
    ) &&
    /^(好|對|棒|行|可以|錯|大|小|多|少|高|低|快|慢|遠|近|久|難|易|滿|亂|重要|清楚|可能|確定|喜歡|想要|需要|應該|願意|擔心)/.test(
      currClean
    )
  ) {
    return true;
  }

  // 7. 句末語氣助詞、時態助詞或組織/名詞詞尾
  if (
    /^(的|得|地|了|嗎|呢|吧|啊|呀|喔|哦|啦|囉|嘛|齁|吼|過|著|們|組|處|署|師|店|標|案|長|員)$/.test(
      currClean
    )
  ) {
    return true;
  }

  // 8. 數詞/指示詞 + 量詞
  if (
    /[一二三四五六七八九十百千萬兩這那某每各幾]$/.test(prevClean) &&
    /^(個|位|名|次|件|種|些|點|處|家|間|筆|項|套|場|人|天|年|月|週|案)/.test(
      currClean
    )
  ) {
    return true;
  }

  return false;
}

/**
 * 從 `FineDiarizationTurn[]` 建構時間解析度 20ms (0.02s) 之平滑幀級語者後驗機率矩陣
 */
export function synthesizePosteriorMatrixFromFineTurns(
  fineTurns: FineDiarizationTurn[],
  windowStartSec: number,
  windowEndSec: number,
  frameStepSec = 0.02
): FrameSpeakerPosteriorMatrix {
  const speakerSet = new Set<string>();
  for (const ft of fineTurns) {
    if (ft.speakerId) speakerSet.add(ft.speakerId);
    if (ft.runnerUpSpeakerId) speakerSet.add(ft.runnerUpSpeakerId);
  }
  if (speakerSet.size === 0) {
    speakerSet.add('Speaker_01');
  }
  const speakerIds = Array.from(speakerSet);
  const K = speakerIds.length;
  const spkIdxMap = new Map<string, number>();
  speakerIds.forEach((id, idx) => spkIdxMap.set(id, idx));

  const durationSec = Math.max(0.5, windowEndSec - windowStartSec);
  const numFrames = Math.max(10, Math.ceil(durationSec / frameStepSec));
  const flatPosteriors = new Float32Array(numFrames * K);
  const osdProbs = new Float32Array(numFrames);

  const uniformVal = 1.0 / K;
  for (let i = 0; i < flatPosteriors.length; i++) {
    flatPosteriors[i] = uniformVal;
  }

  const relevantTurns = fineTurns.filter(
    (ft) => ft.end >= windowStartSec - 0.5 && ft.start <= windowEndSec + 0.5
  );

  for (let f = 0; f < numFrames; f++) {
    const tSec = windowStartSec + (f + 0.5) * frameStepSec;
    const rowOff = f * K;

    let matchedTurn: FineDiarizationTurn | null = null;
    for (const ft of relevantTurns) {
      if (tSec >= ft.start && tSec <= ft.end) {
        matchedTurn = ft;
        break;
      }
    }

    if (!matchedTurn && relevantTurns.length > 0) {
      let minDist = Infinity;
      for (const ft of relevantTurns) {
        const d = Math.min(Math.abs(tSec - ft.start), Math.abs(tSec - ft.end));
        if (d < minDist) {
          minDist = d;
          matchedTurn = ft;
        }
      }
    }

    if (matchedTurn) {
      const top1Idx = spkIdxMap.get(matchedTurn.speakerId) ?? 0;
      const top2Idx =
        spkIdxMap.get(matchedTurn.runnerUpSpeakerId || matchedTurn.speakerId) ??
        top1Idx;

      const p1 = Math.max(
        0.58,
        Math.min(0.96, matchedTurn.speakerConfidence ?? 0.85)
      );
      const margin = Math.max(
        0.10,
        Math.min(0.85, matchedTurn.speakerMargin ?? 0.24)
      );
      const p2 = K > 1 ? Math.max(0.02, Math.min(1.0 - p1, p1 - margin)) : 0.0;
      const rem =
        K > 2 ? Math.max(0.005, (1.0 - p1 - p2) / Math.max(1, K - 2)) : 0.0;

      let rowSum = 0.0;
      for (let k = 0; k < K; k++) {
        let val = rem;
        if (k === top1Idx) val = p1;
        else if (k === top2Idx) val = p2;
        flatPosteriors[rowOff + k] = val;
        rowSum += val;
      }
      if (rowSum > 1e-8) {
        for (let k = 0; k < K; k++) {
          flatPosteriors[rowOff + k] /= rowSum;
        }
      }

      const distToLeft = Math.abs(tSec - matchedTurn.start);
      const distToRight = Math.abs(tSec - matchedTurn.end);
      const boundaryScd =
        distToLeft < 0.12
          ? matchedTurn.boundaryScdLeft || 0
          : distToRight < 0.12
          ? matchedTurn.boundaryScdRight || 0
          : 0;
      osdProbs[f] =
        matchedTurn.overlapRatio ??
        (margin < 0.08 && boundaryScd > 0.25 ? 0.32 : 0.04);
    }
  }

  return {
    chunkStartSec: windowStartSec,
    frameStepSec,
    numFrames,
    speakerIds,
    flatPosteriors,
    osdProbs,
  };
}

/**
 * 句法與語用感知之字級受限 Viterbi 語者解碼器（具備物理時長與音節質量發射加權）
 */
export function decodeWordLevelSpeakersViterbi(
  words: RawWordTimestampItem[],
  posteriorMatrix: FrameSpeakerPosteriorMatrix,
  options?: {
    lambdaTrans?: number;
    backchannelMarginThreshold?: number;
    pingPongMaxDurSec?: number;
  }
): AlignedWordSlot[] {
  const M = words.length;
  const K = posteriorMatrix.speakerIds.length;
  if (M === 0 || K === 0) return [];

  const lambdaTrans = options?.lambdaTrans ?? 0.56;
  const frameStepSec = posteriorMatrix.frameStepSec || 0.02;
  const matrixStartSec = posteriorMatrix.chunkStartSec || 0;
  const numFrames = posteriorMatrix.numFrames;
  const flatPost = posteriorMatrix.flatPosteriors;
  const osdProbs = posteriorMatrix.osdProbs;

  if (K === 1) {
    const spkId = posteriorMatrix.speakerIds[0];
    return words.map((w) => ({
      word: w.text,
      start: w.start,
      end: w.end,
      confidence: w.confidence ?? 0.9,
      assignedSpeakerIdx: 0,
      assignedSpeakerId: spkId,
      top1SpeakerIdx: 0,
      top1SpeakerId: spkId,
      top1Prob: 0.99,
      top2SpeakerIdx: 0,
      top2SpeakerId: spkId,
      top2Prob: 0.01,
      logOddsMargin: 4.5,
      linearMargin: 0.98,
      overlapRatio: 0.0,
    }));
  }

  const emissions: Float64Array[] = new Array(M);
  const wordOverlap = new Float32Array(M);
  const fluxBetweenWords = new Float32Array(M);
  const wordEmissionWeights = new Float64Array(M);

  for (let i = 0; i < M; i++) {
    const relStart = Math.max(0, words[i].start - matrixStartSec);
    const relEnd = Math.max(relStart + 0.02, words[i].end - matrixStartSec);
    const durSec = Math.max(0.06, words[i].end - words[i].start);
    const pureCharLen = Math.max(
      1,
      words[i].text.replace(/[，。？！、；：,.?!\s]/g, '').length
    );
    wordEmissionWeights[i] = Math.max(
      1.0,
      Math.min(4.5, Math.max(durSec / 0.22, pureCharLen * 0.85))
    );

    const fStart = Math.max(
      0,
      Math.min(numFrames - 1, Math.floor(relStart / frameStepSec))
    );
    const fEnd = Math.max(
      fStart,
      Math.min(numFrames - 1, Math.ceil(relEnd / frameStepSec))
    );
    const spanLen = Math.max(1, fEnd - fStart + 1);

    const scores = new Float64Array(K).fill(1e-6);
    let osdSum = 0.0;

    for (let f = fStart; f <= fEnd; f++) {
      const relPos = spanLen > 1 ? (f - fStart) / (spanLen - 1) : 0.5;
      const weight = 0.35 + 0.65 * Math.sin(Math.PI * relPos);
      const rowOff = f * K;
      for (let k = 0; k < K; k++) {
        scores[k] += weight * (flatPost[rowOff + k] || 0);
      }
      osdSum += osdProbs[f] || 0;
    }

    let total = 0.0;
    for (let k = 0; k < K; k++) total += scores[k];
    for (let k = 0; k < K; k++) scores[k] /= Math.max(1e-9, total);

    emissions[i] = scores;
    wordOverlap[i] = osdSum / spanLen;

    if (i > 0) {
      let l1Diff = 0.0;
      for (let k = 0; k < K; k++) {
        l1Diff += Math.abs(emissions[i][k] - emissions[i - 1][k]);
      }
      fluxBetweenWords[i] = l1Diff;
    }
  }

  const V: Float64Array[] = Array.from(
    { length: M },
    () => new Float64Array(K).fill(-Infinity)
  );
  const Ptr: Int32Array[] = Array.from({ length: M }, () => new Int32Array(K));

  for (let k = 0; k < K; k++) {
    V[0][k] = wordEmissionWeights[0] * Math.log(emissions[0][k] + 1e-9);
  }

  for (let i = 1; i < M; i++) {
    const prevText = words[i - 1].text.trim();
    const currText = words[i].text.trim();
    const pauseSec = Math.max(0.0, words[i].start - words[i - 1].end);
    const flux = fluxBetweenWords[i];
    const prevHasPunct = /[，。？！、；：,.?!]$/.test(prevText);
    const isPrevQuestion = QUESTION_MARKERS.test(prevText);
    const isCurrBackchannel =
      (prevHasPunct || pauseSec >= 0.14 || flux >= 0.45) &&
      BACKCHANNEL_SHORT_TOKEN.test(currText);
    const isCurrInterruptionStarter =
      UNIVERSAL_DISCOURSE_STARTER.test(currText);
    const isBoundSyntacticPair = isSyntacticallyBoundPair(prevText, currText);
    const isClauseClosingTail =
      !prevHasPunct &&
      !isCurrBackchannel &&
      !Boolean(words[i].isAcousticTurnBoundary) &&
      /^[\u4e00-\u9fa5]{1,2}[，。？！、；：,.?!]+$/.test(currText) &&
      flux < 0.48;
    const isTurnBoundary =
      Boolean(words[i].isAcousticTurnBoundary) &&
      !isBoundSyntacticPair &&
      !isClauseClosingTail;

    let psiPunct = -5.5;
    if (isBoundSyntacticPair || isClauseClosingTail) {
      psiPunct = -10.5;
    } else if (isPrevQuestion) {
      psiPunct = 2.6;
    } else if (
      isTurnBoundary &&
      (prevHasPunct ||
        isCurrInterruptionStarter ||
        isCurrBackchannel ||
        pauseSec >= 0.14 ||
        flux >= 0.42)
    ) {
      psiPunct = 2.5;
    } else if (TERMINAL_PUNCT.test(prevText)) {
      psiPunct = flux >= 0.38 || isCurrBackchannel ? 2.1 : 1.1;
    } else if (CLAUSE_PUNCT.test(prevText)) {
      psiPunct =
        isCurrBackchannel || isCurrInterruptionStarter || flux >= 0.38
          ? 1.7
          : -1.0;
    } else if (flux >= 0.46 || pauseSec >= 0.28) {
      psiPunct =
        isCurrInterruptionStarter || isCurrBackchannel || flux >= 0.56
          ? 1.6
          : 0.8;
    }

    const logit =
      -1.0 +
      1.0 * psiPunct +
      2.2 * Math.min(pauseSec, 1.5) +
      2.8 * flux +
      (wordOverlap[i] >= 0.25 ? 0.9 : 0.0);
    const pSwitch = Math.min(
      0.88,
      Math.max(0.0005, 1.0 / (1.0 + Math.exp(-logit)))
    );

    // 當聲學後驗通量 flux >= 0.46 或停頓 >= 0.28s 且非強語法價鍵時，解除句中換人封鎖，讓無標點之快速問答也能精準換人！
    const blockMidClauseSwitch =
      isBoundSyntacticPair ||
      isClauseClosingTail ||
      (!prevHasPunct &&
        !isTurnBoundary &&
        flux < 0.46 &&
        pauseSec < 0.28);

    const wEm = wordEmissionWeights[i];

    for (let kCurr = 0; kCurr < K; kCurr++) {
      let bestPrevScore = -Infinity;
      let bestPrevSpk = 0;

      for (let kPrev = 0; kPrev < K; kPrev++) {
        if (blockMidClauseSwitch && kCurr !== kPrev) {
          continue;
        }
        let transProb: number;
        if (kCurr === kPrev) {
          transProb = blockMidClauseSwitch ? 1.0 : 1.0 - pSwitch;
        } else if (
          prevHasPunct ||
          isPrevQuestion ||
          isTurnBoundary ||
          flux >= 0.46
        ) {
          const otherMass = Math.max(1e-6, 1.0 - emissions[i][kPrev]);
          transProb = pSwitch * (emissions[i][kCurr] / otherMass);
        } else {
          transProb = pSwitch / Math.max(1, K - 1);
        }

        const candidateScore =
          V[i - 1][kPrev] + lambdaTrans * Math.log(transProb + 1e-9);
        if (candidateScore > bestPrevScore) {
          bestPrevScore = candidateScore;
          bestPrevSpk = kPrev;
        }
      }

      V[i][kCurr] = wEm * Math.log(emissions[i][kCurr] + 1e-9) + bestPrevScore;
      Ptr[i][kCurr] = bestPrevSpk;
    }
  }

  const bestPath = new Int32Array(M);
  let lastBestSpk = 0;
  let maxFinalScore = -Infinity;
  for (let k = 0; k < K; k++) {
    if (V[M - 1][k] > maxFinalScore) {
      maxFinalScore = V[M - 1][k];
      lastBestSpk = k;
    }
  }
  bestPath[M - 1] = lastBestSpk;
  for (let i = M - 1; i >= 1; i--) {
    bestPath[i - 1] = Ptr[i][bestPath[i]];
  }

  for (let i = 1; i < M; i++) {
    if (/^[，。？！、；：,.?!\s]+$/.test(words[i].text)) {
      bestPath[i] = bestPath[i - 1];
    }
  }

  return words.map((w, i) => {
    const em = emissions[i];
    let top1Idx = 0;
    let top1Prob = -1.0;
    let top2Idx = 0;
    let top2Prob = -1.0;

    for (let k = 0; k < K; k++) {
      const p = em[k];
      if (p > top1Prob) {
        top2Prob = top1Prob;
        top2Idx = top1Idx;
        top1Prob = p;
        top1Idx = k;
      } else if (p > top2Prob) {
        top2Prob = p;
        top2Idx = k;
      }
    }
    if (top2Prob < 0) {
      top2Prob = 1e-6;
      top2Idx = top1Idx;
    }

    const logOddsMargin = Number(
      (Math.log(top1Prob + 1e-6) - Math.log(top2Prob + 1e-6)).toFixed(3)
    );
    const linearMargin = Number(Math.max(0.0, top1Prob - top2Prob).toFixed(3));
    const assignedIdx = bestPath[i];

    return {
      word: w.text,
      start: Number(w.start.toFixed(2)),
      end: Number(w.end.toFixed(2)),
      confidence:
        w.confidence ??
        Number(Math.min(0.98, Math.max(0.55, top1Prob)).toFixed(2)),
      assignedSpeakerIdx: assignedIdx,
      assignedSpeakerId:
        posteriorMatrix.speakerIds[assignedIdx] || 'Speaker_01',
      top1SpeakerIdx: top1Idx,
      top1SpeakerId: posteriorMatrix.speakerIds[top1Idx] || 'Speaker_01',
      top1Prob: Number(top1Prob.toFixed(3)),
      top2SpeakerIdx: top2Idx,
      top2SpeakerId: posteriorMatrix.speakerIds[top2Idx] || 'Speaker_01',
      top2Prob: Number(top2Prob.toFixed(3)),
      logOddsMargin,
      linearMargin,
      overlapRatio: Number(wordOverlap[i].toFixed(3)),
    };
  });
}

export function classifySegmentAcousticSlotTier(
  avgLogOddsMargin: number,
  avgOverlapRatio: number,
  maxWordOverlapRatio: number,
  durationSec: number
): AcousticSlotTier {
  if (avgOverlapRatio >= 0.25 || maxWordOverlapRatio >= 0.4) {
    return 'tier3_overlap_split';
  }
  if (avgLogOddsMargin > 1.8 && avgOverlapRatio < 0.15 && durationSec >= 1.0) {
    return 'tier1_locked_anchor';
  }
  if (avgLogOddsMargin < 0.45) {
    return 'tier3_overlap_split';
  }
  return 'tier2_candidate_constrained';
}

/**
 * 將字級 Viterbi 解碼後的 `AlignedWordSlot[]` 聚合成 `Segment[]`
 */
export function groupAlignedWordsIntoSegments(
  alignedWords: AlignedWordSlot[],
  chunkIndex: number,
  windowIndex: number,
  acousticThreshold = 0.72
): Segment[] {
  if (alignedWords.length === 0) return [];

  interface TempGroup {
    speakerId: string;
    runnerUpSpeakerId: string;
    slots: AlignedWordSlot[];
  }

  const groups: TempGroup[] = [];

  const SEMANTIC_TOPIC_SHIFT_STARTER =
    /^(?:然後|那|另外|再來|此外|最後)?(?:第[二三四五六七八九十][個點項部分步]|另外一個(?:部分|問題|重點|面向)|再來(?:就是|是|一個)|那另外(?:就是|一個)|接下來(?:我們|就是))/;

  for (const slot of alignedWords) {
    const prevGroup = groups[groups.length - 1];
    const lastSlot = prevGroup?.slots[prevGroup.slots.length - 1];
    const pauseGap = lastSlot ? slot.start - lastSlot.end : 0;
    const prevChars = prevGroup
      ? prevGroup.slots.reduce((acc, s) => acc + s.word.length, 0)
      : 0;

    // 【同語者連續發言不碎切】：只要相鄰字詞屬於同一語者（代表中間完全沒有其他語者插話），
    // 即便翻頁或換氣停頓也不拆成多段，除非累積較長段落 (>= 150字) 且遇到明確語意分段標記！
    const isExplicitTopicBreak =
      prevChars >= 150 &&
      lastSlot &&
      /[。？！]$/.test(lastSlot.word) &&
      SEMANTIC_TOPIC_SHIFT_STARTER.test(slot.word.trim());

    if (
      prevGroup &&
      prevGroup.speakerId === slot.assignedSpeakerId &&
      !isExplicitTopicBreak &&
      prevChars + slot.word.length <= 360
    ) {
      if (
        lastSlot &&
        pauseGap >= 0.48 &&
        !/[，。？！、；：,.?!]$/.test(lastSlot.word) &&
        !isSyntacticallyBoundPair(lastSlot.word, slot.word)
      ) {
        lastSlot.word = `${lastSlot.word}，`;
      }
      prevGroup.slots.push(slot);
    } else {
      const runnerUp =
        slot.top1SpeakerId !== slot.assignedSpeakerId
          ? slot.top1SpeakerId
          : slot.top2SpeakerId;
      groups.push({
        speakerId: slot.assignedSpeakerId,
        runnerUpSpeakerId: runnerUp || slot.assignedSpeakerId,
        slots: [slot],
      });
    }
  }

  return groups.map((grp, sIdx) => {
    const segId = `seg_c${chunkIndex}_w${windowIndex}_s${sIdx}`;
    const firstSlot = grp.slots[0];
    const lastSlot = grp.slots[grp.slots.length - 1];
    const startSec = Number(firstSlot.start.toFixed(2));
    const endSec = Number(Math.max(startSec + 0.18, lastSlot.end).toFixed(2));
    const durSec = endSec - startSec;

    let sumLogOdds = 0.0;
    let sumLinearMargin = 0.0;
    let sumOverlap = 0.0;
    let maxOverlap = 0.0;
    const runnerUpCounts = new Map<string, number>();

    const wordSpans: WordSpan[] = grp.slots.map((sl, wIdx) => {
      sumLogOdds += sl.logOddsMargin;
      sumLinearMargin += sl.linearMargin;
      sumOverlap += sl.overlapRatio;
      if (sl.overlapRatio > maxOverlap) maxOverlap = sl.overlapRatio;

      const altSpk =
        sl.top1SpeakerId !== grp.speakerId
          ? sl.top1SpeakerId
          : sl.top2SpeakerId;
      if (altSpk && altSpk !== grp.speakerId) {
        runnerUpCounts.set(altSpk, (runnerUpCounts.get(altSpk) || 0) + 1);
      }

      const isPunct = /^[^\u4e00-\u9fa5\u3040-\u30ffa-zA-Z0-9]+$/.test(sl.word);
      const conf = isPunct ? 0.99 : sl.confidence;

      return {
        id: `${segId}_w_${wIdx}`,
        text: sl.word,
        start: sl.start,
        end: sl.end,
        confidence: conf,
        status: conf < acousticThreshold ? 'acoustic_uncertain' : 'normal',
        speakerId: sl.assignedSpeakerId,
        runnerUpSpeakerId: altSpk || grp.speakerId,
        logOddsMargin: sl.logOddsMargin,
        overlapRatio: sl.overlapRatio,
      };
    });

    if (wordSpans.length > 0) {
      const lastW = wordSpans[wordSpans.length - 1];
      const endsWithIncompleteSyntax =
        /(?:[的之比在跟對向把被讓給為從到]|要能夠|能夠|例如說|比方說|因為|就是|那個|這個)$/.test(
          lastW.text.trim()
        );
      if (
        /[\u4e00-\u9fa5]/.test(lastW.text) &&
        !/[，。？！、；：,.?!]$/.test(lastW.text) &&
        !endsWithIncompleteSyntax
      ) {
        wordSpans[wordSpans.length - 1] = {
          ...lastW,
          text: `${lastW.text}。`,
        };
      }
    }

    const count = Math.max(1, grp.slots.length);
    const avgLogOdds = Number((sumLogOdds / count).toFixed(3));
    const avgLinearMargin = Number((sumLinearMargin / count).toFixed(3));
    const avgOverlap = Number((sumOverlap / count).toFixed(3));

    let dominantRunnerUp = grp.runnerUpSpeakerId;
    let maxVotes = 0;
    for (const [candId, votes] of runnerUpCounts.entries()) {
      if (votes > maxVotes) {
        maxVotes = votes;
        dominantRunnerUp = candId;
      }
    }

    const slotTier = classifySegmentAcousticSlotTier(
      avgLogOdds,
      avgOverlap,
      maxOverlap,
      durSec
    );

    const compatibleAcousticMargin =
      slotTier === 'tier1_locked_anchor'
        ? Math.max(0.22, avgLinearMargin)
        : slotTier === 'tier3_overlap_split'
        ? Math.min(0.08, avgLinearMargin)
        : Math.min(0.15, Math.max(0.09, avgLinearMargin));

    return {
      id: segId,
      chunkIndex,
      speakerId: grp.speakerId,
      runnerUpSpeakerId: dominantRunnerUp,
      acousticMargin: Number(compatibleAcousticMargin.toFixed(3)),
      logOddsMargin: avgLogOdds,
      overlapRatio: avgOverlap,
      slotTier,
      start: startSec,
      end: endSec,
      words: wordSpans,
      isCorrected: false,
      relistenCount: 0,
    };
  });
}
