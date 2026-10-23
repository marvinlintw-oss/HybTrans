// src/utils/viterbiWordAligner.ts
// 句法與語用感知之字級受限 Viterbi 語者解碼器 (Syntax- & Pragmatics-Aware Word-Level Viterbi Decoder)
// 修正重點：
// 1. 當兩字交界處出現顯著聲學語者跳變 (\Phi_i >= 0.38) 或搶話轉折詞 (如「就只要」、「那如果」、「不算」) 時，
//    自動解除句中無標點鎖定 (\psi_{punct})，確保如 [00:14] 女聲2 打斷男聲2 時能即時換人！
// 2. 將同語者聚合門檻收緊至停頓 <= 0.85 秒 (原本 2.6 秒)，確保獨立發言段落不會被跨時間黏合成長段。

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
  /[？?]|(嗎|呢|吧|對不對|是不是|好不好|可以嗎|是嗎|對嗎|誰要|怎麼辦|怎麼|為什麼|算不算|か|でしょうか|ますか|ですか|right\?|okay\?)[。，？?!\s]*$/i;
const TERMINAL_PUNCT = /[。！!？?\.]+$/;
const CLAUSE_PUNCT = /[，,、；;：:]+$/;
const BACKCHANNEL_SHORT_TOKEN =
  /^(不算[啦]?|對[啊呀喔]?|對對對|沒錯|可以[啊]?|沒有[啊啦]?|不用|好[的啊喔]?|是[啊的]?|了解|はい|ええ|うん|なるほど|yeah|yes|right|sure|ok|okay)[，。！？!?,.\s]*$/i;
const UNIVERSAL_DISCOURSE_STARTER =
  /^(不是|可是|不過|但是|然而|因為|所以|如果|要是|那|就|沒錯|可以|算|不算|はい|ええ|いや|しかし|yeah|yes|no|but|right|okay|ok)/i;

/**
 * 檢查兩相鄰詞是否屬於不可切斷之「強語法價鍵結構 (Syntactic Valence Bound Pair)」：
 * 1. 動賓/介賓結構：如「寄給｜他」、「提醒｜他們」、「告訴｜我」
 * 2. 狀中程度結構：如「都很｜好」、「非常｜好」、「太｜遠」
 * 3. 詞幹+結構/句末助詞：如「...｜的」、「...｜了」、「...｜嗎」
 */
export function isSyntacticallyBoundPair(
  prevRaw: string,
  currRaw: string
): boolean {
  const prevTrim = prevRaw.trim();
  const currTrim = currRaw.trim();
  if (!prevTrim || !currTrim) return false;
  // 若前詞已帶有標點符號，代表語法上已斷開，不屬於緊密連綴結構
  if (/[，。？！、；：,.?!]$/.test(prevTrim)) return false;

  const prevClean = prevTrim.replace(/[，。？！、；：,.?!\s]+$/g, '');
  const currClean = currTrim.replace(/[，。？！、；：,.?!\s]+$/g, '');
  if (!prevClean || !currClean) return false;

  // 1. 動詞/介詞 + 代名詞受詞 (如：寄給+他、提醒+他們、交給+你)
  if (
    /[給寄送交找問請跟對向讓叫幫替被把為愛看聽說告訴通知聯絡協助感謝謝謝陪帶]$/.test(
      prevClean
    ) &&
    /^(我|你|您|他|她|它|我們|你們|他們|她們|大家|誰|自己)$/.test(currClean)
  ) {
    return true;
  }

  // 2. 程度/範圍/否定副詞 + 形容詞/助動詞補語 (如：都很+好、非常+好、不+對)
  if (
    /(很|太|更|最|超|蠻|挺|都|還|不|沒|別|真|非常|相當|特別|十分|比較|稍微|有點|已經)$/.test(
      prevClean
    ) &&
    /^(好|對|棒|行|可以|錯|大|小|多|少|高|低|快|慢|遠|近|久|難|易|重要|清楚|可能|確定|喜歡|想要|需要|應該|願意)/.test(
      currClean
    )
  ) {
    return true;
  }

  // 3. 句末語氣助詞、時態助詞或複數詞尾
  if (
    /^(的|得|地|了|嗎|呢|吧|啊|呀|喔|哦|啦|囉|嘛|齁|吼|過|著|們)$/.test(
      currClean
    )
  ) {
    return true;
  }

  // 4. 數詞/指示詞 + 量詞
  if (
    /[一二三四五六七八九十百千萬兩這那某每各幾]$/.test(prevClean) &&
    /^(個|位|名|次|件|種|些|點|處|家|間|筆|項|套|場|人|天|年|月|週)/.test(
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
        0.55,
        Math.min(0.96, matchedTurn.speakerConfidence ?? 0.85)
      );
      const margin = Math.max(
        0.08,
        Math.min(0.85, matchedTurn.speakerMargin ?? 0.22)
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
 * 句法與語用感知之字級受限 Viterbi 語者解碼器
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

  const lambdaTrans = options?.lambdaTrans ?? 0.55;
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

  for (let i = 0; i < M; i++) {
    const relStart = Math.max(0, words[i].start - matrixStartSec);
    const relEnd = Math.max(relStart + 0.02, words[i].end - matrixStartSec);

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
    V[0][k] = Math.log(emissions[0][k] + 1e-9);
  }

  for (let i = 1; i < M; i++) {
    const prevText = words[i - 1].text.trim();
    const currText = words[i].text.trim();
    const pauseSec = Math.max(0.0, words[i].start - words[i - 1].end);
    const flux = fluxBetweenWords[i];
    const prevHasPunct = /[，。？！、；：,.?!]$/.test(prevText);
    const isPrevQuestion = QUESTION_MARKERS.test(prevText);
    // 唯有在前一個詞已結束 (帶有標點或停頓 >= 0.20s) 時，當前短詞才可能是另一人的獨立附和詞 (防止把「都很+好。」的「好」當成附和詞切開！)
    const isCurrBackchannel =
      (prevHasPunct || pauseSec >= 0.2) &&
      BACKCHANNEL_SHORT_TOKEN.test(currText);
    const isCurrInterruptionStarter =
      UNIVERSAL_DISCOURSE_STARTER.test(currText);
    const isBoundSyntacticPair = isSyntacticallyBoundPair(prevText, currText);
    // 句尾收束字防脫落 (Clause-Final Tail Cohesion)：若前詞無標點，且當前詞為 1~2 字並帶有子句/句末標點之收束尾字 (如「寄給+他。」、「都很+好，」)，
    // 真正的換人點必然在當前詞的標點之後 (i+1)，絕不可在收束尾字之前提早換人！
    const isClauseClosingTail =
      !prevHasPunct &&
      /^[\u4e00-\u9fa5]{1,3}[，。？！、；：,.?!]+$/.test(currText);
    const isTurnBoundary =
      Boolean(words[i].isAcousticTurnBoundary) &&
      !isBoundSyntacticPair &&
      !isClauseClosingTail;

    let psiPunct = -4.2; // 預設：子句中強烈傾向維持同語者
    if (isBoundSyntacticPair || isClauseClosingTail) {
      psiPunct = -8.5; // 強語法價鍵約束或句尾收束字：鎖定同一語者，禁止尾字往下掉！
    } else if (isTurnBoundary) {
      psiPunct = 2.0; // 已對齊至合法聲學話輪切點，開放換人
    } else if (isPrevQuestion) {
      psiPunct = 2.2;
    } else if (TERMINAL_PUNCT.test(prevText)) {
      psiPunct = 0.8;
    } else if (CLAUSE_PUNCT.test(prevText)) {
      psiPunct =
        isCurrBackchannel || isCurrInterruptionStarter || flux >= 0.35
          ? 1.2
          : -0.8;
    } else if (flux >= 0.45 && isCurrInterruptionStarter) {
      psiPunct = 1.2;
    }

    const logit =
      -1.0 +
      1.0 * psiPunct +
      2.4 * Math.min(pauseSec, 1.5) +
      3.0 * flux +
      (wordOverlap[i] >= 0.25 ? 1.1 : 0.0);
    const pSwitch = Math.min(
      0.88,
      Math.max(0.002, 1.0 / (1.0 + Math.exp(-logit)))
    );

    for (let kCurr = 0; kCurr < K; kCurr++) {
      let bestPrevScore = -Infinity;
      let bestPrevSpk = 0;

      for (let kPrev = 0; kPrev < K; kPrev++) {
        if ((isBoundSyntacticPair || isClauseClosingTail) && kCurr !== kPrev) {
          continue;
        }
        let transProb: number;
        if (kCurr === kPrev) {
          transProb = isBoundSyntacticPair || isClauseClosingTail ? 1.0 : 1.0 - pSwitch;
        } else if (isPrevQuestion || isTurnBoundary) {
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

      V[i][kCurr] = Math.log(emissions[i][kCurr] + 1e-9) + bestPrevScore;
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
 * 嚴格保持時間軸單調遞增，且僅在停頓 <= 0.85 秒內合併同語者字詞，避免不同輪次的發言被黏成同一長段！
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

  for (const slot of alignedWords) {
    const prevGroup = groups[groups.length - 1];
    const lastSlot = prevGroup?.slots[prevGroup.slots.length - 1];
    const pauseGap = lastSlot ? slot.start - lastSlot.end : 0;
    const prevChars = prevGroup
      ? prevGroup.slots.reduce((acc, s) => acc + s.word.length, 0)
      : 0;

    if (
      prevGroup &&
      prevGroup.speakerId === slot.assignedSpeakerId &&
      pauseGap <= 0.85 &&
      prevChars + slot.word.length <= 160
    ) {
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
      if (
        /[\u4e00-\u9fa5]/.test(lastW.text) &&
        !/[，。？！、；：,.?!]$/.test(lastW.text)
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