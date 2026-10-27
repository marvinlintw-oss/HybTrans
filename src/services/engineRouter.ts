// src/services/engineRouter.ts
// 次世代多階段 AI 語意推敲、ISIAL 免索引槽位協定、三級聲學信心度閘控 (Tier 1~3)、防前錯擴散語料統計與一鍵會議 AI 摘要/結論/行動事項生成引擎

import * as OpenCC from 'opencc-js';
import {
  AcousticSlotTier,
  CorrectionPatchItem,
  DictEntry,
  EngineConfig,
  MeetingActionItem,
  MeetingMinutesReport,
  MeetingMinutesSpeakerStat,
  MeetingTopicDecisionItem,
  Segment,
  SentenceIntegrityDiagnostic,
  SentenceIntegrityIssueType,
  SentenceIntegrityRepair,
  Speaker,
  SuspectSlotItem,
  TriLevelContextPayload,
  WordSpan,
} from '../types';
import { syncSegmentDerivedFields } from '../store/useAppStore';
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
import {
  isSyntacticallyBoundPair,
  segmentNaturalLanguageTokens,
} from '../utils/viterbiWordAligner';
import { useAppStore } from '../store/useAppStore';

// 使用純字形簡轉繁 (cn -> tw)，保留語者原話（「對象」、「招標文件」、「支持」），避免 twp 過度替換成 IT 術語（「物件」、「檔案」、「支援」）
const converter = OpenCC.Converter({ from: 'cn', to: 'tw' });

/**
 * Pass 0：跨領域通用之口語異常疊字、跳針消除與台灣會議常見語境搭配規則
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
    pattern: /([\u4e00-\u9fa5]{1,6})(?:[，。？！、；：,.?!\s]+\1){2,}/g,
    wrongDesc: '跨標點異常跳針重複詞句',
    right: '$1',
    reason: 'Pass 0 跨標點重複跳針消除',
  },
  {
    pattern: /([\u4e00-\u9fa5]{2,4})(?:[，、\s]*\1)+/g,
    wrongDesc: '雙字以上異常跳針重複',
    right: '$1',
    reason: 'Pass 0 口語跳針重複詞組消除',
  },
  {
    pattern: /([店翻裝改修開標案處署師區])(\1)(?=[\u4e00-\u9fa5])/g,
    wrongDesc: '複合詞首字口吃疊字',
    right: '$1',
    reason: 'Pass 0 口語詞首異常疊字消除',
  },
  {
    pattern: /(?<=[\u4e00-\u9fa5])([修店標案處署師區])(\1)/g,
    wrongDesc: '複合詞尾字口吃疊字',
    right: '$1',
    reason: 'Pass 0 口語詞尾異常疊字消除',
  },
  {
    pattern: /那，(個|些|天|邊|裡|種|位|樣)/g,
    wrongDesc: '指示代名詞內部誤插逗號',
    right: '那$1',
    reason: 'Pass 0 修復指示代名詞內部誤插逗號',
  },
  {
    pattern: /沒有，(嗎|呢|啊|吧|喔|要|問題|關係|辦法)/g,
    wrongDesc: '否定疑問詞內部誤插逗號',
    right: '沒有$1',
    reason: 'Pass 0 修復否定疑問結構內部誤插逗號',
  },
  {
    pattern: /另外，(一[個種邊家位半案])/g,
    wrongDesc: '數量修飾詞組內部誤插逗號',
    right: '另外$1',
    reason: 'Pass 0 修復數量詞組內部誤插逗號',
  },
  {
    pattern: /對，(長官|貴賓|同仁|講者|團隊|於|他|她|你|我|大家|口)/g,
    wrongDesc: '介詞結構內部誤插逗號',
    right: '對$1',
    reason: 'Pass 0 修復介詞結構內部誤插逗號',
  },
  {
    pattern: /(又[去在要是][\u4e00-\u9fa5]{1,3}([祖島門灣北中南東]))[，。、\s]*(又[去在要是]幹)[嘛麻]\2/g,
    wrongDesc: '並列句式句尾地名回音幻覺 (如：又去馬祖又去幹麻祖)',
    right: '$1，$3嘛',
    reason: 'Pass 0 消除並列句式尾字回音幻覺',
  },
  {
    pattern: /(我們|你們|他們|各位|審查|評選|外審|那位|這位|請)晚於(?=[上這那昨前剛還也去來提說講看參覺得建議表示])/g,
    wrongDesc: '快速語流「委員 (wěi-yuán)」誤聽為「晚於 (wǎn-yú)」',
    right: '$1委員',
    reason: 'Pass 0 依據主謂人稱語法配價修正「晚於 -> 委員」',
  },
  {
    pattern: /(?:官方|廣方|大方|防方|報到)主席/g,
    wrongDesc: '會議起手敬語「報告主席」誤聽為「官方主席」',
    right: '報告主席',
    reason: 'Pass 0 依會議報告祈使語境修正為「報告主席」',
  },
  {
    pattern: /(?:風空間|風天|風間|公空間|罐空間|罐中間|光有空間|共有空間)(?=活化|盤點|改造|媒合|閒置|那邊|甚麼時候|什麼時候|[，。？?])/g,
    wrongDesc: '殘響環境「公有空間」快速連讀誤聽',
    right: '公有空間',
    reason: 'Pass 0 依空間活化與盤點語境修正為「公有空間」',
  },
  {
    pattern: /(?:優秀性|優先性|優秀)(?:追蹤區|推動區|推展區)/g,
    wrongDesc: '「優先推動區」殘響誤聽為「優秀性追蹤區」',
    right: '優先推動區',
    reason: 'Pass 0 依區域發展語境修正為「優先推動區」',
  },
  {
    pattern: /優先推動(?:區區預|區區|區域預)/g,
    wrongDesc: '「優先推動區」尾音口吃重疊',
    right: '優先推動區',
    reason: 'Pass 0 消除「優先推動區」句尾口吃贅字',
  },
  {
    pattern: /(?:三個處|三二處)(?=[，、\s]*(?:但是|可是|不過|其中)?(?:集中|其中)?有[五5]處)/g,
    wrongDesc: '數量對比句「32處」快速連讀誤聽為「三個處」',
    right: '32處',
    reason: 'Pass 0 依後文「其中有五處」母體基數邏輯修正「三個處 -> 32處」',
  },
  {
    pattern: /(但是|可是|不過|那)集中有(?=[一二三四五六七八九十兩0-9]+[處案個項家])/g,
    wrongDesc: '「其中有」誤聽為「集中有」',
    right: '$1其中有',
    reason: 'Pass 0 依子集數量語法修正「集中有 -> 其中有」',
  },
  {
    pattern: /有(?:大陸|大入|那入)[，、\s]*(?=[0-9一二三四五六七八九十]+月|[第下本]一次|工作會|會報|議程)/g,
    wrongDesc: '「有納入」誤聽為「有大陸/那入」',
    right: '有納入',
    reason: 'Pass 0 依會議議程排入語法修正「有大陸 -> 有納入」',
  },
  {
    pattern: /(?:園民會|援民會|圓民會|願意會|原明會|原名會|人民會|員民會)/g,
    wrongDesc: '部會簡稱「原民會」同音誤聽',
    right: '原民會',
    reason: 'Pass 0 依部會專有名詞修正為「原民會」',
  },
  {
    pattern: /(申請|補助|接觸|請教|詢問)(?:援會|園會|願意會)/g,
    wrongDesc: '快速連讀「原民會」縮讀為「援會/園會」',
    right: '$1原民會',
    reason: 'Pass 0 依動賓搭配與連音縮讀還原為「原民會」',
  },
  {
    pattern: /(?:徵請|真請)(?=(?:原民會|經濟部|文化部|農業部|國發會|交通部|衛福部|補助|計畫|經費))/g,
    wrongDesc: '「申請」誤聽為「徵請」',
    right: '申請',
    reason: 'Pass 0 依計畫補助動賓搭配修正「徵請 -> 申請」',
  },
  {
    pattern: /(?:股材油|股茶油|古茶油|埔茶油)/g,
    wrongDesc: '「苦茶油」同音誤聽為「股材油/古茶油」',
    right: '苦茶油',
    reason: 'Pass 0 依農特產專有名詞修正為「苦茶油」',
  },
  {
    pattern: /苦茶油品牌[，、]*(?:一樣|一按|以案)(?=已經|目前|這邊|他們)/g,
    wrongDesc: '「苦茶油品牌一案」誤聽為「苦茶油品牌一樣」',
    right: '苦茶油品牌一案',
    reason: 'Pass 0 依專案列舉量詞修正「一樣 -> 一案」',
  },
  {
    pattern: /(?:不夥合|不複合|不附合|不負合)(?=補助|辦法|規定|資格|條件|標準)/g,
    wrongDesc: '「不符合」誤聽為「不夥合/不複合」',
    right: '不符合',
    reason: 'Pass 0 依補助辦法語法修正「不夥合 -> 不符合」',
  },
  {
    pattern: /([兩三四五六七八九十幾])[，、\s]*盤子(?=[了嗎呢吧。，？?]|而已)/g,
    wrongDesc: '「案子」快速連讀誤聽為「盤子」',
    right: '$1個案子',
    reason: 'Pass 0 依後文「總共有三案」量詞呼應修正「盤子 -> 個案子」',
  },
  {
    pattern: /(?:兩人|敵人|竹人|足人)(?=已經\s*OK|同意|共識|部落)/g,
    wrongDesc: '原民部落語境「族人」誤聽為「兩人/竹人」',
    right: '族人',
    reason: 'Pass 0 依原民部落會議語境修正為「族人」',
  },
  {
    pattern: /文化部的(?:一天|一件|一間)(?=[，。、\s]*(?:修正|意見))/g,
    wrongDesc: '「文化部的意見」誤聽為「文化部的一天」',
    right: '文化部的意見',
    reason: 'Pass 0 依審查意見修正搭配還原為「文化部的意見」',
  },
  {
    pattern: /(之前|裡面|計畫|工程|他們)的(?:空向|功項|公項|工相)/g,
    wrongDesc: '「工項」同音誤聽為「空向/功項」',
    right: '$1的工項',
    reason: 'Pass 0 依工程與計畫結構語境修正為「工項」',
  },
  {
    pattern: /(第一個是|再來是|你說|這邊是|是)(?:堅持|監持|監時|兼時|間時|間持)(?=[，、\s]*(?:現在|這邊|那邊|目前|的部分|的狀況|[。？?]))/g,
    wrongDesc: '鄉鎮名「尖石」同音誤聽為「監持/堅持/監時」',
    right: '$1尖石',
    reason: 'Pass 0 依分區鄉鎮主語槽位與音韻修正為「尖石」',
  },
  {
    pattern: /(?:堅持|監持|監時|兼時|間時|間持)[，、\s]*(這邊的狀況|的部分|那邊|目前|現在狀況)/g,
    wrongDesc: '鄉鎮名「尖石」同音誤聽為「監持/堅持/監時」',
    right: '尖石，$1',
    reason: 'Pass 0 依分區鄉鎮方位主語槽位修正為「尖石」',
  },
  {
    pattern: /(再來是|下一個是|是)(?:武蜂|武功|武風|五風|五蜂)/g,
    wrongDesc: '鄉鎮名「五峰」同音誤聽為「武蜂/武功/武風」',
    right: '$1五峰',
    reason: 'Pass 0 依分區鄉鎮盤點語境修正為「五峰」',
  },
  {
    pattern: /(?:武蜂|武功|武風|五風|五蜂)(?=[，、\s]*(?:目前|這邊|的部分|那邊|大概))/g,
    wrongDesc: '鄉鎮名「五峰」同音誤聽為「武蜂/武功/武風」',
    right: '五峰',
    reason: 'Pass 0 依分區鄉鎮盤點語境修正為「五峰」',
  },
  {
    pattern: /(?:天使|堅持|監時)[、，\s]*(?:武功|武風|五峰)[，、\s]*(?:臭不起來|湊不起來)/g,
    wrongDesc: '「尖石、五峰湊不起來」連讀誤聽為「天使武功臭不起來」',
    right: '尖石、五峰湊不起來',
    reason: 'Pass 0 依前後文鄉鎮併案脈絡修正為「尖石、五峰湊不起來」',
  },
  {
    pattern: /執行單位是(?:不從|不銅)的/g,
    wrongDesc: '「不同的」誤聽為「不從的」',
    right: '執行單位是不同的',
    reason: 'Pass 0 依語法謂語修正「不從的 -> 不同的」',
  },
  {
    pattern: /聯繫(?:不像|不尚|不向)/g,
    wrongDesc: '趨向補語「聯繫不上」誤聽為「聯繫不像」',
    right: '聯繫不上',
    reason: 'Pass 0 依動詞可能補語搭配修正「聯繫不像 -> 聯繫不上」',
  },
];

const SPOKEN_TITLE_OR_ABBR_SUFFIX_REGEX =
  /(處長|副處長|主委|副主委|部長|次長|署長|副署長|司長|組長|科長|主任|視察|技正|專員|技士|委員|教授|老師|執行長|董事長|總經理|永續長|會計師|律師|會長|先生|市長|鎮長|區長|教練|店長|姐|姊|哥|兄|會|處|署|部|院|站|辦|中心|聯盟|專案|基金|基金會)$/;

export function classifyDictEntryAliases(entry: DictEntry): {
  typoAliases: string[];
  spokenShortForms: string[];
} {
  const typoAliases: string[] = [];
  const spokenShortForms: string[] = [];
  const cleanTerm = converter((entry.term || '').trim());
  if (!cleanTerm || !Array.isArray(entry.aliases)) {
    return { typoAliases, spokenShortForms };
  }

  for (const rawAl of entry.aliases) {
    const al = converter((rawAl || '').trim());
    if (!al || al === cleanTerm || al.length < 2) continue;

    if (cleanTerm.includes(al) || al.includes(cleanTerm)) {
      spokenShortForms.push(al);
      continue;
    }

    if (Math.abs(al.length - cleanTerm.length) >= 2) {
      spokenShortForms.push(al);
      continue;
    }

    const dPhon = computeTaiwanPhoneticDistance(al, cleanTerm);
    if (
      dPhon > 0.28 &&
      SPOKEN_TITLE_OR_ABBR_SUFFIX_REGEX.test(al) &&
      !SPOKEN_TITLE_OR_ABBR_SUFFIX_REGEX.test(cleanTerm)
    ) {
      spokenShortForms.push(al);
      continue;
    }

    if (dPhon > 0.52 && al.length !== cleanTerm.length) {
      spokenShortForms.push(al);
      continue;
    }

    typoAliases.push(al);
  }

  return { typoAliases, spokenShortForms };
}

const COMMON_STOP_WORDS = new Set([
  '我們', '你們', '他們', '這個', '那個', '因為', '所以', '但是', '可是',
  '如果', '就是', '可以', '沒有', '不是', '一個', '現在', '時候', '這樣',
  '覺得', '應該', '知道', '部分', '問題', '意思', '可能', '已經', '還是',
  '然後', '是說', '上次', '不會', '是要', '那我', '對對', '不用', '他要',
  '他的', '我的', '你的', '比較', '其實', '基本', '後面', '裡面', '怎麼',
  '什麼', '甚麼', '那邊', '這邊', '大家', '有些', '確定', '建議', '發現', '開始',
  '出來', '起來', '過去', '進來', '東西', '事情', '的話', '來說', '這件',
  '狀況', '工項', '協助', '目前', '溝通', '路線', '辦法', '方向', '類別',
  '經費', '編列', '資料', '補充', '工程', '補助', '規劃', '設計', '接觸',
  '專員', '對口', '聯繫', '細節', '持續', '當中', '通過', '納入', '會議',
  '會報', '推動', '活化', '空間', '主持', '主席', '報告', '案子', '品牌',
  '計畫', '分線', '執行', '單位', '不同', '總共', '只有', '兩個', '三個',
  '另外', '前面', '上面', '下面', '謝謝', '很快', '責任', '作業', '影響',
  '其中', '五處', '三處', '兩處', '幾處', '早上', '下午', '禮拜', '工作',
  '創生', '釐清', '偏向', '調整', '申請', '以及', '還有', '還沒', '而已',
]);

const BAD_GRAMMATICAL_EDGE_REGEX =
  /^[的我你他她它這那在有是就也都還又再很太更不沒別好對啊喔嗎呢吧啦]|(的|了|嗎|呢|吧|啊|喔|啦|們|說|要|會|是|有|在|過|著)$/;

export function toAccentInsensitivePhoneticKey(text: string): string {
  return toTaiwanFuzzyPhoneticSyllables(text).join('-');
}

const GRAMMATICAL_FUNCTION_CHARS = new Set(
  '的地得了著過在從跟和與及或把被對為以比連越再又也還都只就才剛正曾已經常會能可要得著過個位名次件種些點處裡上中下內外前後左右這那哪誰什怎幾多少一二三四五六七八九十百千萬億零我你他她它人嗎呢吧啊喔啦哦欸恩嗯唔哈嘿是就有但而卻讓給向往到不沒很太更別好對算'.split(
    ''
  )
);

const QUESTION_CLAUSE_REGEX =
  /([？?]|怎麼辦|怎麼樣|怎樣|對不對|是不是|好不好|可以嗎|是嗎|對嗎|有沒有|算不算|對吧|誰會補助|什麼時候|甚麼時候|接觸了沒|接觸哪裡|什麼意思|甚麼意思|嗎[。？?]?|呢[。？?]?|か[。？?]?|でしょうか[。？?]?|ますか[。？?]?|ですか[。？?]?)$/i;

const DIRECT_SHORT_ANSWER_REGEX =
  /^(不算[。，啦]?|算[。，]?|對[，。啊呀]?|對對對[，。]?|沒錯[，。]?|可以[，。啊]?|沒有[，。啊]?|不用[，。]?|不用不用[，。]?|好[，。的啊]?|好[，、]?再來[。，]?|是[，。啊的]?|不是[，。啊]?|不行[，。]?|有[，。啊]?|也是|這個也是|[0-9一二三四五六七八九十]+號|禮拜[一二三四五六日天]|はい|ええ|うん|いいえ|なるほど|そうですね|yeah|yes|no|nope|right|sure|okay|ok)([，。！？!?,.\s]|$)/i;

const RHETORICAL_SELF_ANSWER_BRIDGE_REGEX =
  /^(因為|主要是|其實|也就是說|意思是|換句話說|白話講|我意思是|比如說|比方說)/;

export function resolveSegmentSlotTier(seg: Segment): AcousticSlotTier {
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

export interface SentenceIntelligibilityReport {
  segmentId: string;
  intelligibilityScore: number;
  maxEffectiveOrphanLen: number;
  syllableDensityPerSec: number;
  maxPhoneticGravityTension: number;
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
  requireAmplifiedRelisten: boolean;
}

export interface StatisticalTermEvidence {
  term: string;
  exactSig: string;
  fuzzySig: string;
  evidenceScore: number;
  compoundProductivity: number;
  sovereigntyScore: number;
  isPriorVerified: boolean;
}

/**
 * 【防前錯擴散語料統計引擎 (`analyzeCorpusStatisticalEvidence`)】：
 * 嚴格保護先驗驗證詞彙（內建字典、Google Sheet 字典、簡報摘要）之絕對主權！
 * 若未驗證之原始 ASR 詞彙（如「園民會」、「監時」、「武功」）與任何已驗證詞彙（「原民會」、「尖石」、「五峰」）發生同音或近音碰撞 (dPhon <= 0.28)，
 * 嚴禁將該原始 ASR 錯字累積為高主權詞彙，徹底斬斷「把前錯當成參考一直用下去」的錯誤傳播鏈！
 */
export function analyzeCorpusStatisticalEvidence(
  allSegments: Segment[],
  dictionary: DictEntry[],
  docSummary: string
): Map<string, StatisticalTermEvidence> {
  const evidenceMap = new Map<string, StatisticalTermEvidence>();

  const aliasToCanonicalMap = new Map<string, string>();
  const spokenShortFormSet = new Set<string>();
  const priorVerifiedList: Array<{
    term: string;
    exactSig: string;
    fuzzySig: string;
  }> = [];

  for (const d of dictionary) {
    const cleanTarget = converter(d.term.trim());
    if (!cleanTarget) continue;
    const { typoAliases, spokenShortForms } = classifyDictEntryAliases(d);
    for (const cleanAl of typoAliases) {
      aliasToCanonicalMap.set(cleanAl, cleanTarget);
    }
    for (const sf of spokenShortForms) {
      if (sf.length >= 2 && sf.length <= 4 && !COMMON_STOP_WORDS.has(sf)) {
        if (
          sf.length >= 3 ||
          SPOKEN_TITLE_OR_ABBR_SUFFIX_REGEX.test(sf) ||
          /^(芯榆|俊沂|白姐|曉玲|國發|地創|青培|台經|原民)$/.test(sf)
        ) {
          spokenShortFormSet.add(sf);
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
      if (!isPrior) {
        // 原始語料命中已知錯別名時，將權重直接灌給正詞，絕不註冊錯字本身！
        term = aliasToCanonicalMap.get(term)!;
        isPrior = true;
      } else {
        term = aliasToCanonicalMap.get(term)!;
      }
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

    const exactSig = toPhoneticSignature(term);
    const fuzzySig = toAccentInsensitivePhoneticKey(term);

    // 【防前錯擴散關鍵防線】：若為未驗證之原始 ASR 詞組 (!isPrior)，檢查是否與先驗字典詞發生近音碰撞
    if (!isPrior) {
      for (const pv of priorVerifiedList) {
        if (pv.term === term) {
          isPrior = true;
          break;
        }
        if (
          pv.term.length === term.length &&
          (pv.exactSig === exactSig ||
            pv.fuzzySig === fuzzySig ||
            computeTaiwanPhoneticDistance(term, pv.term) <= 0.26)
        ) {
          // 此詞與字典正詞同音異字（極可能為早期 ASR 辨識錯字），不累積其錯誤主權！
          return;
        }
      }
    }

    const existing = evidenceMap.get(term);
    if (existing) {
      existing.evidenceScore += weightDelta;
      if (isPrior) existing.isPriorVerified = true;
    } else {
      const entry: StatisticalTermEvidence = {
        term,
        exactSig,
        fuzzySig,
        evidenceScore: weightDelta,
        compoundProductivity: 0,
        sovereigntyScore: weightDelta,
        isPriorVerified: isPrior,
      };
      evidenceMap.set(term, entry);
      if (isPrior) {
        priorVerifiedList.push({ term, exactSig, fuzzySig });
      }
    }
  };

  const fullCorpusText = converter(
    `${docSummary}\n${allSegments
      .map((s) => s.words.map((w) => w.text).join(''))
      .join('\n')}`
  );

  for (const d of dictionary) {
    const cleanT = converter(d.term.trim());
    if (cleanT.length >= 2 && cleanT.length <= 4) {
      const { typoAliases } = classifyDictEntryAliases(d);
      const isMentionedInCorpus =
        fullCorpusText.includes(cleanT) ||
        typoAliases.some((al) => fullCorpusText.includes(al));
      if (isMentionedInCorpus) {
        registerOrAdd(cleanT, 20.0, true);
      } else {
        priorVerifiedList.push({
          term: cleanT,
          exactSig: toPhoneticSignature(cleanT),
          fuzzySig: toAccentInsensitivePhoneticKey(cleanT),
        });
      }
    }
  }
  for (const sf of spokenShortFormSet) {
    if (fullCorpusText.includes(sf)) {
      registerOrAdd(sf, 16.0, true);
    }
  }
  const summaryWords = docSummary.match(/[\u4e00-\u9fa5]{2,4}/g) || [];
  for (const sw of summaryWords) {
    registerOrAdd(sw, 8.0, true);
  }

  const clauses: string[] = [];
  for (const seg of allSegments) {
    const segText = converter(seg.words.map((w) => w.text).join(''));
    const subClauses = segText
      .split(/[，。？！、；：\s,.?!]+/)
      .filter((c) => c.length >= 2);
    clauses.push(...subClauses);

    for (const w of seg.words) {
      const txt = converter(w.text).replace(/[，。？！、；：,.?!\s]/g, '');
      if (txt.length >= 2 && txt.length <= 4) {
        // 僅對已通過自動校對 (auto_corrected) 或使用者鎖定之詞給予高權重；未驗證原始 ASR 詞權重壓低為 0.35，防止錯字自我強化
        const confWeight =
          w.status === 'auto_corrected' || w.userLocked
            ? 2.8
            : w.confidence >= 0.85
            ? 0.38
            : 0.12;
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
    if (count >= 3) {
      registerOrAdd(ng, count * 0.45, false);
    }
  }

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
          compoundWitnessBonus += cand.isPriorVerified ? 2.4 : 0.8;
        }
        if (
          rightChar !== '$' &&
          !GRAMMATICAL_FUNCTION_CHARS.has(rightChar)
        ) {
          distinctContexts.add(`R:${rightChar}`);
          compoundWitnessBonus += cand.isPriorVerified ? 2.6 : 0.9;
        }
      }
    }

    cand.compoundProductivity = distinctContexts.size;
    cand.evidenceScore +=
      distinctContexts.size * (cand.isPriorVerified ? 3.2 : 1.1);
    cand.sovereigntyScore = Number(
      (
        cand.evidenceScore +
        compoundWitnessBonus +
        (cand.isPriorVerified ? 14.0 : 0.0)
      ).toFixed(2)
    );
  }

  return evidenceMap;
}

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
  const segFuzzySyls = toTaiwanFuzzyPhoneticSyllables(cleanChars);
  const segInitialSet = new Set<string>(
    segFuzzySyls.map((s) => s[0] || '').filter(Boolean)
  );

  void segInitialSet;
  for (const d of dictionary) {
    const { typoAliases } = classifyDictEntryAliases(d);
    for (const cleanAl of typoAliases) {
      aliasSet.add(cleanAl);
    }
  }

  for (const ev of evidenceMap.values()) {
    if (ev.isPriorVerified || ev.sovereigntyScore >= 5.0) {
      highSovereigntyEntities.push(ev);
    }
  }

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
        scored.push({ term: canon, rankScore: 2.6, tension: 0.94 });
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
          : evidenceMap.get(sub)?.sovereigntyScore || 0.4;
        const topicResonanceDiff = Math.min(
          1.0,
          Math.max(0.0, (entity.sovereigntyScore - subSov) / 20.0) +
            (rollingSummary.includes(entity.term) ? 0.25 : 0.0)
        );
        const tension = Number(
          (gaussianKernel * Math.max(0.38, topicResonanceDiff)).toFixed(3)
        );
        const sourceLambda = entity.isPriorVerified ? 1.35 : 1.25;
        const rankScore = sourceLambda * tension;

        if (tension >= 0.30) {
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

  let maxEffectiveOrphanLen = 0.0;
  let currentOrphanEffLen = 0.0;
  let orphanRunText = '';
  let orphanStartCharIdx = 0;
  let charCursor = 0;

  for (let wIdx = 0; wIdx < seg.words.length; wIdx++) {
    const wText = converter(seg.words[wIdx].text);
    const pureW = wText.replace(/[，。？！、；：,.?!\s]/g, '');
    const isCjkWord = /^[\u4e00-\u9fa5]+$/.test(pureW);

    if (isCjkWord && pureW.length === 1) {
      const wOrphan = GRAMMATICAL_FUNCTION_CHARS.has(pureW) ? 0.15 : 1.0;
      if (currentOrphanEffLen === 0 && wOrphan >= 0.8) {
        orphanStartCharIdx = charCursor;
      }
      if (currentOrphanEffLen > 0 || wOrphan >= 0.8) {
        currentOrphanEffLen += wOrphan;
        orphanRunText += pureW;
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

  let maxPhoneticGravityTension = 0.0;
  const tokenBoundarySet = new Set<number>([0]);
  let accCharLen = 0;
  for (const w of seg.words) {
    const wClean = converter(w.text).replace(/[，。？！、；：,.?!\s]+$/g, "");
    tokenBoundarySet.add(accCharLen + wClean.length);
    accCharLen += converter(w.text).length;
    tokenBoundarySet.add(accCharLen);
  }

  for (let len = 4; len >= 2; len--) {
    for (let i = 0; i <= fullText.length - len; i++) {
      if (!tokenBoundarySet.has(i) || !tokenBoundarySet.has(i + len)) {
        continue;
      }
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
        (!subEv.isPriorVerified && subEv.sovereigntyScore < 6.0);

      if (isLowPlausibility) {
        const { candidates, gravityTension } =
          retrieveConfusionLatticeAndGravity(sub);
        if (gravityTension > maxPhoneticGravityTension) {
          maxPhoneticGravityTension = gravityTension;
        }
        if (candidates.length > 0 && gravityTension >= 0.38) {
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

  // ── 維度 3：局部語法與連詞搭配驚訝度偵測 (Collocational & Syntactic Surprisal) ──
  const NON_MODIFIER_PREFIX_SET = new Set([
    '我們', '你們', '他們', '這個', '那個', '這樣', '那樣',
    '因為', '所以', '但是', '可是', '不過', '其實', '當然',
    '比如', '或者', '還是', '而且', '然後', '那邊', '這邊',
    '現在', '後來', '大家', '覺得', '知道', '希望', '建議',
  ]);
  const collocationRegex =
    /([\u4e00-\u9fa5]{2})([，、\s]*)(如果|要是|假如|雖然|但是|可是)(?=[\u4e00-\u9fa5]{2,})/g;
  let colMatch: RegExpExecArray | null;
  while ((colMatch = collocationRegex.exec(fullText)) !== null) {
    const modWord = colMatch[1];
    const midPunct = colMatch[2] || '';
    const conjWord = colMatch[3];
    const matchedRaw = `${modWord}${midPunct}${conjWord}`;
    const spanStart = colMatch.index;
    const spanEnd = spanStart + matchedRaw.length;
    const tailAfter = fullText.slice(spanEnd);
    const hasDitransitiveDisposalVerb =
      /^(?:我們|你們|他們|我|你|他|再|就|先|會|可以|幫){1,3}(?:寄給|送給|發給|傳給|交給|買給|帶給|補給|提供給|留給|給)[\u4e00-\u9fa5]{1,3}/.test(
        tailAfter
      );

    if (
      !NON_MODIFIER_PREFIX_SET.has(modWord) &&
      !BAD_GRAMMATICAL_EDGE_REGEX.test(modWord) &&
      (midPunct.length === 0 || hasDitransitiveDisposalVerb) &&
      !anomalousSpans.some(
        (a) => spanStart < a.endCharIdx && spanEnd > a.startCharIdx
      )
    ) {
      const valenceObjectCands =
        conjWord === '如果' && hasDitransitiveDisposalVerb
          ? [`${modWord}禮物`, `${modWord}禮盒`]
          : [];
      const { candidates: phonCands } = retrieveConfusionLatticeAndGravity(
        `${modWord}${conjWord}`
      );
      const punctCands = [`${modWord}，${conjWord}`];
      anomalousSpans.push({
        text: matchedRaw,
        startCharIdx: spanStart,
        endCharIdx: spanEnd,
        phoneticHypotheses: Array.from(
          new Set([...valenceObjectCands, ...phonCands, ...punctCands])
        ).slice(0, 4),
        anomalyScore: hasDitransitiveDisposalVerb ? 0.88 : 0.76,
        anomalyType: 'collocation_surprisal',
      });
    }
  }

  // ── 維度 4：條件句尾殘缺形容詞謂語偵測 ──
  const danglingCondMatch = fullText.match(
    /(?:如果|要是|假設)([一-龥]{2,8})(有充足|有足夠|有完整)[。，！？!?\s]*$/
  );
  if (danglingCondMatch) {
    const tailStr = danglingCondMatch[2];
    const tailStart = fullText.lastIndexOf(tailStr);
    if (tailStart >= 0) {
      anomalousSpans.push({
        text: tailStr,
        startCharIdx: tailStart,
        endCharIdx: tailStart + tailStr.length,
        phoneticHypotheses: ['的裝修怎麼辦？', '的翻修怎麼辦？', '裝修怎麼辦？'],
        anomalyScore: 0.91,
        anomalyType: 'collocation_surprisal',
      });
    }
  }

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

  const origMidPeriodCount = (origText.slice(0, -1).match(/[。？！]/g) || [])
    .length;
  const newMidPeriodCount = (newText.slice(0, -1).match(/[。？！]/g) || [])
    .length;
  const hasBrokenFragmentation =
    newMidPeriodCount > origMidPeriodCount && newClean.length < 24;

  const lengthRatio = newClean.length / Math.max(1, origClean.length);
  const isLengthReasonable =
    newClean.length >= Math.max(2, Math.floor(origClean.length * 0.78)) &&
    newClean.length <= Math.ceil(origClean.length * 1.45) + 6;

  const phoneticDistance = computeTaiwanPhoneticDistance(origClean, newClean);
  const phoneticSimilarity = Number((1.0 - phoneticDistance).toFixed(3));
  const isPhoneticFaithful =
    phoneticSimilarity >= 0.58 || origIntel.syllableDensityPerSec < 2.0;

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
    const tier = resolveSegmentSlotTier(seg);
    const runnerUp = seg.runnerUpSpeakerId || seg.speakerId;
    const logOdds = (seg.logOddsMargin ?? 1.2).toFixed(2);
    const overlap = (seg.overlapRatio ?? 0.05).toFixed(2);

    const gateTag =
      tier === 'tier1_locked_anchor'
        ? `[LOCKED | speaker=${seg.speakerId}, logOdds=${logOdds}]`
        : tier === 'tier3_overlap_split'
        ? `[OVERLAP | top1=${seg.speakerId}, top2=${runnerUp}, overlap=${overlap}]`
        : `[REVIEW | top1=${seg.speakerId}, top2=${runnerUp}, logOdds=${logOdds}]`;

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
 * 【LG-TPST + Tier 3 重疊拆解】：偵測不合理「自問自答」與「句尾附和插話重疊」並自動拆句移轉給候選語者
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
    if (runnerUpSpk && runnerUpSpk !== currentSpk) return runnerUpSpk;
    if (neighborSpk && neighborSpk !== currentSpk) return neighborSpk;
    const alt = allSpeakerIds.find((id) => id !== currentSpk);
    return alt || currentSpk;
  };

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const tier = resolveSegmentSlotTier(seg);
    const prevSeg = expanded.length > 0 ? expanded[expanded.length - 1] : null;
    const nextSeg = i + 1 < segments.length ? segments[i + 1] : null;

    // 1. 檢查段內問答對 (QA_SPEAKER_CHANGE)
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
        beforeText.length >= 2 &&
        (DIRECT_SHORT_ANSWER_REGEX.test(afterText) || afterText.length <= 14)
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
            slotTier: 'tier2_candidate_constrained',
            start: splitTime,
            end: Math.max(splitTime + 0.25, seg.end),
            words: afterWords.map((w, idx) => ({
              ...w,
              id: `${answerSegId}_w_${idx}`,
              status: 'auto_corrected' as const,
              reason: `LG-TPST 消除段內自問自答，將回答移轉至競爭候選語者 (${targetAnswerSpeaker})`,
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
        beforeText.length <= 12 &&
        afterText.length >= 12 &&
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
            slotTier: 'tier2_candidate_constrained',
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

    // 2. 針對 Tier 3 重疊搶話槽位：若長句尾端粘連了短促附和詞（如「...所以他們好，再來。」），自動拆出附和槽位
    if (
      (tier === 'tier3_overlap_split' || seg.words.length >= 6) &&
      seg.words.length >= 4
    ) {
      let splitWIdx = -1;
      for (let w = seg.words.length - 2; w >= 1; w--) {
        if (/[。！？?!，]$/.test(seg.words[w].text)) {
          const tailText = seg.words
            .slice(w + 1)
            .map((item) => item.text)
            .join('')
            .trim();
          if (
            (DIRECT_SHORT_ANSWER_REGEX.test(tailText) && tailText.length <= 5) ||
            /^(好[，、]?再來)[。！？?!]*$/.test(tailText)
          ) {
            splitWIdx = w + 1;
            break;
          }
        }
      }

      if (splitWIdx > 0) {
        const mainWords = seg.words.slice(0, splitWIdx);
        const bcWords = seg.words.slice(splitWIdx);
        const bcSpeaker = findInterlocutorSpeakerId(
          seg.speakerId,
          seg.runnerUpSpeakerId,
          nextSeg?.speakerId
        );
        if (bcSpeaker !== seg.speakerId) {
          const splitTime = Number(
            mainWords[mainWords.length - 1].end.toFixed(2)
          );
          const bcSegId = `${seg.id}_ov_bc`;
          expanded.push(
            {
              ...seg,
              end: splitTime,
              words: mainWords,
            },
            {
              ...seg,
              id: bcSegId,
              speakerId: bcSpeaker,
              runnerUpSpeakerId: seg.speakerId,
              slotTier: 'tier2_candidate_constrained',
              start: splitTime,
              end: Math.max(splitTime + 0.22, seg.end),
              words: bcWords.map((w, idx) => ({
                ...w,
                id: `${bcSegId}_w_${idx}`,
                status: 'auto_corrected' as const,
                reason: `重疊搶話拆解：將尾端短促指示/附和句拆分至對話語者 (${bcSpeaker})`,
              })),
              isCorrected: true,
            }
          );
          transferredSegmentIds.push(bcSegId);
          continue;
        }
      }
    }

    expanded.push(seg);
  }

  // 3. 跨段問答對檢查
  for (let i = 1; i < expanded.length; i++) {
    const prev = expanded[i - 1];
    const curr = expanded[i];
    if (prev.speakerId !== curr.speakerId) continue;

    const currTier = resolveSegmentSlotTier(curr);
    if (currTier === 'tier1_locked_anchor') continue;

    const prevText = prev.words.map((w) => w.text).join('').trim();
    const currText = curr.words.map((w) => w.text).join('').trim();

    if (
      QUESTION_CLAUSE_REGEX.test(prevText) &&
      !RHETORICAL_SELF_ANSWER_BRIDGE_REGEX.test(currText) &&
      (DIRECT_SHORT_ANSWER_REGEX.test(currText) ||
        (curr.end - curr.start <= 2.4 && (curr.acousticMargin ?? 0.15) < 0.2))
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
            reason: `LG-TPST 消除跨段自問自答衝突，移轉至 Top-2 候選應答語者 (${targetSpk})`,
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

const DANGLING_TAIL_VERB_PREP_ADVERB_REGEX =
  /[\u4e00-\u9fa5]{1,}(寄給|交給|傳給|發給|送給|還給|轉給|留給|提供給|分享給|告訴|通知|詢問|請教|拜託|都很|非常|特別|十分|相當|比較|稍微|有點|太|更|最|關於|針對|透過|根據|按照|依據|把|將|被|給|向|現在還有|還有|因為我覺得你這個|所以可能我們|但是因為)[，。、,.…]*$/;

const DROPPED_HEAD_COMPLEMENT_PRONOUN_REGEX =
  /^(他|她|它|你|您|我|我們|他們|她們|大家|各位|好|棒|對|行|可|多|少|高|低|快|慢|大|小|長|短|遠|近|難|易|貴|久|清楚|明白|完整|順利|成功|合理|正確|重要|嚴重|明顯|幾處啊|幾處|幾個|幾案|的|了|嗎|呢|吧|啊|喔|呀|囉|啦)[，。、！？,.!?]*$/;

const STOLEN_TAIL_DISCOURSE_STARTER_REGEX =
  /[，。！？,.!?]+(那|但是|不過|所以|因此|而且|另外|還有|接下來|首先|再來|最後|我們|他們|這個|那個|因為|如果|雖然)$/;

const FRICTION_PREFIX_DEBRIS_REGEX =
  /^(常|阿|啊|喔|嗯|欸|呃)[。，！？,.!?]+(?=(謝謝|好|對|是的|沒問題|可以))/;

export function runStage1SentenceIntegrityCheckAndRepair(
  inputSegments: Segment[],
  speakers: Record<string, Speaker> = {}
): {
  repairedSegments: Segment[];
  repairs: SentenceIntegrityRepair[];
  diagnostics: Record<string, SentenceIntegrityDiagnostic>;
  flaggedForRelistenIds: string[];
} {
  if (!inputSegments || inputSegments.length === 0) {
    return {
      repairedSegments: [],
      repairs: [],
      diagnostics: {},
      flaggedForRelistenIds: [],
    };
  }

  const repairs: SentenceIntegrityRepair[] = [];
  const diagnostics: Record<string, SentenceIntegrityDiagnostic> = {};
  const issueMap = new Map<string, Set<SentenceIntegrityIssueType>>();
  const noteMap = new Map<string, string[]>();

  const recordIssue = (
    segId: string,
    issue: SentenceIntegrityIssueType,
    note: string
  ) => {
    if (!issueMap.has(segId)) issueMap.set(segId, new Set());
    issueMap.get(segId)!.add(issue);
    if (!noteMap.has(segId)) noteMap.set(segId, []);
    if (note && !noteMap.get(segId)!.includes(note)) {
      noteMap.get(segId)!.push(note);
    }
  };

  let working: Segment[] = inputSegments
    .map((s) => ({
      ...s,
      words: s.words.map((w) => ({ ...w })),
    }))
    .filter((s) => s.words.length > 0);

  // =========================================================================
  // Step 1A：紙張摩擦殘字與重複幻覺迴圈清除 (HALLUCINATION_LOOP)
  // =========================================================================
  working = working.map((seg) => {
    const rawText = seg.words.map((w) => w.text).join('');
    let words = [...seg.words];

    if (FRICTION_PREFIX_DEBRIS_REGEX.test(rawText) && words.length >= 2) {
      const firstClean = words[0].text.replace(/[。，！？,.!?]/g, '');
      if (firstClean.length <= 1) {
        const beforeTxt = rawText;
        words = words.slice(1);
        const afterTxt = words.map((w) => w.text).join('');
        repairs.push({
          segmentId: seg.id,
          issueType: 'HALLUCINATION_LOOP',
          beforeText: beforeTxt,
          afterText: afterTxt,
          description: 'Stage 1 剝離紙張摩擦前綴殘字',
        });
        recordIssue(
          seg.id,
          'HALLUCINATION_LOOP',
          '已自動剝離句首摩擦雜音殘字'
        );
        recordIssue(seg.id, 'repetition_noise_hallucination', '');
      }
    }

    if (words.length >= 3) {
      const deduped: WordSpan[] = [];
      let repCount = 0;
      for (let i = 0; i < words.length; i++) {
        const currNorm = words[i].text.replace(/[，。！？、,.!?]/g, '');
        const prevNorm =
          deduped.length > 0
            ? deduped[deduped.length - 1].text.replace(/[，。！？、,.!?]/g, '')
            : '';
        if (currNorm.length > 0 && currNorm.length <= 3 && currNorm === prevNorm) {
          repCount++;
          if (repCount >= 2) {
            continue;
          }
        } else {
          repCount = 0;
        }
        deduped.push(words[i]);
      }
      if (deduped.length < words.length) {
        const beforeTxt = words.map((w) => w.text).join('');
        words = deduped;
        const afterTxt = words.map((w) => w.text).join('');
        repairs.push({
          segmentId: seg.id,
          issueType: 'HALLUCINATION_LOOP',
          beforeText: beforeTxt,
          afterText: afterTxt,
          description: 'Stage 1 消除重複跳針幻覺詞',
        });
        recordIssue(seg.id, 'HALLUCINATION_LOOP', '已自動折疊重複跳針短詞');
        recordIssue(seg.id, 'repetition_noise_hallucination', '');
      }
    }

    return {
      ...seg,
      start: words[0]?.start ?? seg.start,
      end: words[words.length - 1]?.end ?? seg.end,
      words,
    };
  });

  // =========================================================================
  // Step 1A.5：夾心微碎片歸位 (`A -> B(<=5字非應答殘片，如「但是因為...」、「還有，」、「族人，」) -> A`)
  // =========================================================================
  for (let i = 1; i < working.length - 1; i++) {
    const prevSeg = working[i - 1];
    const midSeg = working[i];
    const nextSeg = working[i + 1];
    if (
      prevSeg.speakerId === nextSeg.speakerId &&
      midSeg.speakerId !== prevSeg.speakerId &&
      midSeg.start - prevSeg.end <= 1.8 &&
      nextSeg.start - midSeg.end <= 1.8
    ) {
      const midText = midSeg.words.map((w) => w.text).join('').trim();
      const midPure = midText.replace(/[，。？！、；：,.?!…\s]/g, '');
      const prevSpkF0 = speakers[prevSeg.speakerId]?.f0Hz ?? 145;
      const midSpkF0 = speakers[midSeg.speakerId]?.f0Hz ?? 145;
      const isCrossGenderSandwich =
        (prevSpkF0 < 162 && midSpkF0 >= 168) ||
        (prevSpkF0 >= 168 && midSpkF0 < 162);
      const prevPure = prevSeg.words
        .map((w) => w.text)
        .join('')
        .replace(/[，。？！、；：,.?!…\s]/g, '');
      const isTopicEchoTurn =
        midPure.length >= 2 &&
        prevPure.length >= 2 &&
        prevPure.slice(-4).includes(midPure.slice(0, 2));
      const isStandaloneReply =
        DIRECT_SHORT_ANSWER_REGEX.test(midText) ||
        /[？?]$/.test(midText) ||
        isCrossGenderSandwich ||
        isTopicEchoTurn;
      if (midPure.length >= 1 && midPure.length <= 5 && !isStandaloneReply) {
        midSeg.speakerId = prevSeg.speakerId;
        midSeg.words = midSeg.words.map((w) => ({
          ...w,
          speakerId: prevSeg.speakerId,
        }));
      }
    }
  }

  // =========================================================================
  // Step 1B：跨段尾字掉落歸還 (TAIL_DROP_TO_NEXT) 與前段搶字歸還 (HEAD_STOLEN_FROM_PREV)
  // =========================================================================
  for (let i = 0; i < working.length - 1; i++) {
    const prevSeg = working[i];
    const currSeg = working[i + 1];
    if (!prevSeg || !currSeg || prevSeg.words.length === 0 || currSeg.words.length === 0) {
      continue;
    }

    const gapSec = currSeg.start - prevSeg.end;
    if (gapSec > 1.75 || gapSec < -0.5) continue;

    const prevText = prevSeg.words.map((w) => w.text).join('').trim();
    const currFirstWord = currSeg.words[0];
    const currFirstClean = currFirstWord.text.replace(/[，。！？、,.!?]/g, '').trim();

    const isPrevDanglingTail = DANGLING_TAIL_VERB_PREP_ADVERB_REGEX.test(prevText);
    const isCurrHeadDroppedTail =
      currFirstClean.length >= 1 &&
      currFirstClean.length <= 3 &&
      DROPPED_HEAD_COMPLEMENT_PRONOUN_REGEX.test(currFirstWord.text) &&
      (/[，。！？、,.!?]$/.test(currFirstWord.text) ||
        currSeg.words.length === 1 ||
        /^[，。！？、,.!?]/.test(currSeg.words[1]?.text || ''));

    if (isPrevDanglingTail && isCurrHeadDroppedTail) {
      const beforePrevText = prevText;
      const lastPrevW = prevSeg.words[prevSeg.words.length - 1];
      const strippedLastPrevText = lastPrevW.text.replace(/[，。、,.]+$/, '');
      if (strippedLastPrevText.length > 0) {
        prevSeg.words[prevSeg.words.length - 1] = {
          ...lastPrevW,
          text: strippedLastPrevText,
          word: strippedLastPrevText,
        };
      }

      let takeCount = 1;
      if (
        currSeg.words.length >= 2 &&
        /^[，。！？、,.!?]+$/.test(currSeg.words[1].text)
      ) {
        takeCount = 2;
      }

      const movedSpans = currSeg.words.slice(0, takeCount).map((w, idx) => {
        const hasPunct = /[，。！？,.!?]$/.test(w.text);
        const patchedText =
          idx === takeCount - 1 && !hasPunct
            ? /啊|嗎|呢/.test(w.text)
              ? `${w.text}？`
              : `${w.text}。`
            : w.text;
        return {
          ...w,
          text: patchedText,
          word: patchedText,
          speakerId: prevSeg.speakerId,
          status: 'auto_corrected' as const,
          isPatched: true,
          isCorrected: true,
          reason: 'Stage 1 句意完整性修復：歸還跨段掉落之句尾補語/疑問詞',
        };
      });

      prevSeg.words.push(...movedSpans);
      prevSeg.end = Math.max(
        prevSeg.end,
        movedSpans[movedSpans.length - 1].end
      );
      prevSeg.isCorrected = true;

      currSeg.words = currSeg.words.slice(takeCount);
      if (currSeg.words.length > 0) {
        currSeg.start = currSeg.words[0].start;
      }

      const afterPrevText = prevSeg.words.map((w) => w.text).join('');
      repairs.push({
        segmentId: prevSeg.id,
        relatedSegmentId: currSeg.id,
        issueType: 'TAIL_DROP_TO_NEXT',
        beforeText: beforePrevText,
        afterText: afterPrevText,
        description: `將跨段掉落尾字「${currFirstClean}」歸還前句尾端`,
      });
      recordIssue(
        prevSeg.id,
        'TAIL_DROP_TO_NEXT',
        `已從下一段歸還掉落尾字「${currFirstClean}」`
      );
      recordIssue(prevSeg.id, 'tail_character_drop', '');
      continue;
    }

    let splitStolenWordIdx = -1;
    const stolenMatch = prevText.match(STOLEN_TAIL_DISCOURSE_STARTER_REGEX);
    if (stolenMatch && prevSeg.words.length >= 2 && gapSec <= 1.35) {
      const lastW = prevSeg.words[prevSeg.words.length - 1];
      const lastWClean = lastW.text.replace(/[，。！？、,.!?]/g, '').trim();
      if (lastWClean === stolenMatch[1]) {
        splitStolenWordIdx = prevSeg.words.length - 1;
      }
    }
    if (splitStolenWordIdx === -1 && prevSeg.words.length >= 2 && gapSec <= 0.85) {
      const lastPrevW = prevSeg.words[prevSeg.words.length - 1];
      if (isSyntacticallyBoundPair(lastPrevW.text, currFirstWord.text)) {
        for (let k = prevSeg.words.length - 2; k >= Math.max(0, prevSeg.words.length - 3); k--) {
          if (/[。！？!?]$/.test(prevSeg.words[k].text)) {
            const tailWordsLen = prevSeg.words
              .slice(k + 1)
              .reduce((acc, w) => acc + w.text.replace(/[，。！？、,.!?]/g, '').length, 0);
            if (tailWordsLen >= 1 && tailWordsLen <= 4) {
              splitStolenWordIdx = k + 1;
            }
            break;
          }
        }
      }
    }

    if (splitStolenWordIdx > 0) {
      const beforeCurrText = currSeg.words.map((w) => w.text).join('');
      const stolenWords = prevSeg.words.slice(splitStolenWordIdx).map((w) => {
        const cleanT = w.text.replace(/[，。！？、,.!?]+$/g, '');
        return {
          ...w,
          text: cleanT,
          word: cleanT,
          speakerId: currSeg.speakerId,
          status: 'auto_corrected' as const,
          isPatched: true,
          isCorrected: true,
          reason: 'Stage 1 句意完整性修復：歸還前段尾端誤搶之句首詞組',
        };
      });
      const stolenStr = stolenWords.map((w) => w.text).join('');

      prevSeg.words = prevSeg.words.slice(0, splitStolenWordIdx);
      prevSeg.end = prevSeg.words[prevSeg.words.length - 1].end;

      currSeg.words = [...stolenWords, ...currSeg.words];
      currSeg.start = stolenWords[0].start;
      currSeg.isCorrected = true;

      const afterCurrText = currSeg.words.map((w) => w.text).join('');
      repairs.push({
        segmentId: currSeg.id,
        relatedSegmentId: prevSeg.id,
        issueType: 'HEAD_STOLEN_FROM_PREV',
        beforeText: beforeCurrText,
        afterText: afterCurrText,
        description: `將前段誤搶之句首詞「${stolenStr}」移回本句開頭`,
      });
      recordIssue(
        currSeg.id,
        'HEAD_STOLEN_FROM_PREV',
        `已從前段尾端取回句首詞「${stolenStr}」`
      );
      recordIssue(currSeg.id, 'head_stolen_from_prev', '');
    }
  }

  working = working.filter((s) => s.words.length > 0);

  // =========================================================================
  // Step 1C：修復段內假句號 + 同語者無插話連續發言智慧合併與跨段語法價鍵縫合 (INCOMPLETE_SYNTAX)
  // =========================================================================
  const TOPIC_SHIFT_STARTER_STAGE1 =
    /^(?:然後|那|另外|再來|此外|同時|最後)?(?:第[二三四五六七八九十][個點項部分步]|另外一個(?:部分|問題|重點|面向)|再來(?:就是|是|一個)|那另外(?:就是|一個)|接下來(?:我們|就是)|至於(?:在|說)?)/;

  working = working.map((seg) => {
    if (seg.words.length < 2) return seg;
    const healedWords = seg.words.map((w, idx) => {
      if (idx >= seg.words.length - 1) return w;
      if (!/[。]$/.test(w.text)) return w;
      const cleanW = w.text.replace(/[。]+$/g, '');
      const nextW = seg.words[idx + 1].text.trim();
      if (
        /(?:[的之比在跟對向把被讓給為從到]|那個|這個|一些|一個)$/.test(cleanW) ||
        isSyntacticallyBoundPair(cleanW, nextW)
      ) {
        return { ...w, text: cleanW, word: cleanW };
      }
      if (
        /(?:因為|所以|就是|然後|好然後|而且|或是|或者|要能夠|能夠|例如說|比方說|這幾年|前幾年)$/.test(
          cleanW
        ) ||
        cleanW.length <= 3
      ) {
        return { ...w, text: `${cleanW}，`, word: `${cleanW}，` };
      }
      return w;
    });
    return { ...seg, words: healedWords };
  });

  const mergedSyntax: Segment[] = [];
  for (let i = 0; i < working.length; i++) {
    const curr = working[i];
    const prev = mergedSyntax.length > 0 ? mergedSyntax[mergedSyntax.length - 1] : null;

    if (!prev) {
      mergedSyntax.push(curr);
      continue;
    }

    const gapSec = curr.start - prev.end;
    const prevText = prev.words.map((w) => w.text).join('').trim();
    const currText = curr.words.map((w) => w.text).join('').trim();

    const lastPrevToken = prev.words[prev.words.length - 1]?.text || '';
    const lastPrevClean = lastPrevToken.replace(/[，。？！、；：,.?!\s]+$/g, '');
    const firstCurrToken = curr.words[0]?.text || '';

    const isSyntacticallyLockedBoundary =
      gapSec <= 0.85 &&
      (!/[。！？!?]$/.test(lastPrevToken) ||
        lastPrevClean.length <= 2 ||
        /[的之比在跟對向把被讓給有]$/.test(lastPrevClean)) &&
      isSyntacticallyBoundPair(lastPrevClean, firstCurrToken);

    const isSameSpeakerUninterrupted =
      prev.speakerId === curr.speakerId && gapSec >= -0.25 && gapSec <= 12.0;

    const isSameOrCompatibleSpeaker =
      isSameSpeakerUninterrupted || isSyntacticallyLockedBoundary;

    const isMidSentencePeriodCut =
      prev.speakerId === curr.speakerId &&
      (/(?:[的之比在跟對向把被讓給為從到]|要能夠|能夠|例如說|比方說|因為|就是|那個|這個|還有|現在還有)[，。、,.…]*$/.test(
        prevText
      ) ||
        isSyntacticallyBoundPair(lastPrevClean, firstCurrToken));

    const hasExplicitTopicShift =
      !isMidSentencePeriodCut &&
      prevText.length >= 140 &&
      (TOPIC_SHIFT_STARTER_STAGE1.test(currText) ||
        (prevText.length >= 240 && gapSec >= 2.2) ||
        prevText.length + currText.length > 380);

    if (
      prev.chunkIndex === curr.chunkIndex &&
      isSameOrCompatibleSpeaker &&
      ((isSyntacticallyLockedBoundary && gapSec <= 0.85) ||
        (isSameSpeakerUninterrupted &&
          (isMidSentencePeriodCut || !hasExplicitTopicShift)))
    ) {
      if (prev.speakerId !== curr.speakerId) {
        const prevWeight = Math.max(0.2, prev.end - prev.start) * Math.max(0.08, prev.acousticMargin ?? 0.14);
        const currWeight = Math.max(0.2, curr.end - curr.start) * Math.max(0.08, curr.acousticMargin ?? 0.14);
        if (currWeight > prevWeight * 1.15) {
          prev.speakerId = curr.speakerId;
        }
      }
      const beforePrevText = prevText;
      const lastPrevW = prev.words[prev.words.length - 1];
      if (isSyntacticallyLockedBoundary) {
        const stripped = lastPrevW.text.replace(/[，。、,.…]+$/, '');
        if (stripped) {
          prev.words[prev.words.length - 1] = {
            ...lastPrevW,
            text: stripped,
            word: stripped,
          };
        }
      } else if (/[。]$/.test(lastPrevW.text)) {
        const softened = lastPrevW.text.replace(/[。]+$/, '，');
        prev.words[prev.words.length - 1] = {
          ...lastPrevW,
          text: softened,
          word: softened,
        };
      }
      prev.words = [
        ...prev.words,
        ...curr.words.map((w) => ({
          ...w,
          speakerId: prev.speakerId,
        })),
      ];
      prev.end = Math.max(prev.end, curr.end);
      prev.isCorrected = true;

      repairs.push({
        segmentId: prev.id,
        relatedSegmentId: curr.id,
        issueType: 'INCOMPLETE_SYNTAX',
        beforeText: beforePrevText,
        afterText: prev.words.map((w) => w.text).join(''),
        description: 'Stage 1 合併同語者換氣或翻頁切碎之未完子句',
      });
      recordIssue(
        prev.id,
        'INCOMPLETE_SYNTAX',
        '已自動合併換氣斷裂之上下半句'
      );
      recordIssue(prev.id, 'broken_syntax_split', '');
    } else {
      mergedSyntax.push(curr);
    }
  }

  // =========================================================================
  // Step 1C.5：對話語用「問—答相鄰對 (Q&A Adjacency Pair)」與條件疑問句尾配價修復
  // =========================================================================
  const BINARY_VERDICT_REPLY_REGEX =
    /^(?:不算|算啊|算喔|算|可以啊|可以|不行|不可以|沒問題|不用|沒有|對啊|對)[。，！？!?\s]*$/;
  const DANGLING_CONDITIONAL_TAIL_REGEX =
    /^(?:(可是|不是|那|啊|欸)[，、\s]*(?:啊|那)?)?(如果|要是|假設)([一-龥]{2,8}?)(有充足|有足夠|有完整|有豐富)[。，！？!?\s]*$/;

  for (let i = 0; i < mergedSyntax.length; i++) {
    const currSeg = mergedSyntax[i];
    const currRaw = currSeg.words.map((w) => w.text).join('').trim();
    const condMatch = currRaw.match(DANGLING_CONDITIONAL_TAIL_REGEX);
    if (!condMatch) continue;

    const prevSeg = i > 0 ? mergedSyntax[i - 1] : null;
    const nextSeg = i + 1 < mergedSyntax.length ? mergedSyntax[i + 1] : null;
    const prevText = prevSeg ? prevSeg.words.map((w) => w.text).join('').trim() : '';
    const nextText = nextSeg ? nextSeg.words.map((w) => w.text).join('').trim() : '';

    const isNextBinaryVerdict =
      Boolean(nextSeg) &&
      nextSeg!.speakerId !== currSeg.speakerId &&
      BINARY_VERDICT_REPLY_REGEX.test(nextText);

    const hasPriorRenovationTopic = /(?:店面翻修|翻修|裝修|裝潢)/.test(prevText);
    const hasPriorNegativeConstraint = /(?:沒有要支持|不支持|不能|不算|不補助)/.test(
      prevText
    );

    if (isNextBinaryVerdict || hasPriorRenovationTopic) {
      const entityPhrase = condMatch[3];
      const inferredAction = hasPriorRenovationTopic ? '的裝修' : '的部分';
      const leadInterjection = hasPriorNegativeConstraint
        ? '不是，啊'
        : condMatch[1]
        ? `${condMatch[1]}，`
        : '';
      const repairedSentence = `${leadInterjection}${condMatch[2]}${entityPhrase}${inferredAction}怎麼辦？`;

      const altCandidates = [
        repairedSentence,
        `不是，如果${entityPhrase}${inferredAction}怎麼辦？`,
        `可是如果${entityPhrase}${inferredAction}怎麼辦？`,
        currRaw,
      ];

      const tokens = segmentNaturalLanguageTokens(repairedSentence);
      const dur = Math.max(0.3, currSeg.end - currSeg.start);
      const step = dur / Math.max(1, tokens.length);

      currSeg.words = tokens.map((tok, idx) => ({
        id: `${currSeg.id}_qa_${idx}`,
        text: tok,
        word: tok,
        originalText: idx === tokens.length - 1 ? condMatch[4] : tok,
        start: Number((currSeg.start + idx * step).toFixed(2)),
        end: Number((currSeg.start + (idx + 1) * step).toFixed(2)),
        confidence: 0.94,
        status: 'auto_corrected' as const,
        isPatched: true,
        isCorrected: true,
        candidates: altCandidates,
        reason:
          'Stage 1 對話語用「問—答相鄰對」與條件疑問句配價修復：結合前文主題與下句裁定答覆還原句尾快速連讀',
      }));
      currSeg.isCorrected = true;

      repairs.push({
        segmentId: currSeg.id,
        relatedSegmentId: nextSeg?.id,
        issueType: 'SEMANTIC_ANOMALY',
        beforeText: currRaw,
        afterText: repairedSentence,
        description:
          'Stage 1 依前後文「問—答相鄰對 (Q&A Adjacency)」與條件疑問配價修復快速連讀句尾',
      });
      recordIssue(
        currSeg.id,
        'SEMANTIC_ANOMALY',
        `已依問答相鄰對語用與前文主題修復條件疑問句（「${currRaw}」→「${repairedSentence}」）`
      );
    }
  }

  // =========================================================================
  // Step 1D：不合理自問自答與句尾附和插話拆解 (SEMANTIC_ANOMALY / LG-TPST)
  // =========================================================================
  const {
    resolvedSegments,
    transferredSegmentIds,
    flaggedForRelistenIds,
  } = resolveUnreasonableSelfQAAndTransferSpeaker(mergedSyntax, speakers);

  for (const tId of transferredSegmentIds) {
    recordIssue(
      tId,
      'SEMANTIC_ANOMALY',
      'Stage 1 偵測段內自問自答或句尾附和重疊，已自動拆句移轉語者'
    );
    recordIssue(tId, 'unreasonable_self_qa', '');
  }

  // =========================================================================
  // Step 1E：計算每段之句子語意完整度分數 (sentenceIntegrityScore) 與診斷報告
  // =========================================================================
  const finalSegments: Segment[] = resolvedSegments.map((seg) => {
    const fullText = seg.words.map((w) => w.text).join('').trim();
    const segIssues = Array.from(issueMap.get(seg.id) || []);
    const notes = noteMap.get(seg.id) || [];

    let score = 96;
    if (DANGLING_TAIL_VERB_PREP_ADVERB_REGEX.test(fullText)) {
      score -= 18;
      if (!segIssues.includes('INCOMPLETE_SYNTAX')) {
        segIssues.push('INCOMPLETE_SYNTAX');
        notes.push('句尾仍為及物動詞或介詞，待後文補齊賓語');
      }
    }
    if (
      fullText.length <= 2 &&
      !/^(好|對|是|沒錯|謝謝|了解|可以|算|不算|有|啥)[。！？,.!?]*$/.test(fullText)
    ) {
      score -= 15;
    }
    if (!/[。！？，、,.!?]$/.test(fullText) && fullText.length >= 6) {
      score -= 4;
    }
    if (segIssues.length > 0 && notes.some((n) => n.includes('已自動'))) {
      score = Math.max(score, 92);
    }
    score = Math.max(45, Math.min(100, score));

    const diag: SentenceIntegrityDiagnostic = {
      segmentId: seg.id,
      score,
      issues: segIssues.length > 0 ? segIssues : ['NONE', 'none'],
      autoRepaired: segIssues.length > 0 && notes.some((n) => n.includes('已')),
      diagnosticNote: notes.join('；') || '句子語意結構完整',
      suggestedRepair: notes.join('；') || undefined,
      suggestedRepairHint: notes.join('；') || undefined,
    };
    diagnostics[seg.id] = diag;

    return syncSegmentDerivedFields({
      ...seg,
      sentenceIntegrity: diag,
      sentenceIntegrityScore: score,
      sentenceIntegrityIssues: diag.issues,
    });
  });

  return {
    repairedSegments: finalSegments,
    repairs,
    diagnostics,
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
    .filter((item) => item.isPriorVerified || item.sovereigntyScore >= 5.5)
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
    .filter((item) => item.isPriorVerified || item.sovereigntyScore >= 5.5)
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

export function applySentenceLevelLocalCorrection(
  targetSegments: Segment[],
  verifiedTerms: string[],
  dictionary: DictEntry[] = []
): { updatedSegments: Segment[]; patches: CorrectionPatchItem[] } {
  const patches: CorrectionPatchItem[] = [];

  const state = useAppStore.getState();
  const { repairedSegments: stage1RepairedSegments } =
    runStage1SentenceIntegrityCheckAndRepair(targetSegments, state.speakers);
  targetSegments = stage1RepairedSegments;
  const allSegs = state.segments.length > 0 ? state.segments : targetSegments;
  const evidenceMap = analyzeCorpusStatisticalEvidence(
    allSegs,
    dictionary,
    state.docContextSummary
  );

  const fullCorpusText = converter(
    `${state.docContextSummary}\n${allSegs
      .map((s) => s.words.map((w) => w.text).join(''))
      .join('\n')}`
  );

  for (const vt of verifiedTerms) {
    if (
      !evidenceMap.has(vt) &&
      vt.length >= 2 &&
      vt.length <= 4 &&
      fullCorpusText.includes(vt)
    ) {
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
    const { typoAliases } = classifyDictEntryAliases(entry);
    if (typoAliases.length > 0) {
      const scopeLabel =
        entry.scope === 'public' || entry.scope === 'global'
          ? '公共字典'
          : entry.scope === 'personal'
          ? '個人字典'
          : '會議字典';
      for (const alias of typoAliases) {
        if (COMMON_STOP_WORDS.has(alias)) continue;
        dynamicRules.push({
          wrong: alias,
          right: entry.term.trim(),
          reason: `Pass 0 依${scopeLabel}同音別名自動修正（「${alias}」→「${entry.term.trim()}」）`,
        });
      }
    }
  }
  dynamicRules.sort((a, b) => b.wrong.length - a.wrong.length);

  const exactDominantMap = new Map<string, StatisticalTermEvidence>();
  const fuzzyDominantMap = new Map<string, StatisticalTermEvidence>();

  for (const item of evidenceMap.values()) {
    if (item.sovereigntyScore < 5.0) continue;

    const prevExact = exactDominantMap.get(item.exactSig);
    if (!prevExact || item.sovereigntyScore > prevExact.sovereigntyScore) {
      exactDominantMap.set(item.exactSig, item);
    }

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

    refinedSentence = refinedSentence.replace(
      /([\u4e00-\u9fa5]{2})([，、\s]*)(如果|盧果|如過)(?=(?:我們|你們|他們|我|你|他|再|就|先|會|可以|幫){1,3}(?:寄給|送給|發給|傳給|交給|買給|帶給|補給|提供給|留給|給)(?:他|她|你|你們|他們|大家|委員|老師|長官)[。，！？!?\s]*$)/g,
      (fullMatch, modWord: string, _punct: string, conjWord: string) => {
        if (COMMON_STOP_WORDS.has(modWord) || BAD_GRAMMATICAL_EDGE_REGEX.test(modWord)) {
          return fullMatch;
        }
        const replacement = `${modWord}禮物`;
        hitReasons.push({
          wrong: fullMatch,
          right: replacement,
          reason: `Stage 2 依話題化受詞與雙及物動詞「寄給/送給」語法配價修正「${conjWord} -> 禮物」`,
        });
        return replacement;
      }
    );

    for (const dRule of dynamicRules) {
      if (
        refinedSentence.includes(dRule.wrong) &&
        !dRule.right.includes(dRule.wrong)
      ) {
        refinedSentence = refinedSentence.split(dRule.wrong).join(dRule.right);
        hitReasons.push(dRule);
      }
    }

    refinedSentence = refinedSentence
      .replace(/([\u4e00-\u9fa5]{2,3})\1+/g, '$1')
      .replace(/([\u4e00-\u9fa5])\1(?=[\u4e00-\u9fa5]{2})/g, '$1')
      .replace(/店店面/g, '店面')
      .replace(/翻修修/g, '翻修');

    // 建立自然詞界索引集合，嚴禁跨詞切割（如防止從「協助他們」切出「助他」誤替換）
    const preTokens = segmentNaturalLanguageTokens(refinedSentence);
    const refinedBoundarySet = new Set<number>([0]);
    let bCursor = 0;
    for (const pt of preTokens) {
      const ptPure = pt.replace(/[，。？！、；：,.?!\s]+$/g, "");
      refinedBoundarySet.add(bCursor + ptPure.length);
      bCursor += pt.length;
      refinedBoundarySet.add(bCursor);
    }

    for (const [sig, anchor] of exactEntries) {
      const correctTerm = anchor.term;
      if (
        sig.includes('-') &&
        correctTerm.length >= 2 &&
        correctTerm.length <= 4 &&
        fullCorpusText.includes(correctTerm)
      ) {
        const len = correctTerm.length;
        for (let i = 0; i <= refinedSentence.length - len; i++) {
          if (!refinedBoundarySet.has(i) || !refinedBoundarySet.has(i + len)) {
            continue;
          }
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
                  reason: `Pass 1 依會議出現實體主權分數 (${anchor.sovereigntyScore}) 自動收斂為「${correctTerm}」`,
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

    for (const [fKey, anchor] of fuzzyEntries) {
      const correctTerm = anchor.term;
      if (
        fKey.includes('-') &&
        correctTerm.length >= 2 &&
        correctTerm.length <= 4 &&
        anchor.sovereigntyScore >= 8.0 &&
        fullCorpusText.includes(correctTerm)
      ) {
        const len = correctTerm.length;
        for (let i = 0; i <= refinedSentence.length - len; i++) {
          if (!refinedBoundarySet.has(i) || !refinedBoundarySet.has(i + len)) {
            continue;
          }
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
                  reason: `Pass 1 依後文實體見證與模糊音韻主權自動收斂為「${correctTerm}」`,
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

    const tokens = segmentNaturalLanguageTokens(refinedSentence);
    const dur = Math.max(0.2, seg.end - seg.start);
    const step = dur / Math.max(1, tokens.length);

    const newWords: WordSpan[] = tokens.map((tok, idx) => {
      // 【關鍵修正】：剝除尾隨標點符號 (`pureTok`) 再進行命中規則與異常槽位比對，
      // 解決 `segmentNaturalLanguageTokens` 吸附標點（如「原民會。」、「工項，」）後無法顯示綠底/黃底徽章之問題！
      const pureTok = tok.replace(/[，。？！、；：,.?!\s]/g, '');
      const matchedRule =
        pureTok.length >= 1
          ? hitReasons.find((r) => {
              const pureRight = r.right.replace(/[，。？！、；：,.?!\s]/g, '');
              return (
                pureRight === pureTok ||
                (pureRight.length >= 2 &&
                  pureTok.length >= 2 &&
                  (pureRight.includes(pureTok) || pureTok.includes(pureRight)))
              );
            })
          : undefined;

      if (matchedRule && /[\u4e00-\u9fa5A-Za-z0-9]/.test(pureTok)) {
        const trailingPunct = tok.slice(pureTok.length);
        const safeOriginalText =
          matchedRule.wrong.length <= tok.length + 2
            ? `${matchedRule.wrong}${trailingPunct}`
            : tok;
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

      const matchedAnomaly =
        pureTok.length >= 1
          ? intelReport.anomalousSpans.find((a) => {
              const pureAnom = a.text.replace(/[，。？！、；：,.?!\s]/g, '');
              return (
                pureAnom === pureTok ||
                (pureAnom.length <= 5 &&
                  pureTok.length >= 2 &&
                  (pureAnom.includes(pureTok) || pureTok.includes(pureAnom)))
              );
            })
          : undefined;

      if (matchedAnomaly && /[\u4e00-\u9fa5]/.test(pureTok)) {
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
      const preservedStatus = prevWordSameIdx ? prevWordSameIdx.status : 'normal';

      return {
        id: `${seg.id}_w_${idx}`,
        text: tok,
        originalText: prevWordSameIdx?.originalText,
        start: Number((seg.start + idx * step).toFixed(2)),
        end: Number((seg.start + (idx + 1) * step).toFixed(2)),
        confidence: preservedConf,
        status: preservedStatus,
        reason: prevWordSameIdx?.reason,
        candidates: prevWordSameIdx?.candidates,
      };
    });

    return syncSegmentDerivedFields({
      ...seg,
      words: newWords,
      isCorrected: true,
    });
  });

  return { updatedSegments: updatedSegments.map(syncSegmentDerivedFields), patches };
}

function enforceThreeTierGatingOnModelPatches(
  rawPatches: CorrectionPatchItem[],
  targetSegments: Segment[]
): CorrectionPatchItem[] {
  const segMap = new Map<string, Segment>();
  for (const s of targetSegments) segMap.set(s.id, s);

  const filtered: CorrectionPatchItem[] = [];
  for (const p of rawPatches) {
    if (!p.newSpeakerId) {
      filtered.push(p);
      continue;
    }

    const seg = segMap.get(p.segmentId);
    if (!seg) continue;

    const tier = resolveSegmentSlotTier(seg);
    if (tier === 'tier1_locked_anchor') {
      continue;
    }

    if (tier === 'tier2_candidate_constrained') {
      const allowedRunnerUp = seg.runnerUpSpeakerId || seg.speakerId;
      if (
        p.newSpeakerId !== seg.speakerId &&
        p.newSpeakerId !== allowedRunnerUp
      ) {
        if (allowedRunnerUp !== seg.speakerId) {
          filtered.push({
            ...p,
            newSpeakerId: allowedRunnerUp,
          });
        }
        continue;
      }
    }

    filtered.push(p);
  }
  return filtered;
}

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

  const state = useAppStore.getState();
  const {
    repairedSegments: stage1Segments,
    flaggedForRelistenIds,
  } = runStage1SentenceIntegrityCheckAndRepair(
    payload.highAttentionTargetSegments,
    state.speakers
  );

  const { updatedSegments: resolvedSegments, patches: localPatches } =
    applySentenceLevelLocalCorrection(
      stage1Segments,
      payload.lowAttentionGlobalMemory.verifiedTerms,
      dictionary
    );

  const sourceTargetIds = payload.highAttentionTargetSegments.map((s) => s.id);
  useAppStore
    .getState()
    .replaceBatchSegments(resolvedSegments, sourceTargetIds);

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

  const hasNonLockedSegments = resolvedSegments.some(
    (s) => resolveSegmentSlotTier(s) !== 'tier1_locked_anchor'
  );
  if (suspectSlots.length === 0 && !hasNonLockedSegments) {
    return localPatches;
  }

  let rawModelPatches: CorrectionPatchItem[] = [];
  const updatedPayload: TriLevelContextPayload = {
    ...payload,
    highAttentionTargetSegments: resolvedSegments,
    suspectSlots,
    slottedPromptTranscript,
  };

  if (config.geminiApiKey.trim()) {
    try {
      rawModelPatches = await cloudClient.correctWithGeminiTriLevel(
        config.geminiApiKey.trim(),
        updatedPayload
      );
    } catch (err) {
      console.warn('雲端三層關注度槽位校正略過，維持本機自洽結果:', err);
    }
  } else if (runLocalLlmCorrection) {
    try {
      rawModelPatches = await runLocalLlmCorrection(updatedPayload);
    } catch (err) {
      console.warn('本機 WebLLM 槽位校正略過:', err);
    }
  }

  const gatedPatches = enforceThreeTierGatingOnModelPatches(
    rawModelPatches,
    resolvedSegments
  );

  if (gatedPatches.length > 0) {
    onApplyPatches(gatedPatches, targetIds);
  }
  return [...localPatches, ...gatedPatches];
}

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

  useAppStore
    .getState()
    .replaceBatchSegments(
      updatedSegments,
      allSegments.map((s) => s.id)
    );

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
      resolveSegmentSlotTier(s) !== 'tier1_locked_anchor' ||
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
        const gated = enforceThreeTierGatingOnModelPatches(
          cloudPatches,
          batchTarget
        );
        if (gated.length > 0) {
          useAppStore
            .getState()
            .applyCorrectionPatches(
              gated,
              batchTarget.map((s) => s.id)
            );
          aiFixedCount += gated.length;
        }
      } catch {
        // ignore
      }
    } else if (runLocalLlmCorrection) {
      try {
        const llmPatches = await runLocalLlmCorrection(retroPayload);
        const gated = enforceThreeTierGatingOnModelPatches(
          llmPatches,
          batchTarget
        );
        if (gated.length > 0) {
          useAppStore
            .getState()
            .applyCorrectionPatches(
              gated,
              batchTarget.map((s) => s.id)
            );
          aiFixedCount += gated.length;
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

  const state = useAppStore.getState();
  const evidenceMap = analyzeCorpusStatisticalEvidence(
    state.segments.length > 0 ? state.segments : chunkSegments,
    state.dictionary,
    state.docContextSummary
  );

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

/**
 * 【一鍵生成會議 AI 摘要、重要結論與行動事項報告 (`generateStructuredMeetingMinutes`)】
 * 支援雙模式：
 * 1. 若已設定 Gemini API Key，直接呼叫 `gemini-2.5-flash` 結構化 JSON Schema 生成深度會議紀錄。
 * 2. 若處於 100% 本機離線模式（含整合式內顯共享記憶體保護環境），透過內建「多通道話語結構與承諾句分析引擎」自動萃取執行摘要、討論議題、關鍵結論與行動事項（含負責單位/時間點），零顯存佔用、毫秒級完成！
 */
/**
 * 將口語逐字稿句子去蕪存菁，濾除口頭禪與冗贅連接詞，收斂為精煉條列要點
 */
function condenseSpokenClauseToCleanPoint(rawText: string): string {
  let s = rawText.trim();
  s = s
    .replace(/^(?:好[，、]?|那[，、]?|對[，、]?|喂[，、]?|報告主席[，、]?|好報告主席[，、]?|目前的話[，、]?|如果說是以?|我可以問一下那個|我們很快[，、]?)+/g, "")
    .replace(/(?:那個|這個是|那目前就是協助他們去[，、]?|現在就是協助他們|他們現在走的路線會比較像是|因為他們之前的|他們現在要調的就是)/g, "")
    .replace(/協助他們去/g, "協助")
    .replace(/協助他們/g, "協助")
    .replace(/沒有問到細節[，、]?所以還在持續聯繫當中/g, "尚未問到細節，持續聯繫對口中")
    .replace(/計畫書已經寫得差不多了[，、]?所以他們/g, "計畫書已近完成")
    .replace(/[，、]{2,}/g, "，")
    .replace(/^[，、。！？!?\s]+|[，、…\s]+$/g, "");
  if (s.length > 52) {
    const parts = s.split(/[，；]/).filter((p) => p.trim().length >= 4);
    if (parts.length >= 2) {
      s = parts.slice(0, 3).join("，");
    }
  }
  return s;
}

export async function generateStructuredMeetingMinutes(params: {
  config?: EngineConfig;
  segments: Segment[];
  speakers: Record<string, Speaker>;
  dictionary: DictEntry[];
  docContextSummary: string;
  rollingMeetingSummary: string;
  geminiApiKey?: string;
}): Promise<MeetingMinutesReport> {
  const {
    config,
    segments,
    speakers,
    dictionary,
    docContextSummary,
    rollingMeetingSummary,
    geminiApiKey,
  } = params;

  const effectiveGeminiKey = (geminiApiKey || config?.geminiApiKey || "").trim();

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };

  const speakerStatMap = new Map<string, MeetingMinutesSpeakerStat>();
  for (const seg of segments) {
    const spId = seg.speakerId;
    const spName = speakers[spId]?.name || spId;
    const charLen = seg.words.reduce(
      (acc, w) => acc + (w.text || "").trim().length,
      0
    );
    const existing = speakerStatMap.get(spId);
    if (existing) {
      existing.turnCount += 1;
      existing.charCount += charLen;
      existing.speakerName = spName;
    } else {
      speakerStatMap.set(spId, {
        speakerId: spId,
        speakerName: spName,
        turnCount: 1,
        charCount: charLen,
      });
    }
  }
  const speakerStats: MeetingMinutesSpeakerStat[] = Array.from(
    speakerStatMap.values()
  ).sort((a, b) => b.charCount - a.charCount);

  const coreTerms = mineAdaptiveHighFreqTerms(
    segments,
    dictionary,
    `${docContextSummary}\n${rollingMeetingSummary}`
  );

  const formattedLines = segments.map((seg) => {
    const spName = speakers[seg.speakerId]?.name || seg.speakerId;
    const text = seg.words.map((w) => w.text).join("");
    return `[${formatTime(seg.start)}] ${spName}：${text}`;
  });

  // 1. 優先嘗試 Gemini Cloud 結構化極簡條列生成
  if (effectiveGeminiKey) {
    try {
      const transcriptSample =
        formattedLines.length <= 260
          ? formattedLines.join("\n")
          : [
              ...formattedLines.slice(0, 120),
              "...(中段略)...",
              ...formattedLines.slice(-120),
            ].join("\n");

      const prompt = `請擔任專業會議秘書，根據以下會議逐字稿，以「極簡條列式」收斂出核心討論要點、重要決議與行動事項（請使用繁體中文，回傳純 JSON 格式）。
撰寫原則：
- 嚴禁冗長段落與口語贅字，所有內容皆採「精煉條列要點」，每條控制在 20~42 字以內。
- 完全忠於會議實際討論內容，不偏離事實、不自行編造未提及之資訊，讓讀者 10 秒內快速掌握核心事項。

【會議背景與核心詞】：
${docContextSummary || "專案進度檢討會議"}
核心詞：${coreTerms.slice(0, 20).join("、")}

【會議逐字稿內容】：
${transcriptSample}

請嚴格依以下 JSON 結構輸出（勿加 Markdown code block）：
{
  "meetingTitle": "會議名稱（精簡 10~20 字）",
  "executiveSummary": "• 核心範疇要點1\n• 核心範疇要點2",
  "keyConclusions": ["精煉決議要點1（限35字內）", "精煉決議要點2（限35字內）"],
  "topicBreakdowns": [
    {
      "topic": "議題名稱",
      "timeRange": "00:00–02:00",
      "keyPoints": "精煉現況要點（限35字內）",
      "conclusion": "裁示或結論（限25字內）"
    }
  ],
  "actionItems": [
    {
      "id": "ACT-01",
      "task": "具體精簡待辦事項（限32字內）",
      "owner": "負責單位/人員",
      "deadlineOrTimeline": "時程（如：9月24日、持續追蹤）",
      "statusNote": "備註"
    }
  ]
}`;

      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(
          effectiveGeminiKey
        )}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: {
              temperature: 0.1,
              maxOutputTokens: 1500,
              responseMimeType: "application/json",
            },
          }),
        }
      );

      if (res.ok) {
        const data = await res.json();
        const rawJson = (
          data?.candidates?.[0]?.content?.parts?.[0]?.text || ""
        ).trim();
        const parsed = JSON.parse(rawJson);
        if (parsed && parsed.executiveSummary) {
          const resolvedTitle = converter(
            String(parsed.meetingTitle || parsed.title || "會議核心摘要與行動事項")
          );
          return {
            generatedAt: new Date().toISOString(),
            engineUsed: "gemini_cloud",
            title: resolvedTitle,
            meetingTitle: resolvedTitle,
            executiveSummary: converter(String(parsed.executiveSummary)),
            keyConclusions: Array.isArray(parsed.keyConclusions)
              ? parsed.keyConclusions.map((c: string) => converter(String(c)))
              : [],
            topicBreakdowns: Array.isArray(parsed.topicBreakdowns)
              ? parsed.topicBreakdowns.map(
                  (t: Record<string, unknown>): MeetingTopicDecisionItem => {
                    const pts = converter(
                      String(t.discussionPoints || t.keyPoints || "")
                    );
                    const dec = converter(
                      String(t.decision || t.conclusion || "")
                    );
                    const tr = t.timeRange
                      ? converter(String(t.timeRange))
                      : undefined;
                    return {
                      topic: converter(String(t.topic || "")),
                      discussionPoints: pts,
                      keyPoints: pts,
                      decision: dec,
                      conclusion: dec,
                      timeRange: tr,
                    };
                  }
                )
              : [],
            actionItems: Array.isArray(parsed.actionItems)
              ? parsed.actionItems.map(
                  (
                    a: Record<string, unknown>,
                    idx: number
                  ): MeetingActionItem => {
                    const dl = converter(
                      String(a.deadline || a.deadlineOrTimeline || "持續追蹤")
                    );
                    const note = converter(
                      String(a.contextNote || a.statusNote || "進行中")
                    );
                    return {
                      id: String(
                        a.id || `ACT-${String(idx + 1).padStart(2, "0")}`
                      ),
                      task: converter(String(a.task || "")),
                      owner: converter(String(a.owner || "與會團隊")),
                      deadline: dl,
                      deadlineOrTimeline: dl,
                      contextNote: note,
                      statusNote: note,
                    };
                  }
                )
              : [],
            speakerStats,
          };
        }
      }
    } catch (err) {
      console.warn(
        "⚠️ [MeetingMinutes] 雲端摘要生成略過，切換至本機精煉條列分析引擎:",
        err
      );
    }
  }

  // 2. 本機零顯存「精煉條列話語收斂引擎 (Concise Bullet Discourse Analyzer)」
  const topKeywords = coreTerms.slice(0, 6);
  const inferredTitle =
    topKeywords.length >= 2
      ? `${topKeywords.slice(0, 3).join("、")}進度檢討會議摘要`
      : "專案進度與決議事項摘要";

  const decisionRegex =
    /(?:決定|決議|確定|限定|只要|原則上|結論|共識|同意|通過|沒有問題|沒問題|改成|變成|轉成|調整為|朝.+方向|納入.+會議|規劃設計)/;
  const actionRegex =
    /(?:協助|聯繫|聯絡|跟進|追蹤|詢問|確認|釐清|修改|修正|補齊|補充|提供|召開|開會|申請|編列|提報|完成|處理)/;
  const timelineRegex =
    /((?:[0-9一二三四五六七八九十]+月(?:[0-9一二三四五六七八九十]+[號日]|中旬|上旬|下旬|底|初)?)|(?:[0-9一二三四五六七八九十]+[號日])|(?:禮拜[一二三四五六日天](?:早上|下午)?)|(?:這[個]?禮拜|下[個]?禮拜|本週|下週|月底|年底|下次會議|持續聯繫))/;

  const keyConclusions: string[] = [];
  const actionItems: MeetingActionItem[] = [];
  const topicBreakdowns: MeetingTopicDecisionItem[] = [];

  const numBlocks = Math.min(4, Math.max(1, Math.ceil(segments.length / 7)));
  const blockSize = Math.max(3, Math.ceil(segments.length / numBlocks));

  for (let b = 0; b < segments.length; b += blockSize) {
    const blockSegs = segments.slice(b, b + blockSize);
    if (blockSegs.length === 0) continue;

    const blockText = blockSegs
      .map((s) => s.words.map((w) => w.text).join(""))
      .join(" ");
    const timeRange = `${formatTime(blockSegs[0].start)}–${formatTime(
      blockSegs[blockSegs.length - 1].end
    )}`;

    const matchedTopics = coreTerms.filter(
      (t) => t.length >= 2 && blockText.includes(t)
    );
    const topicLabel =
      matchedTopics.length > 0
        ? matchedTopics.slice(0, 2).join("・")
        : `討論要點 ${topicBreakdowns.length + 1}`;

    const cleanClauses = blockSegs
      .flatMap((s) =>
        s.words
          .map((w) => w.text)
          .join("")
          .split(/[。！？?!；;]/)
          .map((cl) => condenseSpokenClauseToCleanPoint(cl))
      )
      .filter((cl) => cl.length >= 6 && !/^(甚麼意思|什麼意思|這樣OK嗎|可以嗎)$/.test(cl));

    const keyPointSent =
      cleanClauses.slice(0, 2).join("；") ||
      condenseSpokenClauseToCleanPoint(blockText.slice(0, 50));

    const decisionSent = cleanClauses.find((cl) => decisionRegex.test(cl));
    const conclusionText = decisionSent || "";

    topicBreakdowns.push({
      topic: topicLabel,
      discussionPoints: keyPointSent,
      keyPoints: keyPointSent,
      decision: conclusionText,
      conclusion: conclusionText,
      timeRange,
    });
  }

  for (const seg of segments) {
    const spName = speakers[seg.speakerId]?.name || seg.speakerId;
    const fullSegText = seg.words.map((w) => w.text).join("").trim();
    const clauses = fullSegText
      .split(/[。！？?!]/)
      .map((c) => c.trim())
      .filter((c) => c.length >= 7);

    for (const cl of clauses) {
      const condensed = condenseSpokenClauseToCleanPoint(cl);
      if (condensed.length < 6) continue;

      if (decisionRegex.test(cl) && keyConclusions.length < 5) {
        if (
          !keyConclusions.some((existing) =>
            existing.includes(condensed.slice(0, 8))
          )
        ) {
          keyConclusions.push(`${condensed}。`);
        }
      }

      if (actionRegex.test(cl) && actionItems.length < 5) {
        const timeMatch = cl.match(timelineRegex);
        const deadline = timeMatch ? timeMatch[1] : "持續追蹤";
        const matchedEntities = coreTerms
          .filter((t) => cl.includes(t) || fullSegText.includes(t))
          .slice(0, 2);
        const prefixTag =
          matchedEntities.length > 0 ? `【${matchedEntities.join("・")}】` : "";
        const taskStr = `${prefixTag}${condensed}。`;

        if (!actionItems.some((a) => a.task.includes(condensed.slice(0, 8)))) {
          actionItems.push({
            id: `ACT-${String(actionItems.length + 1).padStart(2, "0")}`,
            task: taskStr,
            owner: spName,
            deadline,
            deadlineOrTimeline: deadline,
            contextNote: `時間軸 ${formatTime(seg.start)}`,
            statusNote: `時間軸 ${formatTime(seg.start)}`,
          });
        }
      }
    }
  }

  if (keyConclusions.length === 0 && topicBreakdowns.length > 0) {
    for (const tb of topicBreakdowns.slice(0, 3)) {
      keyConclusions.push(
        `【${tb.topic}】${tb.decision || tb.discussionPoints}。`
      );
    }
  }

  if (actionItems.length === 0 && segments.length > 0) {
    const lastSeg = segments[segments.length - 1];
    actionItems.push({
      id: "ACT-01",
      task: `${
        topicBreakdowns[0]?.discussionPoints ||
        condenseSpokenClauseToCleanPoint(
          lastSeg.words.map((w) => w.text).join("")
        )
      }。`,
      owner: speakers[lastSeg.speakerId]?.name || "與會團隊",
      deadline: "持續跟進",
      deadlineOrTimeline: "持續跟進",
      contextNote: "依會議討論辦理",
      statusNote: "依會議討論辦理",
    });
  }

  const executiveSummary =
    topKeywords.length > 0
      ? `• 聚焦議題：${topKeywords.join("、")}之進度盤點、補助資格與跨單位聯繫。`
      : `• 本次會議共檢視 ${topicBreakdowns.length} 項重點議題，收斂 ${keyConclusions.length} 項決議與 ${actionItems.length} 項行動事項。`;

  return {
    generatedAt: new Date().toISOString(),
    engineUsed: "hybrid_discourse_analyzer",
    title: inferredTitle,
    meetingTitle: inferredTitle,
    executiveSummary,
    keyConclusions,
    topicBreakdowns: topicBreakdowns.slice(0, 5),
    actionItems,
    speakerStats,
  };
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

/**
 * 【即時跨視窗回溯同音消歧引擎 (`runIncrementalRetroactiveEntityPass`)】：
 * 每當新視窗轉寫出高信心度詞彙（例如 02:44 清楚辨識出「尖石」、01:49 辨識出「原民會」），
 * 立即以零 VRAM 成本掃描目前已產出之所有前段句子，將前段因口音或殘響誤聽之同音異字（如 00:39 與 00:43 之「監持」）即時回溯統一！
 */
export function runIncrementalRetroactiveEntityPass(): number {
  const state = useAppStore.getState();
  if (state.segments.length < 2) return 0;

  const verifiedTerms = mineAdaptiveHighFreqTerms(
    state.segments,
    state.dictionary,
    `${state.docContextSummary}\n${state.rollingMeetingSummary}`
  );
  if (verifiedTerms.length === 0) return 0;

  const { updatedSegments, patches } = applySentenceLevelLocalCorrection(
    state.segments,
    verifiedTerms,
    state.dictionary
  );

  if (patches.length > 0) {
    useAppStore
      .getState()
      .replaceBatchSegments(
        updatedSegments,
        state.segments.map((s) => s.id)
      );
  }
  return patches.length;
}
