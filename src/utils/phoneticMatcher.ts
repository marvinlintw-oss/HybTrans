// src/utils/phoneticMatcher.ts
import { Segment, PhoneticMatchCandidate } from '../types';

/**
 * 台灣會議常見姓氏、領域詞根、職稱與高頻語素之聲韻母同音群組映射表
 * （涵蓋無平捲舌 z/zh, c/ch, s/sh、前後鼻音 in/ing, en/eng, an/ang、流音滑音弱化 l/r/n/y 與圓唇鬆化）
 */
const PHONETIC_GROUPS: Record<string, string> = {
  // 常見姓氏與高頻會議字根同音群組
  '林臨淋琳霖零玲鈴齡伶翎': 'LIN',
  '陳塵臣辰晨沉成城誠呈承程懲乘': 'CHEN',
  '張章彰璋髒商傷湯': 'ZHANG',
  '李里理禮裡鯉履哩': 'LI',
  '王亡汪網往枉晚碗挽宛婉琬椀灣彎頑丸完玩': 'WAN',
  '吳吾無蕪梧巫務物誤悟霧勿屋': 'WU',
  '劉留流琉榴瘤': 'LIU',
  '蔡菜採彩踩財材裁才': 'CAI',
  '楊陽揚洋羊佯樣養仰': 'YANG',
  '許栩詡徐須需虛噓詢尋巡': 'XU',
  '鄭正政證症爭徵真針珍診陣鎮振': 'ZHENG',
  '謝械卸屑蟹寫血協脅鞋': 'XIE',
  '郭鍋國果過裹擴闊': 'GUO',
  '洪紅宏虹鴻哄烘': 'HONG',
  '曾增爭箏蒸真偵': 'ZENG',
  '廖料瞭療聊撩': 'LIAO',
  '賴來萊瀨藍蘭攔欄': 'LAI',
  '蘇疏輸書舒樞叔屬署鼠數術述束': 'SU',
  '周洲舟州週驟軸': 'ZHOU',
  '葉頁業夜液腋咽謁': 'YE',
  // 常見職稱、政府標案與會議高頻同音群組 (含前後鼻音/平捲舌/流音滑音/元音鬆化容錯)
  '處觸畜築儲礎楚出初除廚': 'CHU',
  '長場廠掌漲常償嚐腸': 'ZHANG_TITLE',
  '委偉尾瑋偽緯位未味衛為圍違唯維微危薇': 'WEI',
  '員元園原源圓援緣沿延研言嚴岩': 'YUAN',
  '於余妤魚漁娛逾愚榆瑜渝雲云勻芸': 'YU',
  '科顆棵磕苛課客克刻': 'KE',
  '主組祖阻詛煮囑助住注駐祝著': 'ZHU',
  '任認韌刃妊人仁忍': 'REN',
  '秘密蜜覓冪密': 'MI',
  '神什沈審嬸慎滲深身申伸升生聲勝聖盛': 'SHEN',
  '如儒茹汝乳入辱盧蘆爐': 'RU',
  '總種腫塚中忠終鐘衷眾重': 'ZONG',
  '工功攻公宮弓恭供躬共貢': 'GONG',
  '項向象像相巷享響想箱廂鄉香': 'XIANG',
  '馬碼瑪螞麻嘛罵媽抹': 'MA',
  '經精晶京驚鯨金今津斤禁近進盡勁景警井': 'JING',
  '俊駿峻浚郡菌軍均君鈞': 'JUN',
  '穎影營迎贏盈瑩螢沂儀宜疑移姨遺頤': 'YING_YI',
  '博薄搏駁伯帛泊白百擺柏拍排牌': 'BO',
  '姐姊解節結潔借界戒介屆街皆接': 'JIE',
  '士事式市試視室是師詩失施獅濕十石時實食識史使始示世勢適釋': 'SHI',
  '教較校酵覺交郊嬌澆焦礁角腳剿叫轎笑效肖孝': 'JIAO',
  '授受售瘦獸收手守首壽': 'SHOU',
  '店電墊殿奠佃甸點典': 'DIAN',
  '面麵棉眠免勉緬': 'MIAN',
  '翻番凡煩繁樊販範犯飯泛': 'FAN',
  '修羞休朽秀袖繡鏽': 'XIU',
  '開揩慨楷凱': 'KAI',
  '新心辛欣薪馨星興腥芯': 'XIN',
  '標彪飆錶表裱': 'BIAO',
  '不布步部怖捕補普譜': 'BU',
  '要耀曜藥鑰搖遙謠姚咬': 'YAO',
  '廊狼郎良涼糧梁量藍蘭籃欄': 'LANG',
  '帶代袋貸待怠殆大太態泰台臺抬苔': 'DAI',
  '件見建健鍵漸踐劍薦鑑間肩堅監兼簡檢減撿': 'JIAN',
  '群裙尋巡詢循': 'QUN',
  '輔腐府俯釜付負附赴副富復腹覆': 'FU',
  '導島倒禱蹈到道稻盜': 'DAO',
  '央泱秧英應鷹迎營贏盈': 'YANG_YING',
  '切且竊妾企契氣棄汽器': 'QIE',
  '小曉筱少稍梢紹哨笑校效': 'XIAO',
  '企起豈啟氣器棄契': 'QI',
  '支之知肢織汁資姿滋諮只指止址紙至志制致置質治炙': 'ZHI',
  '晰析息悉惜昔夕希稀犧吸西膝': 'XI',
  '就舊救舅咎久九酒究糾': 'JIU',
  '加家佳嘉夾挾袈': 'JIA',
  '盟萌蒙濛盲忙茫氓': 'MENG',
  '議意義藝亦異役疫譯翼億憶益誼一衣醫依壹架價假嫁稼駕寄記計既際季紀': 'YI',
  '評平瓶憑萍屏貧頻品聘': 'PING',
  '選癬險顯憲縣現限線獻': 'XUAN',
  '核合何河荷盒和禾吉及級即極急集籍': 'HE',
  '銷消宵霄削囂小曉': 'XIAO_2',
  '給北被備背倍輩貝': 'GEI_BEI',
  '裝樁莊妝裝潢': 'ZHUANG',
  '充衝沖聰匆蔥煙': 'CHONG',
  '足族卒竹築逐燭': 'ZU_FOOT',
  '怎診枕整貞針': 'ZEN',
  '麼嗎嘛麻馬媽': 'ME_PARTICLE',
  '辦半伴扮絆版板': 'BAN',
};

const charToPhoneticCode = new Map<string, string>();
Object.entries(PHONETIC_GROUPS).forEach(([chars, code]) => {
  for (const ch of chars) {
    if (!charToPhoneticCode.has(ch)) {
      charToPhoneticCode.set(ch, code);
    }
  }
});

/**
 * 細粒度音韻四元組結構 (聲母 Initial, 介音 Medial, 韻腹與韻尾 Nucleus/Coda)
 * 用於動態規劃加權音韻編輯距離與跨音節連音縮讀 (Syllable Contraction)
 */
interface SyllablePhonemeTuple {
  rawCode: string;
  fuzzyHash: string;
  initial: string;
  medial: string;
  rime: string;
}

const CODE_TO_TUPLE_MAP: Record<
  string,
  { initial: string; medial: string; rime: string; fuzzyHash: string }
> = {
  LIN: { initial: 'l', medial: 'i', rime: 'in', fuzzyHash: 'lin' },
  CHEN: { initial: 'c', medial: '', rime: 'en', fuzzyHash: 'cen' },
  ZHANG: { initial: 'z', medial: '', rime: 'an', fuzzyHash: 'zan' },
  ZHANG_TITLE: { initial: 'z', medial: '', rime: 'an', fuzzyHash: 'zan' },
  LI: { initial: 'l', medial: 'i', rime: 'i', fuzzyHash: 'li' },
  WAN: { initial: 'w', medial: 'u', rime: 'an', fuzzyHash: 'wan' },
  WU: { initial: 'w', medial: 'u', rime: 'u', fuzzyHash: 'wu' },
  LIU: { initial: 'l', medial: 'i', rime: 'ou', fuzzyHash: 'liou' },
  CAI: { initial: 'c', medial: '', rime: 'ai', fuzzyHash: 'cai' },
  YANG: { initial: 'y', medial: 'i', rime: 'an', fuzzyHash: 'yan' },
  YANG_YING: { initial: 'y', medial: 'i', rime: 'in', fuzzyHash: 'yin' },
  XU: { initial: 'x', medial: 'u', rime: 'u', fuzzyHash: 'xu' },
  ZHENG: { initial: 'z', medial: '', rime: 'en', fuzzyHash: 'zen' },
  XIE: { initial: 'x', medial: 'i', rime: 'ie', fuzzyHash: 'xie' },
  GUO: { initial: 'g', medial: 'u', rime: 'o', fuzzyHash: 'guo' },
  HONG: { initial: 'h', medial: 'u', rime: 'on', fuzzyHash: 'hon' },
  ZENG: { initial: 'z', medial: '', rime: 'en', fuzzyHash: 'zen' },
  LIAO: { initial: 'l', medial: 'i', rime: 'ao', fuzzyHash: 'liao' },
  LAI: { initial: 'l', medial: '', rime: 'ai', fuzzyHash: 'lai' },
  SU: { initial: 's', medial: 'u', rime: 'u', fuzzyHash: 'su' },
  ZHOU: { initial: 'z', medial: '', rime: 'ou', fuzzyHash: 'zou' },
  YE: { initial: 'y', medial: 'i', rime: 'ie', fuzzyHash: 'ye' },
  CHU: { initial: 'c', medial: 'u', rime: 'u', fuzzyHash: 'cu' },
  WEI: { initial: 'w', medial: 'u', rime: 'ei', fuzzyHash: 'wei' },
  YUAN: { initial: 'y', medial: 'u', rime: 'an', fuzzyHash: 'yuan' },
  YU: { initial: 'y', medial: 'u', rime: 'u', fuzzyHash: 'yu' },
  KE: { initial: 'k', medial: '', rime: 'e', fuzzyHash: 'ke' },
  ZHU: { initial: 'z', medial: 'u', rime: 'u', fuzzyHash: 'zu' },
  REN: { initial: 'l', medial: '', rime: 'en', fuzzyHash: 'len' },
  MI: { initial: 'm', medial: 'i', rime: 'i', fuzzyHash: 'mi' },
  SHEN: { initial: 's', medial: '', rime: 'en', fuzzyHash: 'sen' },
  RU: { initial: 'l', medial: 'u', rime: 'u', fuzzyHash: 'lu' },
  ZONG: { initial: 'z', medial: 'u', rime: 'on', fuzzyHash: 'zon' },
  GONG: { initial: 'g', medial: 'u', rime: 'on', fuzzyHash: 'gon' },
  XIANG: { initial: 'x', medial: 'i', rime: 'an', fuzzyHash: 'xian' },
  MA: { initial: 'm', medial: '', rime: 'a', fuzzyHash: 'ma' },
  JING: { initial: 'j', medial: 'i', rime: 'in', fuzzyHash: 'jin' },
  JUN: { initial: 'j', medial: 'u', rime: 'un', fuzzyHash: 'jun' },
  YING_YI: { initial: 'y', medial: 'i', rime: 'i', fuzzyHash: 'yi' },
  BO: { initial: 'b', medial: '', rime: 'o', fuzzyHash: 'bo' },
  JIE: { initial: 'j', medial: 'i', rime: 'ie', fuzzyHash: 'jie' },
  SHI: { initial: 's', medial: '', rime: 'i', fuzzyHash: 'si' },
  JIAO: { initial: 'j', medial: 'i', rime: 'ao', fuzzyHash: 'jiao' },
  SHOU: { initial: 's', medial: '', rime: 'ou', fuzzyHash: 'sou' },
  DIAN: { initial: 'd', medial: 'i', rime: 'an', fuzzyHash: 'dian' },
  MIAN: { initial: 'm', medial: 'i', rime: 'an', fuzzyHash: 'mian' },
  FAN: { initial: 'f', medial: '', rime: 'an', fuzzyHash: 'fan' },
  XIU: { initial: 'x', medial: 'i', rime: 'ou', fuzzyHash: 'xiou' },
  KAI: { initial: 'k', medial: '', rime: 'ai', fuzzyHash: 'kai' },
  XIN: { initial: 'x', medial: 'i', rime: 'in', fuzzyHash: 'xin' },
  BIAO: { initial: 'b', medial: 'i', rime: 'ao', fuzzyHash: 'biao' },
  BU: { initial: 'b', medial: 'u', rime: 'u', fuzzyHash: 'bu' },
  YAO: { initial: 'y', medial: 'i', rime: 'ao', fuzzyHash: 'yao' },
  LANG: { initial: 'l', medial: '', rime: 'an', fuzzyHash: 'lan' },
  DAI: { initial: 'd', medial: '', rime: 'ai', fuzzyHash: 'dai' },
  JIAN: { initial: 'j', medial: 'i', rime: 'an', fuzzyHash: 'jian' },
  QUN: { initial: 'q', medial: 'u', rime: 'un', fuzzyHash: 'qun' },
  FU: { initial: 'f', medial: 'u', rime: 'u', fuzzyHash: 'fu' },
  DAO: { initial: 'd', medial: '', rime: 'ao', fuzzyHash: 'dao' },
  QIE: { initial: 'q', medial: 'i', rime: 'ie', fuzzyHash: 'qie' },
  XIAO: { initial: 'x', medial: 'i', rime: 'ao', fuzzyHash: 'xiao' },
  XIAO_2: { initial: 'x', medial: 'i', rime: 'ao', fuzzyHash: 'xiao' },
  QI: { initial: 'q', medial: 'i', rime: 'i', fuzzyHash: 'qi' },
  ZHI: { initial: 'z', medial: '', rime: 'i', fuzzyHash: 'zi' },
  XI: { initial: 'x', medial: 'i', rime: 'i', fuzzyHash: 'xi' },
  JIU: { initial: 'j', medial: 'i', rime: 'ou', fuzzyHash: 'jiou' },
  JIA: { initial: 'j', medial: 'i', rime: 'a', fuzzyHash: 'jia' },
  MENG: { initial: 'm', medial: '', rime: 'en', fuzzyHash: 'men' },
  YI: { initial: 'y', medial: 'i', rime: 'i', fuzzyHash: 'yi' },
  PING: { initial: 'p', medial: 'i', rime: 'in', fuzzyHash: 'pin' },
  XUAN: { initial: 'x', medial: 'u', rime: 'an', fuzzyHash: 'xuan' },
  HE: { initial: 'h', medial: '', rime: 'e', fuzzyHash: 'he' },
  GEI_BEI: { initial: 'g', medial: '', rime: 'ei', fuzzyHash: 'gei' },
  ZHUANG: { initial: 'z', medial: 'u', rime: 'an', fuzzyHash: 'zuan' },
  CHONG: { initial: 'c', medial: 'u', rime: 'on', fuzzyHash: 'con' },
  ZU_FOOT: { initial: 'z', medial: 'u', rime: 'u', fuzzyHash: 'zu' },
  ZEN: { initial: 'z', medial: '', rime: 'en', fuzzyHash: 'zen' },
  ME_PARTICLE: { initial: 'm', medial: '', rime: 'e', fuzzyHash: 'me' },
  BAN: { initial: 'b', medial: '', rime: 'an', fuzzyHash: 'ban' },
};

function charToSyllableTuple(ch: string): SyllablePhonemeTuple {
  const code = charToPhoneticCode.get(ch);
  if (code && CODE_TO_TUPLE_MAP[code]) {
    const t = CODE_TO_TUPLE_MAP[code];
    return {
      rawCode: code,
      fuzzyHash: t.fuzzyHash,
      initial: t.initial,
      medial: t.medial,
      rime: t.rime,
    };
  }
  return {
    rawCode: ch,
    fuzzyHash: ch.toLowerCase(),
    initial: ch,
    medial: '',
    rime: ch,
  };
}

export function toPhoneticSignature(text: string): string {
  return Array.from(text)
    .map((ch) => charToPhoneticCode.get(ch) || ch)
    .join('-');
}

/**
 * 取得台式華語模糊音韻序列 (供 UncertainCandidateOption.pinyinSequence 與雙向重疊消融使用)
 */
export function toTaiwanFuzzyPhoneticSyllables(text: string): string[] {
  return Array.from(text).map((ch) => charToSyllableTuple(ch).fuzzyHash);
}

/**
 * 計算兩個單音節之台式腔調與快速連讀替換成本 c_sub(s_i, t_j) \in [0, 1]
 * 支援：
 * 1. 平翹舌零成本、前後鼻音零成本
 * 2. 高元音鬆化 (si <-> su, 如「商業室」與「商業署」) 低成本 0.14
 * 3. 快速語流之舌尖流音與圓唇滑音同化 (如「禮 li <-> 如 lu/ru」、「物 wu <-> 果 guo」、「委 wei <-> 晚 wan」、「員 yuan <-> 於 yu」)
 */
function computeSyllableSubCost(
  a: SyllablePhonemeTuple,
  b: SyllablePhonemeTuple
): number {
  if (a.fuzzyHash === b.fuzzyHash) return 0.0;

  if (
    (a.fuzzyHash === 'su' && b.fuzzyHash === 'si') ||
    (a.fuzzyHash === 'si' && b.fuzzyHash === 'su')
  ) {
    return 0.14;
  }
  if (
    (a.fuzzyHash === 'zu' && b.fuzzyHash === 'li') ||
    (a.fuzzyHash === 'li' && b.fuzzyHash === 'zu') ||
    (a.fuzzyHash === 'lu' && b.fuzzyHash === 'li') ||
    (a.fuzzyHash === 'li' && b.fuzzyHash === 'lu')
  ) {
    return 0.18;
  }
  // 快速語流圓唇滑音與軟顎塞音弱化 (如「物 wu <-> 果 guo」、「委 wei <-> 晚 wan」、「員 yuan <-> 於 yu」、「裝 zuan <-> 充 con」、「修 xiou <-> 足 zu」)
  if (
    (a.fuzzyHash === 'wu' && b.fuzzyHash === 'guo') ||
    (a.fuzzyHash === 'guo' && b.fuzzyHash === 'wu') ||
    (a.fuzzyHash === 'wei' && b.fuzzyHash === 'wan') ||
    (a.fuzzyHash === 'wan' && b.fuzzyHash === 'wei') ||
    (a.fuzzyHash === 'yuan' && b.fuzzyHash === 'yu') ||
    (a.fuzzyHash === 'yu' && b.fuzzyHash === 'yuan') ||
    (a.fuzzyHash === 'zuan' && b.fuzzyHash === 'con') ||
    (a.fuzzyHash === 'con' && b.fuzzyHash === 'zuan') ||
    (a.fuzzyHash === 'xiou' && b.fuzzyHash === 'zu') ||
    (a.fuzzyHash === 'zu' && b.fuzzyHash === 'xiou')
  ) {
    return 0.22;
  }

  let initialCost = 1.0;
  if (a.initial === b.initial) {
    initialCost = 0.0;
  } else if (
    (a.initial === 'b' && b.initial === 'p') ||
    (a.initial === 'p' && b.initial === 'b') ||
    (a.initial === 'd' && b.initial === 't') ||
    (a.initial === 't' && b.initial === 'd') ||
    (a.initial === 'g' && b.initial === 'k') ||
    (a.initial === 'k' && b.initial === 'g') ||
    (['l', 'r', 'n', 'y'].includes(a.initial) &&
      ['l', 'r', 'n', 'y'].includes(b.initial)) ||
    (['w', 'g', 'h'].includes(a.initial) &&
      ['w', 'g', 'h'].includes(b.initial) &&
      a.medial === 'u' &&
      b.medial === 'u') ||
    (['j', 'q', 'x'].includes(a.initial) &&
      ['j', 'q', 'x'].includes(b.initial)) ||
    (['z', 'c', 's'].includes(a.initial) &&
      ['z', 'c', 's'].includes(b.initial)) ||
    (['j', 'q', 'x', 'z', 'c', 's'].includes(a.initial) &&
      ['j', 'q', 'x', 'z', 'c', 's'].includes(b.initial))
  ) {
    initialCost = 0.24;
  }

  let rimeCost = 1.0;
  if (a.rime === b.rime) {
    rimeCost = 0.0;
  } else if (
    (a.rime === 'an' && b.rime === 'en') ||
    (a.rime === 'en' && b.rime === 'an') ||
    (a.rime === 'in' && b.rime === 'un') ||
    (a.rime === 'un' && b.rime === 'in') ||
    (a.rime === 'in' && b.rime === 'i') ||
    (a.rime === 'i' && b.rime === 'in') ||
    (a.rime === 'u' && b.rime === 'o') ||
    (a.rime === 'o' && b.rime === 'u') ||
    (a.rime === 'ei' && b.rime === 'an') ||
    (a.rime === 'an' && b.rime === 'ei') ||
    (a.rime === 'i' && b.rime === 'u') ||
    (a.rime === 'u' && b.rime === 'i') ||
    (a.rime === 'ie' && b.rime === 'i') ||
    (a.rime === 'i' && b.rime === 'ie')
  ) {
    rimeCost = 0.22;
  }

  const medialCost = a.medial === b.medial ? 0.0 : 0.18;
  return Math.min(1.0, 0.45 * initialCost + 0.45 * rimeCost + 0.1 * medialCost);
}

/**
 * 判斷雙音節 (s_{i-1}, s_i) 是否可經由快速連音縮讀 (Syllable Contraction 2 -> 1) 合併為單音節 t_j
 * 典型實例：「不 (bu) + 要 (yao)」連音縮讀為「標 (biao)」，即首字聲母 b == b 且次字韻母 ao == ao！
 */
function canContractTwoSyllablesIntoOne(
  sPrev: SyllablePhonemeTuple,
  sCurr: SyllablePhonemeTuple,
  target: SyllablePhonemeTuple
): boolean {
  if (sPrev.initial !== target.initial) return false;
  const secondHasYOrI =
    sCurr.initial === 'y' || sCurr.medial === 'i' || sCurr.rime === target.rime;
  return secondHasYOrI && sCurr.rime === target.rime;
}

/**
 * 支援連音縮讀與標點忽略之動態規劃加權音韻編輯距離 d_phon(S, T) \in [0, 1]
 */
export function computeTaiwanPhoneticDistance(
  sourceText: string,
  targetText: string
): number {
  const cleanS = sourceText.replace(/[，。？！、；：,.?!\s]/g, '').trim();
  const cleanT = targetText.replace(/[，。？！、；：,.?!\s]/g, '').trim();
  const sChars = Array.from(cleanS);
  const tChars = Array.from(cleanT);
  const n = sChars.length;
  const m = tChars.length;
  if (n === 0 && m === 0) return 0.0;
  if (n === 0 || m === 0) return 1.0;
  if (cleanS === cleanT) return 0.0;

  const sTuples = sChars.map(charToSyllableTuple);
  const tTuples = tChars.map(charToSyllableTuple);

  const indelCost = 0.75;
  const contractCost = 0.22;

  const dp: number[][] = Array.from({ length: n + 1 }, () =>
    new Array(m + 1).fill(Infinity)
  );
  dp[0][0] = 0.0;

  for (let i = 1; i <= n; i++) dp[i][0] = i * indelCost;
  for (let j = 1; j <= m; j++) dp[0][j] = j * indelCost;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const sub =
        dp[i - 1][j - 1] + computeSyllableSubCost(sTuples[i - 1], tTuples[j - 1]);
      const del = dp[i - 1][j] + indelCost;
      const ins = dp[i][j - 1] + indelCost;
      let best = Math.min(sub, del, ins);

      // 2 -> 1 跨音節連音縮讀轉移項 (例如「開心不要」4字 對齊 「開新標」3字)
      if (
        i >= 2 &&
        canContractTwoSyllablesIntoOne(
          sTuples[i - 2],
          sTuples[i - 1],
          tTuples[j - 1]
        )
      ) {
        best = Math.min(best, dp[i - 2][j - 1] + contractCost);
      }

      // 1 -> 2 逆向連音展開轉移項
      if (
        j >= 2 &&
        canContractTwoSyllablesIntoOne(
          tTuples[j - 2],
          tTuples[j - 1],
          sTuples[i - 1]
        )
      ) {
        best = Math.min(best, dp[i - 1][j - 2] + contractCost);
      }

      dp[i][j] = best;
    }
  }

  const normDenom = Math.max(n, m);
  return Number(Math.min(1.0, dp[n][m] / normDenom).toFixed(3));
}

/**
 * 當使用者更名語者（如 Speaker_01 -> 林處長）時，掃描全文找出同音誤植詞（如「林築長」、「臨處長」）
 */
export function scanHomophoneSpeakerMentions(
  targetSpeakerName: string,
  segments: Segment[]
): PhoneticMatchCandidate[] {
  const cleanTarget = targetSpeakerName.trim();
  if (cleanTarget.length < 2) return [];

  const targetSig = toPhoneticSignature(cleanTarget);
  const candidates: PhoneticMatchCandidate[] = [];

  for (const seg of segments) {
    for (let i = 0; i < seg.words.length; i++) {
      const wordText = seg.words[i].text;
      if (wordText.length === cleanTarget.length && wordText !== cleanTarget) {
        if (
          toPhoneticSignature(wordText) === targetSig ||
          computeTaiwanPhoneticDistance(wordText, cleanTarget) <= 0.18
        ) {
          const prev = seg.words[i - 1]?.text || '';
          const next = seg.words[i + 1]?.text || '';
          candidates.push({
            segmentId: seg.id,
            wordIndex: i,
            matchedText: wordText,
            targetSpeakerName: cleanTarget,
            contextPreview: `...${prev}【${wordText}】${next}...`,
            selected: true,
          });
        }
      }
    }
  }
  return candidates;
}

export const computePhoneticSignature = toPhoneticSignature;
export const scanTranscriptForPhoneticMatches = scanHomophoneSpeakerMentions;
export const findPhoneticCandidates = scanHomophoneSpeakerMentions;

export function toAccentInsensitivePhoneticKey(text: string): string {
  const syls = toTaiwanFuzzyPhoneticSyllables(text);
  return syls.join('-');
}

export function applyPhoneticDictionaryCorrection(
  text: string,
  dictionary: Array<{ term: string; aliases?: string[] }>
): string {
  let out = text;
  for (const d of dictionary) {
    if (d.aliases) {
      for (const a of d.aliases) {
        if (a && out.includes(a)) {
          out = out.split(a).join(d.term);
        }
      }
    }
  }
  return out;
}