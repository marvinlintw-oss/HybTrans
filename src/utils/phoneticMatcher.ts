// src/utils/phoneticMatcher.ts
import { Segment, PhoneticMatchCandidate } from '../types';

/**
 * 台灣會議常見姓氏、領域詞根、職稱與高頻語素之精準同音群組映射表
 * 設計原則：
 * 1. 每個群組嚴格限制為「同聲母且同韻母（含台灣國語平捲舌 z/zh, c/ch, s/sh 與前後鼻音 in/ing, en/eng, an/ang 合流）」之真正同音/近音字。
 * 2. 絕不將不同聲母（如 d/t, g/k）或一般常見動詞/虛詞混入同一代碼，徹底杜絕與千詞級外部字典發生假陽性碰撞！
 */
const PHONETIC_GROUPS: Record<string, string> = {
  // 常見姓氏與高頻會議字根同音群組
  '林臨淋琳霖零玲鈴齡伶翎鄰鱗磷': 'LIN',
  '陳塵臣辰晨沉成城誠呈承程懲乘丞': 'CHEN',
  '張章彰璋樟蟑': 'ZHANG',
  '李里理禮裡鯉履哩離梨黎麗利例立力歷曆隸粒勵': 'LI',
  '王亡汪網往枉晚碗挽宛婉琬椀灣彎頑丸完玩': 'WAN',
  '吳吾無蕪梧巫務物誤悟霧勿屋五武午舞伍': 'WU',
  '劉留流琉榴瘤柳六陸': 'LIU',
  '蔡菜採彩踩財材裁才猜': 'CAI',
  '楊陽揚洋羊佯樣養仰央泱秧': 'YANG',
  '許栩詡徐須需虛噓詢尋巡敘序緒續蓄恤': 'XU',
  '鄭正政證症爭徵真針珍診陣鎮振震貞偵': 'ZHENG',
  '謝械卸屑蟹寫血協脅鞋斜邪攜些歇': 'XIE',
  '郭鍋國果過裹': 'GUO',
  '洪紅宏虹鴻哄烘轟弘': 'HONG',
  '曾增爭箏蒸真偵憎贈綜': 'ZENG',
  '廖料瞭療聊撩遼僚': 'LIAO',
  '賴來萊瀨藍蘭攔欄': 'LAI',
  '蘇疏輸書舒樞叔屬署鼠數術述束俗宿速塑訴': 'SU',
  '周洲舟州週驟軸宙晝咒奏': 'ZHOU',
  '葉頁業夜液腋咽謁野也冶耶': 'YE',
  // 常見職稱、政府標案與會議高頻同音群組
  '處觸畜築儲礎楚出初除廚': 'CHU',
  '長場廠掌漲常償嚐腸昌暢唱倡': 'ZHANG_TITLE',
  '委偉尾瑋偽緯位未味衛為圍違唯維微危薇威慰胃謂': 'WEI',
  '員元園原源圓援緣沿延研言嚴岩顏鹽院願怨苑': 'YUAN',
  '於余妤魚漁娛逾愚榆瑜渝雲云勻芸育域預遇裕愈喻玉雨語羽與宇嶼': 'YU',
  '科顆棵磕苛課客克刻可渴殼': 'KE',
  '主組祖阻詛煮囑助住注駐祝著柱珠株諸豬竹築逐燭': 'ZHU',
  '任認韌刃妊人仁忍': 'REN',
  '秘密蜜覓冪米迷謎彌靡': 'MI',
  '神什沈審嬸慎滲深身申伸升生聲勝聖盛省剩甚': 'SHEN',
  '如儒茹汝乳入辱盧蘆爐陸錄鹿路露魯': 'RU',
  '總種腫塚中忠終鐘衷眾重宗蹤棕縱': 'ZONG',
  '工功攻公宮弓恭供躬共貢拱鞏': 'GONG',
  '項向象像相巷享響想箱廂鄉香詳祥翔': 'XIANG',
  '馬碼瑪螞麻嘛罵媽抹': 'MA',
  '經精晶京驚鯨金今津斤禁近進盡勁景警井境鏡敬靜淨競徑竟': 'JING',
  '俊駿峻浚郡菌軍均君鈞': 'JUN',
  '穎影營迎贏盈瑩螢沂儀宜疑移姨遺頤銀吟飲引印應鷹英': 'YING_YI',
  '博薄搏駁伯帛泊白百擺柏': 'BO',
  '姐姊解節結潔借界戒介屆街皆接階揭劫傑竭截': 'JIE',
  '士事式市試視室是師詩失施獅濕十石時實食識史使始示世勢適釋持池遲馳': 'SHI',
  '教較校酵覺交郊嬌澆焦礁角腳剿叫轎笑效肖孝': 'JIAO',
  '授受售瘦獸收手守首壽熟': 'SHOU',
  '店電墊殿奠佃甸點典顛巔': 'DIAN',
  '面麵棉眠免勉緬綿': 'MIAN',
  '翻番凡煩繁樊販範犯飯泛反返': 'FAN',
  '修羞休朽秀袖繡鏽': 'XIU',
  '開揩慨楷凱': 'KAI',
  '新心辛欣薪馨星興腥芯信釁': 'XIN',
  '標彪飆錶表裱': 'BIAO',
  '不布步部怖捕補普譜堡埠簿': 'BU',
  '要耀曜藥鑰搖遙謠姚咬邀腰': 'YAO',
  '廊狼郎良涼糧梁量藍蘭籃欄朗浪': 'LANG',
  '帶代袋貸待怠殆大太態泰台臺抬苔': 'DAI',
  '件見建健鍵漸踐劍薦鑑間肩堅監兼簡檢減撿尖奸殲艱': 'JIAN',
  '群裙尋巡詢循旬': 'QUN',
  '輔腐府俯釜付負附赴副富復腹覆服福幅浮符扶夫膚敷賦': 'FU',
  '導島倒禱蹈到道稻盜刀': 'DAO',
  '央泱秧英應鷹迎營贏盈映硬': 'YANG_YING',
  '切且竊妾企契氣棄汽器': 'QIE',
  '小曉筱少稍梢紹哨笑校效': 'XIAO',
  '企起豈啟氣器棄契期欺漆戚妻棲七': 'QI',
  '支之知肢織汁資姿滋諮只指止址紙至志制致置質治炙職直值植執殖': 'ZHI',
  '晰析息悉惜昔夕希稀犧吸西膝溪熙錫襲習席媳洗喜係系細戲': 'XI',
  '就舊救舅咎久九酒究糾': 'JIU',
  '加家佳嘉夾挾袈甲假價架駕嫁稼': 'JIA',
  '盟萌蒙濛盲忙茫氓猛孟夢': 'MENG',
  '議意義藝亦異役疫譯翼億憶益誼一衣醫依壹椅已以倚': 'YI',
  '評平瓶憑萍屏貧頻品聘拼': 'PING',
  '選癬險顯憲縣現限線獻羨鮮仙先掀': 'XUAN',
  '核合何河荷盒和禾吉及級即極急集籍': 'HE',
  '銷消宵霄削囂小曉蕭瀟': 'XIAO_2',
  '給北被備背倍輩貝悲杯碑': 'GEI_BEI',
  '裝樁莊妝潢狀撞壯': 'ZHUANG',
  '充衝沖聰匆蔥從叢崇蟲': 'CHONG',
  '足族卒竹築逐燭': 'ZU_FOOT',
  '怎診枕整貞針': 'ZEN',
  '麼嗎嘛麻馬媽': 'ME_PARTICLE',
  '辦半伴扮絆版板盼判叛盤磐班般搬斑頒': 'BAN',
  '峰蜂風封豐瘋鋒逢縫馮鳳奉諷': 'FENG',
  '苦股古谷鼓骨庫褲酷哭枯': 'KU',
  '油由游遊尤郵猶友有右又幼誘憂優幽悠': 'YOU',
  '民明名銘鳴敏閩眠綿': 'MIN',
  '會匯繪惠慧賄悔毀回迴灰揮輝恢徽': 'HUI',
  '案按暗岸胺安鞍氨': 'AN',
  '培陪賠裴肥配佩沛': 'PEI',
  '訓訊迅遜尋巡詢': 'XUN',
  '吊釣調掉弔雕': 'DIAO',
  '橋瞧喬僑翹竅巧悄': 'QIAO',
  '巴吧疤芭八拔跋把靶爸罷霸': 'BA',
  '連聯憐簾蓮廉臉練煉戀鏈': 'LIAN',
  '專磚轉傳撰賺': 'ZHUAN',
  '單擔丹耽但淡彈誕旦膽': 'DAN',
  '獨讀毒牘度渡鍍肚堵賭睹': 'DU',
  '月越粵悅閱躍嶽岳約': 'YUE',
  '號好耗浩皓毫豪壕': 'HAO',
  '早澡藻棗找爪造皂燥躁': 'ZAO',
  '上尚賞裳商傷': 'SHANG',
  '天添田甜填': 'TIAN',
  '三山衫刪珊散傘閃善扇': 'SAN',
  '兩良涼糧梁量亮諒輛': 'LIANG',
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
  FENG: { initial: 'f', medial: 'u', rime: 'on', fuzzyHash: 'fon' },
  KU: { initial: 'k', medial: 'u', rime: 'u', fuzzyHash: 'ku' },
  YOU: { initial: 'y', medial: 'i', rime: 'ou', fuzzyHash: 'you' },
  MIN: { initial: 'm', medial: 'i', rime: 'in', fuzzyHash: 'min' },
  HUI: { initial: 'h', medial: 'u', rime: 'ei', fuzzyHash: 'hui' },
  AN: { initial: 'y', medial: '', rime: 'an', fuzzyHash: 'an' },
  PEI: { initial: 'p', medial: '', rime: 'ei', fuzzyHash: 'pei' },
  XUN: { initial: 'x', medial: 'u', rime: 'un', fuzzyHash: 'xun' },
  DIAO: { initial: 'd', medial: 'i', rime: 'ao', fuzzyHash: 'diao' },
  QIAO: { initial: 'q', medial: 'i', rime: 'ao', fuzzyHash: 'qiao' },
  BA: { initial: 'b', medial: '', rime: 'a', fuzzyHash: 'ba' },
  LIAN: { initial: 'l', medial: 'i', rime: 'an', fuzzyHash: 'lian' },
  ZHUAN: { initial: 'z', medial: 'u', rime: 'an', fuzzyHash: 'zuan' },
  DAN: { initial: 'd', medial: '', rime: 'an', fuzzyHash: 'dan' },
  DU: { initial: 'd', medial: 'u', rime: 'u', fuzzyHash: 'du' },
  YUE: { initial: 'y', medial: 'u', rime: 'ie', fuzzyHash: 'yue' },
  HAO: { initial: 'h', medial: '', rime: 'ao', fuzzyHash: 'hao' },
  ZAO: { initial: 'z', medial: '', rime: 'ao', fuzzyHash: 'zao' },
  SHANG: { initial: 's', medial: '', rime: 'an', fuzzyHash: 'san' },
  TIAN: { initial: 't', medial: 'i', rime: 'an', fuzzyHash: 'tian' },
  SAN: { initial: 's', medial: '', rime: 'an', fuzzyHash: 'san' },
  LIANG: { initial: 'l', medial: 'i', rime: 'an', fuzzyHash: 'lian' },
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
    (a.fuzzyHash === 'li' && b.fuzzyHash === 'lu') ||
    (a.fuzzyHash === 'fon' && b.fuzzyHash === 'gon') ||
    (a.fuzzyHash === 'gon' && b.fuzzyHash === 'fon')
  ) {
    return 0.18;
  }
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
