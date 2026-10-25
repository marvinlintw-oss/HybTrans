// src/data/taiwanMeetingDictionary.ts
// ============================================================================
// 台灣商業、政府公務、地方創生、採購標案、學術研發與科技專案內建大詞典 (獨立維護檔)
// ----------------------------------------------------------------------------
// 設計目的：
// 1. 獨立於核心邏輯之外，方便日後直接新增、修改或匯出台灣在地會議常用詞彙。
// 2. 每個詞條僅配置真正的「台灣國語口音/快速連讀同音誤聽別名 (aliases)」，
//    杜絕將不同語意詞彙混入 aliases 造成誤替換。
// 3. 針對特定領域之千詞級完整名單（如 1,394 筆之《HybTransPlus 地方創生字典》），
//    系統支援透過網址參數 `?dict=1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E` 動態載入，
//    本內建詞庫則精選高頻、零碰撞風險之核心機構、計畫、專詞與常駐與會者。
// ============================================================================

import { DictCategory, DictEntry } from '../types';

export interface BuiltinDictSeed {
  term: string;
  category: DictCategory;
  aliases?: string[];
}

export const TAIWAN_MEETING_DICTIONARY_SEEDS: BuiltinDictSeed[] = [
  // ==========================================================================
  // 一、 政府機關、部會簡稱與公務組織稱謂
  // ==========================================================================
  {
    term: '國家發展委員會',
    category: 'project',
    aliases: ['國家發展委員匯', '國家發展委原會'],
  },
  {
    term: '國發會',
    category: 'person',
    aliases: ['國法會', '過發會', '果發會', '國發匯', '廣發會'],
  },
  {
    term: '國土區域離島發展處',
    category: 'project',
    aliases: ['國土區域離島發展室', '國土區域離島發展署'],
  },
  {
    term: '國土處',
    category: 'person',
    aliases: ['國土室', '國土署', '國土觸'],
  },
  {
    term: '中小企業署',
    category: 'person',
    aliases: [
      '中央企業師',
      '中小企業處',
      '中小企業屬',
      '中央切屬',
      '中小企署',
      '中小企業絲',
      '中小企業司',
      '中小企業師',
    ],
  },
  {
    term: '商業署',
    category: 'person',
    aliases: [
      '商業室',
      '商業司',
      '商業屬',
      '商業處',
      '商業絲',
      '商業師',
      '商業業絲',
      '商業業師',
    ],
  },
  {
    term: '產業發展署',
    category: 'person',
    aliases: ['產發署', '產發室', '產業發展室', '產發屬'],
  },
  {
    term: '青年發展署',
    category: 'person',
    aliases: ['青發署', '青年發展室', '青年發展處', '清發署'],
  },
  {
    term: '數位發展部',
    category: 'person',
    aliases: ['數位發展處', '數位部'],
  },
  {
    term: '數發部',
    category: 'person',
    aliases: ['速發部', '樹發部', '書發部'],
  },
  {
    term: '國科會',
    category: 'person',
    aliases: ['國科匯', '過科會'],
  },
  {
    term: '經濟部',
    category: 'person',
    aliases: ['金濟部', '經紀部'],
  },
  {
    term: '勞動部',
    category: 'person',
    aliases: ['勞工部'],
  },
  {
    term: '農業部',
    category: 'person',
    aliases: ['農委會', '農業處'],
  },
  {
    term: '文化部',
    category: 'person',
    aliases: ['文化處'],
  },
  {
    term: '環境部',
    category: 'person',
    aliases: ['環保署'],
  },
  {
    term: '金管會',
    category: 'person',
    aliases: ['經管會', '金管匯'],
  },
  {
    term: '主計總處',
    category: 'person',
    aliases: ['主計處', '主記處', '主計室'],
  },
  {
    term: '審計部',
    category: 'person',
    aliases: ['省計部', '審計室', '審計處'],
  },
  {
    term: '立法院',
    category: 'person',
    aliases: ['利法院', '立法院會'],
  },
  {
    term: '立法委員',
    category: 'person',
    aliases: ['利法委員', '立院委員'],
  },
  {
    term: '委員',
    category: 'person',
    aliases: ['委原', '偉員', '尾員', '緯員'],
  },
  {
    term: '行政院',
    category: 'person',
    aliases: ['行正院'],
  },
  {
    term: '處內',
    category: 'jargon',
    aliases: ['出內', '處類', '初內'],
  },
  {
    term: '署內',
    category: 'jargon',
    aliases: ['屬內', '室內', '數內'],
  },
  {
    term: '局處',
    category: 'jargon',
    aliases: ['局室', '菊處'],
  },
  {
    term: '承辦人',
    category: 'person',
    aliases: ['成辦人', '陳辦人'],
  },
  {
    term: '長官',
    category: 'person',
    aliases: ['掌官', '場官', '常官'],
  },

  // ==========================================================================
  // 二、 政府採購、標案管理與行政流程用語
  // ==========================================================================
  {
    term: '開新標',
    category: 'project',
    aliases: ['開心不要', '開新表', '開心標', '開星標', '一個新表', '改開新標'],
  },
  {
    term: '新標',
    category: 'project',
    aliases: ['星標', '心標', '欣標', '新表'],
  },
  {
    term: '工項',
    category: 'project',
    aliases: ['功項', '工象', '攻項', '公項', '工向', '功象', '供項'],
  },
  {
    term: '後續擴充',
    category: 'project',
    aliases: ['擴充套件', '後續過充', '再去過充'],
  },
  {
    term: '擴充',
    category: 'project',
    aliases: ['過充', '闊充', '擴衝'],
  },
  {
    term: '議價',
    category: 'project',
    aliases: ['一架', '意價', '義價', '議假'],
  },
  {
    term: '評選',
    category: 'project',
    aliases: ['平選', '瓶選', '憑選'],
  },
  {
    term: '審查會',
    category: 'project',
    aliases: ['省查會', '審核會'],
  },
  {
    term: '審查委員',
    category: 'person',
    aliases: ['省查委員', '審查委原'],
  },
  {
    term: '徵件',
    category: 'project',
    aliases: ['成見', '針見', '真見', '政見', '爭件', '徵見', '真劍'],
  },
  {
    term: '時程',
    category: 'project',
    aliases: ['十層', '10層', '施程', '石城', '時辰'],
  },
  {
    term: '一般組',
    category: 'project',
    aliases: ['一般族', '一班組', '一般主'],
  },
  {
    term: '專案組',
    category: 'project',
    aliases: ['專案族', '專安組', '專案主'],
  },
  {
    term: '獎勵金',
    category: 'project',
    aliases: ['講歷經', '獎勵劍', '講利金', '獎力金', '獎勵計'],
  },
  {
    term: '補助金',
    category: 'project',
    aliases: ['捕助金', '補住金'],
  },
  {
    term: '配合款',
    category: 'project',
    aliases: ['配何款', '配合館'],
  },
  {
    term: '驗收',
    category: 'project',
    aliases: ['燕收', '厭收', '驗收款'],
  },
  {
    term: '核銷',
    category: 'project',
    aliases: ['合銷', '核消', '河銷', '和銷'],
  },
  {
    term: '流標',
    category: 'project',
    aliases: ['留標', '劉標'],
  },
  {
    term: '決標',
    category: 'project',
    aliases: ['絕標', '決表', '覺標'],
  },
  {
    term: '招標文件',
    category: 'project',
    aliases: ['招標檔案', '招標文建'],
  },
  {
    term: '服務對象',
    category: 'project',
    aliases: ['服務物件', '服務對像'],
  },
  {
    term: '契約變更',
    category: 'project',
    aliases: ['氣變', '契變'],
  },
  {
    term: '管考',
    category: 'project',
    aliases: ['館考', '管烤'],
  },
  {
    term: 'KPI',
    category: 'project',
    aliases: ['K P I', 'kpi'],
  },

  // ==========================================================================
  // 三、 地方創生核心計畫、媒體合作與高頻專有名詞（精選自 HybTransPlus 地方創生字典）
  // ==========================================================================
  {
    term: '地方創生',
    category: 'project',
    aliases: ['地方創身', '地方創聲', '地方創深'],
  },
  {
    term: '地方創生專案辦公室',
    category: 'project',
    aliases: ['地方創生專辦', '地創專案辦公室'],
  },
  {
    term: '地創專辦',
    category: 'project',
    aliases: ['地創專班', '地創專辦室'],
  },
  {
    term: '台灣經濟研究院',
    category: 'project',
    aliases: ['臺灣經濟研究院', '台灣經濟研究苑'],
  },
  {
    term: '台經院',
    category: 'project',
    aliases: ['臺經院', '台金院', '台經苑'],
  },
  {
    term: '分區輔導中心',
    category: 'project',
    aliases: ['分區輔導中興', '分輔中心'],
  },
  {
    term: '青年培力工作站',
    category: 'project',
    aliases: ['青年賠力工作站', '青年陪力工作站'],
  },
  {
    term: '青培站',
    category: 'project',
    aliases: ['清陪站', '青賠站', '青陪站', '清培站', '輕培站'],
  },
  {
    term: '青年培力',
    category: 'project',
    aliases: ['青年賠力', '青年陪力'],
  },
  {
    term: '中興新村育成村',
    category: 'project',
    aliases: ['中興新村預成村', '中興育成村'],
  },
  {
    term: '微笑台灣',
    category: 'project',
    aliases: ['微小臺灣', '微小台灣', '微笑臺灣', '微校台灣', '微校臺灣'],
  },
  {
    term: '天下雜誌',
    category: 'project',
    aliases: ['天夏雜誌', '天下雜志'],
  },
  {
    term: '神秘禮物',
    category: 'jargon',
    aliases: ['神秘如果', '神秘，如果', '神祕禮物', '神祕如果', '神秘里物'],
  },
  {
    term: '馬祖',
    category: 'jargon',
    aliases: ['麻祖', '媽祖', '馬組', '麻組'],
  },
  {
    term: '業師',
    category: 'person',
    aliases: ['夜師', '夜食', '業室', '業勢', '夜視', '葉師'],
  },
  {
    term: '業師群',
    category: 'person',
    aliases: ['夜市群', '夜師群', '業室群', '夜食群'],
  },
  {
    term: '廊帶',
    category: 'project',
    aliases: [
      '狼帶',
      '郎蛋',
      '藍帶',
      '蘭代',
      '狼袋',
      '廊代',
      '郎單',
      '囊蛋',
      '強帶',
      '郎帶',
    ],
  },
  {
    term: '店面翻修',
    category: 'project',
    aliases: ['店店面販修修', '店面凡修', '店面煩修', '店面繁修'],
  },
  {
    term: '開新店',
    category: 'project',
    aliases: ['開心店', '開心電', '開辛店', '開新電'],
  },
  {
    term: '加盟',
    category: 'jargon',
    aliases: ['加盲', '加忙', '加茫'],
  },
  {
    term: '商業模式',
    category: 'jargon',
    aliases: ['商業魔式', '商模'],
  },
  {
    term: '淨零碳排',
    category: 'project',
    aliases: ['淨零探排', '近零碳排'],
  },
  {
    term: '碳盤查',
    category: 'project',
    aliases: ['探盤查', '炭盤查'],
  },

  // ==========================================================================
  // 四、 國發會與地方創生專案辦公室核心常駐與會人名與稱謂（零碰撞高識別度）
  // ==========================================================================
  {
    term: '葉俊顯',
    category: 'person',
    aliases: ['葉俊賢', '葉俊憲', '業俊顯'],
  },
  {
    term: '劉鏡清',
    category: 'person',
    aliases: ['劉敬清', '劉靜清'],
  },
  {
    term: '彭立沛',
    category: 'person',
    aliases: ['彭立佩', '彭麗沛', '彭立配'],
  },
  {
    term: '陳美伶',
    category: 'person',
    aliases: ['陳美玲', '陳美琳'],
  },
  {
    term: '黃文彥',
    category: 'person',
    aliases: ['黃文燕', '黃文諺', '黃文彦'],
  },
  {
    term: '林鐘榮',
    category: 'person',
    aliases: ['林忠榮', '林中榮', '林宗榮'],
  },
  {
    term: '康廷嶽',
    category: 'person',
    aliases: ['康廷岳', '康庭嶽', '康庭岳'],
  },
  {
    term: '黃柏偉',
    category: 'person',
    aliases: ['黃博偉', '黃百偉'],
  },
  {
    term: '許芯榆',
    category: 'person',
    aliases: ['許心榆', '許欣榆', '許芯瑜', '許心瑜'],
  },
  {
    term: '芯榆',
    category: 'person',
    aliases: ['星雲', '星妤', '心榆', '欣榆', '芯瑜', '心瑜', '欣瑜', 'Ciri'],
  },
  {
    term: '葉俊沂',
    category: 'person',
    aliases: ['葉俊儀', '葉俊宜', '業俊沂', '葉俊穎'],
  },
  {
    term: '俊沂',
    category: 'person',
    aliases: ['俊穎', '俊儀', '俊宜'],
  },
  {
    term: '白佩玉',
    category: 'person',
    aliases: ['白珮玉', '白佩鈺'],
  },
  {
    term: '白姐',
    category: 'person',
    aliases: ['白姊', '百姐', '白解'],
  },
  {
    term: '曉玲',
    category: 'person',
    aliases: ['少林', '小玲', '曉林', '小淋'],
  },

  // ==========================================================================
  // 五、 學術研究、企業治理與日常會議高頻詞
  // ==========================================================================
  {
    term: '計畫主持人',
    category: 'person',
    aliases: ['計劃主持人', '主池人'],
  },
  {
    term: '協同主持人',
    category: 'person',
    aliases: ['協同主池人', '偕同主持人'],
  },
  {
    term: '產學合作',
    category: 'project',
    aliases: ['慘學合作', '產學合做'],
  },
  {
    term: '深耕計畫',
    category: 'project',
    aliases: ['生根計畫', '深耕計劃'],
  },
  {
    term: '質性研究',
    category: 'jargon',
    aliases: ['直線研究', '執性研究'],
  },
  {
    term: '會議紀錄',
    category: 'project',
    aliases: ['會議記錄', '匯議紀錄'],
  },
  {
    term: '儘快',
    category: 'jargon',
    aliases: ['精快', '經快'],
  },
];

/**
 * 將內建種子詞條轉換為系統標準 DictEntry 陣列
 */
export function getBuiltinTaiwanMeetingDictionary(): DictEntry[] {
  return TAIWAN_MEETING_DICTIONARY_SEEDS.map((seed, index) => ({
    id: `tw_builtin_${String(index + 1).padStart(3, '0')}`,
    term: seed.term.trim(),
    category: seed.category,
    scope: 'public',
    aliases: seed.aliases ? seed.aliases.map((a) => a.trim()).filter(Boolean) : [],
  }));
}

export const DEFAULT_PUBLIC_DICTIONARY: DictEntry[] =
  getBuiltinTaiwanMeetingDictionary();
export const TAIWAN_MEETING_DEFAULT_DICTIONARY: DictEntry[] =
  DEFAULT_PUBLIC_DICTIONARY;