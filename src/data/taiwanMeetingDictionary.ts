// src/data/taiwanMeetingDictionary.ts
// ============================================================================
// 台灣商業、政府公務、地方創生、採購標案、鄉鎮市區與科技專案內建大詞典 (獨立維護檔)
// ----------------------------------------------------------------------------
// 設計目的：
// 1. 獨立於核心邏輯之外，方便日後直接新增、修改或匯出台灣在地會議常用詞彙。
// 2. 每個詞條僅配置真正的「台灣國語口音/快速連讀同音誤聽別名 (aliases)」，
//    杜絕將不同語意詞彙混入 aliases 造成誤替換。
// 3. 針對特定領域之千詞級完整名單（如 1,394 筆之《HybTransPlus 地方創生字典》），
//    系統支援透過網址參數 `?dict=1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E` 動態載入，
//    本內建詞庫則精選高頻、零碰撞風險之核心部會、計畫、鄉鎮地名、專詞與常駐與會者。
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
    term: '原民會',
    category: 'person',
    aliases: [
      '園民會',
      '援民會',
      '圓民會',
      '願意會',
      '原明會',
      '原名會',
      '人民會',
      '員民會',
      '援會',
      '園會',
    ],
  },
  {
    term: '原住民族委員會',
    category: 'project',
    aliases: ['原住民委員會', '園住民族委員會'],
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
    aliases: ['金濟部', '經紀部', '精濟部'],
  },
  {
    term: '勞動部',
    category: 'person',
    aliases: ['勞工部'],
  },
  {
    term: '農業部',
    category: 'person',
    aliases: ['農委會', '農業處', '濃業部'],
  },
  {
    term: '林業保育署',
    category: 'person',
    aliases: ['林務局', '林保署', '林業署', '林業保育處'],
  },
  {
    term: '文化部',
    category: 'person',
    aliases: ['文化處', '文話部'],
  },
  {
    term: '文資司',
    category: 'person',
    aliases: ['文資師', '文資絲', '文字司', '文資局'],
  },
  {
    term: '環境部',
    category: 'person',
    aliases: ['環保署'],
  },
  {
    term: '交通部',
    category: 'person',
    aliases: ['交痛部', '交統部'],
  },
  {
    term: '衛福部',
    category: 'person',
    aliases: ['衛服部', '未福部', '衛生福利部'],
  },
  {
    term: '教育部',
    category: 'person',
    aliases: ['較育部', '教育處'],
  },
  {
    term: '內政部',
    category: 'person',
    aliases: ['內正部'],
  },
  {
    term: '國產署',
    category: 'person',
    aliases: ['國有財產署', '國產處', '國產屬'],
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
    term: '報告主席',
    category: 'jargon',
    aliases: ['官方主席', '廣方主席', '大方主席', '防方主席', '報到主席'],
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
    aliases: ['功項', '工象', '攻項', '公項', '工向', '功象', '供項', '空向'],
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
    term: '工作會報',
    category: 'project',
    aliases: ['工作回報', '工作匯報', '工作會報的'],
  },
  {
    term: '工作會議',
    category: 'project',
    aliases: ['工作匯議', '工作會意'],
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
    term: '機具補助',
    category: 'project',
    aliases: ['機俱補助', '積聚補助', '機具捕助'],
  },
  {
    term: '產業培訓',
    category: 'project',
    aliases: ['產業陪訓', '產業賠訓', '展業培訓', '產業培訊'],
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
  // 三、 地方創生核心計畫、公有空間活化與高頻專有名詞
  // ==========================================================================
  {
    term: '地方創生',
    category: 'project',
    aliases: ['地方創身', '地方創聲', '地方創深'],
  },
  {
    term: '公有空間活化',
    category: 'project',
    aliases: [
      '風空間活化',
      '風天活化',
      '風間活化',
      '公空間活化',
      '罐空間活化',
      '公有空間文化',
      '公有空間火化',
    ],
  },
  {
    term: '公有空間',
    category: 'project',
    aliases: ['罐空間', '罐中間', '風空間', '光有空間', '共有空間', '公空間'],
  },
  {
    term: '優先推動區',
    category: 'project',
    aliases: [
      '優秀性追蹤區',
      '優先性追蹤區',
      '優秀推動區',
      '優先推動區區預',
      '優先推動區域',
    ],
  },
  {
    term: '苦茶油',
    category: 'project',
    aliases: ['股材油', '股茶油', '埔茶油', '古茶油', '苦茶由', '苦茶遊'],
  },
  {
    term: '幸福巴士',
    category: 'project',
    aliases: ['信福巴士', '幸褔巴士'],
  },
  {
    term: '吊橋工程',
    category: 'project',
    aliases: ['釣橋工程', '調橋工程', '一有工程'],
  },
  {
    term: '林下經濟',
    category: 'project',
    aliases: ['林夏經濟', '臨下經濟', '零下經濟'],
  },
  {
    term: '台灣白及',
    category: 'project',
    aliases: ['台灣白吉', '台灣白急', '臺灣白及', '臺灣白吉'],
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
    term: '尖石',
    category: 'jargon',
    aliases: ['監持', '監時', '堅石', '兼時', '間時', '間持'],
  },
  {
    term: '五峰',
    category: 'jargon',
    aliases: ['武蜂', '武功', '武風', '五風', '午峰', '五蜂'],
  },
  {
    term: '石磊部落',
    category: 'jargon',
    aliases: ['十壘部落', '石壘部絡', '石雷部落'],
  },
  {
    term: '南庄',
    category: 'jargon',
    aliases: ['南莊', '難庄', '藍庄'],
  },
  {
    term: '獅潭',
    category: 'jargon',
    aliases: ['師潭', '詩潭', '施潭', '獅壇'],
  },
  {
    term: '卓蘭',
    category: 'jargon',
    aliases: ['竹蘭', '著蘭', '卓藍'],
  },
  {
    term: '秀水',
    category: 'jargon',
    aliases: ['袖水', '鏽水'],
  },
  {
    term: '竹塘',
    category: 'jargon',
    aliases: ['竹堂', '足塘'],
  },
  {
    term: '二林',
    category: 'jargon',
    aliases: ['二霖', '餓林'],
  },
  {
    term: '溪湖',
    category: 'jargon',
    aliases: ['西湖', '希湖'],
  },
  {
    term: '中寮',
    category: 'jargon',
    aliases: ['忠寮', '終寮'],
  },
  {
    term: '台西',
    category: 'jargon',
    aliases: ['臺西', '抬西'],
  },
  {
    term: '崙背',
    category: 'jargon',
    aliases: ['輪背', '倫背', '論背'],
  },
  {
    term: '東山',
    category: 'jargon',
    aliases: ['冬山', '東衫'],
  },
  {
    term: '西港',
    category: 'jargon',
    aliases: ['希港', '溪港'],
  },
  {
    term: '麻豆',
    category: 'jargon',
    aliases: ['麻逗', '馬豆'],
  },
  {
    term: '佳里',
    category: 'jargon',
    aliases: ['家裡', '嘉里', '佳裡'],
  },
  {
    term: '南化',
    category: 'jargon',
    aliases: ['難化', '南話'],
  },
  {
    term: '內門',
    category: 'jargon',
    aliases: ['那門', '內們'],
  },
  {
    term: '關廟',
    category: 'jargon',
    aliases: ['觀廟', '官廟'],
  },
  {
    term: '燕巢',
    category: 'jargon',
    aliases: ['驗巢', '宴巢'],
  },
  {
    term: '田寮',
    category: 'jargon',
    aliases: ['甜寮', '填寮'],
  },
  {
    term: '旗津',
    category: 'jargon',
    aliases: ['齊津', '奇津'],
  },
  {
    term: '茂林',
    category: 'jargon',
    aliases: ['冒林', '貿林'],
  },
  {
    term: '萬丹',
    category: 'jargon',
    aliases: ['萬單', '慢丹'],
  },
  {
    term: '九如',
    category: 'jargon',
    aliases: ['久如', '酒如'],
  },
  {
    term: '鹽埔',
    category: 'jargon',
    aliases: ['炎埔', '言埔', '鹽捕'],
  },
  {
    term: '枋山',
    category: 'jargon',
    aliases: ['方山', '芳山'],
  },
  {
    term: '新埤',
    category: 'jargon',
    aliases: ['心埤', '新皮'],
  },
  {
    term: '崁頂',
    category: 'jargon',
    aliases: ['砍頂', '坎頂'],
  },
  {
    term: '南州',
    category: 'jargon',
    aliases: ['難州', '南洲'],
  },
  {
    term: '南澳',
    category: 'jargon',
    aliases: ['難澳', '南奧'],
  },
  {
    term: '鳳林',
    category: 'jargon',
    aliases: ['奉林', '縫林'],
  },
  {
    term: '玉里',
    category: 'jargon',
    aliases: ['玉裡', '育里'],
  },
  {
    term: '瑞穗',
    category: 'jargon',
    aliases: ['瑞歲', '銳穗'],
  },
  {
    term: '蘭嶼',
    category: 'jargon',
    aliases: ['藍嶼', '攔嶼'],
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
  // 四、 國發會與地方創生專案辦公室核心常駐與會人名與稱謂
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
