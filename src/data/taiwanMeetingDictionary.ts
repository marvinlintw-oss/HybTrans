// src/data/taiwanMeetingDictionary.ts
// ============================================================================
// 台灣商業、政府公務、採購標案、學術研發與科技專案內建大詞典 (獨立維護檔)
// ----------------------------------------------------------------------------
// 設計目的：
// 1. 獨立於核心邏輯之外，方便日後直接新增、修改或匯出台灣在地會議常用詞彙。
// 2. 每個詞條可配置常見的「台灣國語口音/同音誤聽別名 (aliases)」，
//    供 Whisper 提示詞 (Prompt Hint) 與語境校正引擎高速對齊。
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
    term: '國發會',
    category: 'person',
    aliases: ['國法會', '過發會', '果發會', '國發匯'],
  },
  {
    term: '中小企業署',
    category: 'person',
    aliases: ['中央企業師', '中小企業處', '中小企業屬', '中央切屬', '中小企署'],
  },
  {
    term: '商業署',
    category: 'person',
    aliases: ['商業室', '商業司', '商業屬', '商業處'],
  },
  {
    term: '產業發展署',
    category: 'person',
    aliases: ['產發署', '產發室', '產業發展室', '產發屬'],
  },
  {
    term: '數位發展部',
    category: 'person',
    aliases: ['數發部', '速發部', '數位部'],
  },
  {
    term: '數發部',
    category: 'person',
    aliases: ['速發部', '樹發部', '書發部'],
  },
  {
    term: '國科會',
    category: 'person',
    aliases: ['國科匯', '過科會', '科技部'],
  },
  {
    term: '經濟部',
    category: 'person',
    aliases: ['金濟部', '經紀部'],
  },
  {
    term: '勞動部',
    category: 'person',
    aliases: ['勞工部', '勞動力發展署'],
  },
  {
    term: '農業部',
    category: 'person',
    aliases: ['農委會', '農業處'],
  },
  {
    term: '文化部',
    category: 'person',
    aliases: ['文化局', '文化處'],
  },
  {
    term: '環境部',
    category: 'person',
    aliases: ['環保署', '環評會'],
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
    aliases: ['立法委員', '國會聯絡'],
  },
  {
    term: '行政院',
    category: 'person',
    aliases: ['政院', '院本部'],
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
    aliases: ['局室', '菊處', '各局處'],
  },
  {
    term: '承辦人',
    category: 'person',
    aliases: ['成辦人', '承辦同仁', '承辦科'],
  },
  {
    term: '長官',
    category: 'person',
    aliases: ['掌官', '場官'],
  },

  // ==========================================================================
  // 二、 政府採購、標案管理與行政流程用語
  // ==========================================================================
  {
    term: '開新標',
    category: 'project',
    aliases: ['開心不要', '開新表', '開心標', '開星標', '一個新表'],
  },
  {
    term: '新標',
    category: 'project',
    aliases: ['星標', '心標', '欣標', '新表'],
  },
  {
    term: '後續擴充',
    category: 'project',
    aliases: ['擴充套件', '後擴', '後續過充', '再去過充'],
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
    aliases: ['平選', '瓶選', '評選會', '評選委員'],
  },
  {
    term: '審查會',
    category: 'project',
    aliases: ['省查會', '審核會', '期中審查', '期末審查'],
  },
  {
    term: '徵件',
    category: 'project',
    aliases: ['成見', '針見', '真見', '政見', '爭件', '徵見'],
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
    aliases: ['專案族', '專安組'],
  },
  {
    term: '獎勵金',
    category: 'project',
    aliases: ['講歷經', '獎勵劍', '講利金', '獎力金'],
  },
  {
    term: '補助金',
    category: 'project',
    aliases: ['捕助金', '補助款', '補助經費'],
  },
  {
    term: '配合款',
    category: 'project',
    aliases: ['自籌款', '地方配合款'],
  },
  {
    term: '驗收',
    category: 'project',
    aliases: ['燕收', '驗收款', '結案驗收'],
  },
  {
    term: '核銷',
    category: 'project',
    aliases: ['合銷', '核消', '經費核銷', '單據核銷'],
  },
  {
    term: '流標',
    category: 'project',
    aliases: ['留標', '劉標', '廢標'],
  },
  {
    term: '決標',
    category: 'project',
    aliases: ['絕標', '決表', '得標廠商'],
  },
  {
    term: '招標文件',
    category: 'project',
    aliases: ['招標書', '投標須知', '需求說明書', '工作說明書'],
  },
  {
    term: '契約變更',
    category: 'project',
    aliases: ['契變', '氣變', '變更設計'],
  },
  {
    term: '管考',
    category: 'project',
    aliases: ['館考', '管制考核', '進度管考'],
  },
  {
    term: 'KPI',
    category: 'project',
    aliases: ['績效指標', '查核點', '量化指標', '質化指標'],
  },

  // ==========================================================================
  // 三、 地方創生、商圈輔導、商業經營與創業育成
  // ==========================================================================
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
    aliases: ['狼帶', '郎蛋', '藍帶', '蘭代', '狼袋', '廊代'],
  },
  {
    term: '地方創生',
    category: 'project',
    aliases: ['地方創身', '地方創聲', '地方創深', '地創'],
  },
  {
    term: '青年培力',
    category: 'project',
    aliases: ['青年賠力', '青年陪力', '青培站', '培力工作站'],
  },
  {
    term: '育成中心',
    category: 'project',
    aliases: ['育成計畫', '創新育成', '孵化器', '加速器'],
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
    aliases: ['加盲', '加忙', '加茫', '直營店'],
  },
  {
    term: '商圈輔導',
    category: 'project',
    aliases: ['店家輔導', '商業輔導', '實地訪視', '蹲點輔導'],
  },
  {
    term: '青創貸款',
    category: 'project',
    aliases: ['青年創業', '創業輔導', '新創團隊'],
  },
  {
    term: '商業模式',
    category: 'jargon',
    aliases: ['商模', '傷模', '營收模式', '變現模式'],
  },
  {
    term: '淨零碳排',
    category: 'project',
    aliases: ['淨零轉型', '碳盤查', '探盤查', 'ESG', '永續發展'],
  },
  {
    term: '數位轉型',
    category: 'project',
    aliases: ['雲端服務', '智慧商業', '行動支付', '電子發票'],
  },

  // ==========================================================================
  // 四、 學術研究、高教校務與計畫審查常用詞
  // ==========================================================================
  {
    term: '計畫主持人',
    category: 'person',
    aliases: ['共同主持人', '協同主持人', '主池人'],
  },
  {
    term: '產學合作',
    category: 'project',
    aliases: ['產學計畫', '慘學合作', '技轉中心', '技術移轉'],
  },
  {
    term: '深耕計畫',
    category: 'project',
    aliases: ['高教深耕', '生根計畫', 'USR計畫', '大學社會責任'],
  },
  {
    term: '研究倫理',
    category: 'jargon',
    aliases: ['IRB', '人體試驗', '知情同意書'],
  },
  {
    term: '質性研究',
    category: 'jargon',
    aliases: ['直線研究', '深度訪談', '焦點團體', '田野調查'],
  },
  {
    term: '量化分析',
    category: 'jargon',
    aliases: ['問卷調查', '統計顯著', '迴歸分析', '信效度'],
  },
  {
    term: '文獻探討',
    category: 'jargon',
    aliases: ['文獻回顧', '研究方法', '研究架構', '研究假設'],
  },
  {
    term: '專家學者',
    category: 'person',
    aliases: ['審查委員', '外審委員', '口試委員', '指導教授'],
  },
  {
    term: '博士後研究員',
    category: 'person',
    aliases: ['博後', '專任助理', '兼任助理', '研究助理'],
  },

  // ==========================================================================
  // 五、 企業治理、財務會計、科技研發與日常會議高頻詞
  // ==========================================================================
  {
    term: '董事會',
    category: 'person',
    aliases: ['股東會', '常務董事', '獨立董事', '獨董', '監察人'],
  },
  {
    term: '損益表',
    category: 'jargon',
    aliases: ['資產負債表', '現金流量表', '毛利率', '營業利益率'],
  },
  {
    term: '資本支出',
    category: 'jargon',
    aliases: ['費用化', '折舊攤提', '預算編列', '概算表'],
  },
  {
    term: '法說會',
    category: 'project',
    aliases: ['法人說明會', '財報公布', '季報', '年報'],
  },
  {
    term: '概念驗證',
    category: 'project',
    aliases: ['POC', 'PoC', '場域驗證', '試營運'],
  },
  {
    term: '大型語言模型',
    category: 'jargon',
    aliases: ['生成式AI', '語音辨識', '語者分離', '微調模型'],
  },
  {
    term: '資訊安全',
    category: 'jargon',
    aliases: ['資安檢測', '弱點掃描', '滲透測試', '個資保護', '去識別化'],
  },
  {
    term: '會議紀錄',
    category: 'project',
    aliases: ['會議記錄', '逐字稿', '簽到表', '臨時動議', '主席裁示'],
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
    term: seed.term,
    category: seed.category,
    scope: 'public',
    aliases: seed.aliases ? [...seed.aliases] : [],
  }));
}