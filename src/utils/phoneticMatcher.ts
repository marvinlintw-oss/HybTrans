import { Segment, PhoneticMatchCandidate } from '../types';

/**
 * 台灣會議常見姓氏與職稱之聲韻母同音群組映射表（無平捲舌/前後鼻音模糊容錯）
 */
const PHONETIC_GROUPS: Record<string, string> = {
  // 常見姓氏同音群組
  '林臨淋琳霖零玲': 'LIN',
  '陳塵臣辰晨沉': 'CHEN',
  '張章彰璋髒': 'ZHANG',
  '李里理禮裡鯉': 'LI',
  '王亡汪網往': 'WANG',
  '吳吾無蕪梧巫': 'WU',
  '劉留流琉榴': 'LIU',
  '蔡菜採彩踩': 'CAI',
  '楊陽揚洋羊佯': 'YANG',
  '許栩詡徐須': 'XU',
  '鄭正政證症': 'ZHENG',
  '謝械卸屑蟹': 'XIE',
  '郭鍋國果過': 'GUO',
  '洪紅宏虹鴻': 'HONG',
  '曾增爭箏蒸': 'ZENG',
  '廖料瞭療聊': 'LIAO',
  '賴來萊瀨': 'LAI',
  '蘇疏輸書舒': 'SU',
  '周洲舟州週': 'ZHOU',
  '葉頁業夜液': 'YE',
  // 常見職稱與會議稱謂同音群組 (含前後鼻音/平捲舌容錯)
  '處觸畜築儲礎楚': 'CHU',
  '長場廠掌漲常': 'ZHANG_TITLE',
  '委偉尾瑋偽緯': 'WEI',
  '員元園原源圓援': 'YUAN',
  '科顆棵磕苛': 'KE',
  '主組祖阻煮囑': 'ZHU',
  '任認韌刃妊': 'REN',
  '秘密蜜覓冪': 'MI',
  '總種腫塚': 'ZONG',
  '經精晶京驚鯨': 'JING',
  '博薄搏駁': 'BO',
  '士事式市試視室是': 'SHI',
  '教較校酵覺': 'JIAO',
  '授受售瘦獸': 'SHOU',
};

const charToPhoneticCode = new Map<string, string>();
Object.entries(PHONETIC_GROUPS).forEach(([chars, code]) => {
  for (const ch of chars) charToPhoneticCode.set(ch, code);
});

export function toPhoneticSignature(text: string): string {
  return Array.from(text)
    .map((ch) => charToPhoneticCode.get(ch) || ch)
    .join('-');
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
        if (toPhoneticSignature(wordText) === targetSig) {
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