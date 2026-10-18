// src/utils/dictIO.ts
import * as OpenCC from 'opencc-js';
import { DictCategory, DictEntry, DictScope } from '../types';

const toTwp = OpenCC.Converter({ from: 'cn', to: 'twp' });

export const DEFAULT_PUBLIC_DICTIONARY: DictEntry[] = [
  {
    id: 'pub_default_1',
    term: '國發會',
    category: 'jargon',
    scope: 'public',
    aliases: ['國法會', '過發會'],
  },
  {
    id: 'pub_default_2',
    term: '地方創生',
    category: 'project',
    scope: 'public',
    aliases: ['地方創身', '地方創聲'],
  },
  {
    id: 'pub_default_3',
    term: '青年培力工作站',
    category: 'project',
    scope: 'public',
    aliases: ['青年賠力工作站', '青年陪力工作站', '青培站'],
  },
  {
    id: 'pub_default_4',
    term: '利害關係人',
    category: 'jargon',
    scope: 'public',
    aliases: ['厲害關係人'],
  },
  {
    id: 'pub_default_5',
    term: '國土區域離島發展處',
    category: 'jargon',
    scope: 'public',
    aliases: ['國土處'],
  },
];

/**
 * 將使用者貼上的 Google Sheet 共編網址轉換為可直接 Fetch 的 CSV Export URL
 */
export function buildGoogleSheetCsvUrl(rawUrl: string): string | null {
  const trimmed = rawUrl.trim();
  if (!trimmed) return null;

  // 若已經是發佈到網路的 output=csv 或是 export?format=csv 連結
  if (trimmed.includes('output=csv') || trimmed.includes('format=csv')) {
    return trimmed;
  }

  // 擷取 /spreadsheets/d/{ID}
  const match = trimmed.match(/\/spreadsheets\/d\/([a-zA-Z0-9-_]+)/);
  if (!match || !match[1]) return null;

  const sheetId = match[1];
  const gidMatch = trimmed.match(/[#&?]gid=([0-9]+)/);
  const gid = gidMatch && gidMatch[1] ? gidMatch[1] : '0';

  return `https://docs.google.com/spreadsheets/d/${sheetId}/export?format=csv&gid=${gid}`;
}

/**
 * 解析單行 CSV (處理雙引號包覆之逗號)
 */
function parseCsvRow(line: string): string[] {
  const cols: string[] = [];
  let cur = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      cols.push(cur.trim());
      cur = '';
    } else {
      cur += ch;
    }
  }
  cols.push(cur.trim());
  return cols;
}

/**
 * 從 Google Sheet 共編連結同步詞庫 (歸入 'public' 公共字典)
 * 支援欄位：A欄=正確詞彙, B欄=常見同音錯字(選填), C欄=分類(選填)
 */
export async function syncDictionaryFromGoogleSheet(sheetUrl: string): Promise<DictEntry[]> {
  const csvUrl = buildGoogleSheetCsvUrl(sheetUrl);
  if (!csvUrl) {
    throw new Error('請貼上有效的 Google 試算表網址 (https://docs.google.com/spreadsheets/d/...)');
  }

  const res = await fetch(csvUrl, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error('無法讀取 Google 試算表，請確認試算表右上角「共用」已設為「知道連結的任何人皆可檢視」。');
  }

  const csvText = await res.text();
  if (csvText.trim().startsWith('<!DOCTYPE html') || csvText.includes('<html')) {
    throw new Error('試算表權限未公開，請將 Google Sheet 共用權限設為「知道連結的任何人皆可檢視」。');
  }

  const lines = csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  const entries: DictEntry[] = [];
  const headerKeywords = ['詞彙', '正確詞', '名稱', 'term', 'word', '專有名詞'];

  lines.forEach((line, idx) => {
    const cols = parseCsvRow(line);
    const colA = (cols[0] || '').trim();
    const colB = (cols[1] || '').trim();
    const colC = (cols[2] || '').trim().toLowerCase();

    if (!colA) return;
    // 自動略過第一列標題列
    if (idx === 0 && headerKeywords.some((kw) => colA.toLowerCase().includes(kw))) {
      return;
    }

    const cleanTerm = toTwp(colA);
    const aliases = colB
      ? colB
          .split(/[,，、/;\s]+/)
          .map((s) => toTwp(s.trim()))
          .filter((s) => s && s !== cleanTerm)
      : undefined;

    let category: DictCategory = 'jargon';
    if (colC.includes('person') || colC.includes('人') || colC.includes('職稱')) {
      category = 'person';
    } else if (colC.includes('project') || colC.includes('計畫') || colC.includes('專案')) {
      category = 'project';
    }

    entries.push({
      id: `gsheet_pub_${idx}_${cleanTerm}`,
      term: cleanTerm,
      category,
      scope: 'public',
      aliases: aliases && aliases.length > 0 ? aliases : undefined,
    });
  });

  return entries;
}

export async function fetchRemotePublicDictionary(): Promise<DictEntry[]> {
  try {
    const res = await fetch('./public-dictionary.json', { cache: 'no-store' });
    if (!res.ok) return DEFAULT_PUBLIC_DICTIONARY;
    const rawList = await res.json();
    const parsed = normalizeImportedDictArray(rawList, 'public');
    return mergeDictLists(DEFAULT_PUBLIC_DICTIONARY, parsed);
  } catch {
    return DEFAULT_PUBLIC_DICTIONARY;
  }
}

export function normalizeImportedDictArray(
  rawData: unknown,
  fallbackScope: DictScope = 'personal'
): DictEntry[] {
  if (!Array.isArray(rawData)) {
    throw new Error('字典檔案格式錯誤：根節點必須為 JSON Array。');
  }

  const validCategories: DictCategory[] = ['person', 'project', 'jargon'];
  const validScopes: DictScope[] = ['public', 'personal', 'session', 'global'];
  const results: DictEntry[] = [];

  rawData.forEach((item, idx) => {
    if (!item || typeof item !== 'object') return;
    const rec = item as Record<string, unknown>;
    const rawTerm = typeof rec.term === 'string' ? rec.term.trim() : '';
    if (!rawTerm) return;

    const category: DictCategory = validCategories.includes(rec.category as DictCategory)
      ? (rec.category as DictCategory)
      : 'jargon';

    const rawScope = rec.scope as DictScope;
    const scope: DictScope = validScopes.includes(rawScope)
      ? rawScope === 'global'
        ? 'public'
        : rawScope
      : fallbackScope;

    const aliases = Array.isArray(rec.aliases)
      ? rec.aliases
          .map((a) => (typeof a === 'string' ? toTwp(a.trim()) : ''))
          .filter(Boolean)
      : undefined;

    results.push({
      id: typeof rec.id === 'string' && rec.id ? rec.id : `dict_${scope}_${Date.now()}_${idx}`,
      term: toTwp(rawTerm),
      category,
      scope,
      aliases: aliases && aliases.length > 0 ? aliases : undefined,
    });
  });

  return results;
}

export function mergeDictLists(baseList: DictEntry[], incomingList: DictEntry[]): DictEntry[] {
  const map = new Map<string, DictEntry>();

  for (const item of baseList) {
    const normalizedScope = item.scope === 'global' ? 'public' : item.scope;
    const key = `${normalizedScope}::${item.term}`;
    map.set(key, { ...item, scope: normalizedScope });
  }

  for (const item of incomingList) {
    const normalizedScope = item.scope === 'global' ? 'public' : item.scope;
    const key = `${normalizedScope}::${item.term}`;
    const existing = map.get(key);
    if (existing) {
      const mergedAliases = Array.from(
        new Set([...(existing.aliases || []), ...(item.aliases || [])])
      );
      map.set(key, {
        ...existing,
        category: item.category || existing.category,
        aliases: mergedAliases.length > 0 ? mergedAliases : undefined,
      });
    } else {
      map.set(key, { ...item, scope: normalizedScope });
    }
  }

  return Array.from(map.values());
}

export function exportDictionaryToJsonFile(
  entries: DictEntry[],
  scopeFilter: 'public' | 'personal' | 'all'
): void {
  const filtered =
    scopeFilter === 'all'
      ? entries
      : entries.filter((e) => (e.scope === 'global' ? 'public' : e.scope) === scopeFilter);

  const jsonStr = JSON.stringify(filtered, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  const dateTag = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download =
    scopeFilter === 'public'
      ? 'public-dictionary.json'
      : `hybrid-transcriber-${scopeFilter}-dict-${dateTag}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function parseDictionaryJsonFile(
  file: File,
  targetScope: 'public' | 'personal'
): Promise<DictEntry[]> {
  const text = await file.text();
  const raw = JSON.parse(text);
  return normalizeImportedDictArray(raw, targetScope);
}