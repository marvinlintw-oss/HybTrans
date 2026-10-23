// src/utils/dictIO.ts
import { DictCategory, DictEntry, DictScope } from '../types';
import { getBuiltinTaiwanMeetingDictionary } from '../data/taiwanMeetingDictionary';

/**
 * 預設公共詞庫：直接載入獨立維護之《台灣商業、政府與學術會議內建大詞典》
 */
export const DEFAULT_PUBLIC_DICTIONARY: DictEntry[] =
  getBuiltinTaiwanMeetingDictionary();

function safeTrim(val: unknown): string {
  if (typeof val === 'string') return val.trim();
  if (val === null || val === undefined) return '';
  return String(val).trim();
}

/**
 * 合併現有詞庫與新匯入詞條（若遇到相同 term + scope，自動合併其同音別名 aliases）
 */
export function mergeDictLists(
  existing: DictEntry[] = [],
  incoming: DictEntry[] = []
): DictEntry[] {
  const map = new Map<string, DictEntry>();

  const safeExisting = Array.isArray(existing) ? existing : [];
  const safeIncoming = Array.isArray(incoming) ? incoming : [];

  for (const item of safeExisting) {
    if (!item) continue;
    const cleanTerm = safeTrim(item.term);
    if (!cleanTerm) continue;
    const scope: DictScope = item.scope || 'public';
    const key = `${scope}::${cleanTerm}`;
    map.set(key, {
      ...item,
      term: cleanTerm,
      scope,
      aliases: Array.isArray(item.aliases)
        ? Array.from(new Set(item.aliases.map((a) => safeTrim(a)).filter(Boolean)))
        : [],
    });
  }

  for (const item of safeIncoming) {
    if (!item) continue;
    const cleanTerm = safeTrim(item.term);
    if (!cleanTerm) continue;
    const scope: DictScope = item.scope || 'personal';
    const key = `${scope}::${cleanTerm}`;
    const prev = map.get(key);
    const incomingAliases = Array.isArray(item.aliases)
      ? item.aliases.map((a) => safeTrim(a)).filter((a) => a && a !== cleanTerm)
      : [];

    if (prev) {
      const mergedAliases = Array.from(
        new Set([...(prev.aliases || []), ...incomingAliases])
      ).filter((a) => a && a !== cleanTerm);
      map.set(key, {
        ...prev,
        category: item.category || prev.category,
        aliases: mergedAliases,
      });
    } else {
      map.set(key, {
        ...item,
        id: safeTrim(item.id) || `dict_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        term: cleanTerm,
        category: item.category || 'jargon',
        scope,
        aliases: incomingAliases,
      });
    }
  }

  return Array.from(map.values());
}

/**
 * 匯出指定範圍或全部字典為 JSON 字串
 */
export function exportDictionaryToJson(
  entries: DictEntry[] = [],
  scopeFilter?: DictScope | 'all'
): string {
  const safeEntries = Array.isArray(entries) ? entries : [];
  const filtered =
    !scopeFilter || scopeFilter === 'all'
      ? safeEntries
      : safeEntries.filter((e) => e && e.scope === scopeFilter);
  return JSON.stringify(filtered, null, 2);
}

/**
 * 解析使用者上傳或遠端載入之 JSON / CSV / TXT 詞庫文字（具備完整空值防呆）
 */
export function parseImportedDictionaryText(
  rawContent: unknown,
  defaultScope: DictScope = 'personal'
): DictEntry[] {
  if (Array.isArray(rawContent)) {
    return rawContent
      .map((raw, idx) => {
        if (!raw || typeof raw !== 'object') return null;
        const r = raw as Record<string, unknown>;
        const term = safeTrim(r.term || r.name || r.word);
        if (!term) return null;
        const category: DictCategory =
          r.category === 'person' || r.category === 'project' || r.category === 'jargon'
            ? r.category
            : 'jargon';
        const scope: DictScope =
          r.scope === 'public' ||
          r.scope === 'personal' ||
          r.scope === 'session' ||
          r.scope === 'global'
            ? r.scope
            : defaultScope;
        const aliases = Array.isArray(r.aliases)
          ? r.aliases.map((a) => safeTrim(a)).filter((a) => a && a !== term)
          : typeof r.aliases === 'string'
          ? r.aliases
              .split(/[|｜/、,，]/)
              .map((a) => safeTrim(a))
              .filter((a) => a && a !== term)
          : [];
        return {
          id: safeTrim(r.id) || `imp_${Date.now()}_${idx}`,
          term,
          category,
          scope,
          aliases,
        } as DictEntry;
      })
      .filter((item): item is DictEntry => item !== null);
  }

  const trimmed = safeTrim(rawContent);
  if (!trimmed) return [];

  // 1. 嘗試解析 JSON
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed);
      const arr = Array.isArray(parsed)
        ? parsed
        : Array.isArray(parsed?.entries)
        ? parsed.entries
        : Array.isArray(parsed?.dictionary)
        ? parsed.dictionary
        : [parsed];
      return parseImportedDictionaryText(arr, defaultScope);
    } catch {
      // fallback to line-by-line parsing
    }
  }

  // 2. 支援 CSV / 純文字格式：每行「正詞,別名1|別名2,分類」
  const lines = trimmed.split(/\r?\n/);
  const entries: DictEntry[] = [];
  lines.forEach((line, idx) => {
    const clean = safeTrim(line);
    if (!clean || clean.startsWith('#')) return;
    const parts = clean
      .split(/[,，\t]/)
      .map((p) => safeTrim(p.replace(/^["']|["']$/g, '')));
    const term = parts[0];
    if (!term) return;
    // 略過常見 CSV 標題列
    if (
      idx === 0 &&
      /^(term|正詞|詞彙|專有名詞|名稱|關鍵字|標準詞)$/i.test(term)
    ) {
      return;
    }

    const aliases = parts[1]
      ? parts[1]
          .split(/[|｜/、;；]/)
          .map((a) => safeTrim(a))
          .filter((a) => a && a !== term)
      : [];
    const catRaw = safeTrim(parts[2]).toLowerCase();
    const category: DictCategory =
      catRaw === 'person' || catRaw.includes('人') || catRaw.includes('機關')
        ? 'person'
        : catRaw === 'project' || catRaw.includes('案') || catRaw.includes('計畫')
        ? 'project'
        : 'jargon';

    entries.push({
      id: `txt_${Date.now()}_${idx}`,
      term,
      category,
      scope: defaultScope,
      aliases,
    });
  });

  return entries;
}

/**
 * 解析使用者上傳之 JSON / CSV / TXT 字典檔案 (相容 File、Blob 或純文字字串)
 */
export async function parseDictionaryJsonFile(
  fileOrText: File | Blob | string | null | undefined,
  defaultScope: DictScope = 'personal'
): Promise<DictEntry[]> {
  if (!fileOrText) return [];
  if (typeof fileOrText === 'string') {
    return parseImportedDictionaryText(fileOrText, defaultScope);
  }
  if (typeof fileOrText.text === 'function') {
    const rawText = await fileOrText.text();
    return parseImportedDictionaryText(rawText, defaultScope);
  }
  return [];
}

/**
 * 將指定範圍之字典匯出並觸發瀏覽器下載 JSON 檔案 (同時回傳 JSON 字串)
 */
export function exportDictionaryToJsonFile(
  entries: DictEntry[] = [],
  scopeFilter: DictScope | 'all' = 'all',
  filename?: string
): string {
  const safeScope = safeTrim(scopeFilter) || 'all';
  const resolvedFilename =
    safeTrim(filename) || `hybtrans_dictionary_${safeScope}.json`;
  const jsonStr = exportDictionaryToJson(
    entries,
    (safeScope as DictScope | 'all') || 'all'
  );

  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    try {
      const blob = new Blob([jsonStr], {
        type: 'application/json;charset=utf-8',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = resolvedFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      // 若在非 DOM 環境呼叫則僅回傳字串
    }
  }
  return jsonStr;
}

/**
 * 相容別名：與 exportDictionaryToJsonFile 相同功能
 */
export const exportDictionaryAsJson = exportDictionaryToJsonFile;

/**
 * 從公開的 Google Sheet 網址或遠端 CSV/JSON 端點同步詞庫（具備完整空值防呆）
 */
export async function syncDictionaryFromGoogleSheet(
  sheetUrl?: string | null,
  targetScope: DictScope = 'public'
): Promise<DictEntry[]> {
  const trimmedUrl = safeTrim(sheetUrl);
  // 若使用者未輸入網址或傳入 undefined，直接回傳內建台灣會議大詞典而不崩潰
  if (!trimmedUrl) {
    return getBuiltinTaiwanMeetingDictionary().map((item) => ({
      ...item,
      scope: targetScope,
    }));
  }

  let fetchUrl = trimmedUrl;
  const sheetIdMatch = trimmedUrl.match(/\/spreadsheets\/d\/([a-zA-Z0-9-_]+)/);
  if (sheetIdMatch && sheetIdMatch[1]) {
    const spreadsheetId = sheetIdMatch[1];
    const gidMatch = trimmedUrl.match(/[#&?]gid=([0-9]+)/);
    const gidParam = gidMatch && gidMatch[1] ? `&gid=${gidMatch[1]}` : '';
    fetchUrl = `https://docs.google.com/spreadsheets/d/${spreadsheetId}/export?format=csv${gidParam}`;
  }

  const response = await fetch(fetchUrl, { cache: 'no-cache' });
  if (!response.ok) {
    throw new Error(
      `無法讀取遠端詞庫 (${response.status})，請確認連結有效且已開啟公開檢視權限`
    );
  }

  const rawText = await response.text();
  const parsed = parseImportedDictionaryText(rawText, targetScope);
  return parsed.length > 0 ? parsed : getBuiltinTaiwanMeetingDictionary();
}

/**
 * 載入或更新公共詞庫（相容無參數呼叫、遠端 URL 或本地預設台灣會議大詞典）
 */
export async function fetchRemotePublicDictionary(
  remoteUrl?: string | null,
  targetScope: DictScope = 'public'
): Promise<DictEntry[]> {
  const cleanUrl = safeTrim(remoteUrl);
  if (!cleanUrl) {
    return getBuiltinTaiwanMeetingDictionary().map((item) => ({
      ...item,
      scope: targetScope,
    }));
  }

  try {
    return await syncDictionaryFromGoogleSheet(cleanUrl, targetScope);
  } catch (err) {
    console.warn(
      '⚠️ [Dictionary] 遠端詞庫讀取失敗，自動採用內建台灣會議大詞典:',
      err
    );
    return getBuiltinTaiwanMeetingDictionary().map((item) => ({
      ...item,
      scope: targetScope,
    }));
  }
}