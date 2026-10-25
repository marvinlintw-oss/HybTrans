// src/utils/dictIO.ts
import { DictCategory, DictEntry, DictScope } from '../types';
import { getBuiltinTaiwanMeetingDictionary } from '../data/taiwanMeetingDictionary';

/**
 * 預設公共詞庫：直接載入獨立維護之《台灣商業、政府與地方創生會議內建核心詞典》
 */
export const DEFAULT_PUBLIC_DICTIONARY: DictEntry[] =
  getBuiltinTaiwanMeetingDictionary();

/**
 * 預設支援快速掛載之《HybTransPlus 地方創生字典》Google Sheet ID 與完整網址
 */
export const DEFAULT_REGIONAL_REVITALIZATION_SHEET_ID =
  '1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E';
export const DEFAULT_REGIONAL_REVITALIZATION_SHEET_URL =
  'https://docs.google.com/spreadsheets/d/1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E/edit?gid=0#gid=0';

function safeTrim(val: unknown): string {
  if (typeof val === 'string') return val.trim();
  if (val === null || val === undefined) return '';
  return String(val).trim();
}

/**
 * 解析網址列查詢參數 `?dict=...`（同時支援標準 Search Query `?dict=1ra9-...` 與 Hash Routing `#/?dict=1ra9-...`）
 */
export function getUrlQueryDictParam(): {
  dictParam: string;
  gidParam: string | null;
} | null {
  if (typeof window === 'undefined') return null;
  try {
    const searchParams = new URLSearchParams(window.location.search);
    let rawDict = safeTrim(
      searchParams.get('dict') ||
        searchParams.get('sheet') ||
        searchParams.get('dictionary')
    );
    let rawGid = safeTrim(searchParams.get('gid'));

    if (!rawDict && window.location.hash.includes('?')) {
      const hashQuery = window.location.hash.slice(
        window.location.hash.indexOf('?')
      );
      const hashParams = new URLSearchParams(hashQuery);
      rawDict = safeTrim(
        hashParams.get('dict') ||
          hashParams.get('sheet') ||
          hashParams.get('dictionary')
      );
      if (!rawGid) {
        rawGid = safeTrim(hashParams.get('gid'));
      }
    }

    if (!rawDict) return null;
    if (rawDict.toLowerCase() === 'default' || rawDict.toLowerCase() === 'ndc') {
      rawDict = DEFAULT_REGIONAL_REVITALIZATION_SHEET_ID;
    }

    return {
      dictParam: rawDict,
      gidParam: rawGid || null,
    };
  } catch {
    return null;
  }
}

/**
 * 將使用者傳入之「純 Google Sheet ID (如 `1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E`)」、
 * 「完整 Google Sheet 網址」或「遠端 CSV/JSON 網址」解析為標準端點資訊
 */
export function resolveGoogleSheetEndpoint(
  inputUrlOrId: string,
  explicitGid?: string | null
): {
  sheetId: string | null;
  gid: string;
  primaryCsvUrl: string;
  fallbackGvizCsvUrl: string | null;
  canonicalViewUrl: string;
  shareParamValue: string;
} {
  const trimmed = safeTrim(inputUrlOrId);
  if (!trimmed) {
    return {
      sheetId: null,
      gid: '0',
      primaryCsvUrl: '',
      fallbackGvizCsvUrl: null,
      canonicalViewUrl: '',
      shareParamValue: '',
    };
  }

  // 情況 1：完整 Google Sheets URL (https://docs.google.com/spreadsheets/d/<ID>/...)
  const urlMatch = trimmed.match(/\/spreadsheets\/d\/([a-zA-Z0-9-_]+)/);
  if (urlMatch && urlMatch[1]) {
    const sheetId = urlMatch[1];
    const gidMatch = trimmed.match(/[#&?]gid=([0-9]+)/);
    const gid = safeTrim(explicitGid) || (gidMatch && gidMatch[1]) || '0';
    return {
      sheetId,
      gid,
      primaryCsvUrl: `https://docs.google.com/spreadsheets/d/${sheetId}/export?format=csv&gid=${gid}`,
      fallbackGvizCsvUrl: `https://docs.google.com/spreadsheets/d/${sheetId}/gviz/tq?tqx=out:csv&gid=${gid}`,
      canonicalViewUrl: `https://docs.google.com/spreadsheets/d/${sheetId}/edit?gid=${gid}#gid=${gid}`,
      shareParamValue: gid !== '0' ? `${sheetId}:${gid}` : sheetId,
    };
  }

  // 情況 2：純 Google Sheet ID 或 SheetID:gid (如 `1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E` 或 `1ra9-...:0`)
  const bareIdMatch = trimmed.match(/^([a-zA-Z0-9-_]{20,80})(?:[:#&?]gid=([0-9]+)|:([0-9]+))?$/);
  if (bareIdMatch && bareIdMatch[1] && !trimmed.includes('/') && !trimmed.includes('.')) {
    const sheetId = bareIdMatch[1];
    const gid =
      safeTrim(explicitGid) || bareIdMatch[2] || bareIdMatch[3] || '0';
    return {
      sheetId,
      gid,
      primaryCsvUrl: `https://docs.google.com/spreadsheets/d/${sheetId}/export?format=csv&gid=${gid}`,
      fallbackGvizCsvUrl: `https://docs.google.com/spreadsheets/d/${sheetId}/gviz/tq?tqx=out:csv&gid=${gid}`,
      canonicalViewUrl: `https://docs.google.com/spreadsheets/d/${sheetId}/edit?gid=${gid}#gid=${gid}`,
      shareParamValue: gid !== '0' ? `${sheetId}:${gid}` : sheetId,
    };
  }

  // 情況 3：一般外部 CSV / JSON 直連網址
  return {
    sheetId: null,
    gid: '0',
    primaryCsvUrl: trimmed,
    fallbackGvizCsvUrl: null,
    canonicalViewUrl: trimmed,
    shareParamValue: trimmed,
  };
}

/**
 * 產生包含 `?dict=<GoogleSheetID>` 的可分享完整網址（例如 `http://xxx.xxx.xxx/HybTransPlus?dict=1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E`）
 */
export function buildShareableDictUrl(inputUrlOrId?: string | null): string {
  const target = safeTrim(inputUrlOrId) || DEFAULT_REGIONAL_REVITALIZATION_SHEET_ID;
  const resolved = resolveGoogleSheetEndpoint(target);
  if (typeof window === 'undefined') {
    return `?dict=${encodeURIComponent(resolved.shareParamValue)}`;
  }
  try {
    const url = new URL(window.location.href);
    url.searchParams.set('dict', resolved.sheetId || resolved.shareParamValue);
    if (resolved.sheetId && resolved.gid && resolved.gid !== '0') {
      url.searchParams.set('gid', resolved.gid);
    } else {
      url.searchParams.delete('gid');
    }
    return url.toString();
  } catch {
    return `${window.location.origin}${window.location.pathname}?dict=${encodeURIComponent(
      resolved.shareParamValue
    )}`;
  }
}

/**
 * RFC-4180 相容之單行 CSV/TSV 解析器（正確處理雙引號包覆之逗號與分號欄位）
 */
function parseCsvRowRespectingQuotes(line: string): string[] {
  if (line.includes('\t') && !line.includes(',')) {
    return line.split('\t').map((c) => safeTrim(c.replace(/^["']|["']$/g, '')));
  }
  const cells: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      cells.push(safeTrim(current));
      current = '';
    } else {
      current += ch;
    }
  }
  cells.push(safeTrim(current));
  return cells.map((c) => safeTrim(c.replace(/^["']|["']$/g, '')));
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
 * 解析使用者上傳或遠端 Google Sheet 載入之 JSON / CSV / TSV / TXT 詞庫文字（具備完整空值防呆與 Google Sheet 欄位結構支援）
 * 支援試算表欄位：A欄=term(正詞), B欄=aliases(以 ; / | / 、 / , 分隔之別名), C欄=category(person/project/jargon), D欄=notes, E欄=source
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
              .split(/[|｜/、;；,，]/)
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
      // fallback to line-by-line CSV parsing
    }
  }

  // 2. 支援 Google Sheet CSV / TSV / 純文字格式：每行「term,aliases,category,notes,source」
  const lines = trimmed.split(/\r?\n/);
  const entries: DictEntry[] = [];
  lines.forEach((line, idx) => {
    const clean = safeTrim(line);
    if (!clean || clean.startsWith('#')) return;
    const parts = parseCsvRowRespectingQuotes(clean);
    const term = safeTrim(parts[0]);
    if (!term) return;

    // 略過常見 CSV 標題列 (如 term, 正詞, 詞彙)
    if (
      idx === 0 &&
      /^(term|正詞|正確詞彙|詞彙|專有名詞|名稱|關鍵字|標準詞)$/i.test(term)
    ) {
      return;
    }

    const aliases = parts[1]
      ? parts[1]
          .split(/[|｜/、;；,，]/)
          .map((a) => safeTrim(a))
          .filter((a) => a && a !== term)
      : [];
    const catRaw = safeTrim(parts[2]).toLowerCase();
    const category: DictCategory =
      catRaw === 'person' ||
      catRaw.includes('人') ||
      catRaw.includes('機關') ||
      catRaw.includes('組織')
        ? 'person'
        : catRaw === 'project' ||
          catRaw.includes('案') ||
          catRaw.includes('計畫') ||
          catRaw.includes('政策')
        ? 'project'
        : 'jargon';

    entries.push({
      id: `gsheet_${Date.now()}_${idx}`,
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
 * 從公開的 Google Sheet 網址、純 Sheet ID (如 `?dict=1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E`) 或遠端 CSV/JSON 端點同步詞庫
 * 具備雙端點自動備援 (`/export?format=csv` 與 `/gviz/tq?tqx=out:csv`)！
 */
export async function syncDictionaryFromGoogleSheet(
  sheetUrlOrId?: string | null,
  targetScope: DictScope = 'public',
  explicitGid?: string | null
): Promise<DictEntry[]> {
  const trimmedInput = safeTrim(sheetUrlOrId);
  if (!trimmedInput) {
    return getBuiltinTaiwanMeetingDictionary().map((item) => ({
      ...item,
      scope: targetScope,
    }));
  }

  const endpoint = resolveGoogleSheetEndpoint(trimmedInput, explicitGid);
  const candidateUrls = [
    endpoint.primaryCsvUrl,
    endpoint.fallbackGvizCsvUrl,
  ].filter((u): u is string => Boolean(u));

  let lastError: Error | null = null;
  for (const fetchUrl of candidateUrls) {
    try {
      const response = await fetch(fetchUrl, { cache: 'no-cache' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const rawText = await response.text();
      if (rawText.trim().startsWith('<!DOCTYPE html') || rawText.includes('<html')) {
        throw new Error('回傳內容為登入頁面，請確認試算表已設為「知道連結的任何人皆可檢視」');
      }
      const parsed = parseImportedDictionaryText(rawText, targetScope);
      if (parsed.length > 0) {
        return parsed;
      }
    } catch (err) {
      lastError = err as Error;
    }
  }

  throw new Error(
    `無法讀取遠端詞庫 (${lastError?.message || '連線失敗'})，請確認 Google Sheet ID 或連結有效且已開啟公開檢視權限`
  );
}

/**
 * 載入或更新公共詞庫（相容無參數呼叫、遠端 URL、網址 `?dict=` 參數或本地預設台灣會議大詞典）
 */
export async function fetchRemotePublicDictionary(
  remoteUrlOrId?: string | null,
  targetScope: DictScope = 'public'
): Promise<DictEntry[]> {
  const cleanInput = safeTrim(remoteUrlOrId);
  if (!cleanInput) {
    return getBuiltinTaiwanMeetingDictionary().map((item) => ({
      ...item,
      scope: targetScope,
    }));
  }

  try {
    return await syncDictionaryFromGoogleSheet(cleanInput, targetScope);
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

export function exportProjectStateToJson(
  segments: unknown,
  speakers: unknown
): void {
  const payload = JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      speakers,
      segments,
    },
    null,
    2
  );
  const blob = new Blob([payload], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `會議專案備份_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}