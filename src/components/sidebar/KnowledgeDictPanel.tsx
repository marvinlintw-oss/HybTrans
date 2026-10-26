// src/components/sidebar/KnowledgeDictPanel.tsx
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  BookOpen,
  FileText,
  Globe,
  User,
  Sparkles,
  Plus,
  Trash2,
  Download,
  Upload,
  ArrowUpRight,
  RefreshCw,
  Link2,
  Info,
  Share2,
  Check,
} from 'lucide-react';

/**
 * 內建互動式說明浮出視窗（支援滑鼠懸停與點擊切換，解決原生 title 無法即時顯示或看似空白的問題）
 */
const HoverInfoPopover: React.FC<{
  title: string;
  children: React.ReactNode;
  align?: 'left' | 'right';
}> = ({ title, children, align = 'left' }) => {
  const [pinned, setPinned] = useState(false);

  return (
    <div
      className="relative inline-flex items-center group"
      onMouseLeave={() => setPinned(false)}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setPinned((prev) => !prev);
        }}
        aria-label={title}
        className="inline-flex items-center text-slate-400 hover:text-indigo-600 focus:text-indigo-600 transition-colors cursor-pointer focus:outline-none"
      >
        <Info className="w-3.5 h-3.5 shrink-0" />
      </button>

      <div
        className={`absolute ${
          align === 'right' ? 'right-0' : 'left-0'
        } top-full mt-1.5 z-50 w-72 p-3 rounded-xl bg-slate-900/95 text-slate-100 text-[11px] leading-relaxed shadow-xl border border-slate-700 transition-all duration-150 ${
          pinned
            ? 'opacity-100 visible pointer-events-auto'
            : 'opacity-0 invisible pointer-events-none group-hover:opacity-100 group-hover:visible group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:visible'
        }`}
      >
        <div className="font-semibold text-white mb-1 flex items-center gap-1.5 border-b border-slate-700/80 pb-1">
          <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span>{title}</span>
        </div>
        <div className="text-slate-300 space-y-1 font-normal">{children}</div>
      </div>
    </div>
  );
};
import { useAppStore } from '../../store/useAppStore';
import { DictCategory, DictEntry } from '../../types';
import * as docParserMod from '../../utils/docParser';
import {
  DEFAULT_REGIONAL_REVITALIZATION_SHEET_ID,
  buildShareableDictUrl,
  exportDictionaryToJsonFile,
  fetchRemotePublicDictionary,
  getUrlQueryDictParam,
  parseDictionaryJsonFile,
  resolveGoogleSheetEndpoint,
  syncDictionaryFromGoogleSheet,
} from '../../utils/dictIO';

type ActiveTab = 'public' | 'personal' | 'session';

const GSHEET_STORAGE_KEY = 'hybrid_transcriber_gsheet_dict_url';

interface ParsedDocCompatResult {
  summary: string;
  extractedTerms: Array<{ term: string; category: DictCategory }>;
}

async function invokeDocParserCompat(file: File): Promise<ParsedDocCompatResult> {
  const modRecord = docParserMod as Record<string, unknown>;
  const candidateFn =
    modRecord.parseDocumentForContext ||
    modRecord.parseMeetingDocument ||
    modRecord.parseDocument ||
    modRecord.parseDocFile ||
    modRecord.extractDocContext ||
    Object.values(modRecord).find((v) => typeof v === 'function');

  if (typeof candidateFn !== 'function') {
    throw new Error('找不到可用的文件解析函式，請檢查 src/utils/docParser.ts');
  }

  const rawRes = (await (candidateFn as (f: File) => Promise<unknown>)(file)) as Record<
    string,
    unknown
  >;

  const summary =
    typeof rawRes?.summary === 'string'
      ? rawRes.summary
      : typeof rawRes?.contextSummary === 'string'
      ? rawRes.contextSummary
      : typeof rawRes?.text === 'string'
      ? rawRes.text.slice(0, 300)
      : '';

  const rawList = Array.isArray(rawRes?.extractedTerms)
    ? rawRes.extractedTerms
    : Array.isArray(rawRes?.terms)
    ? rawRes.terms
    : Array.isArray(rawRes?.hotwords)
    ? rawRes.hotwords
    : Array.isArray(rawRes?.entries)
    ? rawRes.entries
    : [];

  const extractedTerms: Array<{ term: string; category: DictCategory }> = [];
  rawList.forEach((item: unknown) => {
    if (typeof item === 'string' && item.trim()) {
      extractedTerms.push({ term: item.trim(), category: 'jargon' });
    } else if (item && typeof item === 'object') {
      const rec = item as Record<string, unknown>;
      const tStr = typeof rec.term === 'string' ? rec.term.trim() : '';
      const cat: DictCategory =
        rec.category === 'person' || rec.category === 'project' || rec.category === 'jargon'
          ? rec.category
          : 'jargon';
      if (tStr) {
        extractedTerms.push({ term: tStr, category: cat });
      }
    }
  });

  return { summary, extractedTerms };
}

export const KnowledgeDictPanel: React.FC = () => {
  const {
    dictionary,
    docContextSummary,
    addDictEntries,
    removeDictEntry,
    promoteDictEntryScope,
    setDocContextSummary,
  } = useAppStore();

  const [activeTab, setActiveTab] = useState<ActiveTab>('public');
  const [sheetUrl, setSheetUrl] = useState<string>(() => {
    const queryDict = getUrlQueryDictParam();
    if (queryDict?.dictParam) {
      return queryDict.dictParam;
    }
    try {
      return localStorage.getItem(GSHEET_STORAGE_KEY) || '';
    } catch {
      return '';
    }
  });
  const [isSyncingSheet, setIsSyncingSheet] = useState(false);
  const [urlMountedSheetId, setUrlMountedSheetId] = useState<string | null>(null);
  const [copiedShareLink, setCopiedShareLink] = useState(false);
  const [newTerm, setNewTerm] = useState('');
  const [newAliases, setNewAliases] = useState('');
  const [newCategory, setNewCategory] = useState<DictCategory>('jargon');
  const [isParsingDoc, setIsParsingDoc] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const docInputRef = useRef<HTMLInputElement | null>(null);
  const dictJsonInputRef = useRef<HTMLInputElement | null>(null);

  const handleSyncGoogleSheet = useCallback(
    async (
      targetUrlOrId: string,
      options?: { silent?: boolean; fromUrlQuery?: boolean; explicitGid?: string | null }
    ) => {
      const trimmed = targetUrlOrId.trim();
      if (!trimmed) return;

      const silent = Boolean(options?.silent);
      const fromUrlQuery = Boolean(options?.fromUrlQuery);

      setIsSyncingSheet(true);
      if (!silent) setStatusMsg(null);

      try {
        const endpoint = resolveGoogleSheetEndpoint(trimmed, options?.explicitGid);
        localStorage.setItem(GSHEET_STORAGE_KEY, trimmed);

        if (endpoint.sheetId && typeof window !== 'undefined') {
          try {
            const nextUrl = new URL(window.location.href);
            nextUrl.searchParams.set('dict', endpoint.sheetId);
            if (endpoint.gid && endpoint.gid !== '0') {
              nextUrl.searchParams.set('gid', endpoint.gid);
            } else {
              nextUrl.searchParams.delete('gid');
            }
            window.history.replaceState({}, '', nextUrl.toString());
          } catch {
            // ignore history update error in sandboxed iframe
          }
        }

        const syncedEntries = await syncDictionaryFromGoogleSheet(
          trimmed,
          'public',
          options?.explicitGid
        );
        if (syncedEntries.length > 0) {
          addDictEntries(syncedEntries);
          if (endpoint.sheetId) {
            setUrlMountedSheetId(endpoint.sheetId);
          }
          if (fromUrlQuery) {
            setStatusMsg(
              `🔗 已由網址參數 (?dict=${
                endpoint.sheetId ? `${endpoint.sheetId.slice(0, 10)}...` : '雲端表單'
              }) 自動載入 ${syncedEntries.length} 筆共編詞彙！`
            );
          } else if (!silent) {
            setStatusMsg(`已從 Google Sheet 同步 ${syncedEntries.length} 筆共編詞彙！`);
          }
        } else if (!silent) {
          setStatusMsg('Google Sheet 中尚無有效詞彙列。');
        }
      } catch (err) {
        if (!silent || fromUrlQuery) {
          setStatusMsg((err as Error).message);
        }
      } finally {
        setIsSyncingSheet(false);
      }
    },
    [addDictEntries]
  );

  // 啟動時：1. 先載入內建台灣會議核心字典；2. 若網址帶有 `?dict=1ra9-...` 參數則優先掛載該雲端字典；3. 否則若曾記憶過 Google Sheet 則背景同步
  useEffect(() => {
    fetchRemotePublicDictionary().then((pubEntries) => {
      if (pubEntries.length > 0) {
        addDictEntries(pubEntries);
      }
    });

    const queryDict = getUrlQueryDictParam();
    if (queryDict && queryDict.dictParam) {
      setSheetUrl(queryDict.dictParam);
      handleSyncGoogleSheet(queryDict.dictParam, {
        silent: false,
        fromUrlQuery: true,
        explicitGid: queryDict.gidParam,
      });
      return;
    }

    try {
      const savedUrl = localStorage.getItem(GSHEET_STORAGE_KEY);
      if (savedUrl && savedUrl.trim()) {
        handleSyncGoogleSheet(savedUrl, { silent: true });
      }
    } catch {
      // ignore storage error
    }
  }, [addDictEntries, handleSyncGoogleSheet]);

  const handleCopyShareableDictUrl = () => {
    const shareUrl = buildShareableDictUrl(
      sheetUrl.trim() || DEFAULT_REGIONAL_REVITALIZATION_SHEET_ID
    );
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(shareUrl).catch(() => {});
    }
    setCopiedShareLink(true);
    setStatusMsg(`已複製分享連結：${shareUrl}`);
    setTimeout(() => setCopiedShareLink(false), 2500);
  };

  const publicList = dictionary.filter((d) => d.scope === 'public' || d.scope === 'global');
  const personalList = dictionary.filter((d) => d.scope === 'personal');
  const sessionList = dictionary.filter((d) => d.scope === 'session');

  const currentList =
    activeTab === 'public'
      ? publicList
      : activeTab === 'personal'
      ? personalList
      : sessionList;

  const isPublicTab = activeTab === 'public';
  const isPersonalTab = activeTab === 'personal';
  const isSessionTab = activeTab === 'session';

  const resolvedEndpoint = resolveGoogleSheetEndpoint(sheetUrl);

  const handleAddTerm = (e: React.FormEvent) => {
    e.preventDefault();
    const termTrimmed = newTerm.trim();
    if (!termTrimmed) return;

    const aliasList = newAliases
      .split(/[,，、;；|｜\s]+/)
      .map((s) => s.trim())
      .filter((s) => s && s !== termTrimmed);

    const entry: DictEntry = {
      id: `dict_${activeTab}_${Date.now()}`,
      term: termTrimmed,
      category: newCategory,
      scope: activeTab,
      aliases: aliasList.length > 0 ? aliasList : undefined,
    };

    addDictEntries([entry]);
    setNewTerm('');
    setNewAliases('');
  };

  const handleDocUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsParsingDoc(true);
    setStatusMsg(null);
    try {
      const { summary, extractedTerms } = await invokeDocParserCompat(file);
      setDocContextSummary(summary);
      const sessionEntries: DictEntry[] = extractedTerms.map(
        (t: { term: string; category: DictCategory }, idx: number) => ({
          id: `doc_term_${Date.now()}_${idx}`,
          term: t.term,
          category: t.category,
          scope: 'session',
        })
      );
      addDictEntries(sessionEntries);
      setActiveTab('session');
      setStatusMsg(`已從簡報萃取 ${sessionEntries.length} 個會議詞彙！`);
    } catch (err) {
      setStatusMsg(`解析文件失敗: ${(err as Error).message}`);
    } finally {
      setIsParsingDoc(false);
      e.target.value = '';
    }
  };

  const handleImportJson = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const targetScope: 'public' | 'personal' = isPublicTab ? 'public' : 'personal';
    try {
      const imported = await parseDictionaryJsonFile(file, targetScope);
      addDictEntries(imported);
      setStatusMsg(`已匯入 ${imported.length} 筆詞彙！`);
    } catch (err) {
      setStatusMsg(`匯入失敗: ${(err as Error).message}`);
    } finally {
      e.target.value = '';
    }
  };

  const categoryBadge = (cat: DictCategory) => {
    if (cat === 'person') return { label: '人名', cls: 'bg-purple-50 text-purple-700' };
    if (cat === 'project') return { label: '計畫', cls: 'bg-emerald-50 text-emerald-700' };
    return { label: '專詞', cls: 'bg-blue-50 text-blue-700' };
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-sm space-y-3">
      {/* 標題列 */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <div className="flex items-center gap-1.5">
          <BookOpen className="w-4 h-4 text-indigo-600" />
          <h3 className="font-semibold text-slate-800 text-sm">智慧字典與會議簡報</h3>
          <HoverInfoPopover
            title="三層智慧字典與會議簡報機制"
            align="left"
          >
            <p>
              • <strong>公共字典：</strong>預載台灣會議與地方創生常見專詞，支援連結 Google Sheet 團隊共編表。
            </p>
            <p>
              • <strong>個人字典：</strong>保存在您的瀏覽器本機，適合記錄部門常用人名與縮寫。
            </p>
            <p>
              • <strong>單次簡報：</strong>上傳本次會議的 <code>.pptx / .docx / .pdf</code>，系統會自動萃取投影片內的專有名詞投入先句後詞校對。
            </p>
          </HoverInfoPopover>
        </div>
        <span className="text-[11px] text-slate-500">共 {dictionary.length} 詞</span>
      </div>

      {/* 會議簡報 (.pptx / .docx / .pdf) 上傳區 */}
      <div className="bg-indigo-50/60 border border-dashed border-indigo-200 rounded-lg p-2 flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-4 h-4 text-indigo-600 shrink-0" />
          <div className="truncate">
            <p className="text-xs font-medium text-indigo-950">匯入會議簡報 (.pptx/.docx/.pdf)</p>
            {docContextSummary && (
              <p className="text-[10px] text-indigo-600 truncate">
                已載入背景摘要 ({docContextSummary.length}字)
              </p>
            )}
          </div>
        </div>
        <input
          ref={docInputRef}
          type="file"
          accept=".pptx,.docx,.pdf"
          onChange={handleDocUpload}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => docInputRef.current?.click()}
          disabled={isParsingDoc}
          className="text-xs bg-indigo-600 text-white px-2.5 py-1 rounded hover:bg-indigo-700 shrink-0 disabled:opacity-50"
        >
          {isParsingDoc ? '解析中...' : '上傳'}
        </button>
      </div>

      {/* 三層字典切換 Tabs */}
      <div className="grid grid-cols-3 gap-1 bg-slate-100 p-1 rounded-lg text-xs">
        <button
          type="button"
          onClick={() => setActiveTab('public')}
          className={`flex items-center justify-center gap-1 py-1 rounded font-medium transition-all ${
            isPublicTab ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-600 hover:text-slate-800'
          }`}
        >
          <Globe className="w-3 h-3" />
          <span>公共 ({publicList.length})</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('personal')}
          className={`flex items-center justify-center gap-1 py-1 rounded font-medium transition-all ${
            isPersonalTab
              ? 'bg-white text-emerald-700 shadow-sm'
              : 'text-slate-600 hover:text-slate-800'
          }`}
        >
          <User className="w-3 h-3" />
          <span>個人 ({personalList.length})</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('session')}
          className={`flex items-center justify-center gap-1 py-1 rounded font-medium transition-all ${
            isSessionTab ? 'bg-white text-amber-700 shadow-sm' : 'text-slate-600 hover:text-slate-800'
          }`}
        >
          <Sparkles className="w-3 h-3" />
          <span>單次 ({sessionList.length})</span>
        </button>
      </div>

      {/* 🌐 公共字典專屬：Google Sheet 雲端共編網址與 `?dict=` 網址參數自動掛載列 */}
      {isPublicTab && (
        <div className="bg-blue-50/50 border border-blue-200/80 rounded-lg p-2 space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-blue-900 font-medium">
            <span className="flex items-center gap-1">
              <Link2 className="w-3.5 h-3.5 text-blue-600" />
              Google Sheet 共編與 ?dict= 網址掛載
              <HoverInfoPopover
                title="如何共享與自動掛載 Google Sheet 字典？"
                align="left"
              >
                <p>
                  • <strong>直接輸入：</strong>可貼上完整 Google Sheet 網址或純 Sheet ID（須設為「知道連結的任何人可檢視」）。
                </p>
                <p>
                  • <strong>網址參數分享：</strong>點擊下方「複製分享連結」，會自動產生帶有 <code>?dict=&lt;SheetID&gt;</code> 的專屬網址，其他人開啟該網址即會自動載入同一份字典。
                </p>
                <p>
                  • <strong>表格欄位：</strong>支援「正確詞彙、分類、常見誤聽別名」三欄或「誤聽詞 → 正確詞」兩欄格式。
                </p>
              </HoverInfoPopover>
            </span>
            {resolvedEndpoint.canonicalViewUrl && (
              <a
                href={resolvedEndpoint.canonicalViewUrl}
                target="_blank"
                rel="noreferrer"
                className="text-[10px] text-blue-600 hover:underline"
              >
                開啟共編表 ↗
              </a>
            )}
          </div>

          <div className="flex gap-1">
            <input
              type="text"
              value={sheetUrl}
              onChange={(e) => {
                const val = e.target.value;
                setSheetUrl(val);
                try {
                  localStorage.setItem(GSHEET_STORAGE_KEY, val.trim());
                } catch {
                  // ignore
                }
              }}
              onPaste={(e) => {
                const pasted = e.clipboardData.getData('text').trim();
                if (
                  pasted &&
                  (pasted.includes('docs.google.com/spreadsheets') ||
                    /^[a-zA-Z0-9-_]{20,80}$/.test(pasted))
                ) {
                  setTimeout(() => handleSyncGoogleSheet(pasted, { silent: false }), 100);
                }
              }}
              placeholder="貼上 Google Sheet 網址或 Sheet ID (如 1ra9-ylq...)..."
              className="text-[11px] px-2 py-1 border border-blue-200 rounded flex-1 bg-white focus:outline-none focus:border-blue-500 min-w-0"
            />
            <button
              type="button"
              onClick={() => handleSyncGoogleSheet(sheetUrl, { silent: false })}
              disabled={isSyncingSheet || !sheetUrl.trim()}
              className="bg-blue-600 text-white px-2 py-1 rounded text-[11px] flex items-center gap-1 hover:bg-blue-700 disabled:opacity-40 shrink-0"
            >
              <RefreshCw className={`w-3 h-3 ${isSyncingSheet ? 'animate-spin' : ''}`} />
              同步
            </button>
          </div>

          {/* 快捷掛載與產生 `?dict=` 分享連結按鈕列 */}
          <div className="flex items-center justify-between gap-1 pt-0.5">
            <button
              type="button"
              onClick={() => {
                setSheetUrl(DEFAULT_REGIONAL_REVITALIZATION_SHEET_ID);
                handleSyncGoogleSheet(DEFAULT_REGIONAL_REVITALIZATION_SHEET_ID, {
                  silent: false,
                });
              }}
              disabled={isSyncingSheet}
              className="text-[10px] px-2 py-0.5 rounded bg-indigo-100/80 text-indigo-800 hover:bg-indigo-200/80 font-medium transition-colors truncate"
              title="一鍵載入《HybTransPlus 地方創生字典 (1ra9-ylqZrpBA5d6g0VLW_pENIYJnR1jICVjAi2oLu6E)》"
            >
              ⚡ 載入地方創生共編字典
            </button>

            <button
              type="button"
              onClick={handleCopyShareableDictUrl}
              className="text-[10px] px-2 py-0.5 rounded bg-white border border-blue-200 text-blue-700 hover:bg-blue-50 flex items-center gap-1 shrink-0 transition-colors"
              title="複製帶有 ?dict=<SheetID> 參數的分享連結，別人點開即自動預設載入此字典"
            >
              {copiedShareLink ? (
                <>
                  <Check className="w-3 h-3 text-emerald-600" />
                  <span className="text-emerald-700 font-medium">已複製 ?dict= 連結</span>
                </>
              ) : (
                <>
                  <Share2 className="w-3 h-3" />
                  <span>複製 ?dict= 分享連結</span>
                </>
              )}
            </button>
          </div>

          {urlMountedSheetId && (
            <div className="text-[10px] text-blue-700 bg-blue-100/60 px-2 py-0.5 rounded flex items-center justify-between">
              <span className="truncate">
                已掛載參數：<code>?dict={urlMountedSheetId}</code>
              </span>
            </div>
          )}
        </div>
      )}

      {/* JSON 匯出入小工具列 */}
      {!isSessionTab && (
        <div className="flex items-center justify-end gap-1.5 text-[11px] text-slate-500">
          <input
            ref={dictJsonInputRef}
            type="file"
            accept=".json,.csv,.txt"
            onChange={handleImportJson}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => dictJsonInputRef.current?.click()}
            className="px-1.5 py-0.5 hover:bg-slate-100 rounded text-slate-600 flex items-center gap-0.5"
          >
            <Upload className="w-3 h-3" />
            <span>匯入JSON/CSV</span>
          </button>
          <button
            type="button"
            onClick={() =>
              exportDictionaryToJsonFile(dictionary, isPublicTab ? 'public' : 'personal')
            }
            className="px-1.5 py-0.5 hover:bg-slate-100 rounded text-slate-600 flex items-center gap-0.5"
          >
            <Download className="w-3 h-3" />
            <span>匯出JSON</span>
          </button>
        </div>
      )}

      {/* 新增詞彙表單 */}
      <form
        onSubmit={handleAddTerm}
        className="space-y-1.5 bg-slate-50 p-2 rounded-lg border border-slate-200/80"
      >
        <div className="flex gap-1">
          <select
            value={newCategory}
            onChange={(e) => setNewCategory(e.target.value as DictCategory)}
            className="text-xs border border-slate-200 rounded px-1 py-1 bg-white text-slate-700"
          >
            <option value="jargon">專詞</option>
            <option value="person">人名</option>
            <option value="project">計畫</option>
          </select>
          <input
            type="text"
            value={newTerm}
            onChange={(e) => setNewTerm(e.target.value)}
            placeholder={`新增${isPublicTab ? '公共' : isPersonalTab ? '個人' : '單次'}詞彙...`}
            className="text-xs px-2 py-1 border border-slate-200 rounded flex-1 bg-white focus:outline-none focus:border-blue-400 min-w-0"
          />
          <button
            type="submit"
            className="bg-slate-800 text-white px-2 py-1 rounded text-xs flex items-center gap-0.5 hover:bg-slate-700 shrink-0"
          >
            <Plus className="w-3 h-3" />
            新增
          </button>
        </div>
        <input
          type="text"
          value={newAliases}
          onChange={(e) => setNewAliases(e.target.value)}
          placeholder="選填：同音錯字，以逗號或分號分隔 (如：臨處長; 凌處長)"
          className="w-full text-[11px] px-2 py-1 border border-slate-200 rounded bg-white focus:outline-none focus:border-blue-400"
        />
      </form>

      {statusMsg && (
        <div className="text-[11px] px-2 py-1 bg-emerald-50 text-emerald-800 rounded border border-emerald-200 break-all">
          {statusMsg}
        </div>
      )}

      {/* 詞彙列表 */}
      <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
        {currentList.length === 0 ? (
          <p className="text-xs text-slate-400 text-center py-3">尚無詞彙</p>
        ) : (
          currentList.map((item) => {
            const badge = categoryBadge(item.category);
            const hasAliases = Boolean(item.aliases && item.aliases.length > 0);
            return (
              <div
                key={item.id}
                className="flex items-center justify-between p-1.5 rounded-lg border border-slate-100 hover:border-slate-200 bg-slate-50/40 text-xs"
              >
                <div className="min-w-0 flex-1 pr-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] px-1 py-0.5 rounded font-medium ${badge.cls}`}>
                      {badge.label}
                    </span>
                    <span className="font-semibold text-slate-800 truncate">{item.term}</span>
                  </div>
                  {hasAliases && (
                    <p className="text-[10px] text-slate-400 truncate mt-0.5">
                      別名/錯字：{item.aliases!.join('、')}
                    </p>
                  )}
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  {isSessionTab && (
                    <button
                      type="button"
                      onClick={() => promoteDictEntryScope(item.id, 'personal')}
                      title="存入個人常駐字典"
                      className="text-[10px] px-1.5 py-0.5 bg-emerald-50 text-emerald-700 rounded hover:bg-emerald-100 flex items-center gap-0.5"
                    >
                      <ArrowUpRight className="w-2.5 h-2.5" />
                      存個人
                    </button>
                  )}
                  {isPersonalTab && (
                    <button
                      type="button"
                      onClick={() => promoteDictEntryScope(item.id, 'public')}
                      title="轉入公共字典"
                      className="text-[10px] px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded hover:bg-blue-100 flex items-center gap-0.5"
                    >
                      <ArrowUpRight className="w-2.5 h-2.5" />
                      轉公共
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => removeDictEntry(item.id)}
                    title="刪除"
                    className="p-1 text-slate-400 hover:text-red-600 rounded"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};