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
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { DictCategory, DictEntry } from '../../types';
import * as docParserMod from '../../utils/docParser';
import {
  exportDictionaryToJsonFile,
  fetchRemotePublicDictionary,
  parseDictionaryJsonFile,
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
    try {
      return localStorage.getItem(GSHEET_STORAGE_KEY) || '';
    } catch {
      return '';
    }
  });
  const [isSyncingSheet, setIsSyncingSheet] = useState(false);
  const [newTerm, setNewTerm] = useState('');
  const [newAliases, setNewAliases] = useState('');
  const [newCategory, setNewCategory] = useState<DictCategory>('jargon');
  const [isParsingDoc, setIsParsingDoc] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const docInputRef = useRef<HTMLInputElement | null>(null);
  const dictJsonInputRef = useRef<HTMLInputElement | null>(null);

  const handleSyncGoogleSheet = useCallback(
    async (targetUrl: string, silent = false) => {
      const trimmed = targetUrl.trim();
      if (!trimmed) return;

      setIsSyncingSheet(true);
      if (!silent) setStatusMsg(null);

      try {
        localStorage.setItem(GSHEET_STORAGE_KEY, trimmed);
        const syncedEntries = await syncDictionaryFromGoogleSheet(trimmed);
        if (syncedEntries.length > 0) {
          addDictEntries(syncedEntries);
          if (!silent) {
            setStatusMsg(`已從 Google Sheet 同步 ${syncedEntries.length} 筆共編詞彙！`);
          }
        } else if (!silent) {
          setStatusMsg('Google Sheet 中尚無有效詞彙列。');
        }
      } catch (err) {
        if (!silent) {
          setStatusMsg((err as Error).message);
        }
      } finally {
        setIsSyncingSheet(false);
      }
    },
    [addDictEntries]
  );

  // 啟動時自動載入預設公共字典，若曾記憶過 Google Sheet 網址則自動背景同步！
  useEffect(() => {
    fetchRemotePublicDictionary().then((pubEntries) => {
      if (pubEntries.length > 0) {
        addDictEntries(pubEntries);
      }
    });

    const savedUrl = localStorage.getItem(GSHEET_STORAGE_KEY);
    if (savedUrl && savedUrl.includes('docs.google.com/spreadsheets')) {
      handleSyncGoogleSheet(savedUrl, true);
    }
  }, [addDictEntries, handleSyncGoogleSheet]);

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

  const handleAddTerm = (e: React.FormEvent) => {
    e.preventDefault();
    const termTrimmed = newTerm.trim();
    if (!termTrimmed) return;

    const aliasList = newAliases
      .split(/[,，、\s]+/)
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
          <span
            title="公共字典支援 Google Sheet 團隊共編自動同步；個人字典儲存在您的瀏覽器中；單次字典由會議簡報自動萃取"
            className="text-slate-400 hover:text-slate-600 cursor-help"
          >
            <Info className="w-3.5 h-3.5" />
          </span>
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

      {/* 🌐 公共字典專屬：Google Sheet 雲端共編網址自動同步列 */}
      {isPublicTab && (
        <div className="bg-blue-50/50 border border-blue-200/80 rounded-lg p-2 space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-blue-900 font-medium">
            <span className="flex items-center gap-1">
              <Link2 className="w-3.5 h-3.5 text-blue-600" />
              Google Sheet 共編字典同步
              <span
                title="將 Google 試算表共用設為「知道連結的任何人皆可檢視」，貼上網址即可自動同步。試算表格式：A欄=正確詞彙、B欄=常見同音錯字(選填)、C欄=分類(人名/計畫/專詞，選填)。網址會自動記憶於瀏覽器中。"
                className="text-blue-400 hover:text-blue-600 cursor-help"
              >
                <Info className="w-3 h-3" />
              </span>
            </span>
            {sheetUrl && (
              <a
                href={sheetUrl}
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
              type="url"
              value={sheetUrl}
              onChange={(e) => {
                const val = e.target.value;
                setSheetUrl(val);
                localStorage.setItem(GSHEET_STORAGE_KEY, val.trim());
              }}
              onPaste={(e) => {
                const pasted = e.clipboardData.getData('text');
                if (pasted && pasted.includes('docs.google.com/spreadsheets')) {
                  setTimeout(() => handleSyncGoogleSheet(pasted, false), 100);
                }
              }}
              placeholder="貼上 Google Sheet 網址自動同步並記憶..."
              className="text-[11px] px-2 py-1 border border-blue-200 rounded flex-1 bg-white focus:outline-none focus:border-blue-500"
            />
            <button
              type="button"
              onClick={() => handleSyncGoogleSheet(sheetUrl, false)}
              disabled={isSyncingSheet || !sheetUrl.trim()}
              className="bg-blue-600 text-white px-2 py-1 rounded text-[11px] flex items-center gap-1 hover:bg-blue-700 disabled:opacity-40 shrink-0"
            >
              <RefreshCw className={`w-3 h-3 ${isSyncingSheet ? 'animate-spin' : ''}`} />
              同步
            </button>
          </div>
        </div>
      )}

      {/* JSON 匯出入小工具列 */}
      {!isSessionTab && (
        <div className="flex items-center justify-end gap-1.5 text-[11px] text-slate-500">
          <input
            ref={dictJsonInputRef}
            type="file"
            accept=".json"
            onChange={handleImportJson}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => dictJsonInputRef.current?.click()}
            className="px-1.5 py-0.5 hover:bg-slate-100 rounded text-slate-600 flex items-center gap-0.5"
          >
            <Upload className="w-3 h-3" />
            <span>匯入JSON</span>
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
          placeholder="選填：同音錯字，以逗號分隔 (如：臨處長, 凌處長)"
          className="w-full text-[11px] px-2 py-1 border border-slate-200 rounded bg-white focus:outline-none focus:border-blue-400"
        />
      </form>

      {statusMsg && (
        <div className="text-[11px] px-2 py-1 bg-emerald-50 text-emerald-800 rounded border border-emerald-200">
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
                      錯字修正：{item.aliases!.join('、')}
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