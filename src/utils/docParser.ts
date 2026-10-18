import JSZip from 'jszip';
import mammoth from 'mammoth';
import * as pdfjsLib from 'pdfjs-dist';
import { DictEntry } from '../types';

// 設定 PDF.js Worker (適配 Vite 6 + Chrome)
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

export async function parseMeetingDocument(file: File): Promise<{
  summaryText: string;
  extractedTerms: DictEntry[];
}> {
  const ext = file.name.split('.').pop()?.toLowerCase();
  let fullText = '';

  if (ext === 'docx') {
    const arrayBuffer = await file.arrayBuffer();
    const res = await mammoth.extractRawText({ arrayBuffer });
    fullText = res.value;
  } else if (ext === 'pptx') {
    const zip = await JSZip.loadAsync(file);
    const slideFiles = Object.keys(zip.files).filter((k) => /^ppt\/slides\/slide\d+\.xml$/.test(k));
    const parser = new DOMParser();
    for (const slidePath of slideFiles) {
      const xmlStr = await zip.files[slidePath].async('string');
      const xmlDoc = parser.parseFromString(xmlStr, 'application/xml');
      const textNodes = Array.from(xmlDoc.getElementsByTagName('a:t'));
      fullText += textNodes.map((n) => n.textContent).join(' ') + '\n';
    }
  } else if (ext === 'pdf') {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    const maxPages = Math.min(pdf.numPages, 30);
    for (let i = 1; i <= maxPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      fullText += content.items.map((item: any) => item.str).join(' ') + '\n';
    }
  }

  // 啟發式規則萃取：職稱姓名（如：X處長、X委員、X科長）與引號內計畫專有名詞
  const titleRegex = /[\u4e00-\u9fa5]{1,3}(?:處長|副處長|委員|科長|專委|技正|主秘|教授|總經理|經理|博士)/g;
  const quoteRegex = /[「『《【]([\u4e00-\u9fa5A-Za-z0-9\-_]{2,16})[」』》】]/g;

  const personMatches = Array.from(new Set(fullText.match(titleRegex) || []));
  const projectMatches = Array.from(new Set(Array.from(fullText.matchAll(quoteRegex), (m) => m[1])));

  const extractedTerms: DictEntry[] = [
    ...personMatches.map((term, i) => ({
      id: `doc-p-${Date.now()}-${i}`,
      term,
      category: 'person' as const,
      scope: 'session' as const,
    })),
    ...projectMatches.map((term, i) => ({
      id: `doc-j-${Date.now()}-${i}`,
      term,
      category: 'project' as const,
      scope: 'session' as const,
    })),
  ];

  const summaryText = fullText.replace(/\s+/g, ' ').slice(0, 1500);
  return { summaryText, extractedTerms };
}