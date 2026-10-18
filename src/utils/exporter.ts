// src/utils/exporter.ts
import JSZip from 'jszip';
import { Segment, Speaker, WordSpan } from '../types';

function formatSrtTime(sec: number): string {
  const hrs = Math.floor(sec / 3600);
  const mins = Math.floor((sec % 3600) / 60);
  const secs = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 1000);
  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

function formatShortTime(sec: number): string {
  const hrs = Math.floor(sec / 3600);
  const mins = Math.floor((sec % 3600) / 60);
  const secs = Math.floor(sec % 60);
  return hrs > 0
    ? `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
    : `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function escapeXml(unsafe: string): string {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * 將單一 WordSpan 轉換為 OpenXML <w:r> 標籤，保留三態視覺標記：
 * - auto_corrected: 柔和綠底 (#D1FAE5) + 深綠字 (#065F46) + 括號備註原字
 * - semantic_uncertain: 醒目黃底 (#FEF08A) + 候選詞提示
 * - acoustic_uncertain: 橘色波浪底線 (w:u val="wave" color="EA580C")
 */
function wordSpanToOpenXmlRun(w: WordSpan): string {
  const safeText = escapeXml(w.text);

  if (w.status === 'auto_corrected') {
    return `
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei" w:hAnsi="Microsoft JhengHei"/>
          <w:color w:val="065F46"/>
          <w:shd w:val="clear" w:color="auto" w:fill="D1FAE5"/>
        </w:rPr>
        <w:t>${safeText}</w:t>
      </w:r>`;
  }

  if (w.status === 'semantic_uncertain') {
    const candNote = w.candidates?.length ? `[候選:${escapeXml(w.candidates.join('/'))}]` : '';
    return `
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei" w:hAnsi="Microsoft JhengHei"/>
          <w:b/>
          <w:color w:val="92400E"/>
          <w:shd w:val="clear" w:color="auto" w:fill="FEF08A"/>
        </w:rPr>
        <w:t>${safeText}${candNote}</w:t>
      </w:r>`;
  }

  if (w.status === 'acoustic_uncertain') {
    return `
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei" w:hAnsi="Microsoft JhengHei"/>
          <w:color w:val="9A3412"/>
          <w:u w:val="wave" w:color="EA580C"/>
        </w:rPr>
        <w:t>${safeText}</w:t>
      </w:r>`;
  }

  return `
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei" w:hAnsi="Microsoft JhengHei"/>
        <w:color w:val="1E293B"/>
      </w:rPr>
      <w:t>${safeText}</w:t>
    </w:r>`;
}

/**
 * 匯出包含「語者色彩標籤、時間戳記表格與三態校對色彩標記」之原生 Microsoft Word (.docx)
 */
export async function exportAsDocx(segments: Segment[], speakers: Record<string, Speaker>): Promise<void> {
  const zip = new JSZip();

  // 1. [Content_Types].xml
  zip.file(
    '[Content_Types].xml',
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default ContentType="application/vnd.openxmlformats-package.relationships+xml" Extension="rels"/>
  <Default ContentType="application/xml" Extension="xml"/>
  <Override ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml" PartName="/word/document.xml"/>
</Types>`
  );

  // 2. _rels/.rels
  zip.folder('_rels')?.file(
    '.rels',
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Target="word/document.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>
</Relationships>`
  );

  // 3. 組裝表格列 (每一段發言一列：左欄為語者與時間戳，右欄為含色彩標記之發言段落)
  const tableRowsXml = segments
    .map((seg) => {
      const sp = speakers[seg.speakerId];
      const spName = escapeXml(sp?.name || seg.speakerId);
      const spHex = (sp?.color || '#2563EB').replace('#', '').toUpperCase();
      const timeRange = `${formatShortTime(seg.start)} - ${formatShortTime(seg.end)}`;
      const runsXml = seg.words.map(wordSpanToOpenXmlRun).join('');

      return `
      <w:tr>
        <w:tc>
          <w:tcPr>
            <w:tcW w:w="2200" w:type="dxa"/>
            <w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/>
          </w:tcPr>
          <w:p>
            <w:r>
              <w:rPr>
                <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei"/>
                <w:b/>
                <w:color w:val="${spHex}"/>
                <w:sz w:val="22"/>
              </w:rPr>
              <w:t>${spName}</w:t>
            </w:r>
          </w:p>
          <w:p>
            <w:r>
              <w:rPr>
                <w:rFonts w:ascii="Calibri" w:eastAsia="Microsoft JhengHei"/>
                <w:color w:val="64748B"/>
                <w:sz w:val="18"/>
              </w:rPr>
              <w:t>${timeRange}</w:t>
            </w:r>
          </w:p>
        </w:tc>
        <w:tc>
          <w:tcPr>
            <w:tcW w:w="7300" w:type="dxa"/>
          </w:tcPr>
          <w:p>
            <w:pPr>
              <w:spacing w:line="320" w:lineRule="auto" w:after="120"/>
            </w:pPr>
            ${runsXml}
          </w:p>
        </w:tc>
      </w:tr>`;
    })
    .join('');

  const exportDateStr = new Date().toLocaleString('zh-TW');

  // 4. word/document.xml
  const documentXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:jc w:val="center"/><w:spacing w:after="200"/></w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei"/>
          <w:b/>
          <w:sz w:val="32"/>
          <w:color w:val="0F172A"/>
        </w:rPr>
        <w:t>會議逐字稿紀錄（含 AI 語境校正標記）</w:t>
      </w:r>
    </w:p>
    <w:p>
      <w:pPr><w:spacing w:after="240"/></w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei"/>
          <w:sz w:val="18"/>
          <w:color w:val="475569"/>
        </w:rPr>
        <w:t>匯出時間：${escapeXml(exportDateStr)} ｜ 色彩圖例：【綠底】AI 語境自動修正 【黃底】多候選詞待確認 【橘波浪底線】低聲學信心度</w:t>
      </w:r>
    </w:p>
    <w:tbl>
      <w:tblPr>
        <w:tblW w:w="9500" w:type="dxa"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:left w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
          <w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
        </w:tblBorders>
        <w:tblCellMar>
          <w:top w:w="140" w:type="dxa"/>
          <w:bottom w:w="140" w:type="dxa"/>
          <w:left w:w="180" w:type="dxa"/>
          <w:right w:w="180" w:type="dxa"/>
        </w:tblCellMar>
      </w:tblPr>
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="2200" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="1E293B"/></w:tcPr>
          <w:p><w:r><w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t>發言語者 / 時間軸</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="7300" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="1E293B"/></w:tcPr>
          <w:p><w:r><w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t>會議發言逐字稿內容</w:t></w:r></w:p>
        </w:tc>
      </w:tr>
      ${tableRowsXml}
    </w:tbl>
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>
    </w:sectPr>
  </w:body>
</w:document>`;

  zip.folder('word')?.file('document.xml', documentXml);

  const blob = await zip.generateAsync({
    type: 'blob',
    mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `會議逐字稿_${new Date().toISOString().slice(0, 10)}.docx`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportAsMarkdown(segments: Segment[], speakers: Record<string, Speaker>): void {
  const lines = segments.map((s) => {
    const spName = speakers[s.speakerId]?.name || s.speakerId;
    const timeTag = `[${formatSrtTime(s.start).slice(0, 8)}]`;
    const text = s.words.map((w) => w.text).join('');
    return `**${spName}** ${timeTag}：${text}`;
  });
  triggerDownload(lines.join('\n\n'), `會議逐字稿_${Date.now()}.md`, 'text/markdown;charset=utf-8');
}

export function exportAsSrt(segments: Segment[], speakers: Record<string, Speaker>): void {
  const blocks = segments.map((s, idx) => {
    const spName = speakers[s.speakerId]?.name || s.speakerId;
    const text = s.words.map((w) => w.text).join('');
    return `${idx + 1}\n${formatSrtTime(s.start)} --> ${formatSrtTime(s.end)}\n[${spName}] ${text}`;
  });
  triggerDownload(blocks.join('\n\n'), `會議字幕_${Date.now()}.srt`, 'text/plain;charset=utf-8');
}

export function exportProjectJson(segments: Segment[], speakers: Record<string, Speaker>): void {
  const payload = JSON.stringify({ exportedAt: new Date().toISOString(), speakers, segments }, null, 2);
  triggerDownload(payload, `會議專案備份_${Date.now()}.json`, 'application/json');
}

function triggerDownload(content: string, filename: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}