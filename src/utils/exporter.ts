// src/utils/exporter.ts
import JSZip from 'jszip';
import { MeetingMinutesReport, Segment, Speaker, WordSpan } from '../types';

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
 * 將會議 AI 摘要、結論與行動事項格式化為精煉條列式 Markdown 字串
 */
export function formatMeetingMinutesToMarkdown(
  report: MeetingMinutesReport
): string {
  const lines: string[] = [];
  lines.push(`# ${report.title}`);
  lines.push(
    `> 產生時間：${new Date(report.generatedAt).toLocaleString('zh-TW')} ｜ 與會語者：${report.speakerStats.length} 位`
  );
  lines.push('');

  lines.push('## 一、會議討論要點摘要');
  const summaryBullets = (report.executiveSummary || '')
    .split(/\n+/)
    .map((s) => s.replace(/^[•\-*]\s*/, '').trim())
    .filter(Boolean);
  for (const b of summaryBullets) {
    lines.push(`- ${b}`);
  }
  if (report.topicBreakdowns.length > 0) {
    for (const t of report.topicBreakdowns) {
      const timeTag = t.timeRange ? ` (${t.timeRange})` : '';
      const detail =
        t.decision && t.decision !== t.discussionPoints
          ? `${t.discussionPoints} ➔ **${t.decision}**`
          : t.discussionPoints;
      lines.push(`- **【${t.topic}】**${timeTag}：${detail}`);
    }
  }
  lines.push('');

  lines.push('## 二、重要結論與裁示');
  if (report.keyConclusions.length > 0) {
    report.keyConclusions.forEach((c) => {
      lines.push(`- ${c.replace(/^[•\-*]\s*/, '')}`);
    });
  } else {
    lines.push('- 依各案討論進度持續推進。');
  }
  lines.push('');

  lines.push('## 三、後續行動事項 (Action Items)');
  if (report.actionItems.length > 0) {
    report.actionItems.forEach((item) => {
      lines.push(
        `- **[${item.owner}｜${item.deadline}]** ${item.task}`
      );
    });
  } else {
    lines.push('- （未偵測到待辦行動事項）');
  }
  lines.push('');

  return lines.join('\n');
}

/**
 * 獨立匯出「會議 AI 摘要、結論與行動事項報告 (.md)」
 */
export function exportMeetingMinutesAsMarkdown(
  report: MeetingMinutesReport
): void {
  const md = formatMeetingMinutesToMarkdown(report);
  triggerDownload(
    md,
    `會議摘要與行動事項_${new Date().toISOString().slice(0, 10)}.md`,
    'text/markdown;charset=utf-8'
  );
}

/**
 * 將結構化會議摘要轉為精煉條列式 OpenXML 段落區段（供單獨匯出或附加於逐字稿 Word 檔頂部）
 */
function buildMeetingMinutesOpenXmlSection(
  report: MeetingMinutesReport
): string {
  const summaryBullets = (report.executiveSummary || '')
    .split(/\n+/)
    .map((s) => s.replace(/^[•\-*]\s*/, '').trim())
    .filter(Boolean);

  const summaryXml = summaryBullets
    .map(
      (b) => `
    <w:p>
      <w:pPr><w:spacing w:after="60"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:color w:val="1E40AF"/><w:sz w:val="21"/></w:rPr>
        <w:t>• </w:t>
      </w:r>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:color w:val="1E293B"/><w:sz w:val="21"/></w:rPr>
        <w:t>${escapeXml(b)}</w:t>
      </w:r>
    </w:p>`
    )
    .join('');

  const topicBulletsXml = report.topicBreakdowns
    .map((t) => {
      const timeTag = t.timeRange ? ` (${t.timeRange})` : '';
      const detail =
        t.decision && t.decision !== t.discussionPoints
          ? `${t.discussionPoints} ➔ ${t.decision}`
          : t.discussionPoints;
      return `
    <w:p>
      <w:pPr><w:spacing w:after="60"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:color w:val="0F172A"/><w:sz w:val="20"/></w:rPr>
        <w:t>• 【${escapeXml(t.topic)}】${escapeXml(timeTag)}：</w:t>
      </w:r>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:color w:val="334155"/><w:sz w:val="20"/></w:rPr>
        <w:t>${escapeXml(detail)}</w:t>
      </w:r>
    </w:p>`;
    })
    .join('');

  const conclusionsXml = report.keyConclusions
    .map(
      (c) => `
    <w:p>
      <w:pPr><w:spacing w:after="60"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:color w:val="065F46"/><w:sz w:val="20"/></w:rPr>
        <w:t>• </w:t>
      </w:r>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:color w:val="0F172A"/><w:sz w:val="20"/></w:rPr>
        <w:t>${escapeXml(c.replace(/^[•\-*]\s*/, ''))}</w:t>
      </w:r>
    </w:p>`
    )
    .join('');

  const actionBulletsXml = report.actionItems
    .map(
      (a) => `
    <w:p>
      <w:pPr><w:spacing w:after="60"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:color w:val="B45309"/><w:sz w:val="20"/></w:rPr>
        <w:t>• [${escapeXml(a.owner)}｜${escapeXml(a.deadline)}] </w:t>
      </w:r>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:color w:val="0F172A"/><w:sz w:val="20"/></w:rPr>
        <w:t>${escapeXml(a.task)}</w:t>
      </w:r>
    </w:p>`
    )
    .join('');

  return `
    <w:p>
      <w:pPr><w:spacing w:before="120" w:after="80"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:sz w:val="24"/><w:color w:val="1E3A8A"/></w:rPr>
        <w:t>壹、會議討論要點摘要</w:t>
      </w:r>
    </w:p>
    ${summaryXml}
    ${topicBulletsXml}
    <w:p>
      <w:pPr><w:spacing w:before="140" w:after="80"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:sz w:val="24"/><w:color w:val="065F46"/></w:rPr>
        <w:t>貳、重要結論與裁示</w:t>
      </w:r>
    </w:p>
    ${conclusionsXml}
    <w:p>
      <w:pPr><w:spacing w:before="140" w:after="80"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:sz w:val="24"/><w:color w:val="B45309"/></w:rPr>
        <w:t>參、後續行動事項 (Action Items)</w:t>
      </w:r>
    </w:p>
    ${actionBulletsXml}
  `;
}

/**
 * 獨立匯出「會議 AI 摘要、結論與行動事項報告 (.docx)」
 */
export async function exportMeetingMinutesAsDocx(
  report: MeetingMinutesReport
): Promise<void> {
  const zip = new JSZip();

  zip.file(
    '[Content_Types].xml',
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default ContentType="application/vnd.openxmlformats-package.relationships+xml" Extension="rels"/>
  <Default ContentType="application/xml" Extension="xml"/>
  <Override ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml" PartName="/word/document.xml"/>
</Types>`
  );

  zip.folder('_rels')?.file(
    '.rels',
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Target="word/document.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>
</Relationships>`
  );

  const exportDateStr = new Date(report.generatedAt).toLocaleString('zh-TW');
  const sectionXml = buildMeetingMinutesOpenXmlSection(report);

  const documentXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:jc w:val="center"/><w:spacing w:after="140"/></w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei"/>
          <w:b/>
          <w:sz w:val="30"/>
          <w:color w:val="0F172A"/>
        </w:rPr>
        <w:t>${escapeXml(report.title)}</w:t>
      </w:r>
    </w:p>
    <w:p>
      <w:pPr><w:jc w:val="center"/><w:spacing w:after="200"/></w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei"/>
          <w:sz w:val="18"/>
          <w:color w:val="475569"/>
        </w:rPr>
        <w:t>產生時間：${escapeXml(exportDateStr)} ｜ 與會語者數：${report.speakerStats.length} 位</w:t>
      </w:r>
    </w:p>
    ${sectionXml}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>
    </w:sectPr>
  </w:body>
</w:document>`;

  zip.folder('word')?.file('document.xml', documentXml);

  const blob = await zip.generateAsync({
    type: 'blob',
    mimeType:
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `會議摘要與行動事項_${new Date().toISOString().slice(0, 10)}.docx`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * 將單一 WordSpan 轉換為 OpenXML <w:r> 標籤
 * 【修正重點】：移除 `[候選:...]` 括號附註文字，保持 Word 逐字稿內文乾淨流暢不干擾閱讀！
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
    return `
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Microsoft JhengHei" w:eastAsia="Microsoft JhengHei" w:hAnsi="Microsoft JhengHei"/>
          <w:color w:val="92400E"/>
          <w:shd w:val="clear" w:color="auto" w:fill="FEF9C3"/>
        </w:rPr>
        <w:t>${safeText}</w:t>
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
 * 匯出包含「(選配) 會議 AI 摘要 + 語者色彩標籤、時間戳記表格與乾淨逐字稿」之原生 Microsoft Word (.docx)
 */
export async function exportAsDocx(
  segments: Segment[],
  speakers: Record<string, Speaker>,
  meetingMinutes?: MeetingMinutesReport | null
): Promise<void> {
  const zip = new JSZip();

  zip.file(
    '[Content_Types].xml',
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default ContentType="application/vnd.openxmlformats-package.relationships+xml" Extension="rels"/>
  <Default ContentType="application/xml" Extension="xml"/>
  <Override ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml" PartName="/word/document.xml"/>
</Types>`
  );

  zip.folder('_rels')?.file(
    '.rels',
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Target="word/document.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>
</Relationships>`
  );

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
  const minutesSectionXml = meetingMinutes
    ? `${buildMeetingMinutesOpenXmlSection(meetingMinutes)}
       <w:p><w:pPr><w:spacing w:before="240" w:after="140"/></w:pPr><w:r><w:rPr><w:rFonts w:eastAsia="Microsoft JhengHei"/><w:b/><w:sz w:val="24"/><w:color w:val="0F172A"/></w:rPr><w:t>肆、會議完整逐字稿紀錄</w:t></w:r></w:p>`
    : '';

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
        <w:t>${escapeXml(meetingMinutes?.title || '會議逐字稿紀錄')}</w:t>
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
        <w:t>匯出時間：${escapeXml(exportDateStr)}</w:t>
      </w:r>
    </w:p>
    ${minutesSectionXml}
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
    mimeType:
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `會議逐字稿_${new Date().toISOString().slice(0, 10)}.docx`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportAsMarkdown(
  segments: Segment[],
  speakers: Record<string, Speaker>,
  meetingMinutes?: MeetingMinutesReport | null
): void {
  const headerBlock = meetingMinutes
    ? `${formatMeetingMinutesToMarkdown(meetingMinutes)}\n---\n\n## 完整會議逐字稿\n\n`
    : '';
  const lines = segments.map((s) => {
    const spName = speakers[s.speakerId]?.name || s.speakerId;
    const timeTag = `[${formatSrtTime(s.start).slice(0, 8)}]`;
    const text = s.words.map((w) => w.text).join('');
    return `**${spName}** ${timeTag}：${text}`;
  });
  triggerDownload(
    headerBlock + lines.join('\n\n'),
    `會議逐字稿_${Date.now()}.md`,
    'text/markdown;charset=utf-8'
  );
}

export function exportAsSrt(
  segments: Segment[],
  speakers: Record<string, Speaker>
): void {
  const blocks = segments.map((s, idx) => {
    const spName = speakers[s.speakerId]?.name || s.speakerId;
    const text = s.words.map((w) => w.text).join('');
    return `${idx + 1}\n${formatSrtTime(s.start)} --> ${formatSrtTime(s.end)}\n[${spName}] ${text}`;
  });
  triggerDownload(
    blocks.join('\n\n'),
    `會議字幕_${Date.now()}.srt`,
    'text/plain;charset=utf-8'
  );
}

export function exportProjectJson(
  segments: Segment[],
  speakers: Record<string, Speaker>,
  meetingMinutes?: MeetingMinutesReport | null
): void {
  const payload = JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      meetingMinutes: meetingMinutes || null,
      speakers,
      segments,
    },
    null,
    2
  );
  triggerDownload(
    payload,
    `會議專案備份_${Date.now()}.json`,
    'application/json'
  );
}

function triggerDownload(
  content: string,
  filename: string,
  mime: string
): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export const exportToDocx = exportAsDocx;
export const exportToMarkdown = exportAsMarkdown;
export const exportToSrt = exportAsSrt;
export const downloadTextFile = triggerDownload;
