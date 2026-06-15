'use strict';

const fs = require('fs');
const path = require('path');
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  AlignmentType,
  PageBreak,
  HeadingLevel,
  BorderStyle,
  ShadingType,
  WidthType,
  Table,
  TableRow,
  TableCell,
  Header,
  Footer,
} = require('docx');

const FONT_BODY = 'Georgia';
const FONT_MATH = 'Cambria Math';
const SIZE_BODY = 24; // 12pt in half-points

function richRuns(parts) {
  if (!Array.isArray(parts)) {
    return [new TextRun({ text: String(parts), size: SIZE_BODY, font: FONT_BODY })];
  }
  return parts.map((part) => {
    if (typeof part === 'string') {
      return new TextRun({ text: part, size: SIZE_BODY, font: FONT_BODY });
    }
    return new TextRun({
      text: part.text ?? '',
      bold: part.bold ?? false,
      italics: part.italics ?? false,
      size: part.size ?? SIZE_BODY,
      font: part.font ?? FONT_BODY,
      color: part.color,
    });
  });
}

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 240 },
    children: [new TextRun({ text, bold: true, size: 32, font: FONT_BODY })],
  });
}

function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 180 },
    children: [new TextRun({ text, bold: true, size: 28, font: FONT_BODY })],
  });
}

function H3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, bold: true, size: 26, font: FONT_BODY })],
  });
}

function P(text, style = {}) {
  if (typeof text === 'string') {
    return new Paragraph({
      spacing: { after: 200, line: 360 },
      alignment: AlignmentType.JUSTIFIED,
      children: [
        new TextRun({
          text,
          size: style.size ?? SIZE_BODY,
          font: style.font ?? FONT_BODY,
          bold: style.bold ?? false,
          italics: style.italics ?? false,
          color: style.color,
        }),
      ],
    });
  }
  return PR(...(Array.isArray(text) ? text : [text]));
}

function PR(...parts) {
  return new Paragraph({
    spacing: { after: 200, line: 360 },
    alignment: AlignmentType.JUSTIFIED,
    children: richRuns(parts),
  });
}

function EQ(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [new TextRun({ text, size: 26, font: FONT_MATH })],
  });
}

function EQ_NUM(text, num) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [
      new TextRun({ text: `${text}    (${num})`, size: 26, font: FONT_MATH }),
    ],
  });
}

function CAPTION(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 240 },
    children: [
      new TextRun({ text, italics: true, size: 20, font: FONT_BODY, color: '555555' }),
    ],
  });
}

function PAGEBREAK() {
  return new Paragraph({ children: [new PageBreak()] });
}

function monoParagraph(line, size = 20) {
  return new Paragraph({
    spacing: { after: 0, line: 240 },
    children: [new TextRun({ text: line, font: 'Consolas', size })],
  });
}

function DIAGRAM(lines, caption) {
  const out = [
    new Paragraph({ spacing: { before: 120 }, children: [] }),
    ...lines.map((line) => monoParagraph(line)),
  ];
  if (caption) out.push(CAPTION(caption));
  return out;
}

function BOX(children, opts = {}) {
  const cell = new TableCell({
    width: { size: 100, type: WidthType.PERCENTAGE },
    shading: opts.shaded
      ? { type: ShadingType.CLEAR, fill: opts.fill ?? 'F2F2F2' }
      : undefined,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: 'AAAAAA' },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: 'AAAAAA' },
      left: { style: BorderStyle.SINGLE, size: 1, color: 'AAAAAA' },
      right: { style: BorderStyle.SINGLE, size: 1, color: 'AAAAAA' },
    },
    children: Array.isArray(children) ? children : [children],
  });
  return [
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [new TableRow({ children: [cell] })],
    }),
    new Paragraph({ spacing: { after: 200 }, children: [] }),
  ];
}

function BULLET(text) {
  return new Paragraph({
    spacing: { after: 120, line: 320 },
    bullet: { level: 0 },
    children: richRuns(typeof text === 'string' ? [text] : text),
  });
}

function NUMBERED(text) {
  return new Paragraph({
    spacing: { after: 120, line: 320 },
    numbering: { reference: 'book-numbered', level: 0 },
    children: richRuns(typeof text === 'string' ? [text] : text),
  });
}

function calloutBox(title, body, borderColor) {
  return BOX([
    new Paragraph({
      spacing: { after: 120 },
      children: [new TextRun({ text: title, bold: true, size: 22, font: FONT_BODY })],
    }),
    new Paragraph({
      spacing: { line: 360, after: 0 },
      alignment: AlignmentType.JUSTIFIED,
      children: richRuns(typeof body === 'string' ? [body] : body),
    }),
  ], { shaded: true, fill: 'FAFAFA' });
}

function OPEN_Q(question, answer) {
  return calloutBox('Open Question', [
    { text: question, bold: true },
    '\n\n',
    answer,
  ]);
}

function PREDICTION(text) {
  return calloutBox('Testable Prediction', text);
}

function flattenChildren(items) {
  const out = [];
  for (const item of items) {
    if (Array.isArray(item)) out.push(...flattenChildren(item));
    else out.push(item);
  }
  return out;
}

function buildDoc(title, subtitle, children) {
  return new Document({
    numbering: {
      config: [
        {
          reference: 'book-numbered',
          levels: [
            {
              level: 0,
              format: 'decimal',
              text: '%1.',
              alignment: AlignmentType.START,
            },
          ],
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
          },
        },
        headers: {
          default: new Header({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({
                    text: subtitle || title,
                    italics: true,
                    size: 18,
                    font: FONT_BODY,
                    color: '888888',
                  }),
                ],
              }),
            ],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({
                    text: 'Packets and Strings',
                    size: 18,
                    font: FONT_BODY,
                    color: '888888',
                  }),
                ],
              }),
            ],
          }),
        },
        children: flattenChildren(children),
      },
    ],
  });
}

async function saveDoc(doc, outPath) {
  const resolved = path.isAbsolute(outPath)
    ? outPath
    : path.join(__dirname, '..', outPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(resolved, buffer);
  console.log(`Wrote ${resolved}`);
}

module.exports = {
  buildDoc,
  saveDoc,
  H1,
  H2,
  H3,
  P,
  PR,
  EQ,
  EQ_NUM,
  CAPTION,
  PAGEBREAK,
  BOX,
  DIAGRAM,
  BULLET,
  NUMBERED,
  OPEN_Q,
  PREDICTION,
  TextRun,
  Paragraph,
  AlignmentType,
  PageBreak,
};
