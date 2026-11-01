#!/usr/bin/env bun
// Splits the interface font into unicode-range slices so the webview fetches only the slices
// whose characters it renders. Every slice keeps all OpenType layout features, IVS sequences
// and the weight axis, so `lang`-driven CJK glyph variants still resolve inside each slice.
// Requires HarfBuzz's hb-info and hb-subset (for example `brew install harfbuzz`).
// bun scripts/subset_ui_font.ts

import { execFileSync } from 'node:child_process'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'

const FONTS = path.resolve(import.meta.dirname, '../packages/koharu/app/fonts')
const SOURCE = path.join(FONTS, 'NotoSansCJK-VF.otf')
const SLICES = 'noto-sans-cjk'
const FAMILY = 'Noto Sans CJK'
// The source is a collection; browsers load its first face (Japanese default forms).
const FACE_INDEX = 0
// Bounds what one rare character costs to fetch.
const MAX_SLICE_CODE_POINTS = 3000

type Range = readonly [number, number]

const HANGUL: Range[] = [
  [0x1100, 0x11ff],
  [0x3130, 0x318f],
  [0xa960, 0xa97f],
  [0xac00, 0xd7ff],
  [0xffa0, 0xffdc],
]
const HAN: Range[] = [
  [0x2e80, 0x2fdf],
  [0x3400, 0x4dbf],
  [0x4e00, 0x9fff],
  [0xf900, 0xfaff],
  [0x20000, 0x3ffff],
]
const CJK_SYMBOLS: Range[] = [
  [0x2ff0, 0x33ff],
  [0xfe10, 0xfe1f],
  [0xfe30, 0xfe6f],
  [0xff00, 0xffef],
  [0x1b000, 0x1b16f],
  [0x1f200, 0x1f2ff],
]
// Ideographic variation sequences are shaped with their base character.
const VARIATION_SELECTORS = [...span([0xfe00, 0xfe0f]), ...span([0xe0100, 0xe01ef])]

function span([start, end]: Range): number[] {
  return Array.from({ length: end - start + 1 }, (_, offset) => start + offset)
}

function within(ranges: Range[], codePoint: number): boolean {
  return ranges.some(([start, end]) => start <= codePoint && codePoint <= end)
}

// Characters a legacy double-byte encoding assigns to the given lead and trail bytes.
function charset(encoding: string, leads: Range, trails: Range[]): Set<number> {
  const decoder = new TextDecoder(encoding, { fatal: true })
  const codePoints = new Set<number>()
  for (const lead of span(leads)) {
    for (const trail of trails.flatMap(span)) {
      try {
        const text = decoder.decode(new Uint8Array([lead, trail]))
        if (Array.from(text).length === 1) codePoints.add(text.codePointAt(0)!)
      } catch {
        // Unassigned code.
      }
    }
  }
  return codePoints
}

const EUC_TRAILS: Range[] = [[0xa1, 0xfe]]
const BIG5_TRAILS: Range[] = [
  [0x40, 0x7e],
  [0xa1, 0xfe],
]

// National character sets list everyday characters first (KS X 1001 Hangul, JIS X 0208 level 1,
// GB 2312 level 1, frequent Big5), so text in one CJK language stays in a few slices. A code
// point joins the first tier that lists it; Hangul and Han outside every tier come last.
const TIERS: Array<[string, Range[], Set<number>]> = [
  ['hangul-ks', HANGUL, charset('euc-kr', [0xa4, 0xc8], EUC_TRAILS)],
  ['han-jis1', HAN, charset('euc-jp', [0xb0, 0xcf], EUC_TRAILS)],
  ['han-gb1', HAN, charset('gbk', [0xb0, 0xd7], EUC_TRAILS)],
  ['han-big5a', HAN, charset('big5', [0xa4, 0xc6], BIG5_TRAILS)],
  ['han-jis2', HAN, charset('euc-jp', [0xd0, 0xf4], EUC_TRAILS)],
  ['han-gb2', HAN, charset('gbk', [0xd8, 0xf7], EUC_TRAILS)],
  ['han-big5b', HAN, charset('big5', [0xc9, 0xf9], BIG5_TRAILS)],
]

function fontCodePoints(): number[] {
  const listing = execFileSync(
    'hb-info',
    [`--face-index=${FACE_INDEX}`, '--list-unicodes', SOURCE],
    { encoding: 'utf8', maxBuffer: 64 << 20 },
  )
  return Array.from(listing.matchAll(/^U\+([0-9A-F]+)\t/gm), ([, hex]) => parseInt(hex, 16))
}

function slices(codePoints: number[]): Array<[string, number[]]> {
  const groups = new Map<string, number[]>()
  const add = (name: string, codePoint: number) => {
    if (!groups.has(name)) groups.set(name, [])
    groups.get(name)!.push(codePoint)
  }
  for (const codePoint of codePoints) {
    const tier = TIERS.find(([, ranges, set]) => within(ranges, codePoint) && set.has(codePoint))
    if (tier) add(tier[0], codePoint)
    else if (within(HANGUL, codePoint)) add('hangul-other', codePoint)
    else if (within(HAN, codePoint)) add('han-other', codePoint)
    else if (within(CJK_SYMBOLS, codePoint)) add('cjk-symbols', codePoint)
    else add('base', codePoint)
  }
  const order = ['base', 'cjk-symbols', ...TIERS.map(([name]) => name), 'hangul-other', 'han-other']
  return order.flatMap((name) => {
    const group = (groups.get(name) ?? []).sort((a, b) => a - b)
    const count = Math.ceil(group.length / MAX_SLICE_CODE_POINTS)
    if (count <= 1) return group.length ? [[name, group] as [string, number[]]] : []
    const size = Math.ceil(group.length / count)
    return Array.from({ length: count }, (_, index): [string, number[]] => [
      `${name}-${index + 1}`,
      group.slice(index * size, (index + 1) * size),
    ])
  })
}

// Formatted as oxfmt prints it, so the generated stylesheet passes the format check.
function unicodeRange(codePoints: number[]): string {
  const ranges: Array<[number, number]> = []
  for (const codePoint of codePoints) {
    const last = ranges.at(-1)
    if (last && codePoint === last[1] + 1) last[1] = codePoint
    else ranges.push([codePoint, codePoint])
  }
  const hex = (codePoint: number) => codePoint.toString(16).toUpperCase()
  const items = ranges.map(([start, end]) =>
    start === end ? `U+${hex(start)}` : `U+${hex(start)}-${hex(end)}`,
  )
  const declaration = `  unicode-range: ${items.join(', ')};`
  if (declaration.length <= 100) return declaration
  const lines = ['  unicode-range:']
  items.forEach((item, index) => {
    const entry = `${item}${index === items.length - 1 ? ';' : ','}`
    const line = lines.at(-1)!
    if (index > 0 && line.length + 1 + entry.length <= 100) lines[lines.length - 1] += ` ${entry}`
    else lines.push(`    ${entry}`)
  })
  return lines.join('\n')
}

const output = path.join(FONTS, SLICES)
await rm(output, { recursive: true, force: true })
await mkdir(output)

const rules: string[] = []
for (const [name, codePoints] of slices(fontCodePoints())) {
  const file = `${SLICES}/${name}.otf`
  const subset = name.startsWith('han') ? [...codePoints, ...VARIATION_SELECTORS] : codePoints
  execFileSync('hb-subset', [
    `--face-index=${FACE_INDEX}`,
    `--font-file=${SOURCE}`,
    `--unicodes=${subset.map((codePoint) => codePoint.toString(16)).join(',')}`,
    '--layout-features=*',
    `--output-file=${path.join(FONTS, file)}`,
  ])
  rules.push(`@font-face {
  font-family: '${FAMILY}';
  font-weight: 100 900;
  font-display: swap;
  src: url('./${file}') format('opentype');
${unicodeRange(codePoints)}
}
`)
}

await writeFile(
  path.join(FONTS, `${SLICES}.css`),
  `/* Generated by scripts/subset_ui_font.ts from NotoSansCJK-VF.otf. Do not edit. */\n\n${rules.join('\n')}`,
)
