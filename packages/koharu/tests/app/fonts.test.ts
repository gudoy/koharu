import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { resources } from '@/lib/i18n'

const fontsDir = path.resolve(import.meta.dirname, '../../app/fonts')
const source = path.join(fontsDir, 'NotoSansCJK-VF.otf')
const stylesheet = readFileSync(path.join(fontsDir, 'noto-sans-cjk.css'), 'utf8')

function uiFontFaces() {
  return Array.from(stylesheet.matchAll(/@font-face\s*{([^}]*)}/g), ([, rule]) => {
    const file = /url\('([^']+)'\)/.exec(rule)![1]
    const ranges = /unicode-range:([^;]+);/
      .exec(rule)?.[1]
      .split(',')
      .map((range) => range.trim().slice(2).split('-'))
      .map(([start, end = start]) => [parseInt(start, 16), parseInt(end, 16)] as const)
    return {
      file,
      bytes: statSync(path.join(fontsDir, file)).size,
      covers: (codePoint: number) =>
        !ranges || ranges.some(([start, end]) => start <= codePoint && codePoint <= end),
    }
  })
}

// `base:selector` for every sequence in the cmap format 14 subtable of the font's first face.
function variationSequences(file: string) {
  const font = readFileSync(file)
  const sfnt = font.toString('latin1', 0, 4) === 'ttcf' ? font.readUInt32BE(12) : 0
  let cmap = 0
  for (let index = 0; index < font.readUInt16BE(sfnt + 4); index++) {
    const record = sfnt + 12 + 16 * index
    if (font.toString('latin1', record, record + 4) === 'cmap') cmap = font.readUInt32BE(record + 8)
  }
  const sequences = new Set<string>()
  for (let index = 0; index < font.readUInt16BE(cmap + 2); index++) {
    const table = cmap + font.readUInt32BE(cmap + 8 + 8 * index)
    if (font.readUInt16BE(table) !== 14) continue
    for (let selectorIndex = 0; selectorIndex < font.readUInt32BE(table + 6); selectorIndex++) {
      const record = table + 10 + 11 * selectorIndex
      const selector = font.readUIntBE(record, 3)
      const defaults = font.readUInt32BE(record + 3)
      const mapped = font.readUInt32BE(record + 7)
      for (let range = 0; defaults && range < font.readUInt32BE(table + defaults); range++) {
        const start = font.readUIntBE(table + defaults + 4 + 4 * range, 3)
        const count = font[table + defaults + 7 + 4 * range]
        for (let base = start; base <= start + count; base++) sequences.add(`${base}:${selector}`)
      }
      for (let mapping = 0; mapped && mapping < font.readUInt32BE(table + mapped); mapping++) {
        sequences.add(`${font.readUIntBE(table + mapped + 4 + 5 * mapping, 3)}:${selector}`)
      }
    }
  }
  return sequences
}

// The webview fetches a face only once it renders a character inside that face's unicode-range.
function fetchedFontBytes(locale: keyof typeof resources) {
  // Interface-language names render only in the settings language picker.
  const text = JSON.stringify({ ...resources[locale].translation, languages: {} })
  const codePoints = Array.from(new Set(text), (character) => character.codePointAt(0)!)
  const faces = uiFontFaces()
  // An uncovered character falls back to a system font and costs nothing, so a lost face or a
  // shrunken range would otherwise pass as a smaller download.
  const uncovered = codePoints.filter((codePoint) => !faces.some((face) => face.covers(codePoint)))
  expect(uncovered.map((codePoint) => String.fromCodePoint(codePoint))).toEqual([])
  return faces
    .filter((face) => codePoints.some(face.covers))
    .reduce((bytes, face) => bytes + face.bytes, 0)
}

describe('interface font', () => {
  it('declares one face per generated slice', () => {
    const slices = readdirSync(path.join(fontsDir, 'noto-sans-cjk'))
      .filter((file) => file.endsWith('.otf'))
      .map((file) => `./noto-sans-cjk/${file}`)
    expect(
      uiFontFaces()
        .map((face) => face.file)
        .sort(),
    ).toEqual(slices.sort())
  })

  it('fetches only the non-CJK slice for an English interface', () => {
    expect(fetchedFontBytes('en-US')).toBeLessThan(1_000_000)
  })

  it('fetches only the common CJK slices for a Japanese interface', () => {
    expect(fetchedFontBytes('ja-JP')).toBeLessThan(6_000_000)
  })

  it('is generated from the current source font', () => {
    const hash = createHash('sha256').update(readFileSync(source)).digest('hex')
    expect(/sha256:([0-9a-f]{64})/.exec(stylesheet)?.[1]).toBe(hash)
  })

  it('keeps every variation sequence in the slice that covers its base', () => {
    const faces = uiFontFaces().map((face) => ({
      ...face,
      sequences: variationSequences(path.join(fontsDir, face.file)),
    }))
    const missing = Array.from(variationSequences(source)).filter((sequence) => {
      const base = parseInt(sequence)
      return !faces.find((face) => face.covers(base))?.sequences.has(sequence)
    })
    expect(missing).toEqual([])
  })

  it('shapes conjoining jamo and tone marks within one slice', () => {
    const faces = uiFontFaces()
    const files = new Set(
      [0x1100, 0x11ff, 0x302e, 0x302f, 0xa960, 0xa97c, 0xd7b0, 0xd7fb].map(
        (codePoint) => faces.find((face) => face.covers(codePoint))?.file,
      ),
    )
    expect(Array.from(files)).toHaveLength(1)
    expect(files.has(undefined)).toBe(false)
  })
})
