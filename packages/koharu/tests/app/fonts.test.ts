import { readFileSync, statSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { resources } from '@/lib/i18n'

const fontsDir = path.resolve(import.meta.dirname, '../../app/fonts')

function uiFontFaces() {
  const css = readFileSync(path.join(fontsDir, 'noto-sans-cjk.css'), 'utf8')
  return Array.from(css.matchAll(/@font-face\s*{([^}]*)}/g), ([, rule]) => {
    const file = /url\('([^']+)'\)/.exec(rule)![1]
    const ranges = /unicode-range:([^;]+);/
      .exec(rule)?.[1]
      .split(',')
      .map((range) => range.trim().slice(2).split('-'))
      .map(([start, end = start]) => [parseInt(start, 16), parseInt(end, 16)] as const)
    return {
      bytes: statSync(path.join(fontsDir, file)).size,
      covers: (codePoint: number) =>
        !ranges || ranges.some(([start, end]) => start <= codePoint && codePoint <= end),
    }
  })
}

// The webview fetches a face only once it renders a character inside that face's unicode-range.
function fetchedFontBytes(locale: keyof typeof resources) {
  // Interface-language names render only in the settings language picker.
  const text = JSON.stringify({ ...resources[locale].translation, languages: {} })
  const codePoints = Array.from(new Set(text), (character) => character.codePointAt(0)!)
  return uiFontFaces()
    .filter((face) => codePoints.some(face.covers))
    .reduce((bytes, face) => bytes + face.bytes, 0)
}

describe('interface font', () => {
  it('fetches only the non-CJK slice for an English interface', () => {
    expect(fetchedFontBytes('en-US')).toBeLessThan(1_000_000)
  })

  it('fetches only the common CJK slices for a Japanese interface', () => {
    expect(fetchedFontBytes('ja-JP')).toBeLessThan(6_000_000)
  })
})
