/**
 * Procedural textures drawn on canvases: wood grain, wicker weave, stone, knit
 * jersey and the T-shirt print. No image files needed.
 */
import * as THREE from 'three'
import { LAUNDRY } from '../lib/stations'

/** Small deterministic random generator so textures look the same on every load. */
function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

function canvas(w: number, h: number) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  return { c, g: c.getContext('2d')! }
}

function tex(c: HTMLCanvasElement, color: boolean, repeat: [number, number] = [1, 1]) {
  const t = new THREE.CanvasTexture(c)
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.repeat.set(...repeat)
  t.anisotropy = 8
  if (color) t.colorSpace = THREE.SRGBColorSpace
  return t
}

const cache = new Map<string, THREE.Texture>()
const once = (key: string, make: () => THREE.Texture) => {
  if (!cache.has(key)) cache.set(key, make())
  return cache.get(key)!.clone()
}

/** Light oak with long grain lines and a few knots. */
export function woodTexture(repeat: [number, number] = [1, 1], base = '#b98b5e') {
  const t = once(`wood${base}`, () => {
    const { c, g } = canvas(1024, 256)
    const r = rng(7)
    g.fillStyle = base
    g.fillRect(0, 0, 1024, 256)
    // Broad colour bands, then many fine, nearly straight grain lines.
    for (let i = 0; i < 18; i++) {
      g.fillStyle = r() < 0.5 ? `rgba(120,80,40,${0.04 + r() * 0.06})` : `rgba(255,235,200,${0.04 + r() * 0.05})`
      g.fillRect(0, r() * 256, 1024, 4 + r() * 18)
    }
    for (let i = 0; i < 260; i++) {
      const y = r() * 256
      const dark = r() < 0.65
      g.strokeStyle = dark ? `rgba(95,58,28,${0.06 + r() * 0.12})` : `rgba(255,232,195,${0.05 + r() * 0.07})`
      g.lineWidth = 0.4 + r() * 1.1
      g.beginPath()
      g.moveTo(0, y)
      const ph = r() * 6
      for (let x = 0; x <= 1024; x += 16) g.lineTo(x, y + Math.sin(x * 0.004 + ph) * 1.1 + Math.sin(x * 0.017 + ph * 2) * 0.35)
      g.stroke()
    }
    for (let k = 0; k < 3; k++) {
      const x = 100 + r() * 820
      const y = 30 + r() * 196
      const grd = g.createRadialGradient(x, y, 1, x, y, 14 + r() * 10)
      grd.addColorStop(0, 'rgba(70,40,15,0.55)')
      grd.addColorStop(1, 'rgba(70,40,15,0)')
      g.fillStyle = grd
      g.beginPath()
      g.ellipse(x, y, 30, 10, 0, 0, Math.PI * 2)
      g.fill()
    }
    return tex(c, true)
  })
  t.repeat.set(...repeat)
  // Boards and legs are long in the other texture direction: turn the grain to run along them.
  t.center.set(0.5, 0.5)
  t.rotation = Math.PI / 2
  t.needsUpdate = true
  return t
}

/** Woven wicker: over-under strands, as a colour map and a matching bump map. */
export function wickerTextures(repeat: [number, number] = [3, 2]) {
  const make = (bump: boolean) => {
    const { c, g } = canvas(256, 256)
    const r = rng(bump ? 3 : 4)
    g.fillStyle = bump ? '#202020' : '#6b4a26'
    g.fillRect(0, 0, 256, 256)
    const n = 8
    const s = 256 / n
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const over = (i + j) % 2 === 0
        const x = i * s
        const y = j * s
        const grd = over ? g.createLinearGradient(x, y, x, y + s) : g.createLinearGradient(x, y, x + s, y)
        if (bump) {
          grd.addColorStop(0, '#404040')
          grd.addColorStop(0.5, '#f0f0f0')
          grd.addColorStop(1, '#404040')
        } else {
          const tone = 150 + r() * 40
          grd.addColorStop(0, `rgb(${tone - 50},${tone - 80},${tone - 120})`)
          grd.addColorStop(0.5, `rgb(${tone + 30},${tone},${tone - 50})`)
          grd.addColorStop(1, `rgb(${tone - 50},${tone - 80},${tone - 120})`)
        }
        g.fillStyle = grd
        g.fillRect(x + 1, y + 1, s - 2, s - 2)
      }
    }
    return tex(c, !bump, repeat)
  }
  return { map: make(false), bump: make(true) }
}

/** Speckled stone for countertops. */
export function stoneTexture(repeat: [number, number] = [2, 1]) {
  const t = once('stone', () => {
    const { c, g } = canvas(512, 512)
    const r = rng(11)
    g.fillStyle = '#e4e2dd'
    g.fillRect(0, 0, 512, 512)
    for (let i = 0; i < 9000; i++) {
      const v = r()
      g.fillStyle = v < 0.5 ? `rgba(120,115,110,${0.15 + r() * 0.3})` : v < 0.8 ? `rgba(255,255,255,${0.3 + r() * 0.4})` : `rgba(60,58,55,${0.2 + r() * 0.3})`
      const s = 0.6 + r() * 2.2
      g.fillRect(r() * 512, r() * 512, s, s)
    }
    return tex(c, true)
  })
  t.repeat.set(...repeat)
  t.needsUpdate = true
  return t
}

/** Fine knit (jersey) bump for fabric. */
export function knitBump(repeat: [number, number] = [6, 6]) {
  const t = once('knit', () => {
    const { c, g } = canvas(128, 128)
    const r = rng(5)
    g.fillStyle = '#808080'
    g.fillRect(0, 0, 128, 128)
    for (let x = 0; x < 128; x += 4) {
      for (let y = 0; y < 128; y += 3) {
        g.fillStyle = `rgba(255,255,255,${0.35 + r() * 0.2})`
        g.beginPath()
        g.ellipse(x + 1, y + 1.5, 1.1, 1.6, (x / 4) % 2 ? 0.5 : -0.5, 0, Math.PI * 2)
        g.fill()
      }
    }
    return tex(c, false)
  })
  t.repeat.set(...repeat)
  t.needsUpdate = true
  return t
}

// ---------------------------------------------------------------------------
// T-shirt
// ---------------------------------------------------------------------------
/** How far down from the shoulders the front of the neckline dips (metres from the hem). */
const NECK_DIP = 0.243

/** Colour, transparency (cut-out) and bump textures for the shirt lying flat. */
export function shirtTextures(color = '#2f4a6d', trim = '#20344f') {
  const { length, sleeve } = LAUNDRY.shirt
  const PPM = 1600 // pixels per metre
  const W = Math.round(2 * sleeve * PPM)
  const H = Math.round(length * PPM)
  const px = (a: number, b: number): [number, number] => [(b + sleeve) * PPM, (length - a) * PPM]

  /** T-shirt outline: straight hem and sides, angled sleeves, sloped shoulders, scoop neck. */
  const outline = (g: CanvasRenderingContext2D) => {
    const R = (b: number) => b
    const P = (a: number, b: number) => px(a, R(b))
    const r = 0.007 * PPM // corner rounding
    const pit = 0.016 * PPM // softer underarm curve
    const [hemC, hemR, pitR, cuffBR, cuffTR, shR, neckR] = [
      [0, 0],
      [0.003, 0.152],
      [0.14, 0.16],
      [0.152, 0.312],
      [0.238, 0.321],
      [0.27, 0.166],
      [0.281, 0.064],
    ] as [number, number][]
    const L = (q: [number, number]): [number, number] => [q[0], -q[1]]
    g.beginPath()
    g.moveTo(...P(...hemC))
    g.arcTo(...P(...hemR), ...P(...pitR), r)
    g.arcTo(...P(...pitR), ...P(...cuffBR), pit)
    g.arcTo(...P(...cuffBR), ...P(...cuffTR), r)
    g.arcTo(...P(...cuffTR), ...P(...shR), r)
    g.arcTo(...P(...shR), ...P(...neckR), r)
    g.lineTo(...P(...neckR))
    g.bezierCurveTo(...P(0.262, 0.06), ...P(NECK_DIP, 0.036), ...P(NECK_DIP, 0))
    g.bezierCurveTo(...P(NECK_DIP, -0.036), ...P(0.262, -0.06), ...P(...L(neckR)))
    g.arcTo(...P(...L(shR)), ...P(...L(cuffTR)), r)
    g.arcTo(...P(...L(cuffTR)), ...P(...L(cuffBR)), r)
    g.arcTo(...P(...L(cuffBR)), ...P(...L(pitR)), r)
    g.arcTo(...P(...L(pitR)), ...P(...L(hemR)), pit)
    g.arcTo(...P(...L(hemR)), ...P(...hemC), r)
    g.closePath()
  }
  /** The collar band follows the neckline curve. */
  const neckline = (g: CanvasRenderingContext2D) => {
    g.beginPath()
    g.moveTo(...px(0.281, 0.064))
    g.bezierCurveTo(...px(0.262, 0.06), ...px(NECK_DIP, 0.036), ...px(NECK_DIP, 0))
    g.bezierCurveTo(...px(NECK_DIP, -0.036), ...px(0.262, -0.06), ...px(0.281, -0.064))
  }

  // Colour
  const col = canvas(W, H)
  const g = col.g
  const r = rng(21)
  g.fillStyle = color
  outline(g)
  g.fill()
  g.save()
  outline(g)
  g.clip()
  // Heathered yarn: tiny lighter and darker flecks.
  for (let i = 0; i < 26000; i++) {
    g.fillStyle = r() < 0.5 ? `rgba(255,255,255,${0.04 + r() * 0.06})` : `rgba(0,0,0,${0.05 + r() * 0.07})`
    g.fillRect(r() * W, r() * H, 1 + r() * 2, 1)
  }
  // Soft wrinkles: blurred light and dark streaks.
  g.filter = 'blur(10px)'
  for (let i = 0; i < 14; i++) {
    const x = r() * W
    const y = r() * H
    g.strokeStyle = r() < 0.55 ? 'rgba(0,0,0,0.16)' : 'rgba(255,255,255,0.12)'
    g.lineWidth = 6 + r() * 14
    g.beginPath()
    g.moveTo(x, y)
    g.quadraticCurveTo(x + (r() - 0.5) * 260, y + (r() - 0.5) * 120, x + (r() - 0.5) * 420, y + (r() - 0.5) * 180)
    g.stroke()
  }
  g.filter = 'none'
  // Seams: shoulders and sides.
  g.strokeStyle = 'rgba(0,0,0,0.22)'
  g.lineWidth = 2
  for (const s of [-1, 1]) {
    g.beginPath()
    g.moveTo(...px(0.281, s * 0.064))
    g.lineTo(...px(0.27, s * 0.166))
    g.moveTo(...px(0.004, s * 0.152))
    g.lineTo(...px(0.14, s * 0.16))
    g.stroke()
  }
  // Double-needle stitching inside the hem and cuffs.
  g.setLineDash([7, 5])
  g.strokeStyle = 'rgba(0,0,0,0.3)'
  g.lineWidth = 1.6
  for (const off of [0.014, 0.019]) {
    g.beginPath()
    g.moveTo(...px(off, -0.15))
    g.quadraticCurveTo(...px(off - 0.003, 0), ...px(off, 0.15))
    g.stroke()
    for (const s of [-1, 1]) {
      g.beginPath()
      g.moveTo(...px(0.153 + off * 0.1, s * (0.312 - off)))
      g.lineTo(...px(0.238 - off * 0.1, s * (0.321 - off)))
      g.stroke()
    }
  }
  g.setLineDash([])
  // Ribbed collar band.
  g.strokeStyle = trim
  g.lineWidth = 0.03 * PPM // half of it lies outside the cut-out, leaving a 1.5 cm band
  neckline(g)
  g.stroke()
  g.strokeStyle = 'rgba(255,255,255,0.09)'
  g.lineWidth = 1
  for (let i = 0; i <= 80; i++) {
    const t = i / 80
    // Point on the neckline (cubic Bezier halves) and a short tick across the band.
    const bez = (p0: number, p1: number, p2: number, p3: number, u: number) => (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3
    const u = t < 0.5 ? t * 2 : (t - 0.5) * 2
    const a = t < 0.5 ? bez(0.281, 0.262, NECK_DIP, NECK_DIP, u) : bez(NECK_DIP, NECK_DIP, 0.262, 0.281, u)
    const b = t < 0.5 ? bez(0.064, 0.06, 0.036, 0, u) : bez(0, -0.036, -0.06, -0.064, u)
    g.beginPath()
    g.moveTo(...px(a + 0.001, b))
    g.lineTo(...px(a - 0.014, b))
    g.stroke()
  }
  g.restore()

  // Cut-out
  const al = canvas(W, H)
  al.g.fillStyle = '#000'
  al.g.fillRect(0, 0, W, H)
  al.g.fillStyle = '#fff'
  outline(al.g)
  al.g.fill()

  // Bump: knit texture plus the collar ribs and hem ridges.
  const bu = canvas(W, H)
  const b = bu.g
  b.fillStyle = '#808080'
  b.fillRect(0, 0, W, H)
  for (let x = 0; x < W; x += 3) {
    b.fillStyle = x % 6 ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.2)'
    b.fillRect(x, 0, 1, H)
  }
  b.strokeStyle = '#d0d0d0'
  b.lineWidth = 0.03 * PPM
  neckline(b)
  b.stroke()
  b.fillStyle = 'rgba(255,255,255,0.35)'
  b.fillRect(0, (length - 0.012) * PPM, W, 0.012 * PPM)

  const map = tex(col.c, true)
  const alpha = tex(al.c, false)
  const bump = tex(bu.c, false)
  for (const t of [map, alpha, bump]) t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping
  return { map, alpha, bump }
}
