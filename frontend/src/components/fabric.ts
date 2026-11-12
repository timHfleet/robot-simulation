import * as THREE from 'three'
import { SHIRT_DIMS, SHIRT_V } from '../lib/geom'

/**
 * Procedural T-shirt textures, drawn on canvases:
 *  - map:   cotton jersey colour with knit lines, seams, stitched hems,
 *           collar ribbing and a small chest logo
 *  - alpha: the shirt's outline (smooth curves), cut out of the cloth grid
 *  - bump:  knit texture and raised seams/stitches for the lighting
 *
 * Canvas x runs across the shirt (v from -SHIRT_V to +SHIRT_V), canvas y runs
 * from the collar (u = 0, top) to the hem (u = length, bottom).
 */

const W = 1024
const H = Math.round((W * SHIRT_DIMS.length) / (2 * SHIRT_V))

const px = (u: number, v: number): [number, number] => [((v + SHIRT_V) / (2 * SHIRT_V)) * W, (u / SHIRT_DIMS.length) * H]
const scale = W / (2 * SHIRT_V) // pixels per metre

/** The shirt outline: crew neck, sloped shoulders, angled short sleeves,
 * curved armholes, slightly flared sides and a gently curved hem. */
function outline(ctx: CanvasRenderingContext2D) {
  const P = (u: number, v: number) => px(u, v)
  ctx.beginPath()
  ctx.moveTo(...P(0.004, -0.062))
  ctx.bezierCurveTo(...P(0.03, -0.05), ...P(0.034, 0.05), ...P(0.004, 0.062)) // front neckline
  ctx.quadraticCurveTo(...P(0.008, 0.11), ...P(0.024, 0.153)) // shoulder slope
  ctx.quadraticCurveTo(...P(0.03, 0.2), ...P(0.052, 0.247)) // sleeve top
  ctx.quadraticCurveTo(...P(0.09, 0.244), ...P(0.13, 0.226)) // sleeve hem
  ctx.bezierCurveTo(...P(0.122, 0.19), ...P(0.118, 0.16), ...P(0.132, 0.146)) // underarm
  ctx.bezierCurveTo(...P(0.22, 0.142), ...P(0.3, 0.148), ...P(0.356, 0.153)) // side seam
  ctx.bezierCurveTo(...P(0.364, 0.08), ...P(0.364, -0.08), ...P(0.356, -0.153)) // hem
  ctx.bezierCurveTo(...P(0.3, -0.148), ...P(0.22, -0.142), ...P(0.132, -0.146))
  ctx.bezierCurveTo(...P(0.118, -0.16), ...P(0.122, -0.19), ...P(0.13, -0.226))
  ctx.quadraticCurveTo(...P(0.09, -0.244), ...P(0.052, -0.247))
  ctx.quadraticCurveTo(...P(0.03, -0.2), ...P(0.024, -0.153))
  ctx.quadraticCurveTo(...P(0.008, -0.11), ...P(0.004, -0.062))
  ctx.closePath()
}

function shade(hex: string, k: number) {
  const c = new THREE.Color(hex)
  c.multiplyScalar(k)
  return `#${c.getHexString()}`
}

function canvas() {
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  return c
}

/** Seams and hems as polylines (in u, v), shared by the colour and bump maps. */
function seams(ctx: CanvasRenderingContext2D, dashed: boolean, width: number) {
  const P = (u: number, v: number) => px(u, v)
  ctx.lineWidth = width
  ctx.setLineDash(dashed ? [5, 4] : [])
  for (const s of [1, -1]) {
    // Shoulder seam
    ctx.beginPath()
    ctx.moveTo(...P(0.006, s * 0.068))
    ctx.quadraticCurveTo(...P(0.01, s * 0.11), ...P(0.026, s * 0.15))
    ctx.stroke()
    // Armhole seam (sleeve set into the body)
    ctx.beginPath()
    ctx.moveTo(...P(0.026, s * 0.152))
    ctx.bezierCurveTo(...P(0.06, s * 0.142), ...P(0.1, s * 0.14), ...P(0.132, s * 0.146))
    ctx.stroke()
    // Sleeve hem stitching
    ctx.beginPath()
    ctx.moveTo(...P(0.05, s * 0.236))
    ctx.quadraticCurveTo(...P(0.088, s * 0.233), ...P(0.122, s * 0.217))
    ctx.stroke()
    // Side seam
    ctx.beginPath()
    ctx.moveTo(...P(0.135, s * 0.144))
    ctx.bezierCurveTo(...P(0.22, s * 0.14), ...P(0.3, s * 0.146), ...P(0.354, s * 0.151))
    ctx.stroke()
  }
  // Hem: double row of stitching
  for (const off of [0.012, 0.016]) {
    ctx.beginPath()
    ctx.moveTo(...P(0.356 - off, -0.15))
    ctx.bezierCurveTo(...P(0.364 - off, -0.08), ...P(0.364 - off, 0.08), ...P(0.356 - off, 0.15))
    ctx.stroke()
  }
  ctx.setLineDash([])
}

function collar(ctx: CanvasRenderingContext2D, color: string) {
  const P = (u: number, v: number) => px(u, v)
  ctx.strokeStyle = color
  ctx.lineWidth = 0.011 * scale
  ctx.beginPath()
  ctx.moveTo(...P(0.006, -0.064))
  ctx.bezierCurveTo(...P(0.034, -0.05), ...P(0.038, 0.05), ...P(0.006, 0.064))
  ctx.stroke()
}

export function shirtTextures(color: string) {
  // --- colour --------------------------------------------------------------
  const cm = canvas()
  const c = cm.getContext('2d')!
  c.save()
  outline(c)
  c.clip()
  c.fillStyle = color
  c.fillRect(0, 0, W, H)
  // Soft light falloff across the body, like fabric that isn't perfectly flat.
  const g = c.createRadialGradient(W / 2, H * 0.45, 20, W / 2, H * 0.45, W * 0.45)
  g.addColorStop(0, 'rgba(255,255,255,0.07)')
  g.addColorStop(1, 'rgba(0,0,0,0.10)')
  c.fillStyle = g
  c.fillRect(0, 0, W, H)
  // Jersey knit: fine vertical wales with tiny alternating loops.
  for (let x = 0; x < W; x += 3) {
    c.fillStyle = x % 6 === 0 ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)'
    c.fillRect(x, 0, 1, H)
  }
  for (let y = 0; y < H; y += 4) {
    c.fillStyle = 'rgba(0,0,0,0.03)'
    c.fillRect(0, y, W, 1)
  }
  // Fibre noise.
  for (let i = 0; i < 9000; i++) {
    c.fillStyle = Math.random() < 0.5 ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)'
    c.fillRect(Math.random() * W, Math.random() * H, 1.5, 1.5)
  }
  // Collar ribbing.
  collar(c, shade(color, 0.72))
  c.strokeStyle = 'rgba(0,0,0,0.18)'
  c.lineWidth = 1
  for (let k = -0.06; k <= 0.06; k += 0.0035) {
    const [x0, y0] = px(0.006 + 0.03 * (1 - (k / 0.064) ** 2), k)
    c.beginPath()
    c.moveTo(x0, y0 - 0.0045 * scale)
    c.lineTo(x0, y0 + 0.0045 * scale)
    c.stroke()
  }
  // Seams and stitching.
  c.strokeStyle = shade(color, 0.6)
  seams(c, false, 1.6)
  c.strokeStyle = 'rgba(255,255,255,0.35)'
  seams(c, true, 1)
  // Small embroidered chest logo (a little robot arm).
  const [lx, ly] = px(0.085, 0.06)
  c.strokeStyle = 'rgba(255,255,255,0.85)'
  c.lineWidth = 2.2
  c.beginPath()
  c.arc(lx, ly, 0.013 * scale, 0, Math.PI * 2)
  c.stroke()
  c.beginPath()
  c.moveTo(lx - 0.007 * scale, ly + 0.007 * scale)
  c.lineTo(lx - 0.002 * scale, ly - 0.003 * scale)
  c.lineTo(lx + 0.007 * scale, ly - 0.006 * scale)
  c.stroke()
  c.restore()
  // Stitched edge just inside the outline.
  c.strokeStyle = shade(color, 0.55)
  c.lineWidth = 2
  outline(c)
  c.stroke()

  // --- alpha (the outline) ---------------------------------------------------
  const am = canvas()
  const a = am.getContext('2d')!
  a.fillStyle = '#000'
  a.fillRect(0, 0, W, H)
  a.fillStyle = '#fff'
  outline(a)
  a.fill()

  // --- bump --------------------------------------------------------------------
  const bm = canvas()
  const b = bm.getContext('2d')!
  b.fillStyle = '#808080'
  b.fillRect(0, 0, W, H)
  for (let x = 0; x < W; x += 3) {
    b.fillStyle = x % 6 === 0 ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.18)'
    b.fillRect(x, 0, 1, H)
  }
  for (let i = 0; i < 12000; i++) {
    b.fillStyle = Math.random() < 0.5 ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)'
    b.fillRect(Math.random() * W, Math.random() * H, 2, 2)
  }
  b.strokeStyle = 'rgba(255,255,255,0.7)'
  seams(b, false, 3)
  collar(b, 'rgba(255,255,255,0.55)')

  const map = new THREE.CanvasTexture(cm)
  map.colorSpace = THREE.SRGBColorSpace
  map.anisotropy = 8
  const alpha = new THREE.CanvasTexture(am)
  const bump = new THREE.CanvasTexture(bm)
  return { map, alpha, bump }
}

/** Terry-cloth bump for the towel. */
export function terryBump() {
  const c = document.createElement('canvas')
  c.width = c.height = 512
  const g = c.getContext('2d')!
  g.fillStyle = '#808080'
  g.fillRect(0, 0, 512, 512)
  for (let i = 0; i < 26000; i++) {
    g.fillStyle = Math.random() < 0.5 ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.25)'
    g.beginPath()
    g.arc(Math.random() * 512, Math.random() * 512, 1.6, 0, Math.PI * 2)
    g.fill()
  }
  const t = new THREE.CanvasTexture(c)
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.repeat.set(4, 4)
  return t
}
