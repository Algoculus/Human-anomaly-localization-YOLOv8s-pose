// Polyfill for CanvasRenderingContext2D.roundRect (for older browsers)
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number | number[]
  ) {
    const radii = typeof radius === 'number' ? [radius, radius, radius, radius] : radius
    const [tl, tr, br, bl] = radii

    this.moveTo(x + tl, y)
    this.lineTo(x + width - tr, y)
    this.arcTo(x + width, y, x + width, y + tr, tr)
    this.lineTo(x + width, y + height - br)
    this.arcTo(x + width, y + height, x + width - br, y + height, br)
    this.lineTo(x + bl, y + height)
    this.arcTo(x, y + height, x, y + height - bl, bl)
    this.lineTo(x, y + tl)
    this.arcTo(x, y, x + tl, y, tl)
    this.closePath()
  }
}

export {}
