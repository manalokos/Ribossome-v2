"""Atlas de sprites dos órgãos (teste do aspeto "microscopia eletrónica").

Lê as imagens geradas (cinzentas, fundo preto, vistas de cima) em
saves/sprites/ e escreve assets/sprites.png: uma fila de mosaicos TILE×TILE,
com a luminância em RGB e a máscara do objeto no alfa. O shader
(agents_view.wgsl) pinta a luminância com a cor do órgão.

Mosaicos (a ordem é a de SPRITE_* no shader):
  0 deposito.png      -> recortado à caixa do objeto e ESTICADO ao mosaico
                         (o shader estica-o ao comprimento real do órgão)
  1 sensor.png        -> coroa de antenas, centrado
  2 sensor_lado.png   -> uma antena para CIMA, com a esfera no centro

Uso: python -X utf8 scripts/sprites_atlas.py
"""
import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "saves", "sprites")
TILE = 256
THRESHOLD = 22  # luminância (0..255) acima da qual já é objeto


def mask_of(lum):
    """Máscara do objeto: tudo o que não é fundo ligado às margens."""
    binary = lum.point(lambda v: 255 if v > THRESHOLD else 0)
    # Enche o fundo a partir dos cantos com um valor à parte; o que sobra a 0
    # são buracos escuros DENTRO do objeto e contam como objeto.
    filled = binary.copy()
    for corner in [(0, 0), (lum.width - 1, 0), (0, lum.height - 1), (lum.width - 1, lum.height - 1)]:
        if filled.getpixel(corner) == 0:
            ImageDraw.floodfill(filled, corner, 128)
    return filled.point(lambda v: 0 if v == 128 else 255)


def load(name, fatten=0):
    """Luminância e máscara. fatten (ímpar, píxeis da origem) engrossa os
    traços finos (antenas), que de outro modo desaparecem ao reduzir."""
    lum = Image.open(os.path.join(SRC, name)).convert("L")
    if fatten:
        lum = lum.filter(ImageFilter.MaxFilter(fatten))
    mask = mask_of(lum)
    # Contraste: estica a luminância DENTRO do objeto a toda a gama.
    lum = ImageOps.autocontrast(lum, cutoff=1, mask=mask)
    return lum, mask


def tile(lum, mask, box):
    """Recorta `box` (pode sair da imagem: enche de fundo) e reduz ao mosaico."""
    w, h = box[2] - box[0], box[3] - box[1]
    out_l = Image.new("L", (w, h), 0)
    out_m = Image.new("L", (w, h), 0)
    out_l.paste(lum, (-box[0], -box[1]))
    out_m.paste(mask, (-box[0], -box[1]))
    out_l = out_l.resize((TILE, TILE), Image.LANCZOS)
    out_m = out_m.resize((TILE, TILE), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
    return Image.merge("RGBA", (out_l, out_l, out_l, out_m))


def square_around(cx, cy, half):
    return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


def main():
    tiles = []
    # 0: depósito, esticado à caixa.
    lum, mask = load("deposito.png")
    tiles.append(tile(lum, mask, mask.getbbox()))
    # 1: coroa, quadrado centrado na caixa.
    lum, mask = load("sensor.png", fatten=9)
    b = mask.getbbox()
    half = max(b[2] - b[0], b[3] - b[1]) / 2 * 1.02
    tiles.append(tile(lum, mask, square_around((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, half)))
    # 2: uma antena: a esfera é a parte larga; fica no centro do mosaico.
    lum, mask = load("sensor_lado.png", fatten=9)
    b = mask.getbbox()
    px = mask.load()
    widths = [(sum(1 for x in range(b[0], b[2]) if px[x, y] > 127), y) for y in range(b[1], b[3])]
    widest = max(w for w, _ in widths)
    rows = [y for w, y in widths if w > 0.5 * widest]
    cy = (rows[0] + rows[-1]) / 2
    cx = (b[0] + b[2]) / 2
    half = max(cy - b[1], b[3] - cy, (b[2] - b[0]) / 2) * 1.02
    tiles.append(tile(lum, mask, square_around(cx, cy, half)))
    print("esfera do sensor de um lado: raio / meio mosaico = %.3f" % ((rows[-1] - rows[0]) / 2 / half))

    atlas = Image.new("RGBA", (TILE * len(tiles), TILE), (0, 0, 0, 0))
    for i, t in enumerate(tiles):
        atlas.paste(t, (i * TILE, 0))
    out = os.path.join(ROOT, "assets", "sprites.png")
    atlas.save(out)
    print("escrito", out, atlas.size)


if __name__ == "__main__":
    main()
