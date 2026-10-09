"""Atlas de sprites do desenho dos agentes (aspeto "microscopia eletrónica").

Lê as imagens geradas em saves/sprites/fontes/ (cinzentas, fundo preto;
NN_nome_K.png = uma variante, NN_nome_grelha.png = matriz 3x3) e escreve
assets/sprites.png: R = cinzento, G = máscara, B = ALTURA (o contorno
"insuflado": cada ponto fica com a altura da maior esfera que cabe na forma
e passa por ele, por isso um disco dá uma cúpula, um anel um toro, um traço
um tubo), em frações de meio mosaico. COLS colunas de variantes por ROWS
linhas, mosaicos de TILE px. O shader (agents_view.wgsl) pinta a luminância
com a cor do órgão e escolhe a coluna pelo código do órgão, por isso órgãos
do mesmo tipo têm pequenas diferenças entre si.

Linhas: 0..21 = tipo de órgão; 22 = troço de aminoácido (cápsula deitada);
23 = um espigão da protease (ponta para cima); 24 = corpo da protease;
25 = grão de entulho; 26 = bloco de rocha; 27 = monómeros ativados e
28 = gastos, colunas A, U, G, C (world_view.wgsl); 29..31 = os 20
aminoácidos, um troço diferente por tipo (aminoácido i na linha 29 + i/9,
coluna i%9; a linha 22 é a versão antiga, já sem uso).
Linhas com menos de COLS variantes repetem-nas.

Encaixe no mosaico:
  quadrado  (por omissão) a caixa do objeto, centrada
  esticado  (depósito, aminoácido, espigão) a caixa esticada ao mosaico
  corpo     (aminoácidos) a caixa do CORPO da cápsula (sem saliências),
            alargada BODY_MARGIN vezes e esticada: as deformações e os
            nós cabem na margem e o shader recorta pela máscara
  haste     (sensores de um lado) a esfera no centro com raio STALK_BODY do
            meio mosaico, a antena para cima

Uso: python -X utf8 scripts/sprites_atlas.py
"""
import glob
import os
import re
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sprites_escolha import ORGANS, mask_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "saves", "sprites", "fontes")
TILE, COLS, ROWS = 512, 9, 32
STALK_BODY = 0.3
STRETCH = {7, 22, 23}
STALK = {8, 9}
BODY = {29}
BODY_MARGIN = 1.3
HOLES = {t for t, o in ORGANS.items() if o[2]}


def fit(lum, mask, box):
    w, h = box[2] - box[0], box[3] - box[1]
    out_l = Image.new("L", (w, h), 0)
    out_m = Image.new("L", (w, h), 0)
    out_l.paste(lum, (-box[0], -box[1]))
    out_m.paste(mask, (-box[0], -box[1]))
    out_l = out_l.resize((TILE, TILE), Image.LANCZOS)
    out_m = out_m.resize((TILE, TILE), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.5))
    hi, lo = inflate(out_l, out_m)
    return Image.merge("RGBA", (out_l, out_m, hi, lo))


def inflate(lum, mask):
    """Altura da forma insuflada (0..255 = 0..meio mosaico): em cada ponto,
    a da maior esfera inscrita que o cobre, mais um relevo fino tirado da
    luminância (as estrias e bossas do desenho)."""
    m = np.array(mask) > 127
    if not m.any():
        return Image.new("L", mask.size, 0), Image.new("L", mask.size, 0)
    d = ndi.distance_transform_edt(m)
    # O raio da maior esfera que cobre cada ponto é um campo suave: calcula-se
    # a um quarto da resolução (64 vezes mais depressa) e amplia-se. O
    # contorno e o perfil vêm de `d`, à resolução inteira.
    k = 4
    ms = m[::k, ::k]
    ds = ndi.distance_transform_edt(ms)
    ws = np.zeros_like(ds)
    r = float(ds.max())
    while r >= 1.0:
        core = ds >= r
        cover = ndi.distance_transform_edt(~core) <= r
        ws = np.where(cover & (ws == 0), r, ws)
        r -= 1.0
    ws = np.maximum(ws, ds) * k
    w = ndi.zoom(ws, k, order=1)[:m.shape[0], :m.shape[1]]
    w = ndi.gaussian_filter(w, k)
    w = np.where(m, np.maximum(w, d), 0.0)
    h = np.sqrt(np.clip(d * (2.0 * w - d), 0.0, None))
    relief = ndi.gaussian_filter(np.array(lum, dtype=np.float32) / 255.0, 1.0)
    h = np.where(m, h + 0.05 * (TILE / 2) * (relief - relief[m].mean()), 0.0)
    h = ndi.gaussian_filter(h, 0.8)
    # ALTURA EM 16 BITS, partida por dois canais (azul = byte alto, alfa = byte
    # baixo): com 8 bits as encostas saíam em degraus no microscópio 3D. A
    # interpolação e as médias dos mipmaps são lineares, por isso continuam certas.
    v = np.clip(h / (TILE / 2) * 65535.0, 0, 65535).astype(np.uint16)
    return Image.fromarray((v >> 8).astype(np.uint8), "L"), Image.fromarray((v & 255).astype(np.uint8), "L")


def tile(lum, row):
    mask = mask_of(lum, row in HOLES)
    clean = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    b = clean.getbbox()
    if b is None:
        return None
    lum = ImageOps.autocontrast(lum, cutoff=1, mask=mask)
    if row in STRETCH:
        return fit(lum, mask, b)
    if row in BODY:
        # O corpo sem as saliências finas: uma abertura larga da máscara.
        k = max((b[3] - b[1]) // 4, 3) | 1
        body = clean.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k)).getbbox() or b
        # AMINO_BBOX=1: formas finas (hélices, fitas) não têm "corpo": usa a caixa toda.
        if os.environ.get("AMINO_BBOX"):
            body = b
        cx, cy = (body[0] + body[2]) / 2, (body[1] + body[3]) / 2
        hw, hh = (body[2] - body[0]) / 2 * BODY_MARGIN, (body[3] - body[1]) / 2 * BODY_MARGIN
        return fit(lum, mask, (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)))
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = max(b[2] - b[0], b[3] - b[1]) / 2 * 1.03
    if row in STALK:
        # A esfera é a parte larga (em baixo): centro e raio pelas linhas largas.
        px = clean.load()
        widths = [(sum(1 for x in range(b[0], b[2], 2) if px[x, y] > 127) * 2, y) for y in range(b[1], b[3])]
        widest = max(w for w, _ in widths)
        rows = [y for w, y in widths if w > 0.6 * widest]
        cy = (rows[0] + rows[-1]) / 2
        half = (widest / 2) / STALK_BODY
    return fit(lum, mask, (int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def main():
    rows = {}
    for path in sorted(glob.glob(os.path.join(SRC, "*.png"))):
        m = re.match(r"(\d+)_(.+)_(grelha|\d+)\.png", os.path.basename(path))
        if not m:
            continue
        row = int(m.group(1))
        lum = Image.open(path).convert("L")
        cells = [lum]
        if m.group(3) == "grelha":
            w, h = lum.size
            cells = [lum.crop((c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3)) for r in range(3) for c in range(3)]
        for cell in cells:
            t = tile(cell, row)
            if t is not None:
                rows.setdefault(row, []).append(t)
    atlas = Image.new("RGBA", (TILE * COLS, TILE * ROWS), (0, 0, 0, 0))
    for row, tiles in rows.items():
        # Mais variantes do que colunas: continuam nas linhas seguintes.
        for r in range((len(tiles) + COLS - 1) // COLS):
            part = tiles[r * COLS:(r + 1) * COLS]
            assert row + r < ROWS, row
            for c in range(COLS):
                atlas.paste(part[c % len(part)], (c * TILE, (row + r) * TILE))
    out = os.path.join(ROOT, "assets", "sprites.png")
    atlas.save(out, optimize=True)
    print("escrito", out, atlas.size, "linhas:", {r: len(t) for r, t in sorted(rows.items())})


if __name__ == "__main__":
    main()
