"""Página para escolher o aspeto de cada órgão.

Lê as imagens geradas em saves/sprites/fontes/ (NN_nome_K.png = uma variante
por imagem; NN_nome_grelha.png = matriz 3x3 de variantes), recorta cada
variante (luminância + máscara, 256 px) para saves/sprites/recortes/ e
escreve saves/sprites/escolha.html, com cada variante pintada com a cor do
órgão como o shader a pinta. As escolhas ficam num texto para copiar.

Uso: python -X utf8 scripts/sprites_escolha.py
"""
import glob
import json
import os
import re
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "saves", "sprites")
SRC = os.path.join(BASE, "fontes")
CUT = os.path.join(BASE, "recortes")
TILE = 256
THRESHOLD = 22

# tipo -> (nome no ecrã, cor do órgão como em organ_color do inspetor, buracos transparentes)
ORGANS = {
    0: ("mouth", (235, 235, 235), False),
    1: ("muscle", (217, 100, 100), False),
    2: ("food sensor", (115, 255, 115), False),
    3: ("light sensor", (255, 242, 102), False),
    4: ("energy sensor", (255, 217, 51), False),
    5: ("clock", (217, 230, 255), False),
    6: ("relay", (153, 230, 255), False),
    7: ("storage", (178, 115, 242), False),
    8: ("food sensor, one-sided", (115, 255, 115), False),
    9: ("light sensor, one-sided", (255, 242, 102), False),
    10: ("photosystem", (89, 242, 89), False),
    11: ("protease", (230, 51, 51), False),
    12: ("anchor", (38, 230, 204), True),
    13: ("bias", (255, 140, 38), False),
    14: ("chemosynthesis", (230, 199, 38), True),
    15: ("proofreading", (200, 170, 255), True),
    16: ("dormancy", (184, 209, 255), False),
    17: ("age bias", (255, 140, 38), False),
    18: ("holdfast", (217, 153, 77), False),
    19: ("chiral", (242, 89, 217), True),
    21: ("inhibitor", (242, 204, 230), False),
}


def mask_of(lum, holes):
    binary = lum.point(lambda v: 255 if v > THRESHOLD else 0)
    if holes:
        return binary
    filled = binary.copy()
    w, h = lum.size
    for x in range(0, w, max(w // 16, 1)):
        for p in ((x, 0), (x, h - 1), (0, min(x, h - 1)), (w - 1, min(x, h - 1))):
            if filled.getpixel(p) == 0:
                ImageDraw.floodfill(filled, p, 128)
    return filled.point(lambda v: 0 if v == 128 else 255)


def cut(lum, holes):
    """Recorta o objeto (o maior bloco) num quadrado e devolve RGBA 256."""
    mask = mask_of(lum, holes)
    # Tira migalhas soltas (pó, restos de vizinhos numa grelha).
    clean = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    box = clean.getbbox()
    if box is None:
        return None
    lum = ImageOps.autocontrast(lum, cutoff=1, mask=mask)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1]) / 2 * 1.03
    b = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    size = (b[2] - b[0], b[3] - b[1])
    out_l = Image.new("L", size, 0)
    out_m = Image.new("L", size, 0)
    out_l.paste(lum, (-b[0], -b[1]))
    out_m.paste(mask, (-b[0], -b[1]))
    out_l = out_l.resize((TILE, TILE), Image.LANCZOS)
    out_m = out_m.resize((TILE, TILE), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
    return Image.merge("RGBA", (out_l, out_l, out_l, out_m))


def tinted(tile, color):
    """Como sem_color no shader: cor * (0,2 + 1,3 L) + 0,4 L^3."""
    l, _, _, a = tile.split()
    chans = []
    for c in color:
        chans.append(l.point(lambda v, c=c: int(min(255, c * (0.2 + 1.3 * v / 255) + 102 * (v / 255) ** 3))))
    return Image.merge("RGBA", (*chans, a.point(lambda v: 255 if v > 127 else 0)))


def main():
    os.makedirs(CUT, exist_ok=True)
    for f in glob.glob(os.path.join(CUT, "*.png")):
        os.remove(f)
    found = {}
    for path in sorted(glob.glob(os.path.join(SRC, "*.png"))):
        m = re.match(r"(\d+)_(.+)_(grelha|\d+)\.png", os.path.basename(path))
        if not m or int(m.group(1)) not in ORGANS:
            continue
        t = int(m.group(1))
        _, color, holes = ORGANS[t]
        lum = Image.open(path).convert("L")
        cells = []
        if m.group(3) == "grelha":
            w, h = lum.size
            for r in range(3):
                for c in range(3):
                    cells.append(lum.crop((c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3)))
        else:
            cells.append(lum)
        for cell in cells:
            tile = cut(cell, holes)
            if tile is None:
                continue
            n = len(found.setdefault(t, []))
            name = "%02d_%d" % (t, n)
            tile.save(os.path.join(CUT, name + ".png"))
            tinted(tile, color).save(os.path.join(CUT, name + "_cor.png"))
            found[t].append(name)
    rows = []
    for t in sorted(found):
        name, color, _ = ORGANS[t]
        cards = "".join(
            '<button class="v" data-o="%d" data-v="%d"><img src="recortes/%s_cor.png"><img class="g" src="recortes/%s.png"><span>%d</span></button>'
            % (t, i, n, n, i + 1)
            for i, n in enumerate(found[t])
        )
        rows.append(
            '<section><h2><i style="background:rgb(%d,%d,%d)"></i>%s <small>organ %d</small></h2><div class="row">'
            '<button class="v keep" data-o="%d" data-v="-1"><span>keep the current drawing</span></button>%s</div></section>'
            % (*color, name, t, t, cards)
        )
    html = PAGE.replace("@ROWS@", "\n".join(rows)).replace("@ORGANS@", json.dumps({t: ORGANS[t][0] for t in found}))
    out = os.path.join(BASE, "escolha.html")
    open(out, "w", encoding="utf-8").write(html)
    print("escrito", out, "com", sum(len(v) for v in found.values()), "variantes de", len(found), "órgãos")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Organ looks</title>
<style>
:root { --bg:#0d0f12; --panel:#161a20; --line:#2a303a; --text:#e6e8ec; --weak:#8a93a3; --pick:#5fd38d; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.4 system-ui, sans-serif; padding:16px 16px 190px; }
h1 { font-size:22px; margin:0 0 4px; } p { color:var(--weak); margin:0 0 18px; max-width:70ch; }
section { margin:0 0 22px; } h2 { font-size:16px; margin:0 0 8px; display:flex; align-items:center; gap:8px; }
h2 i { width:14px; height:14px; border-radius:50%; display:inline-block; } h2 small { color:var(--weak); font-weight:400; }
.row { display:flex; flex-wrap:wrap; gap:8px; }
.v { position:relative; width:132px; height:132px; padding:6px; background:#050607; border:2px solid var(--line); border-radius:10px; cursor:pointer; color:var(--weak); }
.v img { width:100%; height:100%; object-fit:contain; display:block; } .v img.g { display:none; position:absolute; inset:6px; width:calc(100% - 12px); height:calc(100% - 12px); }
body.gray .v img.g { display:block; } body.gray .v img:not(.g) { visibility:hidden; }
.v span { position:absolute; left:6px; top:4px; font-size:12px; } .v.keep span { position:static; font-size:13px; }
.v:hover { border-color:var(--weak); } .v.on { border-color:var(--pick); box-shadow:0 0 0 2px var(--pick) inset; color:var(--pick); }
footer { position:fixed; left:0; right:0; bottom:0; background:var(--panel); border-top:1px solid var(--line); padding:10px 16px; display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap; }
textarea { flex:1 1 320px; min-height:86px; background:#050607; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px; font:13px/1.35 ui-monospace, Consolas, monospace; }
footer button, label { background:#232a34; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px 12px; cursor:pointer; font:inherit; }
#n { color:var(--weak); align-self:center; }
</style></head><body>
<h1>Organ looks</h1>
<p>Click one look per organ (or keep the current drawing). Each picture is the gray microscope image painted with the organ's colour, the way the simulation paints it. When you are done, copy the text at the bottom and paste it in the chat.</p>
@ROWS@
<footer><textarea id="out" readonly></textarea><button id="copy">Copy choices</button><label><input type="checkbox" id="gray"> show gray originals</label><span id="n"></span></footer>
<script>
const ORGANS = @ORGANS@;
let pick = {};
try { pick = JSON.parse(localStorage.getItem("organ-looks") || "{}"); } catch (e) {}
function show() {
  document.querySelectorAll(".v").forEach(b => b.classList.toggle("on", String(pick[b.dataset.o]) === b.dataset.v));
  const lines = Object.keys(ORGANS).map(o => pick[o] === undefined ? null : ORGANS[o] + " (" + o + "): " + (pick[o] < 0 ? "keep current" : "look " + (pick[o] + 1))).filter(Boolean);
  document.getElementById("out").value = lines.join("\\n");
  document.getElementById("n").textContent = lines.length + " of " + Object.keys(ORGANS).length + " chosen";
  try { localStorage.setItem("organ-looks", JSON.stringify(pick)); } catch (e) {}
}
document.querySelectorAll(".v").forEach(b => b.addEventListener("click", () => { pick[b.dataset.o] = Number(b.dataset.v); show(); }));
document.getElementById("copy").addEventListener("click", () => { const t = document.getElementById("out"); t.select(); navigator.clipboard && navigator.clipboard.writeText(t.value); });
document.getElementById("gray").addEventListener("change", e => document.body.classList.toggle("gray", e.target.checked));
show();
</script></body></html>
"""

if __name__ == "__main__":
    main()
