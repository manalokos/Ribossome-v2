"""Extrai da AAindex (GenomeNet) as tabelas reais usadas pelo organismo.

- data/mj1996.csv: energias de contacto de Miyazawa & Jernigan 1996
  (J Mol Biol 256:623), entrada MIYS960101 da AAindex3 (e_ij em unidades
  de RT, inclui o termo hidrofóbico; a matriz própria para dobragem).
- data/flexibility_vihinen1994.csv: flexibilidade normalizada (B-values)
  de Vihinen et al. 1994 (Proteins 19:141), entrada VINM940101 da AAindex1.

Fontes:
  https://www.genome.jp/ftp/db/community/aaindex/aaindex3
  https://www.genome.jp/ftp/db/community/aaindex/aaindex1
Uso: python scripts/aaindex_extract.py [aaindex1] [aaindex3]
Ordem de saída: A C D E F G H I K L M N P Q R S T V W Y (a de src/life/amino.rs).
"""

import pathlib
import sys
import urllib.request

OUT_ORDER = "ACDEFGHIKLMNPQRSTVWY"
BASE = "https://www.genome.jp/ftp/db/community/aaindex/"


def load(path, name):
    if path:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    with urllib.request.urlopen(BASE + name, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def entry(text, key):
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("H " + key))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("//"))
    return lines[start:end]


def mj_matrix(text):
    block = entry(text, "MIYS960101")
    m = next(i for i, l in enumerate(block) if l.startswith("M "))
    order = block[m].split("rows = ")[1].split(",")[0].strip()
    rows = [list(map(float, l.split())) for l in block[m + 1:]]
    full = {}
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            full[(order[i], order[j])] = v
            full[(order[j], order[i])] = v
    return full


def flexibility(text):
    block = entry(text, "VINM940101")
    i = next(k for k, l in enumerate(block) if l.startswith("I "))
    heads = block[i].split()[1:]
    v1 = list(map(float, block[i + 1].split()))
    v2 = list(map(float, block[i + 2].split()))
    out = {}
    for h, a, b in zip(heads, v1, v2):
        x, y = h.split("/")
        out[x] = a
        out[y] = b
    three = {"A": "A", "R": "R", "N": "N", "D": "D", "C": "C", "Q": "Q", "E": "E", "G": "G", "H": "H",
             "I": "I", "L": "L", "K": "K", "M": "M", "F": "F", "P": "P", "S": "S", "T": "T", "W": "W",
             "Y": "Y", "V": "V"}
    return {three[k]: v for k, v in out.items()}


def main():
    a1 = load(sys.argv[1] if len(sys.argv) > 1 else None, "aaindex1")
    a3 = load(sys.argv[2] if len(sys.argv) > 2 else None, "aaindex3")
    data = pathlib.Path(__file__).resolve().parent.parent / "data"
    data.mkdir(exist_ok=True)
    mj = mj_matrix(a3)
    with (data / "mj1996.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Miyazawa & Jernigan 1996, J Mol Biol 256:623 (AAindex3 MIYS960101), e_ij em RT\n")
        f.write("aa," + ",".join(OUT_ORDER) + "\n")
        for a in OUT_ORDER:
            f.write(a + "," + ",".join(f"{mj[(a, b)]:.2f}" for b in OUT_ORDER) + "\n")
    fl = flexibility(a1)
    with (data / "flexibility_vihinen1994.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Vihinen et al. 1994, Proteins 19:141 (AAindex1 VINM940101), B-values normalizados\n")
        f.write("aa,flexibility\n")
        for a in OUT_ORDER:
            f.write(f"{a},{fl[a]:.3f}\n")
    print("LL", mj[("L", "L")], "CC", mj[("C", "C")], "KK", mj[("K", "K")], "flex G", fl["G"], "P", fl["P"])


if __name__ == "__main__":
    main()
