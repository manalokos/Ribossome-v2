"""Propensão catalítica dos 20 aminoácidos, a partir de dados reais.

Fonte dos resíduos catalíticos: M-CSA (Mechanism and Catalytic Site Atlas,
EMBL-EBI; Ribeiro et al. 2018, NAR 46:D618), API pública:
    https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?format=json
Composição de fundo: UniProtKB/Swiss-Prot, release 2026_03 (02-Sep-2026),
    https://web.expasy.org/docs/relnotes/relstat.html

Propensão = (fração do aminoácido entre os resíduos catalíticos que atuam
pela CADEIA LATERAL) / (fração do aminoácido nas proteínas em geral).
Contam só as funções de cadeia lateral (function_location_abv vazio): a
química do resíduo é a da cadeia lateral; as funções de cadeia principal
(main-N, main-C) dependem da geometria, não do tipo de aminoácido.

Uso: python scripts/catalytic_propensity.py [ficheiro.json]
Escreve data/catalytic_propensity.csv.
"""

import collections
import json
import pathlib
import sys
import urllib.request

URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?format=json"

# Swiss-Prot 2026_03, percentagem.
BACKGROUND = {
    "Ala": 8.25, "Arg": 5.52, "Asn": 4.06, "Asp": 5.46, "Cys": 1.38,
    "Gln": 3.93, "Glu": 6.71, "Gly": 7.07, "His": 2.27, "Ile": 5.90,
    "Leu": 9.64, "Lys": 5.79, "Met": 2.41, "Phe": 3.86, "Pro": 4.75,
    "Ser": 6.66, "Thr": 5.36, "Trp": 1.10, "Tyr": 2.92, "Val": 6.85,
}
ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def load(path):
    if path:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    with urllib.request.urlopen(URL, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    data = load(sys.argv[1] if len(sys.argv) > 1 else None)
    counts = collections.Counter()
    for r in data:
        if r.get("function_location_abv", "") != "":
            continue
        seqs = r.get("residue_sequences") or []
        if not seqs:
            continue
        code = seqs[0]["code"]
        if code in BACKGROUND:
            counts[code] += 1
    total = sum(counts.values())
    bg_total = sum(BACKGROUND.values())
    rows = []
    for code in sorted(BACKGROUND, key=lambda c: ONE[c]):
        frac = counts[code] / total
        prop = frac / (BACKGROUND[code] / bg_total)
        rows.append((ONE[code], code, counts[code], frac, prop))
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalytic_propensity.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# M-CSA side-chain catalytic residues: {total} (entries: {len(set(r['mcsa_id'] for r in data))})\n")
        f.write("# background: UniProtKB/Swiss-Prot 2026_03\n")
        f.write("aa,code,count,fraction,propensity\n")
        for one, code, n, frac, prop in rows:
            f.write(f"{one},{code},{n},{frac:.4f},{prop:.3f}\n")
    for one, code, n, frac, prop in sorted(rows, key=lambda r: -r[4]):
        print(f"{one} {code} {n:5d} {frac*100:5.1f}%  propensão {prop:5.2f}")
    print(f"total {total}; escrito {out}")


if __name__ == "__main__":
    main()
