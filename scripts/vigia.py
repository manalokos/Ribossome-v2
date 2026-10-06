"""Vigia da noite: de hora a hora tira um retrato da simulação que está a
correr (pelo servidor MCP da app) e guarda-o numa pasta, para comparar no
dia seguinte.

Uso:  python -X utf8 scripts/vigia.py [pasta] [nº de retratos] [segundos entre eles]

Cada retrato (retrato_NN.json) tem: epoch, parâmetros mudados, a média das
últimas amostras das estatísticas, as espécies (dois limiares), o habitat por
altura e por coluna, e o resultado de duas sondas corridas numa CÓPIA do
autosave (presas: composição e alvos das proteases; predação). Guarda também
duas imagens (mundo inteiro e um pormenor ao centro). A app não é alterada:
só se leem dados.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "saves", "vigia")
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
EVERY = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
PORT = int(os.environ.get("RIBO_MCP_PORT", "8788"))


def call(name, args=None):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args or {}}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/mcp", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["result"]["content"]


def text(name, args=None):
    return json.loads(call(name, args)[0]["text"])


def probe(example, scene):
    env = dict(os.environ, SCENE=scene, CARGO_TARGET_DIR="target/check")
    try:
        r = subprocess.run(["cargo", "run", "--release", "--example", example], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=400)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        return lines if lines else ["(sem saída) " + r.stderr[-300:]]
    except Exception as e:  # noqa: BLE001
        return [f"(falhou: {e})"]


def snapshot(i):
    snap = {"hora": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = text("get_params")
    snap["epoch"] = p["epoch"]
    snap["pausa"] = p["pausa"]
    snap["passos_por_frame"] = p["passos_por_frame"]
    snap["mundo"] = p["mundo"]
    snap["mudados"] = {m["nome"]: m["atual"] for m in p["mudados"]}
    rows = text("get_stats", {"last": 30})["amostras"]
    keys = [k for k in rows[-1] if k != "epoch"]
    snap["stats_media_30"] = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    snap["stats_ultima"] = rows[-1]
    snap["especies_15"] = text("species", {"threshold": 0.15})
    snap["especies_05"] = {k: v for k, v in text("species", {"threshold": 0.05}).items() if k != "maiores"}
    h = text("habitat", {"block": 256})
    snap["altura"] = h["por_altura_de_cima_para_baixo"]
    snap["coluna"] = h["por_coluna"]
    # Sondas numa cópia do autosave (não tocam no mundo a correr).
    src = os.path.join(ROOT, "saves", "autosave.ribo")
    tmp = os.path.join(tempfile.gettempdir(), "vigia_cena.ribo")
    try:
        shutil.copyfile(src, tmp)
        snap["autosave_hora"] = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(src)))
        snap["presas"] = probe("probe_prey", tmp)
        snap["predacao"] = probe("probe_predation", tmp)
        snap["sinais"] = probe("probe_signals", tmp)
    except Exception as e:  # noqa: BLE001
        snap["sondas_erro"] = str(e)
    for name, args in (("mundo", {"size": 1024}), ("pormenor", {"size": 768, "x": 0.5, "y": 0.5, "span": 60})):
        try:
            img = [c for c in call("screenshot", args) if c["type"] == "image"][0]["data"]
            with open(os.path.join(OUT, f"retrato_{i:02d}_{name}.png"), "wb") as f:
                f.write(base64.b64decode(img))
        except Exception as e:  # noqa: BLE001
            snap[f"imagem_{name}_erro"] = str(e)
    with open(os.path.join(OUT, f"retrato_{i:02d}.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    s = snap["stats_media_30"]
    line = f"{snap['hora']}  epoch {snap['epoch']}  vivos {s['vivos']:.0f}  geração {s['geração média']:.0f}  espécies>=1% {snap['especies_15']['grupos_com_1pct']}  protease {s.get('% com protease', 0):.1f}%  mortes por protease {s.get('mortes por protease / 1000 epochs', 0):.0f}"
    with open(os.path.join(OUT, "resumo.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for i in range(COUNT):
        t0 = time.time()
        try:
            snapshot(i)
        except Exception as e:  # noqa: BLE001
            msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  retrato {i} falhou: {e}"
            with open(os.path.join(OUT, "resumo.txt"), "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            print(msg, flush=True)
        if i + 1 < COUNT:
            time.sleep(max(0.0, EVERY - (time.time() - t0)))


if __name__ == "__main__":
    main()
