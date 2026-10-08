"""Vigia leve da noite: de hora a hora lê o mundo que está a correr (pelo MCP
da app) e guarda um retrato em saves/vigia_<data>/: parâmetros, estatísticas,
espécies, habitat, duas imagens e a cena. Não muda nada no mundo nem corre
sondas na placa (ao contrário de scripts/vigia.py).

  python -X utf8 scripts/vigia_leve.py [horas=10] [intervalo_min=60]
"""
import base64, datetime, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8788/mcp"
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 10
EVERY = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 3600
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saves")
TAG = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
OUT = os.path.join(ROOT, "vigia_" + TAG)
os.makedirs(OUT, exist_ok=True)


def call(name, args=None, timeout=120):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args or {}}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["result"]["content"]


def text(content):
    for c in content:
        if c.get("type") == "text":
            try:
                return json.loads(c["text"])
            except ValueError:
                return c["text"]
    return None


def image(content, path):
    for c in content:
        if c.get("type") == "image":
            open(path, "wb").write(base64.b64decode(c["data"]))


def log(msg):
    line = datetime.datetime.now().strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    open(os.path.join(OUT, "resumo.txt"), "a", encoding="utf-8").write(line + "\n")


n = int(HOURS * 3600 / EVERY) + 1
log(f"vigia: {n} retratos de {EVERY / 60:.0f} em {EVERY / 60:.0f} min em {OUT}")
for i in range(n):
    try:
        snap = {"hora": datetime.datetime.now().isoformat(timespec="seconds")}
        snap["params"] = text(call("get_params"))
        snap["stats"] = text(call("get_stats"))
        snap["species"] = text(call("species"))
        snap["habitat"] = text(call("habitat"))
        json.dump(snap, open(os.path.join(OUT, f"retrato_{i:02d}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        image(call("screenshot", {"size": 1024}), os.path.join(OUT, f"retrato_{i:02d}_mundo.png"))
        image(call("screenshot", {"size": 1024, "camera": True}), os.path.join(OUT, f"retrato_{i:02d}_vista.png"))
        call("save_scene", {"name": f"vigia_{TAG}_{i:02d}"})
        p, sp = snap["params"], snap["species"]
        log(f"retrato {i:02d}: epoch {p.get('epoch')}, vivos {sp.get('vivos')}, grupos {sp.get('grupos')} ({sp.get('grupos_com_1pct')} com >=1%)")
    except Exception as e:  # a app pode estar fechada ou ocupada: tenta outra vez na hora seguinte
        log(f"retrato {i:02d}: falhou ({e})")
    if i + 1 < n:
        time.sleep(EVERY)
log("vigia: fim")
