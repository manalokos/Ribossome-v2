//! Servidor MCP local (http://127.0.0.1:8788/mcp, só nesta máquina): deixa
//! um assistente (Claude Code: `claude mcp add --transport http ribossome
//! http://127.0.0.1:8788/mcp`) ler e mudar os parâmetros, ver estatísticas,
//! o habitat e imagens do mundo que está a correr.
//!
//! Transporte "streamable HTTP" mínimo: cada POST traz um pedido JSON-RPC e
//! recebe a resposta em JSON (sem SSE). A thread do servidor só fala o
//! protocolo; as ferramentas correm na thread principal (`poll` entre
//! frames), por isso não há corridas com a GPU nem com a interface.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use serde_json::{Value, json};

pub const PORT: u16 = 8788;
const PROTOCOL: &str = "2025-06-18";
/// Quanto a thread do servidor espera pela app (o habitat lê a grelha toda).
const REPLY_TIMEOUT: Duration = Duration::from_secs(60);

/// Um pedido de ferramenta para a thread principal.
pub struct Call {
    pub tool: String,
    pub args: Value,
    pub reply: Sender<Result<Vec<Value>, String>>,
}

pub struct Mcp {
    pub rx: Receiver<Call>,
    pub url: String,
}

/// Conteúdo de texto de uma resposta.
pub fn text(s: impl Into<String>) -> Value {
    json!({ "type": "text", "text": s.into() })
}

/// Conteúdo JSON (como texto formatado).
pub fn json_text(v: &Value) -> Value {
    text(serde_json::to_string_pretty(v).unwrap_or_default())
}

/// Conteúdo de imagem PNG.
pub fn png(bytes: &[u8]) -> Value {
    json!({ "type": "image", "data": base64(bytes), "mimeType": "image/png" })
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// As ferramentas (nome, descrição, esquema dos argumentos).
fn tools() -> Value {
    let obj = |props: Value, required: &[&str]| json!({ "type": "object", "properties": props, "required": required });
    json!([
        {
            "name": "get_params",
            "description": "Todos os parâmetros da simulação (SimParams) com o valor atual e o valor por omissão do código; 'mudados' lista só os que diferem. Inclui o epoch, a pausa e os passos por frame.",
            "inputSchema": obj(json!({}), &[]),
        },
        {
            "name": "set_params",
            "description": "Muda parâmetros da simulação pelo nome (os de get_params). Aplica-se no passo seguinte e fica no log. Nomes desconhecidos dão erro e nada é mudado.",
            "inputSchema": obj(json!({
                "params": { "type": "object", "description": "nome -> valor, ex. {\"photo_yield\": 0.5}", "additionalProperties": { "type": "number" } }
            }), &["params"]),
        },
        {
            "name": "set_world",
            "description": "Definições do mundo que não são parâmetros: fluido ligado, física do terreno (grãos), repulsão entre agentes e força de todas as fumarolas (0 = sem calor nem química). Só muda o que for dado.",
            "inputSchema": obj(json!({
                "fluid_enabled": { "type": "boolean" },
                "terrain_enabled": { "type": "boolean" },
                "contact_enabled": { "type": "boolean" },
                "fumarole_gain": { "type": "number" }
            }), &[]),
        },
        {
            "name": "get_stats",
            "description": "Estatísticas dos gráficos (vivos, nascimentos e mortes por 1000 epochs, energia, tamanhos, gerações, % de agentes com cada órgão...): as últimas N amostras (uma a cada 'every' epochs).",
            "inputSchema": obj(json!({
                "last": { "type": "integer", "description": "quantas amostras (por omissão 10)" },
                "series": { "type": "array", "items": { "type": "string" }, "description": "só estas séries (nomes como em get_stats sem filtro); vazio = todas" }
            }), &[]),
        },
        {
            "name": "habitat",
            "description": "Mapa do mundo em blocos: por linha de altura (de cima para baixo) e por coluna, e os blocos mais ricos e mais povoados — monómeros ativados e gastos por célula, temperatura, luz, redutor, entulho, agentes e quantos têm fotossistema, boca e quimiossíntese.",
            "inputSchema": obj(json!({
                "block": { "type": "integer", "description": "lado do bloco em células (por omissão 128; o mundo tem 2048)" }
            }), &[]),
        },
        {
            "name": "screenshot",
            "description": "Imagem PNG de uma vista do mundo. Vistas: 0 normal, 1-4 ativados por canal, 5 gastos, 6 terreno, 7 temperatura, 8 UV/luz, 9 fluido, 10 redutor. Por omissão o mundo inteiro; 'camera': true usa o enquadramento que o Filipe está a ver.",
            "inputSchema": obj(json!({
                "view": { "type": "integer", "description": "vista (por omissão 0)" },
                "size": { "type": "integer", "description": "lado em píxeis (por omissão 768, máx. 2048)" },
                "camera": { "type": "boolean", "description": "usar o enquadramento do ecrã" },
                "brightness": { "type": "number", "description": "brilho dos monómeros (por omissão 0.5)" }
            }), &[]),
        },
        {
            "name": "save_scene",
            "description": "Grava a cena atual em saves/<name>.ribo (não mexe no autosave).",
            "inputSchema": obj(json!({ "name": { "type": "string" } }), &["name"]),
        },
        {
            "name": "pause",
            "description": "Pausa (paused: true) ou continua (false) a simulação; opcionalmente muda os passos por frame.",
            "inputSchema": obj(json!({
                "paused": { "type": "boolean" },
                "steps_per_frame": { "type": "integer" }
            }), &[]),
        },
    ])
}

fn rpc_result(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_error(id: &Value, code: i64, msg: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": msg } })
}

/// Responde a uma mensagem JSON-RPC (None = notificação, sem resposta).
fn handle(msg: &Value, tx: &Sender<Call>) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg["method"].as_str().unwrap_or("");
    Some(match method {
        "initialize" => {
            let v = msg["params"]["protocolVersion"].as_str().unwrap_or(PROTOCOL);
            rpc_result(
                &id,
                json!({
                    "protocolVersion": v,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "ribossome", "version": env!("CARGO_PKG_VERSION") },
                    "instructions": "Ribossome v4: simulador de vida artificial a correr na máquina do Filipe. Muda parâmetros com cuidado e diz sempre o que mudaste; ele vê o mundo ao vivo.",
                }),
            )
        }
        "ping" => rpc_result(&id, json!({})),
        "tools/list" => rpc_result(&id, json!({ "tools": tools() })),
        "tools/call" => {
            let (rtx, rrx) = channel();
            let call = Call {
                tool: msg["params"]["name"].as_str().unwrap_or("").to_string(),
                args: msg["params"].get("arguments").cloned().unwrap_or(json!({})),
                reply: rtx,
            };
            if tx.send(call).is_err() {
                return Some(rpc_error(&id, -32603, "a app está a fechar"));
            }
            match rrx.recv_timeout(REPLY_TIMEOUT) {
                Ok(Ok(content)) => rpc_result(&id, json!({ "content": content, "isError": false })),
                Ok(Err(e)) => rpc_result(&id, json!({ "content": [text(e)], "isError": true })),
                Err(_) => rpc_result(
                    &id,
                    json!({ "content": [text("a app não respondeu (janela minimizada ou ocupada?)")], "isError": true }),
                ),
            }
        }
        _ => rpc_error(&id, -32601, &format!("método desconhecido: {method}")),
    })
}

impl Mcp {
    /// Arranca o servidor numa thread. None se a porta estiver ocupada.
    pub fn start() -> Option<Self> {
        let server = match tiny_http::Server::http(("127.0.0.1", PORT)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("mcp: não consegui abrir a porta {PORT}: {e}");
                return None;
            }
        };
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let header = |ct: &str| tiny_http::Header::from_bytes(&b"Content-Type"[..], ct.as_bytes()).unwrap();
            for mut req in server.incoming_requests() {
                if req.url() != "/mcp" {
                    let _ = req.respond(tiny_http::Response::from_string("não encontrado").with_status_code(404));
                    continue;
                }
                if *req.method() != tiny_http::Method::Post {
                    // Sem SSE: só POST.
                    let _ = req.respond(tiny_http::Response::from_string("").with_status_code(405));
                    continue;
                }
                let mut body = String::new();
                let _ = req.as_reader().read_to_string(&mut body);
                let reply = match serde_json::from_str::<Value>(&body) {
                    Err(e) => Some(rpc_error(&Value::Null, -32700, &format!("JSON inválido: {e}"))),
                    Ok(Value::Array(batch)) => {
                        let out: Vec<Value> = batch.iter().filter_map(|m| handle(m, &tx)).collect();
                        (!out.is_empty()).then_some(Value::Array(out))
                    }
                    Ok(msg) => handle(&msg, &tx),
                };
                let _ = match reply {
                    Some(v) => req.respond(tiny_http::Response::from_string(v.to_string()).with_header(header("application/json"))),
                    None => req.respond(tiny_http::Response::from_string("").with_status_code(202)),
                };
            }
        });
        let url = format!("http://127.0.0.1:{PORT}/mcp");
        log::info!("mcp: servidor em {url}");
        Some(Self { rx, url })
    }
}

/// Resumo do habitat por blocos (o que examples/probe_habitat.rs imprime).
pub fn habitat(gpu: &crate::gpu::Gpu, w: &crate::world::World, block: usize) -> Value {
    let cfg = w.cfg;
    let g = cfg.grid_size as usize;
    let f = cfg.fluid_size as usize;
    let b = block.clamp(16, g);
    let nb = g.div_ceil(b);
    let cells = w.read_cells_blocking(gpu);
    let gamma = w.read_gamma_blocking(gpu);
    let temp = w.read_f32_blocking(gpu, &w.temp_buf);
    let redox = w.read_f32_blocking(gpu, &w.redox_buf);
    let light = w.read_f32_blocking(gpu, &w.light_buf);
    let agents = w.read_agents_blocking(gpu);
    let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let ls = g / crate::shaders::LIGHT_DIV as usize;
    #[derive(Default, Clone)]
    struct Blk {
        act: f64,
        spent: f64,
        temp: f64,
        light: f64,
        redox: f64,
        rubble: f64,
        n: f64,
        agents: u32,
        photo: u32,
        mouth: u32,
        chemo: u32,
    }
    let mut blk = vec![Blk::default(); nb * nb];
    for y in 0..g {
        for x in 0..g {
            let i = y * g + x;
            let k = &mut blk[(y / b) * nb + x / b];
            for c in 0..4 {
                k.act += (cells[i * 4 + c] & 0xFFFF) as f64;
                k.spent += (cells[i * 4 + c] >> 16) as f64;
            }
            let fi = (y * f / g) * f + x * f / g;
            k.temp += temp[fi] as f64;
            k.redox += redox[fi] as f64;
            let ld = crate::shaders::LIGHT_DIV as usize;
            k.light += light[(y / ld) * ls + x / ld] as f64;
            k.rubble += (gamma[i] > 0) as u32 as f64;
            k.n += 1.0;
        }
    }
    let wpc = cfg.world_units_per_cell as f32;
    let mut alive = 0u32;
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        alive += 1;
        let (cx, cy) = (((a.pos_x / wpc) as usize).min(g - 1), ((a.pos_y / wpc) as usize).min(g - 1));
        let k = &mut blk[(cy / b) * nb + cx / b];
        k.agents += 1;
        let mut has = [false; 16];
        for r in 0..a.body_len as usize {
            let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
            if o != 0 {
                has[((o & 0xF) - 1) as usize] = true;
            }
        }
        k.photo += has[10] as u32;
        k.mouth += has[0] as u32;
        k.chemo += has[14] as u32;
    }
    let summary = |list: &[&Blk]| -> Value {
        let n: f64 = list.iter().map(|k| k.n).sum::<f64>().max(1.0);
        let r = |x: f64| (x * 100.0).round() / 100.0;
        json!({
            "ativados_por_celula": r(list.iter().map(|k| k.act).sum::<f64>() / n),
            "gastos_por_celula": r(list.iter().map(|k| k.spent).sum::<f64>() / n),
            "temperatura": r(list.iter().map(|k| k.temp).sum::<f64>() / n),
            "luz": r(list.iter().map(|k| k.light).sum::<f64>() / n),
            "redutor": r(list.iter().map(|k| k.redox).sum::<f64>() / n),
            "entulho_pct": r(100.0 * list.iter().map(|k| k.rubble).sum::<f64>() / n),
            "agentes": list.iter().map(|k| k.agents).sum::<u32>(),
            "com_fotossistema": list.iter().map(|k| k.photo).sum::<u32>(),
            "com_boca": list.iter().map(|k| k.mouth).sum::<u32>(),
            "com_quimiossintese": list.iter().map(|k| k.chemo).sum::<u32>(),
        })
    };
    let rows: Vec<Value> = (0..nb)
        .rev()
        .map(|y| {
            let list: Vec<&Blk> = (0..nb).map(|x| &blk[y * nb + x]).collect();
            let mut v = summary(&list);
            v["y"] = json!(y);
            v
        })
        .collect();
    let cols: Vec<Value> = (0..nb)
        .map(|x| {
            let list: Vec<&Blk> = (0..nb).map(|y| &blk[y * nb + x]).collect();
            let mut v = summary(&list);
            v["x"] = json!(x);
            v
        })
        .collect();
    let mut idx: Vec<usize> = (0..nb * nb).collect();
    let top = |idx: &[usize]| -> Vec<Value> {
        idx.iter()
            .take(8)
            .map(|&i| {
                let mut v = summary(&[&blk[i]]);
                v["x"] = json!(i % nb);
                v["y"] = json!(i / nb);
                v
            })
            .collect()
    };
    idx.sort_by(|&a, &c| (blk[c].act / blk[c].n.max(1.0)).total_cmp(&(blk[a].act / blk[a].n.max(1.0))));
    let richest = top(&idx);
    idx.sort_by_key(|&i| std::cmp::Reverse(blk[i].agents));
    let crowded = top(&idx);
    json!({
        "epoch": w.params.epoch,
        "vivos": alive,
        "bloco_celulas": b,
        "blocos_por_lado": nb,
        "nota": "y cresce para cima (y = blocos_por_lado-1 é a superfície); x da esquerda para a direita",
        "por_altura_de_cima_para_baixo": rows,
        "por_coluna": cols,
        "mais_ricos_em_ativados": richest,
        "mais_povoados": crowded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn notifications_get_no_reply_and_tools_are_listed() {
        let (tx, _rx) = channel();
        assert!(handle(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }), &tx).is_none());
        let r = handle(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }), &tx).unwrap();
        let names: Vec<&str> = r["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"set_params") && names.contains(&"screenshot"));
    }
}
