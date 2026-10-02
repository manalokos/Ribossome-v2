//! Editor local (página web em http://127.0.0.1:8787, só nesta máquina):
//! mostra e edita a tabela dos aminoácidos ao vivo e lista os órgãos com
//! todas as variantes. A página envia a tabela inteira a cada alteração; a
//! app aplica-a no passo seguinte (`poll`). "Gravar" escreve
//! assets/aminoacidos.json; "recarregar" lê-o de novo.

use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use crate::life::organs::{GAIN_DEFAULT, ORGAN_NAMES, ORGAN_SYMBOLS, ORGAN_TYPES, describe};
use crate::life::table::{self, AminoRow};

pub const PORT: u16 = 8787;
const PAGE: &str = include_str!("../assets/editor.html");

pub struct Editor {
    rx: Receiver<Vec<AminoRow>>,
    pub url: String,
}

/// Variantes de cada órgão (tipo × parâmetro), para a página.
fn organs_json() -> String {
    let mut out = Vec::new();
    for t in 0..ORGAN_TYPES {
        // O parâmetro vai de 0 a 63 / ORGAN_TYPES (ver translate_organs).
        let max_p = (63 / ORGAN_TYPES) as u8;
        let variants: Vec<serde_json::Value> = (0..=max_p)
            .map(|p| serde_json::json!({ "param": p, "texto": describe(t as u8, p, GAIN_DEFAULT) }))
            .collect();
        out.push(serde_json::json!({
            "tipo": t,
            "nome": ORGAN_NAMES[t],
            "simbolo": ORGAN_SYMBOLS[t].to_string(),
            "variantes": variants,
        }));
    }
    serde_json::Value::Array(out).to_string()
}

impl Editor {
    /// Arranca o servidor numa thread. None se a porta estiver ocupada.
    pub fn start(initial: Vec<AminoRow>) -> Option<Self> {
        let server = match tiny_http::Server::http(("127.0.0.1", PORT)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("editor: não consegui abrir a porta {PORT}: {e}");
                return None;
            }
        };
        let (tx, rx) = channel();
        let shared = Arc::new(Mutex::new(initial));
        let organs = organs_json();
        std::thread::spawn(move || {
            let header = |ct: &str| tiny_http::Header::from_bytes(&b"Content-Type"[..], ct.as_bytes()).unwrap();
            for mut req in server.incoming_requests() {
                let url = req.url().to_string();
                let method = req.method().clone();
                let reply = match (method, url.as_str()) {
                    (tiny_http::Method::Get, "/") => {
                        tiny_http::Response::from_string(PAGE).with_header(header("text/html; charset=utf-8"))
                    }
                    (tiny_http::Method::Get, "/api/table") => {
                        let rows = shared.lock().unwrap().clone();
                        tiny_http::Response::from_string(serde_json::to_string(&rows).unwrap())
                            .with_header(header("application/json"))
                    }
                    (tiny_http::Method::Get, "/api/organs") => {
                        tiny_http::Response::from_string(organs.clone()).with_header(header("application/json"))
                    }
                    (tiny_http::Method::Post, "/api/table") => {
                        let mut body = String::new();
                        let _ = req.as_reader().read_to_string(&mut body);
                        match table::parse(&body) {
                            Ok(rows) => {
                                *shared.lock().unwrap() = rows.clone();
                                let _ = tx.send(rows);
                                tiny_http::Response::from_string("aplicado")
                            }
                            Err(e) => tiny_http::Response::from_string(format!("erro: {e}")).with_status_code(400),
                        }
                    }
                    (tiny_http::Method::Post, "/api/save") => {
                        let rows = shared.lock().unwrap().clone();
                        match table::save(&rows) {
                            Ok(()) => tiny_http::Response::from_string(format!("gravado em {}", table::TABLE_PATH)),
                            Err(e) => tiny_http::Response::from_string(format!("erro: {e}")).with_status_code(500),
                        }
                    }
                    (tiny_http::Method::Post, "/api/reload") => {
                        let (rows, src) = table::load();
                        *shared.lock().unwrap() = rows.clone();
                        let _ = tx.send(rows);
                        tiny_http::Response::from_string(format!("recarregado de {src}"))
                    }
                    _ => tiny_http::Response::from_string("não encontrado").with_status_code(404),
                };
                let _ = req.respond(reply);
            }
        });
        let url = format!("http://127.0.0.1:{PORT}/");
        log::info!("editor dos aminoácidos em {url}");
        Some(Self { rx, url })
    }

    /// A tabela mais recente enviada pela página, se houver.
    pub fn poll(&self) -> Option<Vec<AminoRow>> {
        self.rx.try_iter().last()
    }

    /// Abre a página no browser do sistema.
    pub fn open_browser(&self) {
        #[cfg(windows)]
        let r = std::process::Command::new("cmd").args(["/C", "start", "", &self.url]).spawn();
        #[cfg(not(windows))]
        let r = std::process::Command::new("xdg-open").arg(&self.url).spawn();
        if let Err(e) = r {
            log::error!("não consegui abrir o browser: {e}");
        }
    }
}
