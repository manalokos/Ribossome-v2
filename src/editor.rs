//! Editor local (página web em http://127.0.0.1:8787, só nesta máquina):
//! mostra e edita a tabela dos aminoácidos ao vivo e lista os órgãos com
//! todas as variantes. A página envia a tabela inteira a cada alteração; a
//! app aplica-a no passo seguinte (`poll`). "Gravar" escreve
//! assets/aminoacidos.json; "recarregar" lê-o de novo.

use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use crate::life::organs::{GAIN_DEFAULT, ORGAN_NAMES, ORGAN_SYMBOLS, ORGAN_TYPES, describe};
use crate::life::table::{self, AminoRow, OrganRow};

/// O que a página mudou.
pub enum Update {
    Amino(Vec<AminoRow>),
    Organs(Vec<OrganRow>),
}

pub const PORT: u16 = 8787;
const PAGE: &str = include_str!("../assets/editor.html");

pub struct Editor {
    rx: Receiver<Update>,
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
    pub fn start(initial: Vec<AminoRow>, initial_organs: Vec<OrganRow>) -> Option<Self> {
        let server = match tiny_http::Server::http(("127.0.0.1", PORT)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("editor: não consegui abrir a porta {PORT}: {e}");
                return None;
            }
        };
        let (tx, rx) = channel();
        let shared = Arc::new(Mutex::new(initial));
        let shared_org = Arc::new(Mutex::new(initial_organs));
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
                    (tiny_http::Method::Get, "/api/organ_table") => {
                        let rows = shared_org.lock().unwrap().clone();
                        tiny_http::Response::from_string(serde_json::to_string(&rows).unwrap())
                            .with_header(header("application/json"))
                    }
                    (tiny_http::Method::Post, "/api/organ_table") => {
                        let mut body = String::new();
                        let _ = req.as_reader().read_to_string(&mut body);
                        match table::parse_organs(&body) {
                            Ok(rows) => {
                                *shared_org.lock().unwrap() = rows.clone();
                                let _ = tx.send(Update::Organs(rows));
                                tiny_http::Response::from_string("aplicado")
                            }
                            Err(e) => tiny_http::Response::from_string(format!("erro: {e}")).with_status_code(400),
                        }
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
                                let _ = tx.send(Update::Amino(rows));
                                tiny_http::Response::from_string("aplicado")
                            }
                            Err(e) => tiny_http::Response::from_string(format!("erro: {e}")).with_status_code(400),
                        }
                    }
                    (tiny_http::Method::Post, "/api/save") => {
                        let rows = shared.lock().unwrap().clone();
                        let orgs = shared_org.lock().unwrap().clone();
                        match table::save(&rows).and_then(|_| table::save_organs(&orgs)) {
                            Ok(()) => tiny_http::Response::from_string(format!(
                                "gravado em {} e {}",
                                table::TABLE_PATH,
                                table::ORGANS_PATH
                            )),
                            Err(e) => tiny_http::Response::from_string(format!("erro: {e}")).with_status_code(500),
                        }
                    }
                    (tiny_http::Method::Post, "/api/reload") => {
                        let (rows, src) = table::load();
                        let (orgs, src_o) = table::load_organs();
                        *shared.lock().unwrap() = rows.clone();
                        *shared_org.lock().unwrap() = orgs.clone();
                        let _ = tx.send(Update::Amino(rows));
                        let _ = tx.send(Update::Organs(orgs));
                        tiny_http::Response::from_string(format!("recarregado de {src} e {src_o}"))
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

    /// As alterações enviadas pela página desde a última chamada.
    pub fn poll(&self) -> Vec<Update> {
        self.rx.try_iter().collect()
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
