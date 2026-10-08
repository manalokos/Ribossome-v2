//! Editor local (página web em http://127.0.0.1:8787, só nesta máquina):
//! mostra e edita a tabela dos aminoácidos ao vivo e lista os órgãos com
//! todas as variantes. A página envia a tabela inteira a cada alteração; a
//! app aplica-a no passo seguinte (`poll`). "Gravar" escreve
//! assets/aminoacidos.json; "recarregar" lê-o de novo.

use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use crate::life::organs::{GAIN_DEFAULT, ORGAN_NAMES_EN, ORGAN_PROPS, ORGAN_SYMBOLS, ORGAN_TYPES, VARIANTS, describe};
use crate::life::table::{self, AminoRow, OrganCode, OrganRow};

/// O que a página mudou.
pub enum Update {
    Amino(Vec<AminoRow>),
    Organs(Vec<OrganRow>),
    Code(OrganCode),
}

pub const PORT: u16 = 8787;

/// Porta em uso: `RIBO_EDITOR_PORT` ou PORT.
pub fn port() -> u16 {
    std::env::var("RIBO_EDITOR_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(PORT)
}
const PAGE: &str = include_str!("../assets/editor.html");

pub struct Editor {
    rx: Receiver<Update>,
    pub url: String,
}

/// Esquema e descrição das variantes de cada órgão, com a tabela atual.
fn organs_json(table: &[OrganRow]) -> String {
    let mut out = Vec::new();
    for t in 0..ORGAN_TYPES {
        let props: Vec<serde_json::Value> =
            ORGAN_PROPS[t].iter().map(|p| serde_json::json!({ "nome": p.name, "desc": p.desc })).collect();
        let textos: Vec<String> = (0..VARIANTS as u8).map(|p| describe(t as u8, p, GAIN_DEFAULT, table)).collect();
        out.push(serde_json::json!({
            "tipo": t,
            "nome": ORGAN_NAMES_EN[t],
            "simbolo": ORGAN_SYMBOLS[t].to_string(),
            "props": props,
            "textos": textos,
        }));
    }
    serde_json::Value::Array(out).to_string()
}

impl Editor {
    /// Arranca o servidor numa thread. None se a porta estiver ocupada.
    pub fn start(initial: Vec<AminoRow>, initial_organs: Vec<OrganRow>, initial_code: OrganCode) -> Option<Self> {
        let port = port();
        let server = match tiny_http::Server::http(("127.0.0.1", port)) {
            Ok(s) => s,
            Err(e) => {
                log::error!("editor: could not open port {port}: {e}");
                return None;
            }
        };
        let (tx, rx) = channel();
        let shared = Arc::new(Mutex::new(initial));
        let shared_org = Arc::new(Mutex::new(initial_organs));
        let shared_code = Arc::new(Mutex::new(initial_code));
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
                                tiny_http::Response::from_string("applied")
                            }
                            Err(e) => tiny_http::Response::from_string(format!("error: {e}")).with_status_code(400),
                        }
                    }
                    (tiny_http::Method::Get, "/api/code") => {
                        let code = shared_code.lock().unwrap().clone();
                        tiny_http::Response::from_string(serde_json::to_string(&code).unwrap())
                            .with_header(header("application/json"))
                    }
                    (tiny_http::Method::Post, "/api/code") => {
                        let mut body = String::new();
                        let _ = req.as_reader().read_to_string(&mut body);
                        match table::parse_code(&body) {
                            Ok(code) => {
                                *shared_code.lock().unwrap() = code.clone();
                                let _ = tx.send(Update::Code(code));
                                tiny_http::Response::from_string("applied")
                            }
                            Err(e) => tiny_http::Response::from_string(format!("error: {e}")).with_status_code(400),
                        }
                    }
                    (tiny_http::Method::Get, "/api/organs") => {
                        let rows = shared_org.lock().unwrap().clone();
                        tiny_http::Response::from_string(organs_json(&rows)).with_header(header("application/json"))
                    }
                    (tiny_http::Method::Post, "/api/table") => {
                        let mut body = String::new();
                        let _ = req.as_reader().read_to_string(&mut body);
                        match table::parse(&body) {
                            Ok(rows) => {
                                *shared.lock().unwrap() = rows.clone();
                                let _ = tx.send(Update::Amino(rows));
                                tiny_http::Response::from_string("applied")
                            }
                            Err(e) => tiny_http::Response::from_string(format!("error: {e}")).with_status_code(400),
                        }
                    }
                    (tiny_http::Method::Post, "/api/save") => {
                        let rows = shared.lock().unwrap().clone();
                        let orgs = shared_org.lock().unwrap().clone();
                        let code = shared_code.lock().unwrap().clone();
                        match table::save(&rows).and_then(|_| table::save_organs(&orgs)).and_then(|_| table::save_code(&code)) {
                            Ok(()) => tiny_http::Response::from_string(format!(
                                "saved to {}, {} and {}",
                                table::TABLE_PATH,
                                table::ORGANS_PATH,
                                table::CODE_PATH
                            )),
                            Err(e) => tiny_http::Response::from_string(format!("error: {e}")).with_status_code(500),
                        }
                    }
                    (tiny_http::Method::Post, "/api/reload") => {
                        let (rows, src) = table::load();
                        let (orgs, src_o) = table::load_organs();
                        let (code, src_c) = table::load_code();
                        *shared.lock().unwrap() = rows.clone();
                        *shared_org.lock().unwrap() = orgs.clone();
                        *shared_code.lock().unwrap() = code.clone();
                        let _ = tx.send(Update::Amino(rows));
                        let _ = tx.send(Update::Organs(orgs));
                        let _ = tx.send(Update::Code(code));
                        tiny_http::Response::from_string(format!("reloaded from {src}, {src_o} and {src_c}"))
                    }
                    _ => tiny_http::Response::from_string("not found").with_status_code(404),
                };
                let _ = req.respond(reply);
            }
        });
        let url = format!("http://127.0.0.1:{port}/");
        log::info!("amino acid editor at {url}");
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
            log::error!("could not open the browser: {e}");
        }
    }
}
