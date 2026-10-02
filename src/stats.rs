//! Estatísticas da população ao longo do tempo: amostras de N em N epochs
//! (redução na GPU, `shaders/life/observe.wgsl`, leitura assíncrona), um
//! histórico de tamanho FIXO em memória (quando enche, fica-se com um ponto
//! em cada dois: cobre a sessão toda com resolução decrescente) e um CSV em
//! `logs/estatisticas.csv` com rotação (nunca passa de ~2 × CSV_MAX_BYTES).

use std::io::Write;

use crate::life::organs::{ORGAN_NAMES, ORGAN_TYPES};
use crate::world::{Ledger, LifeCounters};

/// Palavras do buffer de estatísticas (a ordem de observe.wgsl).
pub const STAT_WORDS: usize = 8 + ORGAN_TYPES;
/// Pontos no histórico em memória.
const MAX_POINTS: usize = 3000;
const CSV_PATH: &str = "logs/estatisticas.csv";
const CSV_OLD: &str = "logs/estatisticas.1.csv";
const CSV_MAX_BYTES: u64 = 20 << 20;

/// Séries fixas (as dos órgãos vêm a seguir, uma por tipo).
const BASE: [&str; 12] = [
    "vivos",
    "nascimentos / 1000 epochs",
    "mortes / 1000 epochs",
    "energia média",
    "resíduos por corpo",
    "bases por genoma",
    "geração média",
    "geração máxima",
    "% ligados",
    "órgãos por agente",
    "% monómeros livres ativados",
    "% matéria nos agentes",
];

pub fn series_names() -> Vec<String> {
    let mut v: Vec<String> = BASE.iter().map(|s| s.to_string()).collect();
    v.extend(ORGAN_NAMES.iter().map(|n| format!("% com {n}")));
    v
}

#[derive(Clone)]
pub struct Point {
    pub epoch: u32,
    pub v: Vec<f32>,
}

pub struct History {
    pub names: Vec<String>,
    pub points: Vec<Point>,
    /// Só se guarda uma amostra em cada `stride` (dobra quando enche).
    stride: u32,
    seen: u32,
    /// Contadores da amostra anterior (para as taxas).
    last: Option<(u32, u32, u32)>,
    /// Epochs entre amostras.
    pub every: u32,
    pub next_epoch: u32,
    csv_ok: bool,
}

impl Default for History {
    fn default() -> Self {
        Self {
            names: series_names(),
            points: Vec::new(),
            stride: 1,
            seen: 0,
            last: None,
            every: 2000,
            next_epoch: 0,
            csv_ok: true,
        }
    }
}

impl History {
    /// Junta uma amostra (palavras da GPU + livro-razão + contadores).
    pub fn push(&mut self, epoch: u32, w: &[u32], ledger: Option<Ledger>, c: Option<LifeCounters>) {
        let alive = w[0] as f32;
        let per = |x: u32| if alive > 0.0 { x as f32 / alive } else { 0.0 };
        let (births, deaths) = match (c, self.last) {
            (Some(c), Some((e0, b0, d0))) if epoch > e0 => {
                let k = 1000.0 / (epoch - e0) as f32;
                (c.births.saturating_sub(b0) as f32 * k, c.deaths.saturating_sub(d0) as f32 * k)
            }
            _ => (0.0, 0.0),
        };
        if let Some(c) = c {
            self.last = Some((epoch, c.births, c.deaths));
        }
        let (act, held) = ledger.map_or((0.0, 0.0), |l| {
            let free = l.free_total().max(1) as f32;
            let act: u64 = l.act.iter().map(|&x| x as u64).sum();
            (100.0 * act as f32 / free, 100.0 * l.held_total() as f32 / l.total().max(1) as f32)
        });
        let mut v = vec![
            alive,
            births,
            deaths,
            per(w[1]) / 10.0,
            per(w[2]),
            per(w[3]),
            per(w[5]),
            w[4] as f32,
            100.0 * per(w[6]),
            per(w[7]),
            act,
            held,
        ];
        v.extend((0..ORGAN_TYPES).map(|t| 100.0 * per(w[8 + t])));
        self.write_csv(epoch, &v);
        self.seen += 1;
        if !self.seen.is_multiple_of(self.stride) {
            return;
        }
        self.points.push(Point { epoch, v });
        if self.points.len() >= MAX_POINTS {
            // Cheio: um ponto em cada dois, e passa-se a guardar metade.
            let mut keep = Vec::with_capacity(MAX_POINTS / 2 + 1);
            for (i, p) in self.points.drain(..).enumerate() {
                if i % 2 == 0 {
                    keep.push(p);
                }
            }
            self.points = keep;
            self.stride *= 2;
        }
    }

    fn write_csv(&mut self, epoch: u32, v: &[f32]) {
        if !self.csv_ok {
            return;
        }
        let r = (|| -> std::io::Result<()> {
            std::fs::create_dir_all("logs")?;
            if std::fs::metadata(CSV_PATH).is_ok_and(|m| m.len() > CSV_MAX_BYTES) {
                let _ = std::fs::rename(CSV_PATH, CSV_OLD);
            }
            let new = !std::path::Path::new(CSV_PATH).exists();
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(CSV_PATH)?;
            if new {
                writeln!(f, "epoch,{}", self.names.join(","))?;
            }
            let row: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
            writeln!(f, "{epoch},{}", row.join(","))
        })();
        if let Err(e) = r {
            log::error!("{CSV_PATH}: {e} (deixo de escrever o CSV)");
            self.csv_ok = false;
        }
    }

    /// Para gravar com a cena.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "names": self.names,
            "stride": self.stride,
            "every": self.every,
            "rows": self.points.iter().map(|p| {
                let mut r = vec![p.epoch as f64];
                r.extend(p.v.iter().map(|&x| x as f64));
                r
            }).collect::<Vec<_>>(),
        })
    }

    /// Repõe de uma cena (as séries que mudaram de nome ficam a 0).
    pub fn from_json(v: &serde_json::Value) -> Self {
        let mut h = Self::default();
        let names: Vec<String> =
            v["names"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        h.stride = v["stride"].as_u64().unwrap_or(1).max(1) as u32;
        h.every = v["every"].as_u64().unwrap_or(2000).max(100) as u32;
        for row in v["rows"].as_array().into_iter().flatten() {
            let Some(r) = row.as_array() else { continue };
            let Some(epoch) = r.first().and_then(|x| x.as_f64()) else { continue };
            let vals: Vec<f32> = h
                .names
                .iter()
                .map(|n| {
                    names.iter().position(|m| m == n).and_then(|i| r.get(i + 1)).and_then(|x| x.as_f64()).unwrap_or(0.0)
                        as f32
                })
                .collect();
            h.points.push(Point { epoch: epoch as u32, v: vals });
        }
        h
    }
}

/// Grupos de séries por gráfico (índices em `series_names`).
fn groups() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("População", vec![0]),
        ("Nascimentos e mortes", vec![1, 2]),
        ("Órgãos (% dos agentes com cada um)", (12..12 + ORGAN_TYPES).collect()),
        ("Corpo e genoma", vec![4, 5, 9]),
        ("Energia, gerações e ligações", vec![3, 6, 7, 8]),
        ("Matéria", vec![10, 11]),
    ]
}

/// Os gráficos (separador "Gráficos").
pub fn draw(ui: &mut egui::Ui, h: &mut History) {
    ui.horizontal(|ui| {
        ui.label("amostra de");
        ui.add(egui::DragValue::new(&mut h.every).range(100..=1_000_000).speed(100));
        ui.label("em epochs");
    });
    ui.label(format!(
        "{} pontos (um em cada {} amostras); CSV completo em {CSV_PATH}",
        h.points.len(),
        h.stride
    ));
    if h.points.len() < 2 {
        ui.label("à espera de amostras…");
        return;
    }
    for (title, idx) in groups() {
        ui.separator();
        ui.strong(title);
        egui_plot::Plot::new(title)
            .height(160.0)
            .legend(egui_plot::Legend::default())
            .allow_scroll(false)
            .show(ui, |p| {
                for &i in &idx {
                    let pts: Vec<[f64; 2]> = h.points.iter().map(|q| [q.epoch as f64, q.v[i] as f64]).collect();
                    p.line(egui_plot::Line::new(h.names[i].clone(), egui_plot::PlotPoints::from(pts)));
                }
            });
    }
}
