//! Estatísticas da população ao longo do tempo: amostras de N em N epochs
//! (redução na GPU, `shaders/life/observe.wgsl`, leitura assíncrona), um
//! histórico em memória até MAX_POINTS amostras (quando enche, fica-se com
//! uma em cada duas: cobre a sessão toda com resolução decrescente) e um CSV
//! em `logs/estatisticas.csv` com rotação (nunca passa de ~2 × CSV_MAX_BYTES).
//! Os gráficos desenham uma versão REDUZIDA (médias por intervalo, até
//! PLOT_POINTS por série), refeita só quando entra uma amostra nova.

use std::io::Write;

use crate::life::organs::{ORGAN_NAMES, ORGAN_TYPES};
use crate::world::{Ledger, LifeCounters};

/// Palavras do buffer de estatísticas (a ordem de observe.wgsl).
pub const STAT_WORDS: usize = 8 + ORGAN_TYPES;
/// Amostras no histórico em memória (~110 MB no limite; a 2000 epochs por
/// amostra só se chega lá ao fim de 2 mil milhões de epochs).
const MAX_POINTS: usize = 1_000_000;
/// Pontos por série desenhados.
const PLOT_POINTS: usize = 4000;
const CSV_PATH: &str = "logs/estatisticas.csv";
const CSV_OLD: &str = "logs/estatisticas.1.csv";
const CSV_MAX_BYTES: u64 = 20 << 20;

/// Séries fixas (as dos órgãos vêm a seguir, uma por tipo).
const BASE: [&str; 13] = [
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
    "mortes por protease / 1000 epochs",
];

pub fn series_names() -> Vec<String> {
    let mut v: Vec<String> = BASE.iter().map(|s| s.to_string()).collect();
    v.extend(ORGAN_NAMES.iter().map(|n| format!("% com {n}")));
    v
}

pub struct History {
    pub names: Vec<String>,
    /// Epoch de cada amostra guardada.
    epochs: Vec<u32>,
    /// Valores, por linhas (names.len() por amostra).
    values: Vec<f32>,
    /// Só se guarda uma amostra em cada `stride` (dobra quando enche).
    stride: u32,
    seen: u32,
    /// Contadores da amostra anterior (para as taxas).
    last: Option<(u32, u32, u32, u32)>,
    /// Epochs entre amostras.
    pub every: u32,
    pub next_epoch: u32,
    csv_ok: bool,
    /// Este histórico já escreveu o cabeçalho no CSV (cada recomeço escreve
    /// um novo: marca a fronteira entre corridas no ficheiro).
    csv_header: bool,
    /// Versão reduzida para os gráficos (refeita quando muda o número de amostras).
    plot_cache: Option<(usize, Vec<Vec<[f64; 2]>>)>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            names: series_names(),
            epochs: Vec::new(),
            values: Vec::new(),
            stride: 1,
            seen: 0,
            last: None,
            every: 2000,
            next_epoch: 0,
            csv_ok: true,
            csv_header: false,
            plot_cache: None,
        }
    }
}

impl History {
    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    /// Junta uma amostra (palavras da GPU + livro-razão + contadores).
    pub fn push(&mut self, epoch: u32, w: &[u32], ledger: Option<Ledger>, c: Option<LifeCounters>) {
        let alive = w[0] as f32;
        let per = |x: u32| if alive > 0.0 { x as f32 / alive } else { 0.0 };
        let (births, deaths, bites) = match (c, self.last) {
            (Some(c), Some((e0, b0, d0, m0))) if epoch > e0 => {
                let k = 1000.0 / (epoch - e0) as f32;
                // (As mordidas dão a volta aos 32 bits numa corrida longa.)
                (c.births.saturating_sub(b0) as f32 * k, c.deaths.saturating_sub(d0) as f32 * k, c.bites.wrapping_sub(m0) as f32 * k)
            }
            _ => (0.0, 0.0, 0.0),
        };
        if let Some(c) = c {
            self.last = Some((epoch, c.births, c.deaths, c.bites));
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
            bites,
        ];
        v.extend((0..ORGAN_TYPES).map(|t| 100.0 * per(w[8 + t])));
        self.write_csv(epoch, &v);
        self.seen += 1;
        if !self.seen.is_multiple_of(self.stride) {
            return;
        }
        self.epochs.push(epoch);
        self.values.extend_from_slice(&v);
        if self.epochs.len() >= MAX_POINTS {
            // Cheio: uma amostra em cada duas, e passa-se a guardar metade.
            let n = self.names.len();
            let mut e = Vec::with_capacity(MAX_POINTS / 2 + 1);
            let mut vals = Vec::with_capacity((MAX_POINTS / 2 + 1) * n);
            for i in (0..self.epochs.len()).step_by(2) {
                e.push(self.epochs[i]);
                vals.extend_from_slice(&self.values[i * n..(i + 1) * n]);
            }
            self.epochs = e;
            self.values = vals;
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
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(CSV_PATH)?;
            if !self.csv_header {
                writeln!(f, "epoch,{}", self.names.join(","))?;
                self.csv_header = true;
            }
            let row: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
            writeln!(f, "{epoch},{}", row.join(","))
        })();
        if let Err(e) = r {
            log::error!("{CSV_PATH}: {e} (deixo de escrever o CSV)");
            self.csv_ok = false;
        }
    }

    /// As últimas `n` amostras guardadas: (epoch, valores pela ordem de `names`).
    pub fn last_rows(&self, n: usize) -> Vec<(u32, &[f32])> {
        let k = self.names.len();
        let start = self.epochs.len().saturating_sub(n);
        (start..self.epochs.len()).map(|i| (self.epochs[i], &self.values[i * k..(i + 1) * k])).collect()
    }

    /// Para gravar com a cena: metadados (JSON) e as amostras (binário:
    /// n, epochs[n], valores[n × séries], em u32/f32 little-endian).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "names": self.names, "stride": self.stride, "every": self.every })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.epochs.len() * 4 + self.values.len() * 4);
        out.extend_from_slice(&(self.epochs.len() as u32).to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&self.epochs));
        out.extend_from_slice(bytemuck::cast_slice(&self.values));
        out
    }

    /// Repõe de uma cena (as séries que mudaram de nome ficam a 0).
    pub fn from_saved(meta: &serde_json::Value, data: &[u8]) -> Self {
        let mut h = Self::default();
        let names: Vec<String> =
            meta["names"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        h.stride = meta["stride"].as_u64().unwrap_or(1).max(1) as u32;
        h.every = meta["every"].as_u64().unwrap_or(2000).max(100) as u32;
        if data.len() < 4 || names.is_empty() {
            return h;
        }
        let n = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        let m = names.len();
        if data.len() != 4 + n * 4 + n * m * 4 {
            log::warn!("estatísticas gravadas com tamanho errado: ignoradas");
            return h;
        }
        let epochs: Vec<u32> = bytemuck::pod_collect_to_vec(&data[4..4 + n * 4]);
        let vals: Vec<f32> = bytemuck::pod_collect_to_vec(&data[4 + n * 4..]);
        let map: Vec<Option<usize>> = h.names.iter().map(|x| names.iter().position(|y| y == x)).collect();
        for i in 0..n {
            h.epochs.push(epochs[i]);
            h.values.extend(map.iter().map(|j| j.map_or(0.0, |j| vals[i * m + j])));
        }
        h
    }

    /// Versão reduzida para desenhar: médias por intervalo, até PLOT_POINTS.
    fn plot_series(&mut self) -> &Vec<Vec<[f64; 2]>> {
        let len = self.epochs.len();
        if self.plot_cache.as_ref().is_none_or(|(l, _)| *l != len) {
            let n = self.names.len();
            let bucket = len.div_ceil(PLOT_POINTS).max(1);
            let mut series = vec![Vec::with_capacity(len / bucket + 1); n];
            for start in (0..len).step_by(bucket) {
                let end = (start + bucket).min(len);
                let k = (end - start) as f64;
                let x = self.epochs[start..end].iter().map(|&e| e as f64).sum::<f64>() / k;
                for (s, out) in series.iter_mut().enumerate() {
                    let y = (start..end).map(|i| self.values[i * n + s] as f64).sum::<f64>() / k;
                    out.push([x, y]);
                }
            }
            self.plot_cache = Some((len, series));
        }
        &self.plot_cache.as_ref().unwrap().1
    }
}

/// Grupos de séries por gráfico (índices em `series_names`).
fn groups() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("População", vec![0]),
        ("Nascimentos e mortes", vec![1, 2]),
        ("Órgãos (agentes com cada um)", (BASE.len()..BASE.len() + ORGAN_TYPES).collect()),
        ("Predação", vec![12]),
        ("Corpo e genoma", vec![4, 5, 9]),
        ("Energia, gerações e ligações", vec![3, 6, 7, 8]),
        ("Matéria", vec![10, 11]),
    ]
}

/// O que os dois gráficos mostram e as séries escondidas (clique na legenda).
pub struct ChartSel {
    pub group: [usize; 2],
    pub hidden: std::collections::HashSet<usize>,
    /// Órgãos em NÚMERO de agentes (e não em % da população).
    pub organ_counts: bool,
}

impl Default for ChartSel {
    fn default() -> Self {
        // População em cima, órgãos em baixo.
        Self { group: [0, 2], hidden: Default::default(), organ_counts: true }
    }
}

/// Controlos da amostragem (separador "Gráficos" da barra da esquerda).
pub fn controls(ui: &mut egui::Ui, h: &mut History) {
    ui.label("os gráficos estão no painel do meio (por cima da simulação, que continua a correr). Muda de separador para a voltar a ver.");
    ui.horizontal(|ui| {
        ui.label("amostra de");
        ui.add(egui::DragValue::new(&mut h.every).range(100..=1_000_000).speed(100));
        ui.label("em epochs");
    });
    ui.label(format!(
        "{} amostras guardadas (uma em cada {}; máximo {MAX_POINTS}); CSV completo em {CSV_PATH}",
        h.len(),
        h.stride
    ));
}

/// Cor da série n.º `k` de um gráfico (tons bem separados, pelo ângulo de ouro).
fn series_color(k: usize) -> egui::Color32 {
    egui::ecolor::Hsva::new((k as f32 * 0.618_034).fract(), 0.75, 1.0, 1.0).into()
}

/// Os gráficos (painel central): dois, um por cima do outro, cada um com a
/// sua escolha. A legenda fica FORA do gráfico, por cima dele, com o valor
/// da última amostra; um clique num nome esconde ou mostra a série.
pub fn draw(ui: &mut egui::Ui, h: &mut History, sel: &mut ChartSel) {
    if h.len() < 2 {
        ui.label("à espera de amostras…");
        return;
    }
    let last: Vec<f32> = h.last_rows(1).first().map(|(_, v)| v.to_vec()).unwrap_or_default();
    // ÓRGÃOS EM NÚMERO DE AGENTES: as séries guardam a % da população com
    // cada órgão; em % todas as linhas sobem e descem juntas quando um grupo
    // grande cresce ou encolhe (o ciclo do dia punha tudo às ondas). O
    // número de agentes (% × vivos) de um órgão não depende dos outros.
    let organs = BASE.len()..BASE.len() + ORGAN_TYPES;
    // (O valor deste frame; a caixa de seleção só conta no seguinte.)
    let by_count = sel.organ_counts;
    let as_count = |i: usize| by_count && organs.contains(&i);
    let alive_last = last.first().copied().unwrap_or(0.0);
    let names: Vec<String> = h
        .names
        .iter()
        .enumerate()
        .map(|(i, n)| match last.get(i) {
            Some(v) if as_count(i) => format!("{}: {:.0}", n.trim_start_matches("% "), v * alive_last / 100.0),
            Some(v) if v.abs() >= 100.0 => format!("{n}: {v:.0}"),
            Some(v) => format!("{n}: {v:.1}"),
            None => n.clone(),
        })
        .collect();
    let series = h.plot_series();
    // (Média por intervalo da % × média dos vivos: chega para desenhar.)
    let counts: Vec<Vec<[f64; 2]>> = if by_count {
        organs.clone().map(|i| series[i].iter().zip(&series[0]).map(|(p, a)| [p[0], p[1] * a[1] / 100.0]).collect()).collect()
    } else {
        Vec::new()
    };
    let groups = groups();
    let block = (ui.available_height() - 8.0) / 2.0;
    for slot in 0..2 {
        let top = ui.cursor().top();
        sel.group[slot] = sel.group[slot].min(groups.len() - 1);
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(("grafico", slot)).width(280.0).selected_text(groups[sel.group[slot]].0).show_ui(ui, |ui| {
                for (g, (title, _)) in groups.iter().enumerate() {
                    ui.selectable_value(&mut sel.group[slot], g, *title);
                }
            });
            if groups[sel.group[slot]].1.first().is_some_and(|i| organs.contains(i)) {
                ui.checkbox(&mut sel.organ_counts, "em número de agentes")
                    .on_hover_text("ligado: quantos agentes têm cada órgão (uma linha não mexe nas outras). Desligado: em % da população, que faz todas as linhas ondular juntas quando um grupo grande cresce ou encolhe");
            }
            ui.small("clica num nome para esconder ou mostrar a linha");
        });
        let idx = &groups[sel.group[slot]].1;
        ui.horizontal_wrapped(|ui| {
            for (k, &i) in idx.iter().enumerate() {
                let off = sel.hidden.contains(&i);
                let mut text = egui::RichText::new(format!("■ {}", names[i])).color(if off { egui::Color32::DARK_GRAY } else { series_color(k) });
                if off {
                    text = text.strikethrough();
                }
                if ui.add(egui::Label::new(text).sense(egui::Sense::click())).clicked() && !sel.hidden.remove(&i) {
                    sel.hidden.insert(i);
                }
            }
        });
        let height = (block - (ui.cursor().top() - top) - 6.0).max(80.0);
        egui_plot::Plot::new(("grafico", slot, sel.group[slot])).height(height).allow_scroll(false).show(ui, |p| {
            for (k, &i) in idx.iter().enumerate() {
                if !sel.hidden.contains(&i) {
                    let pts = if as_count(i) { counts[i - organs.start].clone() } else { series[i].clone() };
                    p.line(
                        egui_plot::Line::new(names[i].clone(), egui_plot::PlotPoints::from(pts))
                            .id(egui::Id::new(("serie", i)))
                            .color(series_color(k)),
                    );
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_round_trip() {
        let mut h = History { csv_ok: false, ..Default::default() };
        for i in 0..50u32 {
            let mut w = vec![0u32; STAT_WORDS];
            w[0] = 100 + i;
            w[3] = 5000;
            h.push(i * 1000, &w, None, None);
        }
        let back = History::from_saved(&h.to_json(), &h.to_bytes());
        assert_eq!(back.len(), 50);
        assert_eq!(back.epochs, h.epochs);
        assert_eq!(back.values, h.values);
        assert_eq!(back.values[back.names.len() * 49], 149.0, "vivos da última amostra");
    }
}
