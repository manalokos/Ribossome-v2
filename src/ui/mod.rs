//! Painéis egui. Fase 1: livro-razão, profiler, sliders do mundo, vistas.

use crate::gpu::profiler::Profiler;
use crate::params::SimParams;
use crate::world::{Ledger, MAX_STEPS_PER_FRAME};

pub struct UiState {
    pub paused: bool,
    pub steps_per_frame: u32,
    /// 0 = normal, 1–4 = ativados A U G C, 5 = gastos.
    pub view_mode: u32,
    /// Contagem exata escrita na sementeira (base do Δ).
    pub baseline: Ledger,
    pub ledger: Option<Ledger>,
    pub ledger_epoch: u32,
    pub reseed: bool,
    pub vsync: bool,
}

impl UiState {
    pub fn new(baseline: Ledger) -> Self {
        Self {
            paused: false,
            steps_per_frame: 1,
            view_mode: 0,
            baseline,
            ledger: None,
            ledger_epoch: 0,
            reseed: false,
            vsync: true,
        }
    }
}

const VIEW_NAMES: [&str; 6] = ["normal", "1 A ativ.", "2 U ativ.", "3 G ativ.", "4 C ativ.", "5 gastos"];
const CH: [&str; 4] = ["A", "U", "G", "C"];

pub fn draw(ctx: &egui::Context, st: &mut UiState, params: &mut SimParams, prof: &mut Profiler) {
    egui::Window::new("Ribossome v4").default_pos([12.0, 12.0]).show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button(if st.paused { "▶ continuar" } else { "⏸ pausa" }).clicked() {
                st.paused = !st.paused;
            }
            if ui.button("nova semente").clicked() {
                st.reseed = true;
            }
        });
        ui.label(format!("epoch {}", params.epoch));
        ui.add(egui::Slider::new(&mut st.steps_per_frame, 1..=MAX_STEPS_PER_FRAME).text("passos/frame"));
        ui.checkbox(&mut st.vsync, "vsync");

        ui.separator();
        ui.strong("Mundo");
        ui.add(egui::Slider::new(&mut params.diffusion, 0.0..=50.0).text("difusão ×"));
        egui::ComboBox::from_label("vista").selected_text(VIEW_NAMES[st.view_mode as usize]).show_ui(ui, |ui| {
            for (i, n) in VIEW_NAMES.iter().enumerate() {
                ui.selectable_value(&mut st.view_mode, i as u32, *n);
            }
        });

        ui.separator();
        conservation(ui, st);

        ui.separator();
        ui.checkbox(&mut prof.enabled, "profiler (submit+wait por segmento)");
        ui.label(format!("frame {:.2} ms ({:.0} fps)", prof.frame_ms, 1000.0 / prof.frame_ms.max(1e-3)));
        if prof.enabled {
            egui::Grid::new("prof").striped(true).show(ui, |ui| {
                for s in prof.stats() {
                    ui.label(s.name);
                    ui.label(format!("{:.3} ms", s.avg_ms));
                    ui.end_row();
                }
            });
        }
    });
}

fn conservation(ui: &mut egui::Ui, st: &UiState) {
    ui.strong("Conservação da matéria");
    let Some(l) = st.ledger else {
        ui.label("à espera da primeira leitura…");
        return;
    };
    let base = st.baseline.total() as i64;
    let now = l.total() as i64;
    let d = now - base;
    let pct = if base > 0 { d as f64 / base as f64 * 100.0 } else { 0.0 };
    let color = if d == 0 { egui::Color32::LIGHT_GREEN } else { egui::Color32::LIGHT_RED };
    ui.colored_label(color, format!("total {now}  Δ {d:+} ({pct:+.4}%)"));
    ui.label(format!("(leitura do epoch ~{}, assíncrona)", st.ledger_epoch));
    egui::Grid::new("ledger").striped(true).show(ui, |ui| {
        ui.label("");
        ui.label("ativ.");
        ui.label("gastos");
        ui.label("Δ");
        ui.end_row();
        for (ch, name) in CH.iter().enumerate() {
            ui.label(*name);
            ui.label(l.act[ch].to_string());
            ui.label(l.spent[ch].to_string());
            ui.label(format!("{:+}", l.channel(ch) as i64 - st.baseline.channel(ch) as i64));
            ui.end_row();
        }
    });
}
