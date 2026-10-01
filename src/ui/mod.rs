//! Painéis egui: livro-razão, profiler, sliders do mundo, vistas de debug.

use crate::gpu::profiler::Profiler;
use crate::world::{Ledger, MAX_STEPS_PER_FRAME, World};

pub struct UiState {
    pub paused: bool,
    pub steps_per_frame: u32,
    /// 0 = normal, 1–4 = ativados A U G C, 5 = gastos, 6 terreno, 7 temperatura, 8 UV, 9 fluido.
    pub view_mode: u32,
    /// Contagem exata escrita na sementeira (base do Δ).
    pub baseline: Ledger,
    pub ledger: Option<Ledger>,
    pub ledger_epoch: u32,
    pub reseed: bool,
    pub vsync: bool,
    /// Brilho da camada de monómeros na vista normal.
    pub monomer_brightness: f32,
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
            monomer_brightness: 0.5,
        }
    }
}

pub const VIEW_NAMES: [&str; 10] = [
    "normal",
    "1 A ativ.",
    "2 U ativ.",
    "3 G ativ.",
    "4 C ativ.",
    "5 gastos",
    "6 terreno",
    "7 temperatura",
    "8 luz UV",
    "9 fluido",
];
const CH: [&str; 4] = ["A", "U", "G", "C"];

pub fn draw(ctx: &egui::Context, st: &mut UiState, world: &mut World, prof: &mut Profiler) {
    egui::Window::new("Ribossome v4").default_pos([12.0, 12.0]).show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button(if st.paused { "▶ continuar" } else { "⏸ pausa" }).clicked() {
                st.paused = !st.paused;
            }
            if ui.button("nova semente").clicked() {
                st.reseed = true;
            }
        });
        ui.label(format!("epoch {}", world.params.epoch));
        ui.add(egui::Slider::new(&mut st.steps_per_frame, 1..=MAX_STEPS_PER_FRAME).text("passos/frame"));
        ui.checkbox(&mut st.vsync, "vsync");

        ui.separator();
        egui::ComboBox::from_label("vista").selected_text(VIEW_NAMES[st.view_mode as usize]).show_ui(ui, |ui| {
            for (i, n) in VIEW_NAMES.iter().enumerate() {
                ui.selectable_value(&mut st.view_mode, i as u32, *n);
            }
        });
        ui.add(egui::Slider::new(&mut st.monomer_brightness, 0.0..=1.0).text("brilho dos monómeros"));
        world_panel(ui, world);

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

fn world_panel(ui: &mut egui::Ui, world: &mut World) {
    let mut light_changed = false;
    let p = &mut world.params;
    let st = &mut world.settings;
    egui::CollapsingHeader::new("Monómeros").default_open(true).show(ui, |ui| {
        ui.add(egui::Slider::new(&mut p.diffusion, 0.0..=50.0).text("difusão ×"));
        ui.add(egui::Slider::new(&mut p.settle, 0.0..=10.0).text("assentamento ×"));
        ui.add(egui::Slider::new(&mut p.cohesion, 0.0..=2.0).text("coesão"));
        ui.add(egui::Slider::new(&mut p.uv_strength, 0.0..=10.0).text("força UV"));
        light_changed = ui.add(egui::Slider::new(&mut p.uv_depth, 0.5..=30.0).text("atenuação UV")).changed();
    });
    egui::CollapsingHeader::new("Fluido").default_open(true).show(ui, |ui| {
        ui.checkbox(&mut st.fluid_enabled, "fluido ligado");
        ui.checkbox(&mut st.terrain_enabled, "física do terreno ligada");
        ui.add(egui::Slider::new(&mut st.jacobi_iters, 2..=256).text("iterações Jacobi"));
        ui.add(egui::Slider::new(&mut st.fluid_substep, 1..=4).text("resolve de N em N passos"));
        ui.add(egui::Slider::new(&mut p.fluid_vorticity, 0.0..=10.0).text("vorticidade"));
        ui.add(egui::Slider::new(&mut p.fluid_viscosity, 0.0..=5.0).text("viscosidade"));
        ui.add(egui::Slider::new(&mut p.fluid_decay, 0.9..=1.0).text("decay por frame"));
    });
    egui::CollapsingHeader::new("Fumarolas").default_open(false).show(ui, |ui| {
        for (i, f) in world.fumaroles.iter_mut().enumerate() {
            ui.push_id(i, |ui| {
                let mut on = f.enabled != 0;
                ui.checkbox(&mut on, format!("fumarola {i}"));
                f.enabled = on as u32;
                ui.add(egui::Slider::new(&mut f.x_frac, 0.0..=1.0).text("x"));
                ui.add(egui::Slider::new(&mut f.y_frac, 0.0..=1.0).text("y"));
                ui.add(egui::Slider::new(&mut f.strength, 0.0..=20000.0).text("força"));
                ui.add(egui::Slider::new(&mut f.spread, 60.0..=4000.0).text("raio (mundo)"));
            });
        }
    });
    if light_changed {
        world.invalidate_light();
    }
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
