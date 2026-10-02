//! Painéis egui: livro-razão, profiler, sliders do mundo, vistas de debug, inspetor.

pub mod inspector;

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
    /// Cor dos agentes: 0 química, 1 sinal α, 2 sinal β, 3 α e β.
    pub signal_view: u32,
    /// Semear: quantas sementes, comprimento mínimo/máximo, começar por AUG.
    pub seed_count: u32,
    pub seed_len: [u32; 2],
    pub seed_aug: bool,
    pub seed_now: bool,
    /// Terreno em imagem: caminho, ação pedida e resultado da última.
    pub terrain_path: String,
    pub terrain_action: Option<TerrainAction>,
    pub terrain_msg: String,
    /// Velocidade e população (atualizadas ~2×/s em `update_stats`).
    pub stats: Stats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainAction {
    /// Carrega o PNG e semeia de novo (mesma semente).
    Load,
    /// Grava o terreno atual em PNG.
    Save,
    /// Volta ao terreno gerado e semeia de novo.
    Generated,
}

#[derive(Default)]
pub struct Stats {
    pub epochs_per_sec: f32,
    pub alive: Option<u32>,
    pub births_per_sec: f32,
    pub deaths_per_sec: f32,
    /// Amostra anterior: (instante, epoch, nascimentos, mortes, epoch da leitura).
    last: Option<(std::time::Instant, u32, u32, u32)>,
}

impl Stats {
    /// Atualiza com o epoch atual e os contadores lidos da GPU (assíncronos).
    pub fn update(&mut self, epoch: u32, counters: Option<crate::world::LifeCounters>, max_agents: u32) {
        let now = std::time::Instant::now();
        if let Some(c) = counters {
            self.alive = Some(c.alive(max_agents));
        }
        let (births, deaths) = counters.map_or((0, 0), |c| (c.births, c.deaths));
        match self.last {
            Some((t0, e0, b0, d0)) => {
                let dt = now.duration_since(t0).as_secs_f32();
                if dt >= 0.5 {
                    self.epochs_per_sec = epoch.saturating_sub(e0) as f32 / dt;
                    // Os contadores voltam a zero numa nova semente: nada de taxas negativas.
                    self.births_per_sec = births.saturating_sub(b0) as f32 / dt;
                    self.deaths_per_sec = deaths.saturating_sub(d0) as f32 / dt;
                    self.last = Some((now, epoch, births, deaths));
                }
            }
            None => self.last = Some((now, epoch, births, deaths)),
        }
    }
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
            signal_view: 0,
            stats: Stats::default(),
            terrain_path: std::env::var("RIBO_TERRAIN").unwrap_or_else(|_| "terreno.png".into()),
            terrain_action: None,
            terrain_msg: String::new(),
            seed_count: 500,
            seed_len: [12, 120],
            seed_aug: true,
            seed_now: false,
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
    // Altura máxima e barra de deslocamento: o painel é comprido.
    let max_h = ctx.content_rect().height() - 40.0;
    egui::Window::new("Ribossome v4").default_pos([12.0, 12.0]).max_height(max_h).show(ctx, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| main_panel(ui, st, world, prof));
    });
}

fn main_panel(ui: &mut egui::Ui, st: &mut UiState, world: &mut World, prof: &mut Profiler) {
    {
        ui.horizontal(|ui| {
            if ui.button(if st.paused { "▶ continuar" } else { "⏸ pausa" }).clicked() {
                st.paused = !st.paused;
            }
            if ui.button("nova semente").clicked() {
                st.reseed = true;
            }
        });
        ui.label(format!("epoch {}   ({:.0} epochs/s)", world.params.epoch, st.stats.epochs_per_sec));
        ui.label(match st.stats.alive {
            Some(n) => format!(
                "agentes vivos {n}   (+{:.0}/s nascimentos, −{:.0}/s mortes)",
                st.stats.births_per_sec, st.stats.deaths_per_sec
            ),
            None => "agentes vivos …".into(),
        });
        ui.add(egui::Slider::new(&mut st.steps_per_frame, 1..=MAX_STEPS_PER_FRAME).text("passos/frame"));
        ui.checkbox(&mut st.vsync, "vsync");

        ui.separator();
        egui::ComboBox::from_label("vista").selected_text(VIEW_NAMES[st.view_mode as usize]).show_ui(ui, |ui| {
            for (i, n) in VIEW_NAMES.iter().enumerate() {
                ui.selectable_value(&mut st.view_mode, i as u32, *n);
            }
        });
        ui.add(egui::Slider::new(&mut st.monomer_brightness, 0.0..=1.0).text("brilho dos monómeros"));
        const SIGNAL_VIEWS: [&str; 4] = ["química", "sinal α", "sinal β", "α (vermelho) + β (verde)"];
        egui::ComboBox::from_label("cor dos agentes").selected_text(SIGNAL_VIEWS[st.signal_view as usize]).show_ui(
            ui,
            |ui| {
                for (i, n) in SIGNAL_VIEWS.iter().enumerate() {
                    ui.selectable_value(&mut st.signal_view, i as u32, *n);
                }
            },
        );
        egui::CollapsingHeader::new("Terreno (imagem)").default_open(false).show(ui, |ui| {
            ui.small("PNG em cinzentos: preto água, cinzento escuro entulho, claro rocha, branco rocha maciça; vermelho = fumarola");
            ui.horizontal(|ui| {
                ui.label("ficheiro");
                ui.text_edit_singleline(&mut st.terrain_path);
            });
            ui.horizontal(|ui| {
                if ui.button("carregar (semeia de novo)").clicked() {
                    st.terrain_action = Some(TerrainAction::Load);
                }
                if ui.button("gravar o atual").clicked() {
                    st.terrain_action = Some(TerrainAction::Save);
                }
                if ui.button("terreno gerado").clicked() {
                    st.terrain_action = Some(TerrainAction::Generated);
                }
            });
            if !st.terrain_msg.is_empty() {
                ui.label(&st.terrain_msg);
            }
        });
        world_panel(ui, world);
        life_panel(ui, st, world);

        ui.separator();
        conservation(ui, st);

        ui.separator();
        ui.checkbox(&mut prof.enabled, "profiler (submit+wait por segmento)");
        ui.label(format!("frame {:.2} ms ({:.0} fps)", prof.frame_ms, 1000.0 / prof.frame_ms.max(1e-3)));
        if prof.enabled {
            egui::Grid::new("prof").striped(true).show(ui, |ui| {
                for s in prof.stats() {
                    ui.label(s.name);
                    if s.name == "world" && !st.paused {
                        // O "world" inclui todos os passos do frame.
                        let n = st.steps_per_frame.max(1);
                        ui.label(format!("{:.3} ms  ({n} passos, {:.3} ms/passo)", s.avg_ms, s.avg_ms / n as f64));
                    } else {
                        ui.label(format!("{:.3} ms", s.avg_ms));
                    }
                    ui.end_row();
                }
            });
        }
    }
}

fn world_panel(ui: &mut egui::Ui, world: &mut World) {
    let mut light_changed = false;
    let p = &mut world.params;
    let st = &mut world.settings;
    let seed_density = &mut world.seed_density;
    egui::CollapsingHeader::new("Monómeros").default_open(true).show(ui, |ui| {
        ui.add(egui::Slider::new(seed_density, 0.05..=1.0).text("densidade inicial (na próxima semente)"));
        ui.add(egui::Slider::new(&mut p.diffusion, 0.0..=50.0).text("difusão ×"));
        ui.add(egui::Slider::new(&mut p.settle, 0.0..=10.0).text("assentamento ×"));
        ui.add(egui::Slider::new(&mut p.cohesion, 0.0..=2.0).text("coesão"));
        ui.add(egui::Slider::new(&mut p.uv_strength, 0.0..=10.0).text("força UV"));
        light_changed = ui.add(egui::Slider::new(&mut p.uv_depth, 0.5..=30.0).text("atenuação UV")).changed();
        // Reativação uniforme (modo laboratório): probabilidade por passo de
        // um gasto voltar a ativado; escala logarítmica para afinar valores pequenos.
        ui.add(
            egui::Slider::new(&mut p.reactivation_rate, 0.0..=0.02)
                .logarithmic(true)
                .smallest_positive(1e-5)
                .text("reativação dos gastos"),
        );
        if p.reactivation_rate > 0.0 {
            ui.label(format!("  pousio médio de um gasto: {:.0} passos", 1.0 / p.reactivation_rate));
        }
    });
    egui::CollapsingHeader::new("Fluido").default_open(true).show(ui, |ui| {
        ui.checkbox(&mut st.fluid_enabled, "fluido ligado");
        ui.checkbox(&mut st.terrain_enabled, "física do terreno ligada");
        ui.checkbox(&mut st.multigrid, "pressão por multigrid (senão Jacobi)");
        if st.multigrid {
            ui.add(egui::Slider::new(&mut st.mg_cycles, 1..=4).text("ciclos V"));
        } else {
            ui.add(egui::Slider::new(&mut st.jacobi_iters, 2..=256).text("iterações Jacobi"));
        }
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

fn life_panel(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    egui::CollapsingHeader::new("Vida").default_open(true).show(ui, |ui| {
        ui.add(egui::Slider::new(&mut st.seed_count, 1..=4000).text("sementes"));
        ui.horizontal(|ui| {
            ui.label("bases");
            ui.add(egui::DragValue::new(&mut st.seed_len[0]).range(3..=256));
            ui.label("a");
            ui.add(egui::DragValue::new(&mut st.seed_len[1]).range(3..=256));
        });
        ui.checkbox(&mut st.seed_aug, "começar por AUG (tirado da sopa)");
        if ui.button("semear (geração 0, montada da sopa)").clicked() {
            st.seed_now = true;
        }
        ui.add(egui::Slider::new(&mut world.params.death_probability, 0.0..=0.2).text("mortalidade base"));
        ui.add(egui::Slider::new(&mut world.params.spawn_energy, 0.1..=50.0).text("energia inicial"));
        ui.add(egui::Slider::new(&mut world.params.food_power, 0.0..=20.0).text("energia por monómero"));
        ui.add(egui::Slider::new(&mut world.params.uptake_rate, 0.0..=0.01).text("taxa de hidrólise"));
        ui.add(egui::Slider::new(&mut world.params.pairing_rate, 0.0..=8.0).text("emparelhamento (bases/passo)"));
        ui.add(egui::Slider::new(&mut world.params.mutation_rate, 0.0..=0.05).text("taxa de mutação"));
        ui.add(egui::Slider::new(&mut world.params.uv_damage, 1.0..=50.0).text("dano UV"));
        ui.add(egui::Slider::new(&mut world.params.brownian, 0.0..=20.0).text("movimento browniano"));
        ui.checkbox(&mut world.settings.contact_enabled, "repulsão entre agentes");
        ui.add(egui::Slider::new(&mut world.params.chain_stiffness, 1.0..=100.0).text("rigidez das juntas"));
        ui.add(egui::Slider::new(&mut world.params.thermal_kt, 0.0..=5.0).text("agitação térmica (kT)"));
        ui.add(egui::Slider::new(&mut world.params.motor_amplitude, 0.0..=1.0).text("curso do motor (rad)"));
        ui.add(egui::Slider::new(&mut world.params.joint_coupling, 0.0..=0.95).text("acoplamento entre juntas"));
        let mut hunger = world.params.hunger_regulation != 0;
        ui.checkbox(&mut hunger, "regulação pela fome (v3)");
        world.params.hunger_regulation = hunger as u32;
        let mut aug = world.params.require_start != 0;
        ui.checkbox(&mut aug, "tradução começa no AUG (nascimentos novos)");
        world.params.require_start = aug as u32;
        ui.add(egui::Slider::new(&mut world.params.swim_gain, 0.0..=50.0).text("ganho da natação"));
        let mut rft = world.params.rft_enabled != 0;
        ui.checkbox(&mut rft, "natação (RFT)");
        world.params.rft_enabled = rft as u32;
        ui.add(egui::Slider::new(&mut world.params.phoretic_gain, 0.0..=500.0).text("difusioforese"));
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
    ui.label(format!("livre {}  presa em agentes {}", l.free_total(), l.held_total()));
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
