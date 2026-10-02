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
    /// Pedido para abrir o editor dos aminoácidos no browser.
    pub open_editor: bool,
    /// Separador ativo do painel.
    pub tab: Tab,
    /// Velocidade e população (atualizadas ~2×/s em `update_stats`).
    pub stats: Stats,
    /// Estatísticas ao longo do tempo (gráficos e logs/estatisticas.csv).
    pub history: crate::stats::History,
    /// Cenas: ação pedida, autosave e mensagem da última operação.
    pub scene_action: Option<SceneAction>,
    pub autosave_on: bool,
    pub autosave_every: u32,
    /// Ficheiro do autosave (None = desligado neste arranque, p. ex. testes).
    pub autosave_path: Option<String>,
    pub scene_msg: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneAction {
    /// Escolhe onde gravar (janela) e grava a cena.
    Save,
    /// Escolhe uma cena (janela) e carrega-a.
    Load,
    /// Grava já o autosave.
    AutosaveNow,
    /// Mundo novo com todos os valores por omissão (e o terreno de arranque).
    NewWorld,
}

/// Separadores do painel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Vista,
    Materia,
    Luz,
    Agua,
    Terreno,
    Vida,
    Movimento,
    Cena,
    Graficos,
    Info,
}

const TABS: [(Tab, &str); 10] = [
    (Tab::Vista, "Vista"),
    (Tab::Materia, "Matéria"),
    (Tab::Luz, "Luz"),
    (Tab::Agua, "Água"),
    (Tab::Terreno, "Terreno"),
    (Tab::Vida, "Vida"),
    (Tab::Movimento, "Movimento"),
    (Tab::Cena, "Cena"),
    (Tab::Graficos, "Gráficos"),
    (Tab::Info, "Info"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainAction {
    /// Escolhe um PNG (janela), carrega-o e semeia de novo (mesma semente).
    Load,
    /// Escolhe onde gravar (janela) e grava o terreno atual em PNG.
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
            history: crate::stats::History::default(),
            terrain_path: std::env::var("RIBO_TERRAIN").unwrap_or_else(|_| "assets/terreno.png".into()),
            terrain_action: None,
            terrain_msg: String::new(),
            open_editor: false,
            tab: Tab::Vista,
            seed_count: 500,
            seed_len: [12, 120],
            seed_aug: true,
            seed_now: false,
            scene_action: None,
            autosave_on: true,
            autosave_every: 50_000,
            autosave_path: None,
            scene_msg: String::new(),
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
        main_panel(ui, st, world, prof);
    });
}

fn main_panel(ui: &mut egui::Ui, st: &mut UiState, world: &mut World, prof: &mut Profiler) {
    // ---- Topo, sempre visível ----
    ui.horizontal(|ui| {
        if ui.button(if st.paused { "▶ continuar" } else { "⏸ pausa" }).clicked() {
            st.paused = !st.paused;
        }
        if ui.button("nova semente").clicked() {
            st.reseed = true;
        }
    });
    ui.label(format!(
        "epoch {}   ({:.0} epochs/s)   frame {:.1} ms ({:.0} fps)",
        world.params.epoch,
        st.stats.epochs_per_sec,
        prof.frame_ms,
        1000.0 / prof.frame_ms.max(1e-3)
    ));
    ui.label(match st.stats.alive {
        Some(n) => format!(
            "agentes vivos {n}   (+{:.0}/s nascimentos, −{:.0}/s mortes)",
            st.stats.births_per_sec, st.stats.deaths_per_sec
        ),
        None => "agentes vivos …".into(),
    });
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut st.steps_per_frame, 1..=MAX_STEPS_PER_FRAME).text("passos/frame"));
        ui.checkbox(&mut st.vsync, "vsync");
    });

    // ---- Separadores ----
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        for (t, name) in TABS {
            ui.selectable_value(&mut st.tab, t, name);
        }
    });
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| match st.tab {
        Tab::Vista => tab_view(ui, st),
        Tab::Materia => tab_matter(ui, world),
        Tab::Luz => tab_light(ui, world),
        Tab::Agua => tab_water(ui, world),
        Tab::Terreno => tab_terrain(ui, st, world),
        Tab::Vida => tab_life(ui, st, world),
        Tab::Movimento => tab_motion(ui, world),
        Tab::Cena => tab_scene(ui, st, world),
        Tab::Graficos => crate::stats::draw(ui, &mut st.history),
        Tab::Info => tab_info(ui, st, prof),
    });
}

/// Parâmetros diferentes dos valores do código, com botões para os repor
/// (o autosave retoma-os: um slider esquecido fica preso entre arranques).
fn changed_params(ui: &mut egui::Ui, world: &mut World) {
    let diff = world.params.changed_from_default();
    let title = if diff.is_empty() {
        "parâmetros: todos com os valores do código".to_string()
    } else {
        format!("parâmetros diferentes do código: {}", diff.len())
    };
    egui::CollapsingHeader::new(title).id_salt("changed_params").show(ui, |ui| {
        for &(k, a, b) in &diff {
            ui.horizontal(|ui| {
                ui.label(format!("{k}: {a:.6} (código {b:.6})"));
                if ui.small_button("repor").clicked() {
                    world.params.set_named(k, b);
                }
            });
        }
        if !diff.is_empty() && ui.button("repor todos (mantém o mundo)").clicked() {
            for &(k, _, b) in &diff {
                world.params.set_named(k, b);
            }
        }
    });
}

fn tab_scene(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    changed_params(ui, world);
    ui.separator();
    ui.label("Uma cena = o mundo inteiro (matéria, terreno, água, agentes) e todos os parâmetros.");
    ui.label("As tabelas dos aminoácidos e órgãos vêm sempre de assets/ (não da cena).");
    ui.horizontal(|ui| {
        if ui.button("gravar cena…").clicked() {
            st.scene_action = Some(SceneAction::Save);
        }
        if ui.button("carregar cena…").clicked() {
            st.scene_action = Some(SceneAction::Load);
        }
    });
    ui.separator();
    match &st.autosave_path {
        Some(path) => {
            ui.checkbox(&mut st.autosave_on, format!("autosave em {path}"))
                .on_hover_text("ao arrancar, o programa continua deste ficheiro; ao fechar grava-o");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut st.autosave_every).range(1000..=10_000_000).speed(1000));
                ui.label("epochs entre gravações");
            });
            if ui.button("gravar autosave agora").clicked() {
                st.scene_action = Some(SceneAction::AutosaveNow);
            }
        }
        None => {
            ui.label("autosave desligado neste arranque (cenário de teste)");
        }
    }
    ui.separator();
    if ui
        .button("mundo novo com os valores por omissão")
        .on_hover_text("esquece os parâmetros da cena/autosave: valores do código e o terreno de arranque")
        .clicked()
    {
        st.scene_action = Some(SceneAction::NewWorld);
    }
    if !st.scene_msg.is_empty() {
        ui.separator();
        ui.label(&st.scene_msg);
    }
}

fn tab_view(ui: &mut egui::Ui, st: &mut UiState) {
    egui::ComboBox::from_label("vista").selected_text(VIEW_NAMES[st.view_mode as usize]).show_ui(ui, |ui| {
        for (i, n) in VIEW_NAMES.iter().enumerate() {
            ui.selectable_value(&mut st.view_mode, i as u32, *n);
        }
    });
    ui.add(egui::Slider::new(&mut st.monomer_brightness, 0.0..=1.0).text("brilho dos monómeros"));
    const SIGNAL_VIEWS: [&str; 5] =
        ["química", "sinal α", "sinal β", "α (vermelho) + β (verde)", "parentesco com o selecionado"];
    egui::ComboBox::from_label("cor dos agentes").selected_text(SIGNAL_VIEWS[st.signal_view as usize]).show_ui(
        ui,
        |ui| {
            for (i, n) in SIGNAL_VIEWS.iter().enumerate() {
                ui.selectable_value(&mut st.signal_view, i as u32, *n);
            }
        },
    );
    if st.signal_view == 4 {
        ui.label("clica num organismo: vermelho = genoma igual, azul = sem nada em comum (8-meros partilhados; o filho conta como parente)");
    }
    ui.separator();
    if ui
        .button("editor dos aminoácidos e órgãos (browser)")
        .on_hover_text("http://127.0.0.1:8787 — mudanças aplicadas ao vivo")
        .clicked()
    {
        st.open_editor = true;
    }
}

fn tab_matter(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    ui.add(egui::Slider::new(&mut world.seed_density, 0.05..=1.0).text("densidade inicial (na próxima semente)"));
    ui.add(egui::Slider::new(&mut p.diffusion, 0.0..=50.0).text("difusão ×"));
    ui.add(egui::Slider::new(&mut p.monomer_pressure, 0.0..=20.0).text("pressão dos monómeros"))
        .on_hover_text("a difusão empurra das zonas cheias para as vazias");
    ui.add(
        egui::Slider::new(&mut p.activation_decay, 0.0..=0.002)
            .logarithmic(true)
            .text("decaimento da ativação (por passo)"),
    );
    // Reativação uniforme: probabilidade por passo de um gasto voltar a
    // ativado; escala logarítmica para afinar valores pequenos.
    ui.add(
        egui::Slider::new(&mut p.reactivation_rate, 0.0..=0.02)
            .logarithmic(true)
            .smallest_positive(1e-5)
            .text("reativação dos gastos"),
    );
    if p.reactivation_rate > 0.0 {
        ui.label(format!("  pousio médio de um gasto: {:.0} passos", 1.0 / p.reactivation_rate));
    }
    ui.add(egui::Slider::new(&mut p.settle, 0.0..=10.0).text("assentamento ×"));
    ui.add(egui::Slider::new(&mut p.cohesion, 0.0..=2.0).text("coesão"));
}

fn tab_light(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    ui.add(egui::Slider::new(&mut p.uv_strength, 0.0..=10.0).text("força UV (sol)"));
    ui.add(egui::Slider::new(&mut p.direct_photoactivation, 0.0..=1.0).text("fotoativação direta dos gastos"))
        .on_hover_text("0 = a luz só vira comida pelos fotossistemas dos agentes");
    let light_changed =
        ui.add(egui::Slider::new(&mut p.uv_depth, 0.0..=30.0).text("atenuação UV pela água")).changed();
    ui.add(egui::Slider::new(&mut p.monomer_uv_absorb, 0.0..=5.0).text("absorção UV pelos monómeros"));
    ui.add(egui::Slider::new(&mut p.uv_damage, 1.0..=50.0).text("dano UV"));
    ui.add(
        egui::Slider::new(&mut world.settings.light_rows_per_step, 0..=16)
            .text("velocidade da luz (linhas/passo; 0 = varredura)"),
    )
    .on_hover_text("a luz e as sombras descem N linhas por passo; 0 = varredura inteira de N em N passos");
    if light_changed {
        world.invalidate_light();
    }
}

fn tab_water(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    let st = &mut world.settings;
    ui.checkbox(&mut st.fluid_enabled, "fluido ligado");
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
    ui.separator();
    ui.strong("Fumarolas");
    ui.add(egui::Slider::new(&mut world.fumarole_gain, 0.0..=5.0).text("força das fumarolas ×"));
    if world.heat_image.is_some() {
        ui.small("+ calor dos píxeis vermelhos do terreno carregado");
    }
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
}

fn tab_terrain(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    ui.small("PNG: AZUL = terreno (0 água, fraco entulho, forte rocha, 255 rocha maciça); VERMELHO = calor por píxel (vermelho − verde). Cinzentos também servem.");
    ui.horizontal(|ui| {
        ui.label("ficheiro");
        ui.text_edit_singleline(&mut st.terrain_path);
    });
    ui.horizontal(|ui| {
        if ui.button("carregar…").on_hover_text("escolhe um PNG; o mundo é semeado de novo").clicked() {
            st.terrain_action = Some(TerrainAction::Load);
        }
        if ui.button("gravar…").on_hover_text("grava o terreno atual (e o calor) num PNG").clicked() {
            st.terrain_action = Some(TerrainAction::Save);
        }
        if ui.button("terreno gerado").clicked() {
            st.terrain_action = Some(TerrainAction::Generated);
        }
    });
    if !st.terrain_msg.is_empty() {
        ui.label(&st.terrain_msg);
    }
    ui.separator();
    ui.checkbox(&mut world.settings.terrain_enabled, "física do terreno ligada");
    ui.add(egui::Slider::new(&mut world.params.bioturbation, 0.0..=0.5).text("bioturbação (empurrar entulho)"));
    ui.add(egui::Slider::new(&mut world.params.bioturbation_cost, 0.0..=1.0).text("custo por grão empurrado"));
    ui.add(egui::Slider::new(&mut world.params.sedimentation, 0.0..=0.5).text("sedimentação (afundar ∝ √n)"));
}

fn tab_life(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    ui.strong("Semear");
    ui.add(egui::Slider::new(&mut st.seed_count, 1..=20000).text("sementes"));
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
    ui.separator();
    ui.strong("Metabolismo e ciclo de vida");
    let p = &mut world.params;
    ui.add(egui::Slider::new(&mut p.death_probability, 0.0..=0.2).text("mortalidade base"));
    ui.add(egui::Slider::new(&mut p.spawn_energy, 0.1..=50.0).text("energia inicial"));
    ui.add(egui::Slider::new(&mut p.food_power, 0.0..=20.0).text("energia por monómero"));
    ui.add(
        egui::Slider::new(&mut p.uptake_rate, 0.0..=0.01).logarithmic(true).smallest_positive(1e-5).text("taxa de hidrólise"),
    )
    .on_hover_text("quanto os agentes comem; a 0 ninguém come");
    let mut hunger = p.hunger_regulation != 0;
    ui.checkbox(&mut hunger, "regulação pela carga energética (cheio não come)");
    p.hunger_regulation = hunger as u32;
    ui.add(egui::Slider::new(&mut p.maintenance_cost, 0.0..=0.01).text("manutenção por resíduo"));
    ui.add(egui::Slider::new(&mut p.pairing_rate, 0.0..=8.0).text("emparelhamento (bases/passo)"));
    ui.add(egui::Slider::new(&mut p.pairing_cost, 0.0..=2.0).text("custo por base copiada"));
    ui.add(egui::Slider::new(&mut p.mutation_rate, 0.0..=0.05).text("taxa de mutação"));
    let mut aug = p.require_start != 0;
    ui.checkbox(&mut aug, "tradução começa no AUG (nascimentos novos)");
    p.require_start = aug as u32;
    ui.separator();
    ui.strong("Ligações entre agentes (órgão âncora: + liga a −)");
    ui.add(egui::Slider::new(&mut p.bond_rate, 0.0..=1.0).logarithmic(true).smallest_positive(1e-3).text("formação"))
        .on_hover_text("probabilidade por passo de um agente com uma âncora livre a tentar ligar a uma âncora oposta de um vizinho; a duração vem da variante da âncora (editor)");
    ui.add(egui::Slider::new(&mut p.bond_energy_share, 0.0..=0.2).text("energia partilhada"))
        .on_hover_text("fração da diferença de energia que passa por cada ligação, por passo");
    ui.add(egui::Slider::new(&mut p.bond_signal, 0.0..=1.0).text("sinais pela ligação"));
}

fn tab_motion(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    ui.strong("Natação");
    let mut rft = p.rft_enabled != 0;
    ui.checkbox(&mut rft, "natação (RFT)");
    p.rft_enabled = rft as u32;
    ui.add(egui::Slider::new(&mut p.swim_gain, 0.0..=50.0).text("ganho da natação"));
    ui.add(egui::Slider::new(&mut p.swim_wobble, 0.0..=1.0).text("vaivém da natação"))
        .on_hover_text("1 = balanço físico de cada batida; 0 = só o avanço médio");
    ui.add(egui::Slider::new(&mut p.motion_cost, 0.0..=2.0).text("custo de dissipação do movimento"))
        .on_hover_text("energia = isto × Σ √arrasto·dθ² das juntas: bater depressa custa ao quadrado");
    ui.add(egui::Slider::new(&mut p.agent_fluid_push, -1.0..=1.0).text("agentes empurram a água"))
        .on_hover_text("cada resíduo devolve ao fluido o seu arrasto (só no mundo com fluido)");
    let mut fso = p.fluid_swim_only != 0;
    if ui
        .checkbox(&mut fso, "experiência: natação só pelo fluido")
        .on_hover_text("sem RFT: a forma empurra a água e a água leva o agente")
        .changed()
    {
        p.fluid_swim_only = fso as u32;
    }
    ui.separator();
    ui.strong("Juntas e contacto");
    ui.add(egui::Slider::new(&mut p.chain_stiffness, 1.0..=100.0).text("rigidez das juntas"));
    ui.add(egui::Slider::new(&mut p.thermal_kt, 0.0..=5.0).text("agitação térmica (kT)"));
    ui.add(egui::Slider::new(&mut p.motor_amplitude, 0.0..=1.0).text("curso do motor (rad)"));
    ui.add(egui::Slider::new(&mut p.joint_coupling, 0.0..=0.95).text("acoplamento entre juntas"));
    ui.add(egui::Slider::new(&mut p.brownian, 0.0..=20.0).text("movimento browniano"));
    ui.add(egui::Slider::new(&mut p.phoretic_gain, 0.0..=500.0).text("difusioforese"));
    ui.checkbox(&mut world.settings.contact_enabled, "repulsão entre agentes");
}

fn tab_info(ui: &mut egui::Ui, st: &mut UiState, prof: &mut Profiler) {
    conservation(ui, st);
    ui.separator();
    ui.checkbox(&mut prof.enabled, "profiler (submit+wait por segmento)");
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
