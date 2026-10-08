//! Painéis egui: livro-razão, profiler, sliders do mundo, vistas de debug, inspetor.

pub mod inspector;

use crate::gpu::profiler::Profiler;
use crate::world::{Ledger, MAX_STEPS_PER_FRAME, World};

pub struct UiState {
    pub paused: bool,
    pub steps_per_frame: u32,
    /// Refresco fluido: faz menos passos por frame do que os pedidos quando
    /// a placa não aguenta (o ecrã fica a ~30 imagens por segundo).
    pub smooth_refresh: bool,
    /// Passos feitos no último frame (para mostrar).
    pub steps_done: u32,
    /// 0 = normal, 1–4 = ativados A U G C, 5 = gastos, 6 terreno, 7 temperatura, 8 UV, 9 fluido.
    pub view_mode: u32,
    /// Contagem exata escrita na sementeira (base do Δ).
    pub baseline: Ledger,
    pub ledger: Option<Ledger>,
    pub ledger_epoch: u32,
    pub reseed: bool,
    /// Recomeçar: sopa e vida novas, epoch a zero, mesmo terreno e parâmetros.
    pub restart: bool,
    pub vsync: bool,
    /// Brilho da camada de monómeros na vista normal.
    pub monomer_brightness: f32,
    /// Círculo de confusão dos monómeros (células; 0 = quadrados).
    pub coc_radius: f32,
    /// Cor dos agentes: 0 química, 1 sinal α, 2 sinal β, 3 α e β.
    pub signal_view: u32,
    /// Órgão a marcar no mapa: tipo + 1 (0 = nenhum).
    pub mark_organ: u32,
    /// Semear: quantas sementes, comprimento mínimo/máximo, começar por AUG.
    pub seed_count: u32,
    pub seed_len: [u32; 2],
    pub seed_aug: bool,
    pub seed_now: bool,
    /// "Ativar já": fração dos gastos a ativar e pedido.
    pub activate_frac: f32,
    pub activate_now: bool,
    /// Terreno em imagem: caminho, ação pedida e resultado da última.
    pub terrain_path: String,
    pub terrain_action: Option<TerrainAction>,
    /// Pincel do terreno: ligado, material (ver PAINT_MATERIALS), raio em
    /// células e força das fumarolas (0..1).
    pub paint_on: bool,
    pub paint_material: usize,
    pub paint_radius: f32,
    pub paint_strength: f32,
    pub terrain_msg: String,
    /// Pedido para abrir o editor dos aminoácidos no browser.
    pub open_editor: bool,
    /// Separador ativo do painel.
    pub tab: Tab,
    /// Velocidade e população (atualizadas ~2×/s em `update_stats`).
    pub stats: Stats,
    /// Estatísticas ao longo do tempo (gráficos e logs/estatisticas.csv).
    /// Pedido de relatório (página HTML das espécies e linhagens).
    pub report_now: bool,
    /// Pedido da página só com a árvore das linhagens.
    pub tree_now: bool,
    /// Pedido de captura do mundo inteiro: lado da imagem em píxeis (0 = nada).
    pub big_shot: u32,
    /// AGENTES GUARDADOS: pedido (gravar o selecionado, carregar, espalhar).
    pub agent_action: Option<AgentAction>,
    /// Com um agente carregado: clicar na vista põe lá uma cópia.
    pub place_agent: bool,
    /// O agente carregado, para mostrar (nome do ficheiro e bases).
    pub agent_info: String,
    /// Quantas cópias espalhar de cada vez.
    pub agent_copies: u32,
    /// MODO FOTO/VÍDEO: mostrar a mira de enquadramento na vista.
    pub frame_guide: bool,
    /// Pedido de uma fotografia do enquadramento.
    pub photo_now: bool,
    /// A gravar vídeo (uma imagem de `rec_every` em `rec_every` frames).
    pub rec: bool,
    pub rec_every: u32,
    /// Lado da imagem (fotografia e vídeo), em píxeis.
    pub shot_size: u32,
    /// Estado da gravação, para mostrar (pasta, número de imagens).
    pub rec_info: String,
    /// Epochs entre censos das linhagens e o estado do registo (texto).
    pub lineage_every: u32,
    pub lineage_info: String,
    pub history: crate::stats::History,
    /// O que os dois gráficos mostram.
    pub charts: crate::stats::ChartSel,
    /// Cenas: ação pedida, autosave e mensagem da última operação.
    pub scene_action: Option<SceneAction>,
    pub autosave_on: bool,
    pub autosave_every: u32,
    /// Ficheiro do autosave (None = desligado neste arranque, p. ex. testes).
    pub autosave_path: Option<String>,
    pub scene_msg: String,
    /// Nome do mundo que está a correr (names.rs): mostra-se no separador
    /// da cena e é o nome sugerido ao gravar.
    pub scene_name: String,
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

/// Valor de `mark_organ` que marca os agentes ligados (MARK_BONDED no shader).
const MARK_BONDED: u32 = 255;

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
    (Tab::Vista, "View"),
    (Tab::Materia, "Matter"),
    (Tab::Luz, "Light"),
    (Tab::Agua, "Water and vents"),
    (Tab::Terreno, "Terrain"),
    (Tab::Vida, "Life"),
    (Tab::Movimento, "Body"),
    (Tab::Cena, "Scene"),
    (Tab::Graficos, "Charts"),
    (Tab::Info, "Info"),
];

/// O que fazer com um agente guardado em ficheiro.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentAction {
    /// Grava o genoma do agente selecionado (janela de ficheiros).
    Save,
    /// Carrega um genoma de um ficheiro (janela de ficheiros).
    Load,
    /// Espalha cópias do agente carregado pelo mundo.
    Spread,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainAction {
    /// Escolhe um PNG (janela) e troca o terreno do mundo vivo (os agentes
    /// e os monómeros ficam).
    Load,
    /// Mundo vazio (só água, sem fumarolas), semeado de novo: para pintar.
    Empty,
    /// Escolhe onde gravar (janela) e grava o terreno atual em PNG.
    Save,
    /// Volta ao terreno gerado e semeia de novo.
    Generated,
}

/// Materiais do pincel (o índice é `UiState::paint_material`).
pub const PAINT_MATERIALS: [&str; 7] = [
    "water (erases terrain)",
    "fine rubble (1 grain)",
    "dense rubble (2 grains)",
    "rock",
    "vent: heat",
    "vent: chemistry (reductant)",
    "erase vents",
];

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
            smooth_refresh: true,
            steps_done: 1,
            view_mode: 0,
            baseline,
            ledger: None,
            ledger_epoch: 0,
            reseed: false,
            restart: false,
            vsync: true,
            monomer_brightness: 0.5,
            coc_radius: 0.35,
            signal_view: 0,
            mark_organ: 0,
            stats: Stats::default(),
            report_now: false,
            tree_now: false,
            big_shot: 0,
            agent_action: None,
            place_agent: false,
            agent_info: String::new(),
            agent_copies: 200,
            frame_guide: false,
            photo_now: false,
            rec: false,
            rec_every: 2,
            shot_size: 1024,
            rec_info: String::new(),
            lineage_every: 50_000,
            lineage_info: String::new(),
            history: crate::stats::History::default(),
            charts: Default::default(),
            terrain_path: std::env::var("RIBO_TERRAIN").unwrap_or_else(|_| "assets/terreno.png".into()),
            terrain_action: None,
            paint_on: false,
            paint_material: 3,
            paint_radius: 12.0,
            paint_strength: 1.0,
            terrain_msg: String::new(),
            open_editor: false,
            tab: Tab::Vista,
            seed_count: 500,
            seed_len: [12, 120],
            seed_aug: true,
            seed_now: false,
            activate_frac: 0.3,
            activate_now: false,
            scene_action: None,
            autosave_on: true,
            autosave_every: 50_000,
            autosave_path: None,
            scene_msg: String::new(),
            scene_name: String::new(),
        }
    }
}

pub const VIEW_NAMES: [&str; 11] = [
    "normal",
    "1 A act.",
    "2 U act.",
    "3 G act.",
    "4 C act.",
    "5 spent",
    "6 terrain",
    "7 temperature",
    "8 UV light",
    "9 fluid",
    "vent reductant",
];
const CH: [&str; 4] = ["A", "U", "G", "C"];

/// Deslizador e caixa numérica que só aplicam o valor escrito ao CONFIRMAR
/// (Enter ou sair da caixa). Por omissão o egui aplica a cada tecla; como os
/// parâmetros são f32, "0.9" fica 0,899999976, o egui vê um valor diferente do
/// texto e reescreve a caixa com zeros a meio da escrita ("0.900"), e não se
/// consegue acabar de escrever o número. Arrastar continua igual.
fn slider<'a, N: egui::emath::Numeric>(value: &'a mut N, range: std::ops::RangeInclusive<N>) -> egui::Slider<'a> {
    egui::Slider::new(value, range).update_while_editing(false)
}

fn drag<'a, N: egui::emath::Numeric>(value: &'a mut N) -> egui::DragValue<'a> {
    egui::DragValue::new(value).update_while_editing(false)
}

/// Desenha a interface em BARRAS FIXAS: controlos à esquerda, inspetor à
/// direita (quando há um organismo escolhido) e, no separador "Gráficos", os
/// gráficos no meio. Devolve o retângulo livre para a simulação (em pontos
/// do egui), ou None quando os gráficos o tapam.
pub fn draw(root: &mut egui::Ui, st: &mut UiState, world: &mut World, prof: &mut Profiler, ins: &mut inspector::Inspector) -> Option<egui::Rect> {
    egui::Panel::left("controlos").default_size(400.0).size_range(300.0..=800.0).resizable(true).show(root, |ui| {
        main_panel(ui, st, world, prof);
    });
    // Sempre presente: a simulação não muda de tamanho ao escolher um organismo.
    egui::Panel::right("inspetor").default_size(300.0).size_range(280.0..=600.0).resizable(true).show(root, |ui| {
        inspector::panel(ui, ins, &world.organ_table, &world.amino, &world.organ_code, world.params.require_start != 0);
    });
    let free = root.available_rect_before_wrap();
    if st.tab == Tab::Graficos {
        egui::CentralPanel::default_margins().show(root, |ui| crate::stats::draw(ui, &mut st.history, &mut st.charts));
        return None;
    }
    Some(free)
}

fn main_panel(ui: &mut egui::Ui, st: &mut UiState, world: &mut World, prof: &mut Profiler) {
    // ---- Topo, sempre visível ----
    ui.horizontal(|ui| {
        if ui.button(if st.paused { "▶ resume" } else { "⏸ pause" }).clicked() {
            st.paused = !st.paused;
        }
        if ui.button("new seed").clicked() {
            st.reseed = true;
        }
        if ui
            .button("restart")
            .on_hover_text("starts again in the same place: fresh soup, no agents and epoch at zero, but with the terrain as it is now and all your current parameters. (\"new seed\" also generates a new terrain, unless it comes from an image, and leaves the epoch alone)")
            .clicked()
        {
            st.restart = true;
        }
    });
    if world.params.day_period >= 1.0 {
        let d = world.params.daylight(world.params.epoch);
        ui.label(if d > 0.0 { format!("☀ day ({:.0}% of the sun)", d * 100.0) } else { "☾ night".to_string() });
    }
    ui.label(format!(
        "epoch {}   ({:.0} epochs/s)   frame {:.1} ms ({:.0} fps)",
        world.params.epoch,
        st.stats.epochs_per_sec,
        prof.frame_ms,
        1000.0 / prof.frame_ms.max(1e-3)
    ));
    ui.label(match st.stats.alive {
        Some(n) => format!(
            "living agents {n}   (+{:.0}/s births, −{:.0}/s deaths)",
            st.stats.births_per_sec, st.stats.deaths_per_sec
        ),
        None => "living agents …".into(),
    });
    ui.horizontal(|ui| {
        ui.add(slider(&mut st.steps_per_frame, 1..=MAX_STEPS_PER_FRAME).text("steps/frame"));
        ui.checkbox(&mut st.vsync, "vsync");
        ui.checkbox(&mut st.smooth_refresh, "smooth display")
            .on_hover_text("steps per frame become a maximum: if the graphics card cannot do them all in time, fewer are done per frame so the display does not freeze. The display refreshes at ~30 images per second when drawing is cheap; when it is expensive (many agents in view) it refreshes more slowly, down to 15 per second, so the simulation gets up to 80% of the card's time");
        if st.smooth_refresh && st.steps_done < st.steps_per_frame {
            ui.label(format!("doing {}", st.steps_done));
        }
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
        Tab::Materia => tab_matter(ui, st, world),
        Tab::Luz => tab_light(ui, world),
        Tab::Agua => tab_water(ui, world),
        Tab::Terreno => tab_terrain(ui, st, world),
        Tab::Vida => tab_life(ui, st, world),
        Tab::Movimento => tab_motion(ui, world),
        Tab::Cena => tab_scene(ui, st, world),
        Tab::Graficos => crate::stats::controls(ui, &mut st.history),
        Tab::Info => tab_info(ui, st, prof),
    });
}

/// Parâmetros diferentes dos valores do código, com botões para os repor
/// (o autosave retoma-os: um slider esquecido fica preso entre arranques).
fn changed_params(ui: &mut egui::Ui, world: &mut World) {
    let diff = world.params.changed_from_default();
    let title = if diff.is_empty() {
        "parameters: all at the code values".to_string()
    } else {
        format!("parameters differing from the code: {}", diff.len())
    };
    egui::CollapsingHeader::new(title).id_salt("changed_params").show(ui, |ui| {
        for &(k, a, b) in &diff {
            ui.horizontal(|ui| {
                ui.label(format!("{k}: {a:.6} (code {b:.6})"));
                if ui.small_button("reset").clicked() {
                    world.params.set_named(k, b);
                }
            });
        }
        if !diff.is_empty() && ui.button("reset all (keeps the world)").clicked() {
            for &(k, _, b) in &diff {
                world.params.set_named(k, b);
            }
        }
    });
}

fn tab_scene(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    if !st.scene_name.is_empty() {
        ui.horizontal(|ui| {
            ui.label("world:");
            ui.label(egui::RichText::new(st.scene_name.replace('_', " ")).italics().strong())
                .on_hover_text("name of this world, given when it was created, restarted or seeded again (or taken from the scene file that was loaded). It is the suggested file name when you save the scene");
        });
        ui.separator();
    }
    changed_params(ui, world);
    ui.separator();
    ui.label("A scene = the whole world (matter, terrain, water, agents) and all the parameters.");
    ui.label("The amino acid and organ tables always come from assets/ (not from the scene).");
    ui.horizontal(|ui| {
        if ui.button("save scene…").clicked() {
            st.scene_action = Some(SceneAction::Save);
        }
        if ui.button("load scene…").clicked() {
            st.scene_action = Some(SceneAction::Load);
        }
    });
    ui.separator();
    match &st.autosave_path {
        Some(path) => {
            ui.checkbox(&mut st.autosave_on, format!("autosave to {path}"))
                .on_hover_text("on startup the program continues from this file; on exit it saves it");
            ui.horizontal(|ui| {
                ui.add(drag(&mut st.autosave_every).range(1000..=10_000_000).speed(1000));
                ui.label("epochs between saves");
            });
            if ui.button("save autosave now").clicked() {
                st.scene_action = Some(SceneAction::AutosaveNow);
            }
        }
        None => {
            ui.label("autosave off for this run (test scenario)");
        }
    }
    ui.separator();
    ui.strong("Species and lineages");
    ui.horizontal(|ui| {
        ui.label("lineage census every");
        ui.add(drag(&mut st.lineage_every).range(5_000..=5_000_000).speed(1000));
        ui.label("epochs");
    })
    .response
    .on_hover_text("from time to time the living genomes are grouped into species and each one is linked to the one it descends from: this is the record of this run's tree of life. It is saved with the scene and starts over with a new world. Each census reads all the agents from the graphics card (a pause of hundredths of a second)");
    if !st.lineage_info.is_empty() {
        ui.small(&st.lineage_info);
    }
    if ui
        .button("generate report (HTML page)")
        .on_hover_text("a page with the species and their two forms (portraits, organs, where they live), who can attack whom, attacks and bonds in progress, and the trees: that of the recorded lineages and that of the kinship between the living species. It is written to saves/relatorios/ and opens in the browser. Takes a few seconds")
        .clicked()
    {
        st.report_now = true;
    }
    if ui
        .button("open the lineage tree")
        .on_hover_text("this run's tree of life on an interactive page (zoom, drag, click a branch): the drawing of the two forms of each branch, the organs and the population over time. Takes a few seconds: the portraits are drawn by the simulation in a separate world")
        .clicked()
    {
        st.tree_now = true;
    }
    ui.horizontal(|ui| {
        ui.label("whole-world capture:");
        for (name, side) in [("8k", 8192u32), ("16k", 16384)] {
            if ui
                .button(name)
                .on_hover_text("the whole world in one PNG image (8192 or 16384 pixels per side), with the chosen view and brightness. It is written to saves/capturas/. The simulation stops for a few seconds while it draws; the file finishes saving in the background (the 16k one takes hundreds of MB)")
                .clicked()
            {
                st.big_shot = side;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut st.frame_guide, "framing guide").on_hover_text("shows in the view the framing (a square in the center) that the photo and the video capture, with the rule of thirds");
        if ui.button("photo").on_hover_text("saves what the framing guide frames to saves/capturas/ (without the interface), with the chosen view and brightness").clicked() {
            st.photo_now = true;
        }
        let label = if st.rec { egui::RichText::new("■ stop").color(egui::Color32::from_rgb(255, 90, 80)) } else { egui::RichText::new("● rec") };
        if ui.button(label).on_hover_text("records what the framing guide frames straight into an MP4 video in saves/videos/ (the images go raw to ffmpeg, with no intermediate files). Move and zoom the camera freely while recording; the size stays as it was at the start").clicked() {
            st.rec = !st.rec;
        }
        egui::ComboBox::from_id_salt("shot_size").selected_text(format!("{} px", st.shot_size)).width(70.0).show_ui(ui, |ui| {
            for v in [512u32, 1024, 2048] {
                ui.selectable_value(&mut st.shot_size, v, format!("{v} px"));
            }
        });
    });
    if st.rec || !st.rec_info.is_empty() {
        ui.add(slider(&mut st.rec_every, 1..=30).text("1 image every N frames")).on_hover_text("the video runs at 30 images per second: with 2, one second of video is 60 frames of the simulation");
        ui.small(&st.rec_info);
    }
    ui.separator();
    if ui
        .button("new world with the default values")
        .on_hover_text("forgets the scene/autosave parameters: code values and the startup terrain")
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
    egui::ComboBox::from_label("view").selected_text(VIEW_NAMES[st.view_mode as usize]).show_ui(ui, |ui| {
        for (i, n) in VIEW_NAMES.iter().enumerate() {
            ui.selectable_value(&mut st.view_mode, i as u32, *n);
        }
    });
    ui.add(slider(&mut st.monomer_brightness, 0.0..=1.0).text("monomer brightness"));
    ui.add(slider(&mut st.coc_radius, 0.0..=1.0).text("monomer circle of confusion (cells)"))
        .on_hover_text("up close, each monomer is drawn as a soft disc of this radius, at its own position inside the cell (drawing only: the simulation counts monomers per cell). Small = loose molecules; large = continuous haze; 0 = squares, one color per cell");
    const SIGNAL_VIEWS: [&str; 6] = [
        "chemistry",
        "α signal",
        "β signal",
        "α (red) + β (green)",
        "kinship with the selected one",
        "γ (red) + δ (green)",
    ];
    egui::ComboBox::from_label("agent color").selected_text(SIGNAL_VIEWS[st.signal_view as usize]).show_ui(
        ui,
        |ui| {
            for (i, n) in SIGNAL_VIEWS.iter().enumerate() {
                ui.selectable_value(&mut st.signal_view, i as u32, *n);
            }
        },
    );
    if st.signal_view == 4 {
        ui.label("click an organism: green ball = close genome, yellow = intermediate, red = distant (shared 8-mers; the child counts as kin)");
    }
    let names = crate::life::organs::ORGAN_NAMES_EN;
    let current = match st.mark_organ {
        0 => "none",
        MARK_BONDED => "bonded by anchor",
        m => names[(m as usize - 1).min(names.len() - 1)],
    };
    egui::ComboBox::from_label("mark who has the organ").selected_text(current).show_ui(ui, |ui| {
        ui.selectable_value(&mut st.mark_organ, 0, "none");
        ui.selectable_value(&mut st.mark_organ, MARK_BONDED, "bonded by anchor");
        for (i, n) in names.iter().enumerate() {
            ui.selectable_value(&mut st.mark_organ, i as u32 + 1, *n);
        }
    });
    if st.mark_organ == MARK_BONDED {
        ui.small("golden ball = agent with a live bond to another. Up close, the bond is a thread: light blue = from birth (parent and child), golden = made on touching");
    } else if st.mark_organ != 0 {
        ui.small("cyan ball = agent with this organ (the same size on screen at any zoom)");
    }
    ui.separator();
    if ui
        .button("amino acid and organ editor (browser)")
        .on_hover_text("http://127.0.0.1:8787 — changes applied live")
        .clicked()
    {
        st.open_editor = true;
    }
}

fn tab_matter(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    let p = &mut world.params;
    ui.strong("Initial soup and activation");
    ui.add(slider(&mut world.seed_density, 0.05..=1.0).text("initial density (on the next seed)"));
    ui.add(slider(&mut world.seed_active, 0.0..=1.0).text("initial activated fraction (on the next seed)"))
        .on_hover_text("fraction of the monomers that are born activated when a new world is seeded (0.5 = half)");
    ui.horizontal(|ui| {
        ui.add(slider(&mut st.activate_frac, 0.0..=1.0).text("activate now"));
        if ui.button("activate").on_hover_text("activates this fraction of the free spent monomers now (matter does not change)").clicked() {
            st.activate_now = true;
        }
    });
    ui.add(
        slider(&mut p.activation_decay, 0.0..=0.002)
            .logarithmic(true)
            .text("activation decay (per step)"),
    );
    // Reativação uniforme: probabilidade por passo de um gasto voltar a
    // ativado; escala logarítmica para afinar valores pequenos.
    ui.add(
        slider(&mut p.reactivation_rate, 0.0..=0.02)
            .logarithmic(true)
            .smallest_positive(1e-5)
            .text("reactivation of spent monomers"),
    );
    if p.reactivation_rate > 0.0 {
        ui.label(format!("  mean fallow time of a spent monomer: {:.0} steps", 1.0 / p.reactivation_rate));
    }
    ui.strong("Abiotic activation (without life)");
    ui.add(slider(&mut p.direct_photoactivation, 0.0..=1.0).logarithmic(true).smallest_positive(0.001).text("by the sun (photoactivation of spent monomers)"))
        .on_hover_text("light reactivates spent monomers on its own (it follows day and night and the shadows). 1 = ~1.8% of the spent ones per step in full sun; 0.02 = a trickle. 0 = photosystems only");
    ui.add(slider(&mut p.thermal_activation, 0.0..=2.0).logarithmic(true).smallest_positive(0.01).text("by heat (above T = 2)"))
        .on_hover_text("heat reactivates spent monomers on its own, only in water above T = 2 (vents). 0 = only chemosynthesizers make use of the vents");
    ui.separator();
    ui.strong("Monomer transport");
    ui.add(slider(&mut p.diffusion, 0.0..=50.0).text("diffusion ×"));
    ui.add(slider(&mut p.transport_every, 1..=4).text("transport every N steps"))
        .on_hover_text("monomer transport (current, diffusion, aggregation, reactions) is the most expensive part of each step. With 2, it runs every other step, with twice the displacement each time: the simulation gets ~15% faster and the monomers move in larger, less frequent jumps. Agents eat on every step. The maximum possible diffusion drops in the same proportion. 1 = as always");
    ui.add(slider(&mut p.monomer_pressure, 0.0..=20.0).text("monomer pressure"))
        .on_hover_text("diffusion pushes from full areas to empty ones");
    ui.add(slider(&mut p.cohesion, 0.0..=2.0).text("cohesion of activated monomers of the same type"))
        .on_hover_text("an activated monomer leaves a cell less readily when the neighbors hold activated monomers of the same type: it gathers each type into patches");
    ui.add(
        slider(&mut p.aggregation, 0.0..=1.0).logarithmic(true).smallest_positive(0.01).text("aggregation of activated monomers"),
    )
        .on_hover_text("binding energy between neighboring activated monomers (÷ temperature): they form clumps that the current carries whole; heat dissolves them");
    ui.separator();
    ui.strong("Gravity");
    ui.add(slider(&mut p.settle, 0.0..=100.0).logarithmic(true).smallest_positive(0.1).text("gravity on MONOMERS ×"))
        .on_hover_text("probability per step of a monomer dropping one cell = 0.002 × this (10 = 0.02 cells/step)");
    ui.add(slider(&mut p.sediment_settle, 0.0..=5.0).text("gravity on rubble GRAINS ×"))
        .on_hover_text("fall speed (×0.5 fluid cells/s): a loose grain moves with the current minus the fall — it rises where the upward current is stronger (suspension) and settles where it slows down. 0 = they float");
    ui.add(slider(&mut p.sedimentation, 0.0..=0.5).text("gravity on AGENTS × (∝ √n)"))
        .on_hover_text("agents sink ∝ √(number of residues): large ones go down faster. 0 = they do not sink");
}

fn tab_light(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    ui.add(slider(&mut p.uv_strength, 0.0..=10.0).text("UV strength (sun)"));
    ui.add(slider(&mut p.sun_heat, 0.0..=5.0).text("solar heating"))
        .on_hover_text("the sun heats the surface (infrared absorbed by the water) and whatever absorbs light (rock, agents, monomers). It follows day and night");
    ui.add(slider(&mut p.day_period, 0.0..=200_000.0).text("day and night: period (epochs; 0 = always day)"))
        .on_hover_text("during the day the sun rises and sets as a half sine (dawn and dusk); for the rest of the cycle it is night");
    if p.day_period >= 1.0 {
        ui.add(slider(&mut p.day_fraction, 0.05..=1.0).text("fraction of the cycle that is day"))
            .on_hover_text("0.5 = equal day and night; 0.75 = day lasts 3/4 of the cycle; 1 = no night (but the sun still rises and sets)");
        ui.add(slider(&mut p.sun_angle, 0.0..=85.0).text("sun: maximum angle at sunrise/sunset (degrees)"))
            .on_hover_text("0 = always overhead; 85 = almost grazing light in the morning and evening (long shadows)");
        let d = p.daylight(p.epoch);
        ui.label(format!("  now: {} ({:.0}% of the sun)", if d > 0.0 { "day" } else { "night" }, d * 100.0));
    }
    let light_changed =
        ui.add(slider(&mut p.uv_depth, 0.0..=30.0).text("UV attenuation by water")).changed();
    ui.add(slider(&mut p.monomer_uv_absorb, 0.0..=5.0).text("UV absorption by monomers"));
    ui.add(slider(&mut p.photo_yield, 0.0..=0.5).logarithmic(true).smallest_positive(0.005).text("photosynthetic yield"))
        .on_hover_text("energy per unit of light absorbed by a photosystem (the recycler converts the same energy into activated monomers)");
    ui.add(slider(&mut p.uv_damage, 1.0..=50.0).text("UV damage"))
        .on_hover_text("risk of death per step in the light = base mortality × (this − 1) × light × 0.01, and it falls with the fraction of aromatic amino acids in the body (tryptophan, tyrosine: the sunscreen). 30 in full sun with no protection gives about 140 steps of life; 1 = no damage");
    ui.add(
        slider(&mut world.settings.light_rows_per_step, 0..=16)
            .text("speed of light (rows/step; 0 = sweep)"),
    )
    .on_hover_text("light and shadows descend N rows per step; 0 = a full sweep every N steps");
    if light_changed {
        world.invalidate_light();
    }
}

fn tab_water(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    let st = &mut world.settings;
    ui.strong("Fluid");
    ui.checkbox(&mut st.fluid_enabled, "fluid on");
    ui.checkbox(&mut st.multigrid, "pressure by multigrid (otherwise Jacobi)");
    if st.multigrid {
        ui.add(slider(&mut st.mg_cycles, 1..=4).text("V cycles"));
    } else {
        ui.add(slider(&mut st.jacobi_iters, 2..=256).text("Jacobi iterations"));
    }
    ui.add(slider(&mut st.fluid_substep, 1..=4).text("solve every N steps"));
    ui.add(slider(&mut p.fluid_vorticity, 0.0..=10.0).text("vorticity"));
    ui.add(slider(&mut p.fluid_viscosity, 0.0..=5.0).text("viscosity"));
    ui.add(slider(&mut p.fluid_decay, 0.9..=1.0).text("decay per frame"));
    ui.separator();
    ui.strong("Vents");
    ui.add(slider(&mut world.fumarole_gain, 0.0..=5.0).text("vent strength ×"));
    if world.heat_image.is_some() {
        ui.small("+ heat from the red pixels of the loaded terrain");
    }
    ui.add(slider(&mut p.chemo_take, 0.0..=0.2).logarithmic(true).smallest_positive(0.0005).text("chemosynthesis uptake"))
        .on_hover_text("fraction of the reductant in its fluid cell that each chemosynthesis organ takes per step (× metabolism × intensity × efficiency). The fractions of all the organs in a cell add up: high, and the first agents next to a vent use it all up; low, and the reductant travels further and feeds more agents, each one more slowly. With hunger regulation on, an agent that is full takes only what it has room for");
    ui.add(slider(&mut p.chemo_yield, 0.0..=5.0).text("chemosynthesis yield"))
        .on_hover_text("energy per unit of reductant consumed");
    ui.add(slider(&mut p.redox_decay, 0.0..=0.5).logarithmic(true).smallest_positive(0.001).text("reductant oxidation (1/s)"))
        .on_hover_text("the slower, the farther the reductant reaches ('vent reductant' view)");
    for (i, f) in world.fumaroles.iter_mut().enumerate() {
        ui.push_id(i, |ui| {
            let mut on = f.enabled != 0;
            ui.checkbox(&mut on, format!("vent {i}"));
            f.enabled = on as u32;
            ui.add(slider(&mut f.x_frac, 0.0..=1.0).text("x"));
            ui.add(slider(&mut f.y_frac, 0.0..=1.0).text("y"));
            ui.add(slider(&mut f.strength, 0.0..=20000.0).text("strength"));
            ui.add(slider(&mut f.spread, 60.0..=4000.0).text("radius (world)"));
        });
    }
}

fn tab_terrain(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    ui.small("PNG: BLUE = terrain (0 water, weak rubble, strong rock, 255 solid rock); RED = heat and GREEN = chemistry (reductant) of the vents, per pixel and independent (green without red = cold seep). With no green in the image, the chemistry follows the heat. Grays (r = g = b) give terrain only.");
    ui.horizontal(|ui| {
        ui.label("file");
        ui.text_edit_singleline(&mut st.terrain_path);
    });
    ui.horizontal(|ui| {
        if ui
            .button("load…")
            .on_hover_text("choose a PNG; the terrain changes in the running world (agents and monomers stay; whatever no longer fits moves aside). To start from scratch with it, seed again")
            .clicked()
        {
            st.terrain_action = Some(TerrainAction::Load);
        }
        if ui.button("save…").on_hover_text("saves the current terrain (and the heat) to a PNG").clicked() {
            st.terrain_action = Some(TerrainAction::Save);
        }
        if ui.button("generated terrain").clicked() {
            st.terrain_action = Some(TerrainAction::Generated);
        }
        if ui.button("empty world").on_hover_text("water only, no vents, seeded again: for painting by hand").clicked() {
            st.terrain_action = Some(TerrainAction::Empty);
        }
    });
    ui.separator();
    ui.strong("Brush");
    ui.checkbox(&mut st.paint_on, "paint with the left button (the right one still drags the view)");
    egui::ComboBox::from_label("material")
        .selected_text(PAINT_MATERIALS[st.paint_material.min(PAINT_MATERIALS.len() - 1)])
        .show_ui(ui, |ui| {
            for (i, name) in PAINT_MATERIALS.iter().enumerate() {
                ui.selectable_value(&mut st.paint_material, i, *name);
            }
        });
    ui.add(slider(&mut st.paint_radius, 1.0..=200.0).logarithmic(true).text("radius (cells)"));
    if st.paint_material == 4 || st.paint_material == 5 {
        ui.add(slider(&mut st.paint_strength, 0.05..=1.0).text("vent strength"));
    }
    ui.small("you can also paint while paused; monomers move aside when rock is placed (matter is conserved)");
    if !st.terrain_msg.is_empty() {
        ui.label(&st.terrain_msg);
    }
    ui.separator();
    ui.checkbox(&mut world.settings.terrain_enabled, "terrain physics on");
    ui.strong("Sediments (loose rubble)");
    ui.add(slider(&mut world.params.sediment_transport, 0.0..=5.0).text("drag by the current ×"))
        .on_hover_text("how much the current carries loose rubble (1 = as in v3)");
    ui.add(slider(&mut world.params.sediment_threshold, 0.0..=5.0).text("critical entrainment velocity"))
        .on_hover_text("Shields criterion: below this velocity (fluid cells/s) the current does not lift grains; above it, it lifts them ∝ to the excess");
    ui.add(slider(&mut world.params.bioturbation, 0.0..=0.5).text("bioturbation (pushing rubble)"));
    ui.add(slider(&mut world.params.bioturbation_cost, 0.0..=1.0).text("cost per grain pushed"));
}

fn tab_life(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    ui.strong("Seed");
    ui.add(slider(&mut st.seed_count, 1..=20000).text("seeds"));
    ui.horizontal(|ui| {
        ui.label("bases");
        ui.add(drag(&mut st.seed_len[0]).range(3..=256));
        ui.label("to");
        ui.add(drag(&mut st.seed_len[1]).range(3..=256));
    });
    ui.checkbox(&mut st.seed_aug, "start with AUG (taken from the soup)");
    if ui.button("seed (generation 0, assembled from the soup)").clicked() {
        st.seed_now = true;
    }
    ui.separator();
    ui.strong("Saved agents");
    ui.horizontal(|ui| {
        if ui.button("save the selected one…").on_hover_text("saves the genome of the selected agent to a text file (letters A, U, G, C) in saves/agentes/").clicked() {
            st.agent_action = Some(AgentAction::Save);
        }
        if ui.button("load…").on_hover_text("loads a saved genome; you can then spread it or place it with the mouse").clicked() {
            st.agent_action = Some(AgentAction::Load);
        }
    });
    if !st.agent_info.is_empty() {
        ui.small(&st.agent_info);
        ui.horizontal(|ui| {
            if ui.button("spread").on_hover_text("puts this number of copies at random places in the world (each one assembled from bases in the surrounding soup; where there are no bases, it is not born)").clicked() {
                st.agent_action = Some(AgentAction::Spread);
            }
            ui.add(drag(&mut st.agent_copies).range(1..=20000).suffix(" copies"));
        });
        ui.checkbox(&mut st.place_agent, "place with the mouse").on_hover_text("when on, a click in the view places a copy of the loaded agent there (instead of selecting whatever is there)");
    }
    ui.separator();
    ui.strong("Metabolism");
    let p = &mut world.params;
    ui.add(slider(&mut p.food_power, 0.0..=20.0).text("energy per monomer"));
    ui.add(
        slider(&mut p.uptake_rate, 0.0..=0.01).logarithmic(true).smallest_positive(1e-5).text("hydrolysis rate"),
    )
    .on_hover_text("how much the agents eat; at 0 nobody eats");
    let mut hunger = p.hunger_regulation != 0;
    ui.checkbox(&mut hunger, "regulation by energy charge (a full agent does not eat)");
    p.hunger_regulation = hunger as u32;
    ui.add(slider(&mut p.maintenance_cost, 0.0..=0.01).text("maintenance per residue"));
    ui.add(slider(&mut p.leak_base, 0.0..=1.0).text("base leak (body without a mouth) ×"))
        .on_hover_text("maintenance is multiplied by this + the leak of the mouths. 0.1 = a body without a mouth pays one tenth; 1 with the leak per mouth at 0 = as it was before");
    ui.add(slider(&mut p.mouth_leak, 0.0..=2.0).text("leak per open mouth ×"))
        .on_hover_text("what lets things in also lets them out: each open standard mouth adds this to the maintenance multiplier (stronger mouths add more; a closed mouth adds nothing). With 0.3, a body with three mouths pays the same as before");
    ui.add(slider(&mut p.skin_uptake, 0.0..=5.0).text("uptake without a mouth ×"))
        .on_hover_text("how much residues without a mouth absorb (× the amino acid's catalysis; a mouth is worth 20 to 80×). 0.2 = a 30-residue body without a mouth eats about one tenth of a weak mouth; 0 = only mouths eat");
    ui.add(slider(&mut p.metabolic_q10, 1.0..=4.0).text("metabolism: Q10"))
        .on_hover_text("how much the chemistry of life speeds up for each temperature 'span' (1 = does not depend on temperature). It multiplies maintenance, eating, chemosynthesis and pairing; not light");
    ui.add(slider(&mut p.metabolic_span, 0.5..=12.0).text("metabolism: span (T units per Q10)"));
    ui.add(slider(&mut p.metabolic_ref, 0.0..=8.0).text("metabolism: reference temperature (m = 1)"));
    ui.separator();
    ui.strong("Reproduction");
    ui.add(slider(&mut p.spawn_energy, 0.1..=50.0).text("initial energy"));
    ui.add(slider(&mut p.pairing_rate, 0.0..=8.0).text("pairing (bases/step)"));
    ui.add(slider(&mut p.pairing_cost, 0.0..=2.0).text("cost per base copied"))
        .on_hover_text("energy spent for each base of the genome that is copied");
    let mut salvage = p.salvage > 0.0;
    ui.checkbox(&mut salvage, "recharge: producers copy themselves with spent monomers")
        .on_hover_text("the energy that overflows from a photosystem or a chemosynthesis organ first charges a spent monomer for the copy of its own genome (building with raw material); only what is of no use for that goes on to reactivate monomers in the medium. Off = all the overflow goes to the medium (as it was)");
    p.salvage = salvage as u32 as f32;
    ui.add(slider(&mut p.mutation_rate, 0.0..=0.05).text("mutation rate"));
    let mut aug = p.require_start != 0;
    ui.checkbox(&mut aug, "translation starts at AUG (new births)");
    p.require_start = aug as u32;
    ui.separator();
    ui.strong("Death");
    ui.add(slider(&mut p.death_probability, 0.0..=0.2).text("base mortality"));
    ui.add(slider(&mut p.death_metab, 0.0..=1.0).text("mortality follows the pace of life"))
        .on_hover_text("1 = base mortality is multiplied by the agent's pace (metabolism × leak): dormancy, closed mouths, bodies without a mouth and cold water make it live longer (cysts, spores). 0 = mortality does not depend on the pace");
    ui.add(slider(&mut p.death_energy_cap, 0.0..=200.0).text("cap on the protection by energy"))
        .on_hover_text("base mortality is ÷ energy only up to this value (a reserve protects, hoarding more does not); 0 = no cap (v3)");
    ui.add(slider(&mut p.denature_temp, 0.0..=12.0).text("denaturation temperature"))
        .on_hover_text("above this, heat kills (view 7 = temperature; the core of the vents reaches 12)");
    ui.add(slider(&mut p.heat_kill, 0.0..=1.0).logarithmic(true).smallest_positive(0.001).text("denaturation by heat"))
        .on_hover_text("risk of dying in hot water (above the vent threshold), × (1 − thermostability of the body; a column of the amino acid table)");
    ui.separator();
    ui.strong("Predation");
    ui.add(slider(&mut p.protease_power, 0.0..=30.0).logarithmic(true).smallest_positive(0.1).text("protease strength ×"))
        .on_hover_text("multiplies the energy that proteases take from the victim per step of contact (base: 0.2 × strength × intensity for a victim with 10% target residues). When the victim's energy reaches zero, it dies. Each family cuts certain amino acids and proline defends. 0 = no predation");
    ui.add(slider(&mut p.protease_direct, 0.0..=1.0).text("direct fraction to the predator"))
        .on_hover_text("share of the energy taken that goes straight into the attacker. The rest goes to the medium (see the next slider). 0 = the predator has to eat the remains; 1 = it sucks up everything");
    ui.add(slider(&mut p.lysis_yield, 0.0..=1.0).text("yield of the remains"))
        .on_hover_text("of the energy that does not go straight to the predator, the fraction that stays in the medium as activated monomers next to the victim (one for each 'energy per monomer'); the rest is lost");
    ui.separator();
    ui.strong("Bonds between agents (anchor organ: + binds to −)");
    ui.add(slider(&mut p.bond_rate, 0.0..=1.0).logarithmic(true).smallest_positive(1e-3).text("formation"))
        .on_hover_text("probability per step of an agent with a free anchor trying to bind to an opposite anchor of a neighbor; the duration comes from the anchor's variant (editor)");
    ui.add(slider(&mut p.bond_energy_share, 0.0..=0.5).logarithmic(true).smallest_positive(0.001).text("energy diffusion through the bond"))
        .on_hover_text("energy flows through the bond from the fuller agent (energy ÷ capacity) to the emptier one, until both are equally full. It is the fraction of the difference that passes per step: 0.1 = the difference halves in ~7 steps; 0.01 = in ~70 (the old value)");
    ui.add(slider(&mut p.bond_matter_share, 0.0..=1.0).text("matter sharing through the bond"))
        .on_hover_text("probability per step of a bonded agent receiving from its partner a complement the partner has already captured, from the one whose genome copy is further ahead to the one that is further behind (a leaf feeding the root). It only passes when the base is useful to the receiver, one time in four on average. 0 = they do not share matter");
    ui.add(slider(&mut p.bond_signal, 0.0..=1.0).text("signals through the bond"));
}

fn tab_motion(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    ui.strong("Internal signals");
    ui.add(slider(&mut p.signal_mode, 0.0..=4.0).step_by(1.0).text("signal mode"))
        .on_hover_text("how the α/β signals travel along the chain and bend the joints. 0: conduction and sensitivity of each amino acid (v3). 1: equal diffusion to both sides and all joints respond alike (α bends one way, β the other). 2: the signal only travels from the N side to the C side, same response. 3: it travels from N to C and each joint responds according to its amino acid (the body decides which way it turns). 4: transport from the table (conduction of each amino acid and organ, as in 0) but all joints respond alike");
    ui.add(slider(&mut p.signal_crosstalk, 0.0..=0.5).text("leakage into the other channels"))
        .on_hover_text("an organ that emits on one channel lets this fraction of the emission leak into each of the other three (imperfect specificity): an α sensor also puts a little into β, γ and δ. Relays have no leakage (they serve to separate channels). 0 = clean emission");
    ui.add(slider(&mut p.clock_mute, 0.0..=1.0).text("mute clocks"))
        .on_hover_text("experiment: removes amplitude from all clocks (1 = mute). The organ stays in the body and keeps paying its cost; it is for seeing whether the agents move without it (sensors, emission by contact)");
    ui.small(match p.signal_mode.round() as i32 {
        0 => "  0 = per amino acid (each joint responds in its own way)",
        1 => "  1 = isotropic (diffusion to both sides, equal response)",
        2 => "  2 = directional (from the N side to the C side, equal response)",
        3 => "  3 = directional, response of each amino acid (the body decides)",
        _ => "  4 = transport from the table (amino acids and organs), equal response",
    });
    ui.separator();
    ui.strong("Swimming");
    let mut rft = p.rft_enabled != 0;
    ui.checkbox(&mut rft, "swimming (RFT)");
    p.rft_enabled = rft as u32;
    ui.add(slider(&mut p.swim_grip, 1.0..=30.0).logarithmic(true).text("swimming grip (2 = water)"))
        .on_hover_text("how much more the medium resists a segment moving sideways than lengthwise. 2 = water, the physical limit for a thin body. More = a medium that grips sideways (gel, mucus): each stroke yields more advance, IMMEDIATELY; when the body stops beating, it stops. 5 gives about 6 times the advance of water for an undulating swimmer");
    ui.add(slider(&mut p.swim_gain, 0.0..=50.0).text("swimming gain (with memory; 1 = off)"))
        .on_hover_text("multiplies the AVERAGE of the advance over the last steps (see the memory, next): above 1 the agents keep gliding in the old direction after stopping or turning (it is not physical). 1 = just the physics of the strokes");
    ui.add(slider(&mut p.swim_memory, 1.0..=200.0).logarithmic(true).text("gain memory (steps)"))
        .on_hover_text("how many steps the gliding lasts when the gain is greater than 1. 20 = as in v3 (speed lost 5% per step); 100 = what v4 had until now. With the gain at 1 it does nothing");
    ui.add(slider(&mut p.swim_wobble, 0.0..=1.0).text("swimming sway"))
        .on_hover_text("1 = physical sway of each stroke; 0 = only the mean advance");
    ui.add(slider(&mut p.motion_cost, 0.0..=2.0).logarithmic(true).smallest_positive(0.001).text("movement cost"))
        .on_hover_text("energy spent moving the body: dissipation in the water (this × Σ √drag·dθ² of the joints: beating fast costs quadratically) and the bending of the joints by the signals, which follows the same value. 0.02 = swimming costs about one third of the maintenance of a body without a mouth; 0.1 = the old value (swimming cost more than being alive)");
    ui.add(slider(&mut p.inertia, 0.0..=10.0).text("inertia of heavy bodies (non-physical)"))
        .on_hover_text("0 = physical: at the molecular scale water damps everything, a body that stops beating stops at once. Above 0 the velocity approaches the requested one with weight 1/(1 + this × mass/mass of an average body): heavy ones glide and accelerate slowly, like large swimmers");
    ui.add(slider(&mut p.flow_coupling, 0.0..=1.0).text("drag by the current"))
        .on_hover_text("1 = physical (a free body follows the water); less = experiment: currents carry them less and they also push the water less");
    ui.add(slider(&mut p.flow_mass, 0.0..=4.0).text("heavy bodies follow the current less"))
        .on_hover_text("each agent's drag by the current is divided by 1 + this value × (mean mass per residue ÷ that of a normal residue − 1): a body with heavy organs (stores, proteases with reach) is carried less by the water. Density counts, not length. 0 = all follow the water equally");
    ui.add(slider(&mut p.agent_fluid_push, -1.0..=1.0).text("agents push the water"))
        .on_hover_text("each residue gives its drag back to the fluid (only in the world with fluid)");
    let mut fso = p.fluid_swim_only != 0;
    if ui
        .checkbox(&mut fso, "experiment: swimming through the fluid only")
        .on_hover_text("without RFT: the shape pushes the water and the water carries the agent")
        .changed()
    {
        p.fluid_swim_only = fso as u32;
    }
    ui.separator();
    ui.strong("Joints and contact");
    ui.add(slider(&mut p.chain_stiffness, 1.0..=100.0).text("joint stiffness"));
    ui.add(slider(&mut p.joint_load, 0.0..=5.0).text("joint load (rotational drag)"))
        .on_hover_text("how much the water resists the bending of each joint. Each joint rotates the two sides of the body in opposite directions, and what resists it is the drag of the side that rotates more easily (segment length × organ drag × distance²). The tips bend fast, the trunk of a long body bends slowly and a bulky organ at a tip makes that tip slow. 0 = all joints at the same pace; 1 = the middle joint of an average body bends at half speed");
    ui.add(slider(&mut p.rest_angle_mult, 0.0..=6.0).text("× rest angles"))
        .on_hover_text("multiplies the rest angle of all joints (amino acids and organs). 1 = those in the table (almost straight bodies); 3–4 gives bends of 50–90° as in a real protein: coiled bodies, with more contacts between residues");
    ui.add(slider(&mut p.thermal_kt, 0.0..=5.0).text("thermal agitation (kT)"));
    ui.add(slider(&mut p.motor_amplitude, 0.0..=1.0).text("motor stroke (rad)"));
    ui.add(slider(&mut p.joint_coupling, 0.0..=0.95).text("coupling between joints"));
    ui.add(slider(&mut p.brownian, 0.0..=20.0).text("Brownian motion"));
    ui.add(slider(&mut p.brownian_rot, 0.0..=5.0).text("Brownian rotation"))
        .on_hover_text("thermal agitation also rotates bodies at random: 0.15 rad per step ÷ radius^1.5 (the radius is counted in residues, √n), times this. A naked RNA rotates a lot; a 16-residue body, ~0.02 rad per step. 1 = the usual value; 0 = they only rotate by swimming, by the water or by contact");
    ui.add(slider(&mut p.phoretic_gain, 0.0..=500.0).text("diffusiophoresis"));
    ui.checkbox(&mut world.settings.contact_enabled, "repulsion between agents");
}

fn tab_info(ui: &mut egui::Ui, st: &mut UiState, prof: &mut Profiler) {
    conservation(ui, st);
    ui.separator();
    ui.checkbox(&mut prof.enabled, "profiler (submit+wait per segment)");
    if prof.enabled {
        egui::Grid::new("prof").striped(true).show(ui, |ui| {
            for s in prof.stats() {
                ui.label(s.name);
                if s.name == "world" && !st.paused {
                    // O "world" inclui todos os passos do frame.
                    let n = st.steps_per_frame.max(1);
                    ui.label(format!("{:.3} ms  ({n} steps, {:.3} ms/step)", s.avg_ms, s.avg_ms / n as f64));
                } else {
                    ui.label(format!("{:.3} ms", s.avg_ms));
                }
                ui.end_row();
            }
        });
    }
}

fn conservation(ui: &mut egui::Ui, st: &UiState) {
    ui.strong("Conservation of matter");
    let Some(l) = st.ledger else {
        ui.label("waiting for the first reading…");
        return;
    };
    let base = st.baseline.total() as i64;
    let now = l.total() as i64;
    let d = now - base;
    let pct = if base > 0 { d as f64 / base as f64 * 100.0 } else { 0.0 };
    let color = if d == 0 { egui::Color32::LIGHT_GREEN } else { egui::Color32::LIGHT_RED };
    ui.colored_label(color, format!("total {now}  Δ {d:+} ({pct:+.4}%)"));
    ui.label(format!("free {}  held in agents {}", l.free_total(), l.held_total()));
    ui.label(format!("(reading from epoch ~{}, asynchronous)", st.ledger_epoch));
    egui::Grid::new("ledger").striped(true).show(ui, |ui| {
        ui.label("");
        ui.label("act.");
        ui.label("spent");
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
