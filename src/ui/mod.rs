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
    (Tab::Vista, "Vista"),
    (Tab::Materia, "Matéria"),
    (Tab::Luz, "Luz"),
    (Tab::Agua, "Água e fumarolas"),
    (Tab::Terreno, "Terreno"),
    (Tab::Vida, "Vida"),
    (Tab::Movimento, "Corpo"),
    (Tab::Cena, "Cena"),
    (Tab::Graficos, "Gráficos"),
    (Tab::Info, "Info"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainAction {
    /// Escolhe um PNG (janela) e troca o terreno do mundo vivo (os agentes
    /// e os monómeros ficam).
    Load,
    /// Mundo vazio (só água, sem fumarolas), semeado de novo: para pintar.
    Empty,
    /// Pista de corridas: terreno do circuito, parâmetros do ensaio e recomeço.
    Track,
    /// Escolhe onde gravar (janela) e grava o terreno atual em PNG.
    Save,
    /// Volta ao terreno gerado e semeia de novo.
    Generated,
}

/// Materiais do pincel (o índice é `UiState::paint_material`).
pub const PAINT_MATERIALS: [&str; 7] = [
    "água (apaga terreno)",
    "entulho fino (1 grão)",
    "entulho denso (2 grãos)",
    "rocha",
    "fumarola: calor",
    "fumarola: química (redutor)",
    "apagar fumarolas",
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
        }
    }
}

pub const VIEW_NAMES: [&str; 11] = [
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
    "redutor das fumarolas",
];
const CH: [&str; 4] = ["A", "U", "G", "C"];

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
        inspector::panel(ui, ins, &world.organ_table, &world.amino);
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
        if ui.button(if st.paused { "▶ continuar" } else { "⏸ pausa" }).clicked() {
            st.paused = !st.paused;
        }
        if ui.button("nova semente").clicked() {
            st.reseed = true;
        }
        if ui
            .button("recomeçar")
            .on_hover_text("começa de novo no mesmo sítio: sopa nova, sem agentes e epoch a zero, mas com o terreno tal como está agora e todos os parâmetros que tens. (\"nova semente\" gera também um terreno novo, se não for de imagem, e não mexe na epoch)")
            .clicked()
        {
            st.restart = true;
        }
    });
    if world.params.day_period >= 1.0 {
        let d = world.params.daylight(world.params.epoch);
        ui.label(if d > 0.0 { format!("☀ dia ({:.0}% do sol)", d * 100.0) } else { "☾ noite".to_string() });
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
            "agentes vivos {n}   (+{:.0}/s nascimentos, −{:.0}/s mortes)",
            st.stats.births_per_sec, st.stats.deaths_per_sec
        ),
        None => "agentes vivos …".into(),
    });
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut st.steps_per_frame, 1..=MAX_STEPS_PER_FRAME).text("passos/frame"));
        ui.checkbox(&mut st.vsync, "vsync");
        ui.checkbox(&mut st.smooth_refresh, "ecrã fluido")
            .on_hover_text("os passos por frame passam a ser um máximo: se a placa não os faz todos a tempo, fazem-se menos por frame para o ecrã não ficar preso. O ecrã refresca a ~30 imagens por segundo quando o desenho é barato; quando é caro (muitos agentes à vista) refresca mais devagar, até 15 por segundo, para a simulação ficar com até 80% do tempo da placa");
        if st.smooth_refresh && st.steps_done < st.steps_per_frame {
            ui.label(format!("a fazer {}", st.steps_done));
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
    ui.strong("Espécies e linhagens");
    ui.horizontal(|ui| {
        ui.label("censo das linhagens de");
        ui.add(egui::DragValue::new(&mut st.lineage_every).range(5_000..=5_000_000).speed(1000));
        ui.label("em epochs");
    })
    .response
    .on_hover_text("de tempos a tempos agrupam-se os genomas vivos em espécies e liga-se cada uma àquela de que descende: é o registo da árvore da vida desta corrida. Fica guardado com a cena e recomeça com um mundo novo. Cada censo lê os agentes todos da placa (uma pausa de centésimos de segundo)");
    if !st.lineage_info.is_empty() {
        ui.small(&st.lineage_info);
    }
    if ui
        .button("gerar relatório (página HTML)")
        .on_hover_text("uma página com as espécies e as suas duas formas (retratos, órgãos, onde vivem), quem pode atacar quem, ataques e ligações a decorrer, e as árvores: a das linhagens registadas e a do parentesco entre as espécies vivas. Fica em saves/relatorios/ e abre no browser. Demora uns segundos")
        .clicked()
    {
        st.report_now = true;
    }
    if ui
        .button("abrir a árvore das linhagens")
        .on_hover_text("a árvore da vida desta corrida numa página interativa (zoom, arrastar, clicar num ramo): o desenho das duas formas de cada ramo, os órgãos e a população ao longo do tempo. Demora uns segundos: os retratos são desenhados pela simulação num mundo à parte")
        .clicked()
    {
        st.tree_now = true;
    }
    ui.horizontal(|ui| {
        ui.label("captura do mundo inteiro:");
        for (name, side) in [("8k", 8192u32), ("16k", 16384)] {
            if ui
                .button(name)
                .on_hover_text("o mundo todo numa imagem PNG (8192 ou 16384 píxeis de lado), com a vista e o brilho escolhidos. Fica em saves/capturas/. A simulação pára uns segundos enquanto desenha; o ficheiro acaba de se gravar em segundo plano (o de 16k ocupa centenas de MB)")
                .clicked()
            {
                st.big_shot = side;
            }
        }
    });
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
    ui.add(egui::Slider::new(&mut st.coc_radius, 0.0..=1.0).text("círculo de confusão dos monómeros (células)"))
        .on_hover_text("de perto, cada monómero desenha-se como um disco suave deste raio, numa posição própria dentro da célula (só para o desenho: a simulação conta monómeros por célula). Pequeno = moléculas soltas; grande = névoa contínua; 0 = quadrados, uma cor por célula");
    const SIGNAL_VIEWS: [&str; 6] = [
        "química",
        "sinal α",
        "sinal β",
        "α (vermelho) + β (verde)",
        "parentesco com o selecionado",
        "γ (vermelho) + δ (verde)",
    ];
    egui::ComboBox::from_label("cor dos agentes").selected_text(SIGNAL_VIEWS[st.signal_view as usize]).show_ui(
        ui,
        |ui| {
            for (i, n) in SIGNAL_VIEWS.iter().enumerate() {
                ui.selectable_value(&mut st.signal_view, i as u32, *n);
            }
        },
    );
    if st.signal_view == 4 {
        ui.label("clica num organismo: bola verde = genoma próximo, amarela = meio, vermelha = distante (8-meros partilhados; o filho conta como parente)");
    }
    let names = crate::life::organs::ORGAN_NAMES;
    let current = match st.mark_organ {
        0 => "nenhum",
        MARK_BONDED => "ligados por âncora",
        m => names[(m as usize - 1).min(names.len() - 1)],
    };
    egui::ComboBox::from_label("marcar quem tem o órgão").selected_text(current).show_ui(ui, |ui| {
        ui.selectable_value(&mut st.mark_organ, 0, "nenhum");
        ui.selectable_value(&mut st.mark_organ, MARK_BONDED, "ligados por âncora");
        for (i, n) in names.iter().enumerate() {
            ui.selectable_value(&mut st.mark_organ, i as u32 + 1, *n);
        }
    });
    if st.mark_organ == MARK_BONDED {
        ui.small("bola dourada = agente com uma ligação viva a outro. De perto, a ligação é um fio: azul-claro = de nascimento (pai e filho), dourado = feita ao tocar");
    } else if st.mark_organ != 0 {
        ui.small("bola ciano = agente com este órgão (do mesmo tamanho no ecrã a qualquer zoom)");
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

fn tab_matter(ui: &mut egui::Ui, st: &mut UiState, world: &mut World) {
    let p = &mut world.params;
    ui.strong("Sopa inicial e ativação");
    ui.add(egui::Slider::new(&mut world.seed_density, 0.05..=1.0).text("densidade inicial (na próxima semente)"));
    ui.add(egui::Slider::new(&mut world.seed_active, 0.0..=1.0).text("fração ativada inicial (na próxima semente)"))
        .on_hover_text("fração dos monómeros que nascem ativados quando se semeia um mundo novo (0,5 = metade)");
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut st.activate_frac, 0.0..=1.0).text("ativar já"));
        if ui.button("ativar").on_hover_text("ativa agora esta fração dos monómeros gastos livres (a matéria não muda)").clicked() {
            st.activate_now = true;
        }
    });
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
    ui.strong("Ativação abiótica (sem vida)");
    ui.add(egui::Slider::new(&mut p.direct_photoactivation, 0.0..=1.0).logarithmic(true).smallest_positive(0.001).text("pelo sol (fotoativação dos gastos)"))
        .on_hover_text("a luz reativa gastos sozinha (segue o dia e a noite e as sombras). 1 = ~1,8% dos gastos por passo em sol pleno; 0,02 = um pingo. 0 = só os fotossistemas");
    ui.add(egui::Slider::new(&mut p.thermal_activation, 0.0..=2.0).logarithmic(true).smallest_positive(0.01).text("pelo calor (acima de T = 2)"))
        .on_hover_text("o calor reativa gastos sozinho, só na água acima de T = 2 (fumarolas). 0 = só os quimiossintéticos aproveitam as fumarolas");
    ui.separator();
    ui.strong("Transporte dos monómeros");
    ui.add(egui::Slider::new(&mut p.diffusion, 0.0..=50.0).text("difusão ×"));
    ui.add(egui::Slider::new(&mut p.transport_every, 1..=4).text("transporte de N em N passos"))
        .on_hover_text("o transporte dos monómeros (corrente, difusão, agregação, reações) é a parte mais cara de cada passo. Com 2, corre passo sim, passo não, com o dobro do deslocamento de cada vez: a simulação fica ~15% mais rápida e os monómeros andam em saltos maiores e menos frequentes. Os agentes comem em todos os passos. A difusão máxima possível baixa na mesma proporção. 1 = como sempre");
    ui.add(egui::Slider::new(&mut p.monomer_pressure, 0.0..=20.0).text("pressão dos monómeros"))
        .on_hover_text("a difusão empurra das zonas cheias para as vazias");
    ui.add(egui::Slider::new(&mut p.cohesion, 0.0..=2.0).text("coesão dos ativados do mesmo tipo"))
        .on_hover_text("um ativado sai menos de uma célula quando os vizinhos têm ativados do mesmo tipo: junta cada tipo em manchas");
    ui.add(
        egui::Slider::new(&mut p.aggregation, 0.0..=1.0).logarithmic(true).smallest_positive(0.01).text("agregação dos ativados"),
    )
        .on_hover_text("energia de ligação entre ativados vizinhos (÷ temperatura): formam grumos que a corrente leva inteiros; o calor dissolve-os");
    ui.separator();
    ui.strong("Gravidade");
    ui.add(egui::Slider::new(&mut p.settle, 0.0..=100.0).logarithmic(true).smallest_positive(0.1).text("gravidade dos MONÓMEROS ×"))
        .on_hover_text("probabilidade por passo de um monómero descer uma célula = 0,002 × isto (10 = 0,02 células/passo)");
    ui.add(egui::Slider::new(&mut p.sediment_settle, 0.0..=5.0).text("gravidade dos GRÃOS de entulho ×"))
        .on_hover_text("velocidade de queda (×0,5 células do fluido/s): um grão solto anda com a corrente menos a queda — sobe onde a corrente a subir é mais forte (suspensão) e assenta onde ela abranda. 0 = flutuam");
    ui.add(egui::Slider::new(&mut p.sedimentation, 0.0..=0.5).text("gravidade dos AGENTES × (∝ √n)"))
        .on_hover_text("os agentes afundam ∝ √(nº de resíduos): os grandes descem mais depressa. 0 = não afundam");
}

fn tab_light(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    ui.add(egui::Slider::new(&mut p.uv_strength, 0.0..=10.0).text("força UV (sol)"));
    ui.add(egui::Slider::new(&mut p.sun_heat, 0.0..=5.0).text("aquecimento solar"))
        .on_hover_text("o sol aquece a superfície (infravermelho absorvido pela água) e o que absorve luz (rocha, agentes, monómeros). Segue o dia e a noite");
    ui.add(egui::Slider::new(&mut p.day_period, 0.0..=200_000.0).text("dia e noite: período (epochs; 0 = sempre dia)"))
        .on_hover_text("durante o dia o sol sobe e desce como meio seno (amanhecer e anoitecer); no resto do ciclo é noite");
    if p.day_period >= 1.0 {
        ui.add(egui::Slider::new(&mut p.day_fraction, 0.05..=1.0).text("fração do ciclo que é dia"))
            .on_hover_text("0,5 = dia e noite iguais; 0,75 = dia de 3/4 do ciclo; 1 = sem noite (mas o sol ainda sobe e desce)");
        ui.add(egui::Slider::new(&mut p.sun_angle, 0.0..=85.0).text("sol: ângulo máximo ao nascer/pôr (graus)"))
            .on_hover_text("0 = sempre a pique; 85 = luz quase rasante de manhã e à tarde (sombras compridas)");
        let d = p.daylight(p.epoch);
        ui.label(format!("  agora: {} ({:.0}% do sol)", if d > 0.0 { "dia" } else { "noite" }, d * 100.0));
    }
    let light_changed =
        ui.add(egui::Slider::new(&mut p.uv_depth, 0.0..=30.0).text("atenuação UV pela água")).changed();
    ui.add(egui::Slider::new(&mut p.monomer_uv_absorb, 0.0..=5.0).text("absorção UV pelos monómeros"));
    ui.add(egui::Slider::new(&mut p.photo_yield, 0.0..=0.5).logarithmic(true).smallest_positive(0.005).text("rendimento fotossintético"))
        .on_hover_text("energia por unidade de luz absorvida por um fotossistema (o reciclador converte a mesma energia em ativados)");
    ui.add(egui::Slider::new(&mut p.uv_damage, 1.0..=50.0).text("dano UV"))
        .on_hover_text("risco de morte por passo à luz = mortalidade base × (isto − 1) × luz × 0,01, e cai com a fração de aminoácidos aromáticos do corpo (triptofano, tirosina: o protetor solar). 30 em sol pleno e sem proteção dá cerca de 140 passos de vida; 1 = sem dano");
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
    ui.strong("Fluido");
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
    ui.add(egui::Slider::new(&mut p.chemo_yield, 0.0..=5.0).text("rendimento da quimiossíntese"))
        .on_hover_text("energia por unidade de redutor consumido");
    ui.add(egui::Slider::new(&mut p.redox_decay, 0.0..=0.5).logarithmic(true).smallest_positive(0.001).text("oxidação do redutor (1/s)"))
        .on_hover_text("quanto mais lento, mais longe o redutor chega (vista 'redutor das fumarolas')");
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
    ui.small("PNG: AZUL = terreno (0 água, fraco entulho, forte rocha, 255 rocha maciça); VERMELHO = calor e VERDE = química (redutor) das fumarolas, por píxel e independentes (verde sem vermelho = exsudação fria). Sem verde na imagem, a química segue o calor. Cinzentos (r = g = b) só dão terreno.");
    ui.horizontal(|ui| {
        ui.label("ficheiro");
        ui.text_edit_singleline(&mut st.terrain_path);
    });
    ui.horizontal(|ui| {
        if ui
            .button("carregar…")
            .on_hover_text("escolhe um PNG; o terreno muda no mundo que está a correr (os agentes e os monómeros ficam; o que deixar de caber sai para o lado). Para começar do zero com ele, semeia de novo")
            .clicked()
        {
            st.terrain_action = Some(TerrainAction::Load);
        }
        if ui.button("gravar…").on_hover_text("grava o terreno atual (e o calor) num PNG").clicked() {
            st.terrain_action = Some(TerrainAction::Save);
        }
        if ui.button("terreno gerado").clicked() {
            st.terrain_action = Some(TerrainAction::Generated);
        }
        if ui.button("mundo vazio").on_hover_text("só água, sem fumarolas, semeado de novo: para pintar à mão").clicked() {
            st.terrain_action = Some(TerrainAction::Empty);
        }
    });
    if ui
        .button("pista de corridas (ensaio de natação)")
        .on_hover_text("troca o mundo por um circuito fechado com curvas para os dois lados e recomeça do zero. Não há monómeros, comida, luz do sol, fumarolas nem corrente (os genomas nascem e copiam-se sem matéria): a única energia é AVANÇAR na pista (no sentido contrário ao dos ponteiros do relógio); tocar nas paredes tira energia; as paredes dão luz, para os sensores de luz as verem. Os filhos são cópias iguais ao pai. Todos os parâmetros passam aos do ensaio (valores por omissão + os da pista), para os resultados serem comparáveis; os que tinhas voltam quando saíres com 'terreno gerado' ou 'mundo vazio'")
        .clicked()
    {
        st.terrain_action = Some(TerrainAction::Track);
    }
    ui.separator();
    ui.strong("Pincel");
    ui.checkbox(&mut st.paint_on, "pintar com o botão esquerdo (o direito continua a arrastar a vista)");
    egui::ComboBox::from_label("material")
        .selected_text(PAINT_MATERIALS[st.paint_material.min(PAINT_MATERIALS.len() - 1)])
        .show_ui(ui, |ui| {
            for (i, name) in PAINT_MATERIALS.iter().enumerate() {
                ui.selectable_value(&mut st.paint_material, i, *name);
            }
        });
    ui.add(egui::Slider::new(&mut st.paint_radius, 1.0..=200.0).logarithmic(true).text("raio (células)"));
    if st.paint_material == 4 || st.paint_material == 5 {
        ui.add(egui::Slider::new(&mut st.paint_strength, 0.05..=1.0).text("força da fumarola"));
    }
    ui.small("pinta-se também em pausa; os monómeros saem para o lado quando se põe rocha (a matéria conserva-se)");
    if !st.terrain_msg.is_empty() {
        ui.label(&st.terrain_msg);
    }
    ui.separator();
    ui.checkbox(&mut world.settings.terrain_enabled, "física do terreno ligada");
    ui.strong("Sedimentos (entulho solto)");
    ui.add(egui::Slider::new(&mut world.params.sediment_transport, 0.0..=5.0).text("arrasto pela corrente ×"))
        .on_hover_text("quanto a corrente leva o entulho solto (1 = o do v3)");
    ui.add(egui::Slider::new(&mut world.params.sediment_threshold, 0.0..=5.0).text("velocidade crítica de arranque"))
        .on_hover_text("critério de Shields: abaixo desta velocidade (células do fluido/s) a corrente não arranca grãos; acima, arranca ∝ ao excesso");
    ui.add(egui::Slider::new(&mut world.params.bioturbation, 0.0..=0.5).text("bioturbação (empurrar entulho)"));
    ui.add(egui::Slider::new(&mut world.params.bioturbation_cost, 0.0..=1.0).text("custo por grão empurrado"));
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
    ui.strong("Metabolismo");
    let p = &mut world.params;
    ui.add(egui::Slider::new(&mut p.food_power, 0.0..=20.0).text("energia por monómero"));
    ui.add(
        egui::Slider::new(&mut p.uptake_rate, 0.0..=0.01).logarithmic(true).smallest_positive(1e-5).text("taxa de hidrólise"),
    )
    .on_hover_text("quanto os agentes comem; a 0 ninguém come");
    let mut hunger = p.hunger_regulation != 0;
    ui.checkbox(&mut hunger, "regulação pela carga energética (cheio não come)");
    p.hunger_regulation = hunger as u32;
    ui.add(egui::Slider::new(&mut p.maintenance_cost, 0.0..=0.01).text("manutenção por resíduo"));
    ui.add(egui::Slider::new(&mut p.leak_base, 0.0..=1.0).text("fuga base (corpo sem boca) ×"))
        .on_hover_text("a manutenção é multiplicada por isto + a fuga das bocas. 0,1 = um corpo sem boca paga um décimo; 1 com a fuga por boca a 0 = como era antes");
    ui.add(egui::Slider::new(&mut p.mouth_leak, 0.0..=2.0).text("fuga por boca aberta ×"))
        .on_hover_text("o que deixa entrar também deixa sair: cada boca padrão aberta soma isto ao multiplicador da manutenção (bocas mais fortes somam mais; uma boca fechada não soma). Com 0,3, um corpo de três bocas paga o mesmo que antes");
    ui.add(egui::Slider::new(&mut p.skin_uptake, 0.0..=5.0).text("absorção sem boca ×"))
        .on_hover_text("quanto os resíduos sem boca absorvem (× a catálise do aminoácido; uma boca vale 20 a 80×). 0,2 = um corpo de 30 resíduos sem boca come cerca de um décimo de uma boca fraca; 0 = só as bocas comem");
    ui.add(egui::Slider::new(&mut p.metabolic_q10, 1.0..=4.0).text("metabolismo: Q10"))
        .on_hover_text("quanto a química da vida acelera por cada 'escala' de temperatura (1 = não depende da temperatura). Multiplica manutenção, comer, quimiossíntese e emparelhamento; a luz não");
    ui.add(egui::Slider::new(&mut p.metabolic_span, 0.5..=12.0).text("metabolismo: escala (unidades de T por Q10)"));
    ui.add(egui::Slider::new(&mut p.metabolic_ref, 0.0..=8.0).text("metabolismo: temperatura de referência (m = 1)"));
    ui.separator();
    ui.strong("Reprodução");
    ui.add(egui::Slider::new(&mut p.spawn_energy, 0.1..=50.0).text("energia inicial"));
    ui.add(egui::Slider::new(&mut p.pairing_rate, 0.0..=8.0).text("emparelhamento (bases/passo)"));
    ui.add(egui::Slider::new(&mut p.pairing_cost, 0.0..=2.0).text("custo por base copiada"));
    let mut salvage = p.salvage > 0.0;
    ui.checkbox(&mut salvage, "recarga: os produtores copiam-se com monómeros gastos")
        .on_hover_text("a energia que transborda de um fotossistema ou de uma quimiossíntese carrega primeiro um monómero gasto para a cópia do próprio genoma (construir com matéria-prima); só o que não servir para isso vai reativar monómeros no meio. Desligado = o transbordo vai todo para o meio (como era)");
    p.salvage = salvage as u32 as f32;
    ui.add(egui::Slider::new(&mut p.mutation_rate, 0.0..=0.05).text("taxa de mutação"));
    let mut aug = p.require_start != 0;
    ui.checkbox(&mut aug, "tradução começa no AUG (nascimentos novos)");
    p.require_start = aug as u32;
    ui.separator();
    ui.strong("Morte");
    ui.add(egui::Slider::new(&mut p.death_probability, 0.0..=0.2).text("mortalidade base"));
    ui.add(egui::Slider::new(&mut p.death_metab, 0.0..=1.0).text("mortalidade segue o ritmo de vida"))
        .on_hover_text("1 = a mortalidade base é multiplicada pelo ritmo do agente (metabolismo × fuga): dormência, bocas fechadas, corpos sem boca e água fria fazem viver mais (cistos, esporos). 0 = a mortalidade não depende do ritmo");
    ui.add(egui::Slider::new(&mut p.death_energy_cap, 0.0..=200.0).text("teto da proteção pela energia"))
        .on_hover_text("a mortalidade base é ÷ energia só até este valor (uma reserva protege, acumular mais não); 0 = sem teto (v3)");
    ui.add(egui::Slider::new(&mut p.denature_temp, 0.0..=12.0).text("temperatura de desnaturação"))
        .on_hover_text("acima disto o calor mata (vista 7 = temperatura; o miolo das fumarolas chega a 12)");
    ui.add(egui::Slider::new(&mut p.heat_kill, 0.0..=1.0).logarithmic(true).smallest_positive(0.001).text("desnaturação pelo calor"))
        .on_hover_text("risco de morrer na água quente (acima do limiar das fumarolas), × (1 − termoestabilidade do corpo; coluna da tabela dos aminoácidos)");
    ui.separator();
    ui.strong("Predação");
    ui.add(egui::Slider::new(&mut p.protease_power, 0.0..=30.0).logarithmic(true).smallest_positive(0.1).text("força das proteases ×"))
        .on_hover_text("multiplica a energia que as proteases tiram à vítima por passo de contacto (base: 0,2 × força × intensidade para uma vítima com 10% de resíduos-alvo). Quando a energia da vítima chega a zero, morre. Cada família corta certos aminoácidos e a prolina defende. 0 = sem predação");
    ui.add(egui::Slider::new(&mut p.protease_direct, 0.0..=1.0).text("fração direta para o predador"))
        .on_hover_text("parte da energia tirada que entra diretamente no atacante. O resto vai para o meio (ver o slider seguinte). 0 = o predador tem de comer os restos; 1 = aspira tudo");
    ui.add(egui::Slider::new(&mut p.lysis_yield, 0.0..=1.0).text("rendimento dos restos"))
        .on_hover_text("da energia que não vai direta para o predador, a fração que fica no meio como monómeros ativados junto da vítima (um por cada 'energia por monómero'); o resto perde-se");
    ui.separator();
    ui.strong("Ligações entre agentes (órgão âncora: + liga a −)");
    ui.add(egui::Slider::new(&mut p.bond_rate, 0.0..=1.0).logarithmic(true).smallest_positive(1e-3).text("formação"))
        .on_hover_text("probabilidade por passo de um agente com uma âncora livre a tentar ligar a uma âncora oposta de um vizinho; a duração vem da variante da âncora (editor)");
    ui.add(egui::Slider::new(&mut p.bond_energy_share, 0.0..=0.5).logarithmic(true).smallest_positive(0.001).text("difusão da energia pela ligação"))
        .on_hover_text("a energia corre pela ligação do agente mais cheio (energia ÷ capacidade) para o mais vazio, até ficarem com o mesmo enchimento. É a fração da diferença que passa por passo: 0,1 = a diferença cai para metade em ~7 passos; 0,01 = em ~70 (o valor antigo)");
    ui.add(egui::Slider::new(&mut p.bond_matter_share, 0.0..=1.0).text("partilha de matéria pela ligação"))
        .on_hover_text("probabilidade por passo de um agente ligado receber do parceiro um complemento que este já capturou, do que tem a cópia do genoma mais adiantada para o que a tem mais atrasada (uma folha a alimentar a raiz). Só passa quando a base serve ao que recebe, uma vez em quatro em média. 0 = não partilham matéria");
    ui.add(egui::Slider::new(&mut p.bond_signal, 0.0..=1.0).text("sinais pela ligação"));
}

fn tab_motion(ui: &mut egui::Ui, world: &mut World) {
    let p = &mut world.params;
    ui.strong("Sinais internos");
    ui.add(egui::Slider::new(&mut p.signal_mode, 0.0..=4.0).step_by(1.0).text("modo dos sinais"))
        .on_hover_text("como os sinais α/β andam pela cadeia e dobram as juntas. 0: condução e sensibilidade de cada aminoácido (v3). 1: difusão igual para os dois lados e todas as juntas respondem igual (α dobra para um lado, β para o outro). 2: o sinal só anda do lado N para o C, mesma resposta. 3: anda do N para o C e cada junta responde conforme o seu aminoácido (o corpo decide para que lado vira). 4: transporte da tabela (condução de cada aminoácido e órgão, como no 0) mas todas as juntas respondem igual");
    ui.add(egui::Slider::new(&mut p.signal_crosstalk, 0.0..=0.5).text("resíduo nos outros canais"))
        .on_hover_text("um órgão que emite num canal deixa escapar esta fração da emissão para cada um dos outros três (especificidade imperfeita): um sensor de α também põe um pouco em β, γ e δ. Os relés não têm resíduo (servem para separar canais). 0 = emissão limpa");
    if p.track_mode != 0 {
        ui.strong("Pista de corridas");
        ui.add(egui::Slider::new(&mut p.track_gain, 0.0..=1.0).logarithmic(true).text("energia por unidade avançada"))
            .on_hover_text("energia que um agente ganha por cada unidade do mundo de avanço LÍQUIDO na pista (só rende passar do ponto mais avançado a que já chegou). Quem recua mais de 150 unidades paga o excesso ao mesmo preço. Um agente de 20 resíduos gasta ~0,04 por passo só a manter-se");
        ui.add(egui::Slider::new(&mut p.wall_damage, 0.0..=1.0).logarithmic(true).text("dano das paredes"))
            .on_hover_text("energia perdida por passo com o centro do agente em cima de uma parede (proporcional a quanto o seu raio entra nela)");
        let mut same = p.copy_same != 0;
        if ui.checkbox(&mut same, "filhos iguais ao pai").on_hover_text("o filho é uma cópia do genoma do pai em vez do complemento reverso (uma só forma por linhagem)").changed() {
            p.copy_same = same as u32;
        }
    }
    ui.add(egui::Slider::new(&mut p.clock_mute, 0.0..=1.0).text("silenciar relógios"))
        .on_hover_text("experiência: tira amplitude a todos os relógios (1 = mudos). O órgão continua no corpo e continua a pagar o custo; serve para ver se os agentes se mexem sem ele (sensores, emissão por contacto)");
    ui.small(match p.signal_mode.round() as i32 {
        0 => "  0 = por aminoácido (cada junta responde à sua maneira)",
        1 => "  1 = isotrópico (difusão para os dois lados, resposta igual)",
        2 => "  2 = direcional (do lado N para o C, resposta igual)",
        3 => "  3 = direcional, resposta de cada aminoácido (o corpo decide)",
        _ => "  4 = transporte da tabela (aminoácidos e órgãos), resposta igual",
    });
    ui.separator();
    ui.strong("Natação");
    let mut rft = p.rft_enabled != 0;
    ui.checkbox(&mut rft, "natação (RFT)");
    p.rft_enabled = rft as u32;
    ui.add(egui::Slider::new(&mut p.swim_grip, 1.0..=30.0).logarithmic(true).text("aderência da natação (2 = água)"))
        .on_hover_text("quanto mais o meio trava um segmento a andar de lado do que ao comprido. 2 = água, o limite físico de um corpo fino. Mais = um meio que agarra de lado (gel, muco): cada batida rende mais avanço, NA HORA; quando o corpo pára de bater, pára. 5 dá cerca de 6 vezes o avanço da água num nadador ondulante");
    ui.add(egui::Slider::new(&mut p.swim_gain, 0.0..=50.0).text("ganho da natação (com memória; 1 = desligado)"))
        .on_hover_text("multiplica a MÉDIA do avanço dos últimos passos (ver a memória, a seguir): acima de 1 os agentes continuam a deslizar na direção antiga depois de pararem ou virarem (não é físico). 1 = só a física das batidas");
    ui.add(egui::Slider::new(&mut p.swim_memory, 1.0..=200.0).logarithmic(true).text("memória do ganho (passos)"))
        .on_hover_text("quantos passos dura o deslizar quando o ganho é maior que 1. 20 = como no v3 (a velocidade perdia 5% por passo); 100 = o que o v4 tinha até agora. Com o ganho a 1 não faz nada");
    ui.add(egui::Slider::new(&mut p.swim_wobble, 0.0..=1.0).text("vaivém da natação"))
        .on_hover_text("1 = balanço físico de cada batida; 0 = só o avanço médio");
    ui.add(egui::Slider::new(&mut p.motion_cost, 0.0..=2.0).logarithmic(true).smallest_positive(0.001).text("custo do movimento"))
        .on_hover_text("energia gasta a mexer o corpo: a dissipação na água (isto × Σ √arrasto·dθ² das juntas: bater depressa custa ao quadrado) e o dobrar das juntas pelos sinais, que segue o mesmo valor. 0,02 = nadar custa cerca de um terço da manutenção de um corpo sem boca; 0,1 = o valor antigo (nadar custava mais do que estar vivo)");
    ui.add(egui::Slider::new(&mut p.inertia, 0.0..=10.0).text("inércia dos pesados (não física)"))
        .on_hover_text("0 = físico: à escala molecular a água amortece tudo, um corpo que pára de bater pára logo. Acima de 0 a velocidade aproxima-se da pedida com peso 1/(1 + isto × massa/massa de um corpo médio): os pesados deslizam e aceleram devagar, como nadadores grandes");
    ui.add(egui::Slider::new(&mut p.flow_coupling, 0.0..=1.0).text("arrasto pela corrente"))
        .on_hover_text("1 = físico (um corpo livre segue a água); menos = experiência: as correntes levam-nos menos e eles também empurram menos a água");
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
    ui.add(egui::Slider::new(&mut p.joint_load, 0.0..=5.0).text("carga das juntas (arrasto a rodar)"))
        .on_hover_text("quanto a água trava a dobra de cada junta. Cada junta roda os dois lados do corpo em sentidos opostos e o que a trava é o arrasto do lado que roda mais facilmente (comprimento dos segmentos × arrasto dos órgãos × distância²). As pontas dobram depressa, o tronco de um corpo comprido dobra devagar e um órgão volumoso numa ponta torna essa ponta lenta. 0 = todas as juntas ao mesmo ritmo; 1 = a junta do meio de um corpo médio dobra a metade");
    ui.add(egui::Slider::new(&mut p.rest_angle_mult, 0.0..=6.0).text("× ângulos de repouso"))
        .on_hover_text("multiplica o ângulo de repouso de todas as juntas (aminoácidos e órgãos). 1 = os da tabela (corpos quase direitos); 3–4 dá dobras de 50–90° como numa proteína real: corpos enrolados, com mais contactos entre resíduos");
    ui.add(egui::Slider::new(&mut p.thermal_kt, 0.0..=5.0).text("agitação térmica (kT)"));
    ui.add(egui::Slider::new(&mut p.motor_amplitude, 0.0..=1.0).text("curso do motor (rad)"));
    ui.add(egui::Slider::new(&mut p.joint_coupling, 0.0..=0.95).text("acoplamento entre juntas"));
    ui.add(egui::Slider::new(&mut p.brownian, 0.0..=20.0).text("movimento browniano"));
    ui.add(egui::Slider::new(&mut p.brownian_rot, 0.0..=5.0).text("rotação browniana"))
        .on_hover_text("a agitação térmica também roda os corpos ao acaso: 0,15 rad por passo ÷ raio^1,5 (o raio conta-se em resíduos, √n), vezes isto. Um RNA nu roda muito; um corpo de 16 resíduos, ~0,02 rad por passo. 1 = o valor de sempre; 0 = só rodam por nadar, pela água ou por contacto");
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
