//! Arranque e ciclo de eventos. A lógica vive na biblioteca (`src/lib.rs`).

use std::sync::Arc;

use ribossome::gpu::Gpu;
use ribossome::gpu::profiler::Profiler;
use ribossome::params::WorldConfig;
use ribossome::render::{Camera, WorldView};
use ribossome::ui::{self, UiState};
use ribossome::world::World;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    world: World,
    view: WorldView,
    /// MICROSCÓPIO 3D (só existe enquanto está ligado: ocupa mais de 1 GB
    /// na placa) e o seu painel.
    scope: Option<ribossome::microscope::Scope>,
    scope_panel: bool,
    /// Quanto da transição mapa → microscópio já se fez (0 = só o mapa, 1 =
    /// só o microscópio), pelo zoom; e se o botão direito está a rodar a câmara.
    micro_t: f32,
    /// Zoom ainda por aplicar (logaritmo do fator) e o ponto do ecrã que fica
    /// fixo: a roda do rato aplica-se aos poucos, sem saltos.
    zoom_pending: f32,
    zoom_anchor: [f32; 2],
    orbiting: bool,
    /// Alvo com várias amostras onde o mundo é desenhado (do tamanho da
    /// janela; refeito quando ela muda). Resolve para a imagem da janela.
    msaa: Option<wgpu::Texture>,
    /// Profundidade do desenho do mundo (do tamanho de `msaa`).
    depth: Option<wgpu::Texture>,
    /// ECRÃ DE ENTRADA: a imagem (carregada no primeiro frame) e quando
    /// começou. Some ao fim de SPLASH_SECS ou com um clique ou tecla.
    splash: Option<(Option<egui::TextureHandle>, std::time::Instant)>,
    /// Genoma carregado de um ficheiro (bases 0..3 = A, U, G, C).
    loaded_agent: Option<Vec<u8>>,
    /// MODO FOTO/VÍDEO: alvo de captura (refeito se o tamanho mudar), pasta
    /// e contagem das imagens da gravação em curso.
    shot_cap: Option<ribossome::render::capture::Capture>,
    /// Gravação em curso: canal para a thread que alimenta o ffmpeg (fechar
    /// o canal termina o ficheiro), o ficheiro e o lado da imagem.
    rec_tx: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
    rec_path: std::path::PathBuf,
    rec_side: u32,
    /// A gravação em curso é do microscópio (e o tamanho das imagens dela).
    rec_micro: bool,
    rec_dims: [u32; 2],
    rec_frames: u32,
    rec_tick: u32,
    cam: Camera,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    profiler: Profiler,
    ui: UiState,
    /// Posição do rato em píxeis, RELATIVA ao canto do viewport da simulação.
    cursor: [f32; 2],
    /// Onde a simulação é desenhada na janela: x, y, largura, altura
    /// (píxeis). É todo o espaço livre entre as barras da interface.
    viewport: [f32; 4],
    /// Os gráficos tapam a simulação (não se desenha).
    covered: bool,
    dragging: bool,
    /// Botão esquerdo premido com o pincel ligado.
    painting: bool,
    /// Onde o botão esquerdo foi premido e se o rato já se mexeu (clique vs arrastar).
    press_pos: [f32; 2],
    press_moved: bool,
    inspector: ui::inspector::Inspector,
    seed: u64,
    seed_rng: ribossome::life::SplitMix,
    runlog: ribossome::runlog::RunLog,
    /// Editor local da tabela dos aminoácidos (http://127.0.0.1:8787).
    editor: Option<ribossome::editor::Editor>,
    /// Servidor MCP local (http://127.0.0.1:8788/mcp).
    mcp: Option<ribossome::mcp::Mcp>,
    /// Gravação de cena em curso (a thread que comprime e escreve).
    save_job: Option<std::thread::JoinHandle<Result<String, String>>>,
    /// Epoch do último autosave.
    last_autosave: u32,
    /// Mapa genético: id do agente de referência e frames até recalcular.
    kin_id: Option<u32>,
    kin_frames: u32,
    /// Refresco fluido: instante do frame anterior e passos por frame que a
    /// placa está a aguentar (ver `adaptive_steps`).
    /// Registo das linhagens (árvore da vida da corrida); guardado com a cena.
    lineages: ribossome::lineage::Lineages,
    /// Um preset pediu população: semeia-se no frame a seguir ao mundo novo.
    pending_seed: bool,
    /// Preset pedido por RIBO_PRESET, a lançar no primeiro frame.
    startup_preset: Option<String>,
    last_frame: std::time::Instant,
    steps_eff: f32,
    /// Estimativa de frame = o + n·s (ver adaptive_steps): médias de n, f,
    /// n², n·f e o peso acumulado; passos do frame anterior; alvo atual.
    fit: [f32; 5],
    last_n: f32,
    target_ms: f32,
    dither: u32,
}

#[derive(Default)]
struct App {
    run: Option<Running>,
}

/// Modo laboratório (RIBO_LAB=1): piscina 1024², sem fluido nem terreno,
/// monómeros ativados por igual e reativação uniforme.
fn lab_mode() -> bool {
    std::env::var("RIBO_LAB").map(|v| v != "0").unwrap_or(false)
}

/// CENÁRIO DE TESTE (o mesmo que o exemplo probe_jitter), para ver na janela
/// o que se mede: RIBO_SWIMMERS=n nadadores construídos (relógio + 15
/// glicinas; RIBO_CONTROL=1 = sem relógio), sem morte, fome nem
/// reprodução, energia 60. RIBO_FSO=1 = natação só pelo fluido; RIBO_PUSH
/// = agentes empurram a água. Com RIBO_TERRAIN=plano fica sem terreno.
fn test_scenario(world: &mut World) {
    let env_f = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    if let Some(v) = env_f("RIBO_FSO") {
        world.params.fluid_swim_only = (v != 0.0) as u32;
    }
    if let Some(v) = env_f("RIBO_PUSH") {
        world.params.agent_fluid_push = v;
    }
    let Some(n) = env_f("RIBO_SWIMMERS") else { return };
    world.params.death_probability = 0.0;
    world.params.maintenance_cost = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.sedimentation = 0.0;
    world.params.spawn_energy = 60.0;
    let control = env_f("RIBO_CONTROL").unwrap_or(0.0) != 0.0;
    let text = if control {
        format!("AUG {} UAA", "GGU ".repeat(16))
    } else {
        format!("AUG CAU CUU {} UAA", "GGU ".repeat(15))
    };
    let g: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    let s = world.cfg.sim_size();
    let mut rng = ribossome::life::SplitMix(5);
    let reqs: Vec<ribossome::params::SpawnRequest> = (0..n as u32)
        .map(|_| ribossome::params::SpawnRequest::with_genome(s * (0.2 + 0.6 * rng.f32()), s * (0.3 + 0.5 * rng.f32()), &g))
        .collect();
    world.request_seeds(&reqs);
    log::info!("test scenario: {} swimmers{}", n, if control { " (control, no clock)" } else { "" });
}

/// Terreno carregado por omissão no mundo completo (azul = terreno, vermelho = calor).
const DEFAULT_TERRAIN: &str = "assets/terreno.png";

/// Pasta das cenas gravadas e do autosave.
/// Duração do ecrã de entrada (segundos).
/// Passos de simulação que duram os restos de um agente morto (as peças a
/// separarem-se e a irem na corrente).
const GHOST_STEPS: f32 = 60.0;
const SPLASH_SECS: f32 = 5.0;
const SAVES_DIR: &str = "saves";

/// O autosave deste modo (o laboratório tem outro tamanho de mundo).
fn autosave_file() -> &'static str {
    // Cada tamanho de mundo tem o seu (uma cena de um tamanho não carrega
    // noutro, e não pode ser gravada por cima do autosave do tamanho normal).
    static FILE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let grid = world_config_from_env().grid_size;
        if lab_mode() {
            "saves/autosave_lab.ribo".into()
        } else if grid != WorldConfig::DEFAULT.grid_size {
            format!("saves/autosave_{grid}.ribo")
        } else {
            "saves/autosave.ribo".into()
        }
    })
}

/// Terreno de arranque. De uma imagem: RIBO_TERRAIN=caminho.png; por omissão
/// (mundo completo) o terreno do projeto, assets/terreno.png; "-" = gerado.
fn startup_terrain(world: &mut World) {
    let terrain = std::env::var("RIBO_TERRAIN")
        .ok()
        .or_else(|| (!lab_mode() && std::path::Path::new(DEFAULT_TERRAIN).exists()).then(|| DEFAULT_TERRAIN.into()))
        .filter(|p| p != "-");
    if terrain.as_deref() == Some("plano") {
        // Sem terreno nenhum (testes).
        let n = world.cfg.cells() as usize;
        world.custom_terrain = Some((vec![0; n], vec![0.0; n]));
    } else if let Some(path) = terrain {
        match world.load_terrain_png(std::path::Path::new(&path)) {
            Ok(nf) => log::info!("terrain from {path} ({nf} hot cells)"),
            Err(e) => log::error!("RIBO_TERRAIN: {e}; using the generated terrain"),
        }
    }
}

/// Monómeros ativados por canal e célula na piscina do modo laboratório.
const LAB_PER_CHANNEL: f32 = 1.5;

/// Onde fica o tamanho do mundo escolhido na interface (células de lado).
const WORLD_SIZE_FILE: &str = "saves/world_size.txt";

fn world_config_from_env() -> WorldConfig {
    let mut cfg = WorldConfig::DEFAULT;
    if lab_mode() {
        cfg.grid_size = 1024;
        cfg.fluid_size = 512;
        cfg.max_agents = 100_000;
    }
    // TAMANHO DO MUNDO: RIBO_GRID manda; sem ele, o último tamanho escolhido
    // na interface (guardado em WORLD_SIZE_FILE), para o programa reabrir no
    // tamanho em que se fechou.
    let chosen = std::env::var("RIBO_GRID").ok().or_else(|| if lab_mode() { None } else { std::fs::read_to_string(WORLD_SIZE_FILE).ok() });
    if let Some(g) = chosen.and_then(|v| v.trim().parse::<u32>().ok()).filter(|g| g.is_power_of_two() && (64..=4096).contains(g) && *g != WorldConfig::DEFAULT.grid_size) {
        cfg.grid_size = g;
        cfg.fluid_size = (g / 2).max(16);
        // A capacidade de agentes escala com a área (400 000 a 2048²).
        cfg.max_agents = ((400_000u64 * g as u64 * g as u64) / (2048 * 2048)).max(1024) as u32;
    }
    if let Some(f) = std::env::var("RIBO_FLUID").ok().and_then(|v| v.parse::<u32>().ok()) {
        cfg.fluid_size = f;
    }
    cfg
}

impl Running {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title({
                            let g = world_config_from_env().grid_size;
                            if g == WorldConfig::DEFAULT.grid_size { "Ribossome".to_string() } else { format!("Ribossome ({g} × {g})") }
                        })
                        .with_window_icon({
                            // Ícone da janela: o mesmo do executável (assets/icon.png).
                            let dec = png::Decoder::new(std::io::Cursor::new(&include_bytes!("../assets/icon.png")[..]));
                            dec.read_info().ok().and_then(|mut r| {
                                let mut buf = vec![0; r.output_buffer_size()];
                                let info = r.next_frame(&mut buf).ok()?;
                                buf.truncate(info.buffer_size());
                                winit::window::Icon::from_rgba(buf, info.width, info.height).ok()
                            })
                        })
                        .with_inner_size(winit::dpi::LogicalSize::new(1400, 900)),
                )
                .expect("create window"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let gpu = pollster::block_on(Gpu::new(instance, Some(&surface))).expect("GPU");

        let caps = surface.get_capabilities(&gpu.adapter);
        // O egui quer um formato sem sRGB; o nosso shader também escreve em espaço de ecrã.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&gpu.device, &surface_cfg);

        let cfg = world_config_from_env();
        log::info!("world {}² cells, {} units", cfg.grid_size, cfg.sim_size());
        let seed = 1;
        let mut world = World::new(&gpu, cfg, seed as u32);
        startup_terrain(&mut world);
        let mut baseline = if lab_mode() {
            world.configure_lab();
            world.seed_lab(&gpu, seed, LAB_PER_CHANNEL)
        } else {
            world.seed_matter(&gpu, seed)
        };
        test_scenario(&mut world);
        // Autosave: desligado nos cenários de teste; RIBO_RESUME=0 arranca um
        // mundo novo (mas continua a gravar por cima do autosave).
        let testing = std::env::var("RIBO_SWIMMERS").is_ok();
        let resume = std::env::var("RIBO_RESUME").map(|v| v != "0").unwrap_or(true);
        let autosave_path = (!testing).then(|| autosave_file().to_string());
        let mut scene_msg = String::new();
        let mut resumed = None;
        let mut resumed_stats: Option<Vec<u8>> = None;
        let mut resumed_lineages: Option<ribossome::lineage::Lineages> = None;
        if let Some(path) = autosave_path.as_deref().filter(|p| resume && std::path::Path::new(p).exists()) {
            let t = std::time::Instant::now();
            let scene = ribossome::world::Scene::read(std::path::Path::new(path));
            let stats_bytes = scene.as_ref().ok().and_then(|s| s.extra_block("estatisticas").map(|b| b.to_vec()));
            let lineage_bytes = scene.as_ref().ok().and_then(|s| s.extra_block("linhagens").map(|b| b.to_vec()));
            match scene.and_then(|s| world.load_scene(&gpu, &s)) {
                Ok((extra, notes)) => {
                    resumed_stats = stats_bytes;
                    resumed_lineages = lineage_bytes.and_then(|b| ribossome::lineage::Lineages::from_bytes(&b));
                    scene_msg = format!("resumed from {path} (epoch {})", world.params.epoch);
                    log::info!("{scene_msg} in {:.1} s", t.elapsed().as_secs_f32());
                    for n in notes {
                        log::warn!("autosave: {n}");
                    }
                    let changed = world.params.changed_from_default();
                    if !changed.is_empty() {
                        let list: Vec<String> = changed.iter().map(|(k, a, b)| format!("{k} {a} (code {b})")).collect();
                        log::warn!("autosave: parameters differing from the code: {}", list.join(", "));
                    }
                    if let Some(b) = ribossome::world::ledger_from_json(&extra["baseline"]) {
                        baseline = b;
                    }
                    resumed = Some(extra);
                }
                Err(e) => {
                    scene_msg = format!("could not resume {path}: {e}; new world");
                    log::error!("{scene_msg}");
                }
            }
        }
        let view = WorldView::new(&gpu.device, &gpu.queue, &world, format);
        let mut cam = Camera::fit(&cfg, [surface_cfg.width as f32, surface_cfg.height as f32]);
        // (RIBO_ZOOM = píxeis por unidade do mundo e RIBO_CENTER = x,y: câmara
        // de arranque, para testes.)
        if let Some(z) = std::env::var("RIBO_ZOOM").ok().and_then(|v| v.parse::<f32>().ok()) {
            cam.zoom = z;
        }
        if let Some((x, y)) = std::env::var("RIBO_CENTER").ok().and_then(|v| v.split_once(',').and_then(|(x, y)| Some((x.trim().parse::<f32>().ok()?, y.trim().parse::<f32>().ok()?)))) {
            cam.center = [x, y];
        }

        // O MICROSCÓPIO fica já pronto (pipelines e texturas): criado só ao
        // aproximar, a transição dava um soluço na primeira vez.
        let mut scope = Some(ribossome::microscope::Scope::new(&gpu, &world, format, [64, 64], cam.center, 420.0));
        // (Sem rotação: ao aproximar, a câmara só se inclina. Roda-se à mão.)
        if let Some(s) = scope.as_mut() {
            s.orbit.yaw = 0.0;
        }
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let mut egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
        let info = gpu.adapter.get_info();
        let runlog = ribossome::runlog::RunLog::from_env(format!(
            "mode {}  {:?}  GPU {} ({:?}, driver {})  build {}",
            if lab_mode() { "lab" } else { "full world" },
            world.cfg,
            info.name,
            info.backend,
            info.driver_info,
            if cfg!(debug_assertions) { "debug" } else { "release" }
        ));
        let editor =
            ribossome::editor::Editor::start(world.amino.clone(), world.organ_table.clone(), world.organ_code.clone());
        let mcp = ribossome::mcp::Mcp::start();
        let mut inspector = ui::inspector::Inspector::new(&gpu, &world);
        inspector.register(&gpu.device, &mut egui_renderer);

        let last_autosave = world.params.epoch;
        let mut r = Self {
            window,
            surface,
            surface_cfg,
            gpu,
            world,
            view,
            cam,
            egui_state,
            msaa: None,
            depth: None,
            loaded_agent: None,
            // RIBO_NO_SPLASH=1 salta o ecrã de entrada.
            splash: std::env::var("RIBO_NO_SPLASH").is_err().then(|| (None, std::time::Instant::now())),
            shot_cap: None,
            rec_tx: None,
            rec_path: std::path::PathBuf::new(),
            rec_side: 0,
            rec_micro: false,
            rec_dims: [0; 2],
            rec_frames: 0,
            rec_tick: 0,
            egui_renderer,
            profiler: Profiler::from_env(),
            ui: UiState::new(baseline),
            cursor: [0.0; 2],
            viewport: [0.0; 4],
            scope,
            scope_panel: true,
            micro_t: 0.0,
            zoom_pending: 0.0,
            zoom_anchor: [0.0; 2],
            orbiting: false,
            covered: false,
            dragging: false,
            painting: false,
            press_pos: [0.0; 2],
            press_moved: false,
            inspector,
            seed,
            seed_rng: ribossome::life::SplitMix(seed ^ 0x5EED),
            runlog,
            editor,
            mcp,
            save_job: None,
            last_autosave,
            kin_id: None,
            kin_frames: 0,
            lineages: resumed_lineages.unwrap_or_default(),
            pending_seed: false,
            startup_preset: std::env::var("RIBO_PRESET").ok().filter(|v| !v.is_empty()),
            last_frame: std::time::Instant::now(),
            steps_eff: 1.0,
            fit: [0.0; 5],
            last_n: 1.0,
            target_ms: 33.0,
            dither: 0,
        };
        r.ui.autosave_path = autosave_path;
        r.ui.scene_msg = scene_msg;
        r.ui.scene_name = ribossome::names::new_scene_name(&r.world, r.seed);
        if let Some(extra) = resumed {
            r.apply_interface(&extra);
            // Mundo retomado do autosave: o nome sai só da semente dele, para
            // ser o mesmo de cada vez que o programa arranca.
            r.ui.scene_name = r.loaded_scene_name(std::path::Path::new(""));
            if let Some(b) = resumed_stats {
                r.ui.history = ribossome::stats::History::from_saved(&extra["estatisticas"], &b);
            }
        }
        r
    }

    /// Estado da interface que vai com a cena (vista, câmara, base da matéria).
    fn interface_json(&self) -> serde_json::Value {
        serde_json::json!({
            "baseline": ribossome::world::ledger_json(&self.ui.baseline),
            "seed": self.seed,
            "steps_per_frame": self.ui.steps_per_frame,
            "view_mode": self.ui.view_mode,
            "monomer_brightness": self.ui.monomer_brightness,
            "coc_radius": self.ui.coc_radius,
            "signal_view": self.ui.signal_view,
            "seed_count": self.ui.seed_count,
            "seed_len": self.ui.seed_len,
            "camera": { "center": self.cam.center, "zoom": self.cam.zoom },
            "autosave_every": self.ui.autosave_every,
            "estatisticas": self.ui.history.to_json(),
        })
    }

    fn apply_interface(&mut self, v: &serde_json::Value) {
        let u = |k: &str| v[k].as_u64();
        if let Some(b) = ribossome::world::ledger_from_json(&v["baseline"]) {
            self.ui.baseline = b;
        }
        if let Some(s) = u("seed") {
            self.seed = s;
        }
        if let Some(n) = u("steps_per_frame") {
            self.ui.steps_per_frame = (n as u32).clamp(1, ribossome::world::MAX_STEPS_PER_FRAME);
        }
        if let Some(n) = u("view_mode") {
            self.ui.view_mode = (n as u32).min(10);
        }
        if let Some(x) = v["coc_radius"].as_f64() {
            self.ui.coc_radius = x as f32;
        }
        if let Some(x) = v["monomer_brightness"].as_f64() {
            self.ui.monomer_brightness = x as f32;
        }
        if let Some(n) = u("signal_view") {
            self.ui.signal_view = (n as u32).min(5);
        }
        if let Some(n) = u("seed_count") {
            self.ui.seed_count = n as u32;
        }
        if let Some(a) = v["seed_len"].as_array().filter(|a| a.len() == 2) {
            self.ui.seed_len = [a[0].as_u64().unwrap_or(12) as u32, a[1].as_u64().unwrap_or(120) as u32];
        }
        if let (Some(c), Some(z)) = (v["camera"]["center"].as_array(), v["camera"]["zoom"].as_f64())
            && c.len() == 2
        {
            self.cam.center = [c[0].as_f64().unwrap_or(0.0) as f32, c[1].as_f64().unwrap_or(0.0) as f32];
            self.cam.zoom = z as f32;
        }
        if let Some(n) = u("autosave_every") {
            self.ui.autosave_every = (n as u32).max(1000);
        }

    }

    /// Nome de um mundo lido de um ficheiro: o do ficheiro, se parecer um
    /// nome de cena ("Abyssus_lucidus_417_e52000.ribo"); senão um tirado só
    /// da semente do mundo (o mesmo de cada vez que se abre).
    fn loaded_scene_name(&self, path: &std::path::Path) -> String {
        path.file_stem()
            .and_then(|s| ribossome::names::name_from_stem(&s.to_string_lossy()))
            .unwrap_or_else(|| ribossome::names::scene_name_with(self.seed, &ribossome::names::world_moods(&self.world)))
    }

    /// Começa a gravar uma cena (espera pela gravação anterior, se houver).
    fn start_save(&mut self, path: std::path::PathBuf, keep_previous: bool) {
        self.finish_save(true);
        let t = std::time::Instant::now();
        let extra = self.interface_json();
        let blocks = vec![("estatisticas", self.ui.history.to_bytes()), ("linhagens", self.lineages.to_bytes())];
        self.save_job = Some(self.world.save_scene(&self.gpu, path, extra, blocks, keep_previous));
        log::info!("scene: state read from the GPU in {:.2} s (epoch {})", t.elapsed().as_secs_f32(), self.world.params.epoch);
        self.ui.scene_msg = "saving…".into();
    }

    /// Recolhe o resultado da gravação em curso (`wait`: espera por ela).
    fn finish_save(&mut self, wait: bool) {
        let Some(job) = self.save_job.take_if(|j| wait || j.is_finished()) else { return };
        self.ui.scene_msg = match job.join() {
            Ok(Ok(m)) => {
                log::info!("scene saved: {m}");
                format!("saved {m}")
            }
            Ok(Err(e)) => {
                log::error!("scene: {e}");
                format!("error saving: {e}")
            }
            Err(_) => "error saving (the thread failed)".into(),
        };
    }

    /// Ações do separador "Cena" e o autosave periódico.
    fn scene_tick(&mut self) {
        use ribossome::ui::SceneAction;
        self.finish_save(false);
        let epoch = self.world.params.epoch;
        if epoch < self.last_autosave {
            self.last_autosave = epoch;
        }
        match self.ui.scene_action.take() {
            Some(SceneAction::Save) => {
                let dir = std::path::Path::new(SAVES_DIR);
                let _ = std::fs::create_dir_all(dir);
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Ribossome scene", &["ribo"])
                    .set_directory(dir.canonicalize().unwrap_or_default())
                    .set_file_name(format!("{}_e{epoch}.ribo", self.ui.scene_name))
                    .set_title("Save scene")
                    .save_file()
                {
                    self.start_save(path, false);
                }
            }
            Some(SceneAction::Load) => {
                let dir = std::path::Path::new(SAVES_DIR).canonicalize().unwrap_or_default();
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Ribossome scene", &["ribo"])
                    .set_directory(dir)
                    .set_title("Load scene")
                    .pick_file()
                {
                    self.finish_save(true);
                    let scene = ribossome::world::Scene::read(&path);
                    let stats_bytes = scene.as_ref().ok().and_then(|s| s.extra_block("estatisticas").map(|b| b.to_vec()));
                    let lineage_bytes = scene.as_ref().ok().and_then(|s| s.extra_block("linhagens").map(|b| b.to_vec()));
                    match scene.and_then(|s| self.world.load_scene(&self.gpu, &s)) {
                        Ok((extra, notes)) => {
                            self.apply_interface(&extra);
                            self.ui.scene_name = self.loaded_scene_name(&path);
                            // O registo das linhagens é o da cena (ou vazio, se ela não o tiver).
                            self.lineages = lineage_bytes.and_then(|b| ribossome::lineage::Lineages::from_bytes(&b)).unwrap_or_default();
                            self.ui.history = ribossome::stats::History::from_saved(
                                &extra["estatisticas"],
                                stats_bytes.as_deref().unwrap_or_default(),
                            );
                            self.ui.ledger = None;
                            self.last_autosave = self.world.params.epoch;
                            self.ui.scene_msg = format!("loaded {} (epoch {})", path.display(), self.world.params.epoch);
                            log::info!("{}", self.ui.scene_msg);
                            for n in &notes {
                                log::warn!("scene: {n}");
                            }
                            if !notes.is_empty() {
                                self.ui.scene_msg += &format!("\n{}", notes.join("\n"));
                            }
                        }
                        Err(e) => {
                            self.ui.scene_msg = format!("error: {e}");
                            log::error!("scene: {e}");
                        }
                    }
                }
            }
            Some(SceneAction::AutosaveNow) => {
                if let Some(p) = self.ui.autosave_path.clone() {
                    self.start_save(p.into(), true);
                    self.last_autosave = epoch;
                }
            }
            Some(SceneAction::NewWorld) => {
                self.world.reset_settings();
                self.ui.history = ribossome::stats::History::default();
            self.lineages = ribossome::lineage::Lineages::default();
                startup_terrain(&mut self.world);
                if lab_mode() {
                    self.world.configure_lab();
                }
                self.seed = 0;
                self.ui.reseed = true;
                self.last_autosave = 0;
                self.ui.scene_msg = "new world with the default values".into();
                log::info!("{}", self.ui.scene_msg);
            }
            None => {}
        }
        if self.ui.autosave_on
            && self.save_job.is_none()
            && let Some(p) = self.ui.autosave_path.clone()
            && epoch.wrapping_sub(self.last_autosave) >= self.ui.autosave_every.max(1000)
        {
            self.start_save(p.into(), true);
            self.last_autosave = epoch;
        }
    }

    /// Pincel: enquanto o botão esquerdo estiver premido, pinta à volta do rato.
    fn paint_tick(&mut self) {
        if !(self.painting && self.ui.paint_on) {
            return;
        }
        let w = self.cam.screen_to_world(self.cursor, self.screen());
        let wpc = self.world.cfg.world_units_per_cell as f32;
        let (cx, cy, r) = (w[0] / wpc, w[1] / wpc, self.ui.paint_radius.max(0.5));
        let s = self.ui.paint_strength;
        match self.ui.paint_material {
            4 => self.world.paint_source(cx, cy, r, Some(s), None),
            5 => self.world.paint_source(cx, cy, r, None, Some(s)),
            6 => self.world.paint_source(cx, cy, r, Some(0.0), Some(0.0)),
            m => {
                // 0 água, 1 e 2 entulho, 3 rocha maciça.
                let grains = [0, 1, 2, 6][m.min(3)];
                let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                self.world.encode_paint(&self.gpu.queue, &mut enc, cx, cy, r, grains);
                self.gpu.queue.submit([enc.finish()]);
            }
        }
    }

    /// Atende os pedidos do servidor MCP (entre frames, na thread principal).
    fn mcp_tick(&mut self) {
        let calls: Vec<ribossome::mcp::Call> = match &self.mcp {
            Some(m) => m.rx.try_iter().collect(),
            None => return,
        };
        for c in calls {
            let out = self.mcp_call(&c.tool, &c.args);
            let _ = c.reply.send(out);
        }
    }

    fn mcp_call(&mut self, tool: &str, args: &serde_json::Value) -> Result<Vec<serde_json::Value>, String> {
        use ribossome::mcp::{json_text, png, text};
        use serde_json::json;
        let named = |p: &ribossome::params::SimParams, k: &str| p.to_named().into_iter().find(|(n, _)| *n == k).map_or(0.0, |(_, v)| v);
        // Os parâmetros são f32/u32: mostra-os como tal (0.6, não 0.6000000238).
        let num = |v: f64| if v.fract() == 0.0 && v.abs() < 1e15 { json!(v as i64) } else { json!(format!("{}", v as f32).parse::<f64>().unwrap_or(v)) };
        match tool {
            "get_params" => {
                let p = &self.world.params;
                let d = ribossome::params::SimParams::default().to_named();
                let cur: serde_json::Map<String, serde_json::Value> = p
                    .to_named()
                    .into_iter()
                    .zip(d)
                    .filter(|((n, _), _)| !n.starts_with('_'))
                    .map(|((n, v), (_, dv))| (n.to_string(), json!({ "atual": num(v), "omissao": num(dv) })))
                    .collect();
                let changed: Vec<serde_json::Value> = p
                    .changed_from_default()
                    .into_iter()
                    .map(|(n, v, dv)| json!({ "nome": n, "atual": num(v), "omissao": num(dv) }))
                    .collect();
                let st = &self.world.settings;
                Ok(vec![json_text(&json!({
                    "mundo": {
                        "fluid_enabled": st.fluid_enabled,
                        "terrain_enabled": st.terrain_enabled,
                        "contact_enabled": st.contact_enabled,
                        "fumarole_gain": num(self.world.fumarole_gain as f64),
                        "terreno_carregado": self.world.custom_terrain.is_some(),
                    },
                    "epoch": p.epoch,
                    "pausa": self.ui.paused,
                    "passos_por_frame": self.ui.steps_per_frame,
                    "sol_agora": p.daylight(p.epoch),
                    "mudados": changed,
                    "parametros": cur,
                }))])
            }
            "set_params" => {
                let map = args["params"].as_object().ok_or("falta 'params' (objeto nome -> valor)")?;
                // Valida tudo antes de mudar alguma coisa.
                let mut probe = self.world.params;
                for (k, v) in map {
                    let x = v.as_f64().ok_or(format!("{k}: o valor tem de ser um número"))?;
                    if k.starts_with('_') || !probe.set_named(k, x) {
                        return Err(format!("parâmetro desconhecido: {k}"));
                    }
                }
                let mut lines = Vec::new();
                for (k, v) in map {
                    let old = named(&self.world.params, k);
                    self.world.params.set_named(k, v.as_f64().unwrap_or(0.0));
                    let new = named(&self.world.params, k);
                    log::info!("mcp: {k} {} -> {}", num(old), num(new));
                    lines.push(format!("{k}: {} -> {}", num(old), num(new)));
                }
                // A atenuação muda a luz em todo o mundo: recalcula-a já.
                if map.contains_key("uv_depth") {
                    self.world.invalidate_light();
                }
                Ok(vec![text(lines.join("\n"))])
            }
            "set_world" => {
                let mut lines = Vec::new();
                let st = &mut self.world.settings;
                for (k, slot) in [
                    ("fluid_enabled", &mut st.fluid_enabled),
                    ("terrain_enabled", &mut st.terrain_enabled),
                    ("contact_enabled", &mut st.contact_enabled),
                ] {
                    if let Some(v) = args[k].as_bool() {
                        lines.push(format!("{k}: {} -> {v}", *slot));
                        *slot = v;
                    }
                }
                if let Some(v) = args["fumarole_gain"].as_f64() {
                    lines.push(format!("fumarole_gain: {} -> {}", self.world.fumarole_gain, v as f32));
                    self.world.fumarole_gain = (v as f32).max(0.0);
                }
                for l in &lines {
                    log::info!("mcp: {l}");
                }
                Ok(vec![text(if lines.is_empty() { "nada mudado".into() } else { lines.join("\n") })])
            }
            "get_stats" => {
                let h = &self.ui.history;
                let n = args["last"].as_u64().unwrap_or(10).clamp(1, 1000) as usize;
                let want: Vec<String> = args["series"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if let Some(bad) = want.iter().find(|s| !h.names.contains(s)) {
                    return Err(format!("série desconhecida: {bad}; existem: {}", h.names.join(", ")));
                }
                let rows: Vec<serde_json::Value> = h
                    .last_rows(n)
                    .into_iter()
                    .map(|(e, v)| {
                        let mut m = serde_json::Map::new();
                        m.insert("epoch".into(), json!(e));
                        for (name, x) in h.names.iter().zip(v) {
                            if want.is_empty() || want.contains(name) {
                                m.insert(name.clone(), json!((*x as f64 * 1000.0).round() / 1000.0));
                            }
                        }
                        serde_json::Value::Object(m)
                    })
                    .collect();
                Ok(vec![json_text(&json!({ "amostra_a_cada_epochs": h.every, "amostras": rows }))])
            }
            "habitat" => {
                let block = args["block"].as_u64().unwrap_or(128) as usize;
                Ok(vec![json_text(&ribossome::mcp::habitat(&self.gpu, &self.world, block))])
            }
            "species" => {
                let thr = args["threshold"].as_f64().unwrap_or(0.15).clamp(0.0, 1.0) as f32;
                Ok(vec![json_text(&ribossome::mcp::species(&self.gpu, &self.world, thr))])
            }
            "screenshot" => {
                // Múltiplo de 64 (ver Capture::new), para a câmara bater certo.
                let size = (args["size"].as_u64().unwrap_or(768).clamp(64, 2048) as u32).div_ceil(64) * 64;
                let view = args["view"].as_u64().unwrap_or(0) as u32;
                let bright = args["brightness"].as_f64().unwrap_or(0.5) as f32;
                let s = self.world.cfg.sim_size();
                let cam = if args["camera"].as_bool().unwrap_or(false) {
                    let [_, h] = self.screen();
                    Camera { center: self.cam.center, zoom: self.cam.zoom * size as f32 / h.max(1.0) }
                } else if let Some(span) = args["span"].as_f64().filter(|v| *v > 0.0) {
                    // Enquadramento pedido: centro em frações do mundo e largura em células.
                    let fx = args["x"].as_f64().unwrap_or(0.5) as f32;
                    let fy = args["y"].as_f64().unwrap_or(0.5) as f32;
                    let units = span as f32 * self.world.cfg.world_units_per_cell as f32;
                    Camera { center: [fx * s, fy * s], zoom: size as f32 / units }
                } else {
                    Camera { center: [0.5 * s, 0.5 * s], zoom: size as f32 / s }
                };
                let cap = ribossome::render::capture::Capture::new(&self.gpu, &self.world, size);
                // Marcar um órgão (tipo 0..14), como na vista "marcar quem tem o órgão".
                cap.view.mark_organ.set(args["mark_organ"].as_u64().map_or(0, |t| t as u32 + 1));
                cap.view.coc_radius.set(args["coc"].as_f64().map_or(self.ui.coc_radius, |v| v as f32));
                let rgba = cap.render(&self.gpu, &self.world, &cam, view, bright);
                let bytes = cap.encode_png(&rgba).map_err(|e| format!("png: {e}"))?;
                Ok(vec![png(&bytes), text(format!("vista {view}, epoch {}", self.world.params.epoch))])
            }
            "save_scene" => {
                let name = args["name"].as_str().ok_or("falta 'name'")?;
                if name.is_empty() || name.contains(['/', '\\', '.']) || name.starts_with("autosave") {
                    return Err("nome inválido (sem / \\ . e não pode começar por autosave)".into());
                }
                let _ = std::fs::create_dir_all(SAVES_DIR);
                let path = std::path::Path::new(SAVES_DIR).join(format!("{name}.ribo"));
                self.finish_save(true);
                self.start_save(path.clone(), false);
                Ok(vec![text(format!("a gravar {} (epoch {})", path.display(), self.world.params.epoch))])
            }
            "activate_spent" => {
                let f = args["fraction"].as_f64().ok_or("falta 'fraction' (0..1)")? as f32;
                let n = self.world.activate_spent(&self.gpu, f, self.world.params.epoch as u64);
                log::info!("mcp: activate now {f}: {n} monomers");
                Ok(vec![text(format!("{n} monómeros gastos ativados ({:.0}%)", f.clamp(0.0, 1.0) * 100.0))])
            }
            "paint" => {
                let f = |k: &str| args[k].as_f64().ok_or(format!("falta '{k}'"));
                let n = self.world.cfg.grid_size as f32;
                let (cx, cy, r) = (f("x")? as f32 * n, f("y")? as f32 * n, (f("radius")? as f32).clamp(0.5, n));
                let m = args["material"].as_u64().ok_or("falta 'material'")? as usize;
                let s = args["strength"].as_f64().unwrap_or(1.0) as f32;
                let name = ribossome::ui::PAINT_MATERIALS.get(m).ok_or("material desconhecido (0..6)")?;
                match m {
                    4 => self.world.paint_source(cx, cy, r, Some(s), None),
                    5 => self.world.paint_source(cx, cy, r, None, Some(s)),
                    6 => self.world.paint_source(cx, cy, r, Some(0.0), Some(0.0)),
                    _ => {
                        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                        self.world.encode_paint(&self.gpu.queue, &mut enc, cx, cy, r, [0, 1, 2, 6][m]);
                        self.gpu.queue.submit([enc.finish()]);
                    }
                }
                log::info!("mcp: paint {name} at ({cx:.0}, {cy:.0}) radius {r:.0}");
                Ok(vec![text(format!("pintado: {name}, centro ({cx:.0}, {cy:.0}) células, raio {r:.0}"))])
            }
            "pause" => {
                if let Some(p) = args["paused"].as_bool() {
                    self.ui.paused = p;
                }
                if let Some(n) = args["steps_per_frame"].as_u64() {
                    self.ui.steps_per_frame = (n as u32).clamp(1, ribossome::world::MAX_STEPS_PER_FRAME);
                }
                log::info!("mcp: pause {} steps/frame {}", self.ui.paused, self.ui.steps_per_frame);
                Ok(vec![text(format!("pausa: {}, passos por frame: {}", self.ui.paused, self.ui.steps_per_frame))])
            }
            _ => Err(format!("ferramenta desconhecida: {tool}")),
        }
    }

    /// Ao fechar: grava o autosave e espera que fique escrito.
    fn on_close(&mut self) {
        if self.ui.autosave_on
            && let Some(p) = self.ui.autosave_path.clone()
        {
            self.start_save(p.into(), true);
        }
        self.finish_save(true);
    }

    /// Tamanho, em píxeis, da área onde a simulação é desenhada (o viewport
    /// entre as barras): é o "ecrã" para a câmara.
    fn screen(&self) -> [f32; 2] {
        [self.viewport[2].max(1.0), self.viewport[3].max(1.0)]
    }

    fn reconfigure(&mut self) {
        self.surface_cfg.present_mode =
            if self.ui.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        self.surface.configure(&self.gpu.device, &self.surface_cfg);
    }

    /// Carregar / gravar / repor o terreno (botões do painel "Terreno").
    /// AGENTES GUARDADOS: grava o genoma do selecionado num ficheiro de
    /// texto (A, U, G, C), carrega um, ou espalha cópias do carregado.
    fn agent_action(&mut self, action: ribossome::ui::AgentAction) {
        use ribossome::ui::AgentAction;
        const LETTERS: [char; 4] = ['A', 'U', 'G', 'C'];
        let dir = std::path::Path::new(SAVES_DIR).join("agentes");
        let _ = std::fs::create_dir_all(&dir);
        let dialog = rfd::FileDialog::new().add_filter("genome", &["rna", "txt"]).set_directory(dir.canonicalize().unwrap_or(dir.clone()));
        match action {
            AgentAction::Save => {
                let Some(d) = self.inspector.data.as_ref() else {
                    self.ui.scene_msg = "save agent: none is selected".into();
                    return;
                };
                let name = format!("agente_{}_{}bases.rna", d.agent.id, d.genome.len());
                if let Some(path) = dialog.set_title("Save agent").set_file_name(&name).save_file() {
                    let text: String = d.genome.iter().map(|&b| LETTERS[(b & 3) as usize]).collect();
                    self.ui.scene_msg = match std::fs::write(&path, text + "\n") {
                        Ok(()) => format!("agent saved to {}", path.display()),
                        Err(e) => format!("save agent: {e}"),
                    };
                }
            }
            AgentAction::Load => {
                if let Some(path) = dialog.set_title("Load agent").pick_file() {
                    match std::fs::read_to_string(&path) {
                        Ok(text) => {
                            // Aceita T por U e ignora tudo o que não for base (espaços, linhas).
                            let g: Vec<u8> = text.chars().filter_map(|c| match c.to_ascii_uppercase() { 'A' => Some(0), 'U' | 'T' => Some(1), 'G' => Some(2), 'C' => Some(3), _ => None }).take(256).collect();
                            if g.len() < 3 {
                                self.ui.scene_msg = format!("{}: does not contain a genome (letters A, U, G, C)", path.display());
                            } else {
                                self.ui.agent_info = format!("{} ({} bases)", path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), g.len());
                                self.loaded_agent = Some(g);
                            }
                        }
                        Err(e) => self.ui.scene_msg = format!("load agent: {e}"),
                    }
                }
            }
            AgentAction::Spread => {
                if let Some(g) = &self.loaded_agent {
                    let s = self.world.cfg.sim_size();
                    let reqs: Vec<ribossome::params::SpawnRequest> =
                        (0..self.ui.agent_copies).map(|_| ribossome::params::SpawnRequest::with_genome(self.seed_rng.f32() * s, self.seed_rng.f32() * s, g)).collect();
                    self.world.request_seeds(&reqs);
                    self.ui.scene_msg = format!("{} copies requested (they are born where there are bases in the soup)", reqs.len());
                }
            }
        }
        log::info!("{}", self.ui.scene_msg);
    }

    /// MODO FOTO/VÍDEO: fotografa o enquadramento da mira (o quadrado ao
    /// centro da vista, 90% do lado menor). A gravar, cada imagem vai CRUA
    /// (sem PNG, sem ficheiros intermédios) para um ffmpeg que escreve logo o
    /// MP4: uma thread à parte alimenta-o, e se o codificador se atrasar a
    /// imagem perde-se em vez de travar a simulação.
    /// Foto e vídeo do MICROSCÓPIO (quando é ele que está à vista): a imagem
    /// do frame anterior, do tamanho da vista, lida da textura `shot`.
    fn photo_video_micro(&mut self) {
        if !self.ui.rec && self.rec_tx.take().is_some() {
            self.ui.rec_info = format!("video saved: {} ({} images)", self.rec_path.display(), self.rec_frames);
        }
        // Uma gravação começada no mapa tem outro tamanho: acaba aqui.
        if self.rec_tx.is_some() && !self.rec_micro {
            self.rec_tx = None;
            self.ui.rec = false;
            self.ui.rec_info = format!("view changed, video closed: {} ({} images)", self.rec_path.display(), self.rec_frames);
        }
        let Some(scope) = self.scope.as_mut() else { return };
        if !self.ui.photo_now && !self.ui.rec {
            scope.shot = None;
            scope.shot_ready = false;
            return;
        }
        let (sw, sh) = (self.surface_cfg.width, self.surface_cfg.height);
        if scope.shot.as_ref().is_none_or(|t| t.width() != sw || t.height() != sh) {
            scope.shot = Some(self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scope shot"),
                size: wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_cfg.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }));
            scope.shot_ready = false;
        }
        if !scope.shot_ready {
            // Ainda sem imagem: fica para o frame seguinte.
            return;
        }
        let vp = self.viewport;
        let (x, y) = (vp[0] as u32, vp[1] as u32);
        // (O x264 quer lados pares.)
        let (w, h) = (((vp[2] as u32).min(sw.saturating_sub(x))) & !1, ((vp[3] as u32).min(sh.saturating_sub(y))) & !1);
        if w < 16 || h < 16 {
            return;
        }
        let row = (w * 4).div_ceil(256) * 256;
        let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scope shot"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: scope.shot.as_ref().unwrap(), mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit([enc.finish()]);
        buf.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        self.gpu.wait_idle();
        let bgra = matches!(self.surface_cfg.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = buf.get_mapped_range(..).expect("mapped");
            for line in 0..h as usize {
                for px in data[line * row as usize..line * row as usize + (w * 4) as usize].chunks_exact(4) {
                    rgba.extend_from_slice(&if bgra { [px[2], px[1], px[0], 255] } else { [px[0], px[1], px[2], 255] });
                }
            }
        }
        buf.unmap();
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        if std::mem::take(&mut self.ui.photo_now) {
            let dir = std::path::Path::new(SAVES_DIR).join("capturas");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("microscopio_{}.png", self.world.params.epoch));
            self.ui.scene_msg = format!("photo saved to {}", path.display());
            let data = rgba.clone();
            std::thread::spawn(move || {
                let write = || -> Result<(), Box<dyn std::error::Error>> {
                    let mut e = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&path)?), w, h);
                    e.set_color(png::ColorType::Rgba);
                    e.set_depth(png::BitDepth::Eight);
                    e.write_header()?.write_image_data(&data)?;
                    Ok(())
                };
                if let Err(e) = write() {
                    log::error!("{}: {e}", path.display());
                }
            });
        }
        if !self.ui.rec {
            return;
        }
        if self.rec_tx.is_none() {
            let dir = std::path::Path::new(SAVES_DIR).join("videos");
            let _ = std::fs::create_dir_all(&dir);
            self.rec_path = dir.join(format!("microscopio_{stamp}_epoch{}.mp4", self.world.params.epoch));
            let child = std::process::Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
                .arg(format!("{w}x{h}"))
                .args(["-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"])
                .arg(&self.rec_path)
                .stdin(std::process::Stdio::piped())
                .spawn();
            match child {
                Ok(mut child) => {
                    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(8);
                    std::thread::spawn(move || {
                        use std::io::Write;
                        if let Some(mut pipe) = child.stdin.take() {
                            for img in rx {
                                if pipe.write_all(&img).is_err() {
                                    break;
                                }
                            }
                        }
                        let _ = child.wait();
                    });
                    self.rec_tx = Some(tx);
                    self.rec_micro = true;
                    self.rec_dims = [w, h];
                    self.rec_frames = 0;
                }
                Err(e) => {
                    self.ui.rec = false;
                    self.ui.rec_info = format!("could not start ffmpeg ({e}): it must be installed and on the PATH");
                    return;
                }
            }
        }
        if [w, h] != self.rec_dims {
            // A vista mudou de tamanho: a gravação acaba aqui.
            self.rec_tx = None;
            self.ui.rec = false;
            self.ui.rec_info = format!("view resized, video closed: {} ({} images)", self.rec_path.display(), self.rec_frames);
            return;
        }
        match self.rec_tx.as_ref().map(|tx| tx.try_send(rgba)) {
            Some(Ok(())) => {
                self.rec_frames += 1;
                self.ui.rec_info = format!("recording the microscope: {} images ({:.1} s of video)", self.rec_frames, self.rec_frames as f32 / 60.0);
            }
            Some(Err(std::sync::mpsc::TrySendError::Disconnected(_))) => {
                self.rec_tx = None;
                self.ui.rec = false;
                self.ui.rec_info = "ffmpeg stopped in the middle of the recording".into();
            }
            _ => {}
        }
    }

    fn photo_video(&mut self) {
        // (RIBO_AUTOSHOT=n: fotografa sozinho ao frame n; para testes.)
        static FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let frame = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if std::env::var("RIBO_AUTOSHOT").ok().and_then(|v| v.parse::<u32>().ok()) == Some(frame) {
            self.ui.photo_now = true;
        }
        if self.micro_t > 0.75 && self.scope.is_some() {
            self.photo_video_micro();
            return;
        }
        if let Some(scope) = self.scope.as_mut() {
            scope.shot = None;
            scope.shot_ready = false;
        }
        // Uma gravação começada no microscópio tem outro tamanho: acaba aqui.
        if self.rec_tx.is_some() && self.rec_micro {
            self.rec_tx = None;
            self.ui.rec = false;
            self.ui.rec_info = format!("view changed, video closed: {} ({} images)", self.rec_path.display(), self.rec_frames);
        }
        let photo = std::mem::take(&mut self.ui.photo_now);
        // Paragem: fechar o canal faz a thread fechar o ffmpeg, que termina o ficheiro.
        if !self.ui.rec && self.rec_tx.take().is_some() {
            self.ui.rec_info = format!("video saved: {} ({} images)", self.rec_path.display(), self.rec_frames);
            log::info!("{}", self.ui.rec_info);
        }
        let starting = self.ui.rec && self.rec_tx.is_none();
        let frame = self.ui.rec && {
            self.rec_tick += 1;
            starting || (self.rec_tick - 1) % self.ui.rec_every.max(1) == 0
        };
        if !photo && !frame {
            return;
        }
        // A gravar, o tamanho fica o do arranque (o ffmpeg precisa de imagens iguais).
        let size = if self.rec_tx.is_some() { self.rec_side } else { self.ui.shot_size.clamp(256, 4096).div_ceil(64) * 64 };
        if self.shot_cap.as_ref().is_none_or(|c| c.size() != size) {
            self.shot_cap = Some(ribossome::render::capture::Capture::new(&self.gpu, &self.world, size));
        }
        if starting {
            let dir = std::path::Path::new(SAVES_DIR).join("videos");
            let _ = std::fs::create_dir_all(&dir);
            let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            self.rec_path = dir.join(format!("video_{stamp}_epoch{}.mp4", self.world.params.epoch));
            let child = std::process::Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
                .arg(format!("{size}x{size}"))
                .args(["-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"])
                .arg(&self.rec_path)
                .stdin(std::process::Stdio::piped())
                .spawn();
            match child {
                Ok(mut child) => {
                    // Até 8 imagens em espera; mais do que isso, perdem-se.
                    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(8);
                    std::thread::spawn(move || {
                        use std::io::Write;
                        if let Some(mut pipe) = child.stdin.take() {
                            for rgba in rx {
                                if pipe.write_all(&rgba).is_err() {
                                    break;
                                }
                            }
                        }
                        let _ = child.wait();
                    });
                    self.rec_tx = Some(tx);
                    self.rec_side = size;
                    self.rec_micro = false;
                    self.rec_frames = 0;
                    self.rec_tick = 1;
                }
                Err(e) => {
                    self.ui.rec = false;
                    self.ui.rec_info = format!("could not start ffmpeg ({e}): it must be installed and on the PATH");
                    log::warn!("{}", self.ui.rec_info);
                    if !photo {
                        return;
                    }
                }
            }
        }
        let cap = self.shot_cap.as_ref().unwrap();
        let side = cap.size();
        // A imagem cobre o quadrado da mira: 90% do lado menor da vista.
        let guide = 0.9 * self.viewport[2].min(self.viewport[3]).max(1.0);
        let cam = Camera { center: self.cam.center, zoom: self.cam.zoom * side as f32 / guide };
        cap.view.coc_radius.set(self.ui.coc_radius);
        // (O relógio do tremor das moléculas, como na vista.)
        cap.view.epoch.set(self.world.params.epoch);
        let rgba = cap.render(&self.gpu, &self.world, &cam, self.ui.view_mode, self.ui.monomer_brightness);
        if photo {
            let dir = std::path::Path::new(SAVES_DIR).join("capturas");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("foto_{}.png", self.world.params.epoch));
            self.ui.scene_msg = format!("photo saved to {}", path.display());
            let rgb: Vec<u8> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            // O PNG comprime-se noutra thread, para a simulação não esperar.
            std::thread::spawn(move || {
                if let Err(e) = ribossome::render::capture::save_rgb_png(&rgb, side, &path) {
                    log::error!("{}: {e}", path.display());
                }
            });
        }
        if let Some(tx) = self.rec_tx.as_ref().filter(|_| frame) {
            match tx.try_send(rgba) {
                Ok(()) => {
                    self.rec_frames += 1;
                    self.ui.rec_info = format!("recording {}: {} images ({:.1} s of video)", self.rec_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), self.rec_frames, self.rec_frames as f32 / 30.0);
                }
                // Codificador atrasado: esta imagem perde-se.
                Err(std::sync::mpsc::TrySendError::Full(_)) => {}
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    self.rec_tx = None;
                    self.ui.rec = false;
                    self.ui.rec_info = "ffmpeg stopped in the middle of the recording".into();
                }
            }
        }
    }

    fn terrain_action(&mut self, action: ribossome::ui::TerrainAction) {
        use ribossome::ui::TerrainAction;
        // Janela de ficheiros do sistema, a começar no último caminho usado.
        let last = std::path::PathBuf::from(self.ui.terrain_path.trim());
        let dir = last
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .map(|d| d.to_path_buf())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_default();
        let name = last.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "terreno.png".into());
        let dialog = rfd::FileDialog::new().add_filter("PNG", &["png"]).set_directory(&dir);
        let chosen = match action {
            TerrainAction::Load => dialog.set_title("Load terrain").pick_file(),
            TerrainAction::Save => dialog.set_title("Save terrain").set_file_name(&name).save_file(),
            TerrainAction::Generated | TerrainAction::Empty | TerrainAction::Noise => Some(last.clone()),
        };
        let Some(path) = chosen else {
            self.ui.terrain_msg = "canceled".into();
            return;
        };
        self.ui.terrain_path = path.display().to_string();
        let resow = match action {
            TerrainAction::Load => match self.world.load_terrain_png_live(&self.gpu, &path) {
                Ok(nf) => {
                    self.ui.terrain_msg =
                        format!("loaded {} ({nf} hot cells); the world carries on (seed again to start from scratch)", path.display());
                    log::info!("{}", self.ui.terrain_msg);
                    false
                }
                Err(e) => {
                    self.ui.terrain_msg = format!("error: {e}");
                    false
                }
            },
            TerrainAction::Save => {
                self.ui.terrain_msg = match self.world.save_terrain_png(&self.gpu, &path) {
                    Ok(()) => format!("saved to {}", path.display()),
                    Err(e) => format!("error: {e}"),
                };
                false
            }
            TerrainAction::Generated => {
                self.world.use_generated_terrain();
                self.ui.terrain_msg = "generated terrain; world seeded again".into();
                true
            }
            TerrainAction::Empty => {
                self.world.use_empty_terrain();
                self.ui.terrain_msg = "empty world (water only); seeded again".into();
                true
            }
            TerrainAction::Noise => {
                self.world.use_noise_terrain(&self.ui.noise);
                self.ui.terrain_msg = format!("noise terrain (seed {}); world seeded again", self.ui.noise.seed);
                true
            }
        };
        if resow {
            // Mesma semente: só o terreno muda.
            self.seed -= 1;
            self.ui.reseed = true;
        }
    }

    /// PRESETS DE LANÇAMENTO (ver presets.rs).
    fn preset_action(&mut self, action: ribossome::ui::PresetAction) {
        use ribossome::presets::{self, Preset, PresetTerrain};
        use ribossome::ui::PresetAction;
        match action {
            PresetAction::Refresh => {
                self.ui.presets = presets::list();
                self.ui.preset_msg = format!("{} presets in {}", self.ui.presets.len(), presets::DIR);
            }
            PresetAction::Launch(i) => {
                let Some((_, p)) = self.ui.presets.get(i).cloned() else { return };
                // Tudo por omissão, depois o que o preset muda.
                self.world.reset_settings();
                let mut unknown = Vec::new();
                for (k, v) in &p.params {
                    if k.starts_with('_') || !self.world.params.set_named(k, *v) {
                        unknown.push(k.clone());
                    }
                }
                self.world.settings.fluid_enabled = p.fluid;
                self.world.fumarole_gain = p.fumarole_gain;
                let mut note = String::new();
                match &p.terrain {
                    PresetTerrain::Default => startup_terrain(&mut self.world),
                    PresetTerrain::Empty => self.world.use_empty_terrain(),
                    PresetTerrain::Image { path } => {
                        if let Err(e) = self.world.load_terrain_png(std::path::Path::new(path)) {
                            note = format!(" (terrain {path}: {e}; using the default terrain)");
                            startup_terrain(&mut self.world);
                        }
                    }
                    PresetTerrain::Noise(o) => {
                        self.ui.noise = *o;
                        self.world.use_noise_terrain(o);
                    }
                }
                if lab_mode() {
                    self.world.configure_lab();
                }
                // A população do preset é para o mundo de tamanho normal: num
                // mundo mais pequeno semeia-se na proporção da área.
                let (g, g0) = (self.world.cfg.grid_size as u64, WorldConfig::DEFAULT.grid_size as u64);
                let seeds = ((p.seeds as u64 * g * g) / (g0 * g0)).max(p.seeds.min(50) as u64) as u32;
                self.ui.seed_count = seeds.max(1);
                self.ui.seed_len = p.seed_len;
                if p.steps_per_frame > 0 {
                    self.ui.steps_per_frame = p.steps_per_frame.clamp(1, ribossome::world::MAX_STEPS_PER_FRAME);
                }
                self.pending_seed = p.seeds > 0;
                self.seed = 0;
                self.ui.reseed = true;
                self.last_autosave = 0;
                if !unknown.is_empty() {
                    note += &format!(" (unknown parameters ignored: {})", unknown.join(", "));
                }
                self.ui.preset_msg = format!("launched \"{}\" in the {1} × {1} world, {seeds} seeds{note}", p.name, self.world.cfg.grid_size);
                log::info!("preset: {}", self.ui.preset_msg);
            }
            PresetAction::SaveCurrent => {
                let name = self.ui.preset_name.trim().to_string();
                let stem = presets::file_stem(&name);
                // O terreno atual vai para uma imagem ao lado do preset.
                let png = format!("{}/{stem}.png", presets::DIR);
                let _ = std::fs::create_dir_all(presets::DIR);
                let terrain = match self.world.save_terrain_png(&self.gpu, std::path::Path::new(&png)) {
                    Ok(()) => PresetTerrain::Image { path: png },
                    Err(e) => {
                        log::warn!("preset: terrain not saved ({e}); the preset uses the default terrain");
                        PresetTerrain::Default
                    }
                };
                let preset = Preset {
                    name: name.clone(),
                    description: format!("saved at epoch {} from {}", self.world.params.epoch, self.ui.scene_name.replace('_', " ")),
                    fluid: self.world.settings.fluid_enabled,
                    terrain,
                    fumarole_gain: self.world.fumarole_gain,
                    params: self.world.params.changed_from_default().into_iter().map(|(k, v, _)| (k.to_string(), v)).collect(),
                    seeds: self.ui.seed_count,
                    seed_len: self.ui.seed_len,
                    steps_per_frame: self.ui.steps_per_frame,
                };
                self.ui.preset_msg = match presets::save(&preset) {
                    Ok(path) => format!("preset saved to {}", path.display()),
                    Err(e) => format!("save preset: {e}"),
                };
                self.ui.presets = presets::list();
            }
        }
    }

    /// REFRESCO FLUIDO: os passos por frame pedidos são um MÁXIMO. Se a placa
    /// não os faz todos a tempo, o ecrã ficava a 5–10 imagens por segundo (e
    /// a interface presa); em vez disso faz-se, em cada frame, só os passos
    /// que cabem no tempo-alvo do frame.
    ///
    /// O alvo NÃO é fixo: se o desenho for caro (muitos agentes à vista), um
    /// alvo de 33 ms deixava a simulação com os restos (com 14 ms de desenho
    /// ficava com metade do tempo). Estima-se, dos próprios frames, quanto
    /// custa o que não é simulação (o) e cada passo (s), por mínimos
    /// quadrados sobre frame = o + n·s, e escolhe-se o alvo para a simulação
    /// ficar com SIM_SHARE do tempo: alvo = o / (1 − SIM_SHARE), entre 33 ms
    /// (30 imagens/s) e 66 ms (15 imagens/s).
    fn adaptive_steps(&mut self) -> u32 {
        const TARGET_MS: f32 = 33.0;
        const MAX_TARGET_MS: f32 = 66.0;
        const SIM_SHARE: f32 = 0.8;
        // Memória da estimativa (~40 frames).
        const FORGET: f32 = 0.025;
        let now = std::time::Instant::now();
        let frame_ms = now.duration_since(self.last_frame).as_secs_f32() * 1000.0;
        self.last_frame = now;
        let want = self.ui.steps_per_frame.max(1) as f32;
        if !self.ui.smooth_refresh || self.ui.paused {
            self.steps_eff = want;
            self.fit = [0.0; 5];
        } else if frame_ms > 0.0 && frame_ms < 2000.0 {
            // Regressão com esquecimento: médias de n, f, n², n·f e o peso.
            let n = self.last_n;
            let f = &mut self.fit;
            let k = if f[4] < 1.0 { 1.0 / (f[4] * 40.0 + 1.0).max(1.0) } else { FORGET };
            f[0] += k * (n - f[0]);
            f[1] += k * (frame_ms - f[1]);
            f[2] += k * (n * n - f[2]);
            f[3] += k * (n * frame_ms - f[3]);
            f[4] = (f[4] + 0.025).min(1.0);
            let var = f[2] - f[0] * f[0];
            let cov = f[3] - f[0] * f[1];
            let mut target = TARGET_MS;
            let mut goal = self.steps_eff * TARGET_MS / frame_ms;
            if f[4] >= 1.0 && var > 0.05 && cov > 0.0 {
                let step_ms = (cov / var).max(0.02);
                let other_ms = (f[1] - step_ms * f[0]).max(0.0);
                target = (other_ms / (1.0 - SIM_SHARE)).clamp(TARGET_MS, MAX_TARGET_MS);
                goal = ((target - other_ms) / step_ms).max(1.0);
                // Nunca muito além do que o frame anterior mostrou caber.
                goal = goal.min(self.steps_eff * target / frame_ms * 1.5 + 1.0);
            }
            self.target_ms = target;
            // Aproxima-se aos poucos para não oscilar.
            self.steps_eff = (self.steps_eff + 0.25 * (goal - self.steps_eff)).clamp(1.0, want);
        }
        let mut n = (self.steps_eff.round() as u32).clamp(1, self.ui.steps_per_frame.max(1));
        // De vez em quando um frame com mais passos, para a regressão ter
        // dois pontos mesmo quando o número de passos está parado.
        self.dither = self.dither.wrapping_add(1);
        if self.ui.smooth_refresh && !self.ui.paused && self.dither % 12 == 0 {
            n = (n + (n / 4).max(1)).min(self.ui.steps_per_frame.max(1));
        }
        self.last_n = n as f32;
        self.ui.steps_done = n;
        n
    }

    /// Censo das linhagens, quando chega a altura (ver `lineage.rs`).
    fn lineage_tick(&mut self) {
        let epoch = self.world.params.epoch;
        self.lineages.every = self.ui.lineage_every.max(5_000);
        let l = &self.lineages;
        if l.censuses > 0 && epoch < l.last_epoch {
            // O epoch voltou atrás sem aviso: é outra corrida.
            self.lineages = ribossome::lineage::Lineages { every: l.every, ..Default::default() };
        }
        if self.ui.paused || epoch < self.lineages.next_epoch {
            return;
        }
        let genomes = ribossome::species::living_genomes(&self.gpu, &self.world);
        let species = ribossome::species::cluster(&genomes, 0.15);
        self.lineages.census(epoch, &species, genomes.len());
        let l = &self.lineages;
        self.ui.lineage_info = format!(
            "{} censuses, {} branches recorded ({} alive); last census at epoch {}",
            l.censuses,
            l.branches.len(),
            l.branches.iter().filter(|b| l.alive(b)).count(),
            l.last_epoch
        );
    }

    /// Gera a página do relatório em saves/relatorios/ e abre-a no browser.
    fn write_report(&mut self) {
        let t = std::time::Instant::now();
        let epoch = self.world.params.epoch;
        let html = ribossome::report::generate(&self.gpu, &self.world, Some(&self.lineages), &format!("Ribossome: {}, report at epoch {epoch}", self.ui.scene_name.replace('_', " ")));
        self.write_page(&format!("relatorio_{epoch}.html"), html, &format!("report ({:.1} s)", t.elapsed().as_secs_f32()));
    }

    /// Grava uma página em saves/relatorios/ e abre-a no browser.
    fn write_page(&mut self, name: &str, html: String, what: &str) {
        let dir = std::path::Path::new(SAVES_DIR).join("relatorios");
        let path = dir.join(name);
        let r = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, html));
        match r {
            Ok(()) => {
                self.ui.scene_msg = format!("{what} saved to {}", path.display());
                let full = path.canonicalize().unwrap_or(path.clone());
                #[cfg(windows)]
                let _ = std::process::Command::new("cmd").args(["/C", "start", "", &full.display().to_string()]).spawn();
                #[cfg(not(windows))]
                let _ = std::process::Command::new("xdg-open").arg(&full).spawn();
            }
            Err(e) => self.ui.scene_msg = format!("{what}: {e}"),
        }
        log::info!("{}", self.ui.scene_msg);
    }

    fn redraw(&mut self) {
        self.runlog.before_frame(&mut self.profiler);
        if let Some(ed) = &self.editor {
            for u in ed.poll() {
                match u {
                    ribossome::editor::Update::Amino(rows) => self.world.set_amino(&self.gpu.queue, rows),
                    ribossome::editor::Update::Organs(rows) => self.world.set_organ_table(&self.gpu.queue, rows),
                    ribossome::editor::Update::Code(code) => self.world.set_organ_code(&self.gpu.queue, code),
                }
            }
            if std::mem::take(&mut self.ui.open_editor) {
                ed.open_browser();
            }
        }
        self.mcp_tick();
        self.paint_tick();
        self.scene_tick();
        self.inspector.poll(&self.gpu.device);
        if self.inspector.follow
            && let Some(d) = &self.inspector.data
        {
            self.cam.center = [d.agent.pos_x, d.agent.pos_y];
        }
        if let Some(l) = self.world.poll_ledger(&self.gpu.device) {
            self.ui.ledger = Some(l);
            self.ui.ledger_epoch = self.world.params.epoch;
        }
        self.ui.stats.update(self.world.params.epoch, self.world.last_counters, self.world.cfg.max_agents);
        if self.ui.seed_now {
            self.ui.seed_now = false;
            let reqs = ribossome::life::seed_requests(
                self.ui.seed_count,
                self.ui.seed_len,
                self.ui.seed_aug,
                self.world.cfg.sim_size(),
                &mut self.seed_rng,
            );
            self.world.request_seeds(&reqs);
        }
        if std::mem::take(&mut self.ui.activate_now) {
            let n = self.world.activate_spent(&self.gpu, self.ui.activate_frac, self.world.params.epoch as u64);
            log::info!("activate now: {n} spent monomers activated ({:.0}%)", self.ui.activate_frac * 100.0);
        }
        if let Some(action) = self.ui.terrain_action.take() {
            self.terrain_action(action);
        }
        // RIBO_PRESET=nome do ficheiro (sem .json) ou número: lança esse preset
        // ao arrancar (para atalhos .bat).
        if let Some(want) = self.startup_preset.take() {
            let found = self.ui.presets.iter().position(|(path, _)| path.file_stem().is_some_and(|s| s.to_string_lossy() == want)).or_else(|| want.parse::<usize>().ok().filter(|i| (1..=self.ui.presets.len()).contains(i)).map(|i| i - 1));
            match found {
                Some(i) => self.ui.preset_action = Some(ribossome::ui::PresetAction::Launch(i)),
                None => log::warn!("RIBO_PRESET={want}: no such preset"),
            }
        }
        if let Some(action) = self.ui.preset_action.take() {
            self.preset_action(action);
        }
        if self.ui.restart {
            self.ui.restart = false;
            self.seed += 1;
            self.ui.baseline = if lab_mode() {
                self.world.params.epoch = 0;
                self.world.seed_lab(&self.gpu, self.seed, LAB_PER_CHANNEL)
            } else {
                self.world.restart_keeping_terrain(&self.gpu, self.seed)
            };
            self.ui.ledger = None;
            self.ui.history = ribossome::stats::History::default();
            self.lineages = ribossome::lineage::Lineages::default();
            self.last_autosave = 0;
            self.ui.scene_name = ribossome::names::new_scene_name(&self.world, self.seed);
            log::info!("restart: same terrain and parameters, epoch 0; world {}", self.ui.scene_name);
        }
        if self.ui.reseed {
            self.ui.reseed = false;
            self.seed += 1;
            self.ui.baseline = if lab_mode() {
                self.world.seed_lab(&self.gpu, self.seed, LAB_PER_CHANNEL)
            } else {
                self.world.seed_matter(&self.gpu, self.seed)
            };
            self.ui.ledger = None;
            // Mundo semeado de novo: os gráficos recomeçam (o CSV fica, com
            // uma linha de cabeçalho nova a marcar o recomeço).
            self.ui.history = ribossome::stats::History::default();
            self.lineages = ribossome::lineage::Lineages::default();
            self.ui.history.next_epoch = self.world.params.epoch;
            self.ui.scene_name = ribossome::names::new_scene_name(&self.world, self.seed);
            log::info!("world seeded again: {}", self.ui.scene_name);
            // Preset: a população é lançada no frame seguinte, já no mundo novo.
            if std::mem::take(&mut self.pending_seed) {
                self.ui.seed_now = true;
            }
        }
        let want_vsync = matches!(self.surface_cfg.present_mode, wgpu::PresentMode::AutoVsync);
        if want_vsync != self.ui.vsync {
            self.reconfigure();
        }

        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure();
                return;
            }
            other => {
                log::warn!("no surface texture: {other:?}");
                return;
            }
        };
        let target = tex.texture.create_view(&Default::default());

        // UI primeiro (pode mudar params e vista neste frame).
        let raw = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();
        let mut free = None;
        let cam_now = self.cam;
        let mut micro_asked = ribossome::microscope::Asked::default();
        let mut out = ctx.run_ui(raw, |root| {
            free = ui::draw(root, &mut self.ui, &mut self.world, &mut self.profiler, &mut self.inspector);
            // MICROSCÓPIO: a barra de dados e o painel por cima da vista.
            if let (true, Some(r), Some(scope)) = (self.micro_t >= 0.6, free, self.scope.as_mut()) {
                micro_asked = ribossome::microscope::interface(root, r, scope, &mut self.scope_panel, None, false, "");
            }
            // ECRÃ DE ENTRADA por cima de tudo, a desvanecer no fim.
            if let Some((tex, t0)) = &mut self.splash {
                let ctx = root.ctx().clone();
                let tex = tex.get_or_insert_with(|| {
                    let dec = png::Decoder::new(std::io::Cursor::new(&include_bytes!("../assets/splash.png")[..]));
                    let mut reader = dec.read_info().expect("assets/splash.png");
                    let mut buf = vec![0; reader.output_buffer_size()];
                    let info = reader.next_frame(&mut buf).expect("assets/splash.png");
                    let img = egui::ColorImage::from_rgb([info.width as usize, info.height as usize], &buf[..info.buffer_size()]);
                    ctx.load_texture("splash", img, egui::TextureOptions::LINEAR)
                });
                let t = t0.elapsed().as_secs_f32();
                let fade = ((SPLASH_SECS - t) / 0.7).clamp(0.0, 1.0);
                let screen = ctx.content_rect();
                let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("splash")));
                painter.rect_filled(screen, 0.0, egui::Color32::from_black_alpha((215.0 * fade) as u8));
                // A imagem cabe em 82% do ecrã, sem deformar.
                let size = tex.size_vec2();
                let k = (0.82 * screen.width() / size.x).min(0.82 * screen.height() / size.y);
                let r = egui::Rect::from_center_size(screen.center(), size * k);
                let tint = egui::Color32::from_white_alpha((255.0 * fade) as u8);
                painter.image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), tint);
                painter.text(
                    r.left_bottom() + egui::vec2(14.0, -34.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("version {}  ·  click or press a key to start", env!("CARGO_PKG_VERSION")),
                    egui::FontId::proportional(13.0),
                    egui::Color32::from_white_alpha((170.0 * fade) as u8),
                );
                ctx.request_repaint();
                let skip = ctx.input(|i| i.pointer.any_click() || i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. })));
                if t >= SPLASH_SECS || (skip && t > 0.3) {
                    self.splash = None;
                }
            }
            // Por cima da vista: mira do agente selecionado e do enquadramento
            // (são do mapa: no microscópio a perspetiva é outra).
            if let Some(r) = free.filter(|_| self.micro_t < 0.4) {
                let ppp = root.ctx().pixels_per_point();
                let painter = root.ctx().layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("mira"))).with_clip_rect(r);
                let centre = r.center();
                let dark = egui::Stroke::new(3.0, egui::Color32::from_black_alpha(140));
                if let Some(d) = self.inspector.data.as_ref().filter(|_| !self.inspector.dead) {
                    // Posição lida da placa em todos os frames (inspector.tracked_pos),
                    // avançada pela velocidade o atraso da leitura; sem ela ainda,
                    // a do painel.
                    let w = self.inspector.tracked_pos().unwrap_or([d.agent.pos_x, d.agent.pos_y]);
                    root.ctx().request_repaint();
                    // Mundo -> ecrã (cima no ecrã = +y no mundo).
                    let p = centre + egui::vec2((w[0] - cam_now.center[0]) * cam_now.zoom / ppp, -(w[1] - cam_now.center[1]) * cam_now.zoom / ppp);
                    let rad = (d.agent.radius * cam_now.zoom / ppp + 8.0).max(16.0);
                    let light = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 235, 120));
                    for st in [dark, light] {
                        for (dx, dy) in [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                            let dir = egui::vec2(dx, dy);
                            painter.line_segment([p + dir * rad, p + dir * (rad + 12.0)], st);
                        }
                        painter.circle_stroke(p, rad, egui::Stroke::new(st.width * 0.6, st.color));
                    }
                }
                if self.ui.frame_guide || self.ui.rec {
                    let side = 0.9 * r.width().min(r.height());
                    let q = egui::Rect::from_center_size(centre, egui::vec2(side, side));
                    let col = if self.ui.rec { egui::Color32::from_rgb(255, 80, 70) } else { egui::Color32::from_white_alpha(200) };
                    let line = egui::Stroke::new(1.5, col);
                    let arm = side * 0.08;
                    for st in [dark, line] {
                        for (cx, cy, sx, sy) in [(q.left(), q.top(), 1.0f32, 1.0f32), (q.right(), q.top(), -1.0, 1.0), (q.left(), q.bottom(), 1.0, -1.0), (q.right(), q.bottom(), -1.0, -1.0)] {
                            let c = egui::pos2(cx, cy);
                            painter.line_segment([c, c + egui::vec2(arm * sx, 0.0)], st);
                            painter.line_segment([c, c + egui::vec2(0.0, arm * sy)], st);
                        }
                    }
                    // Regra dos terços, ténue.
                    let thin = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(40));
                    for k in [1.0f32 / 3.0, 2.0 / 3.0] {
                        painter.line_segment([egui::pos2(q.left() + side * k, q.top()), egui::pos2(q.left() + side * k, q.bottom())], thin);
                        painter.line_segment([egui::pos2(q.left(), q.top() + side * k), egui::pos2(q.right(), q.top() + side * k)], thin);
                    }
                    if self.ui.rec {
                        painter.circle_filled(q.left_top() + egui::vec2(16.0, 16.0), 6.0, egui::Color32::from_rgb(255, 60, 50));
                        painter.text(q.left_top() + egui::vec2(28.0, 16.0), egui::Align2::LEFT_CENTER, "REC", egui::FontId::proportional(14.0), egui::Color32::from_rgb(255, 90, 80));
                    }
                }
            }
        });
        // Viewport da simulação: todo o espaço livre entre as barras.
        self.covered = free.is_none();
        if let Some(r) = free {
            let ppp = out.pixels_per_point;
            let (w, h) = ((r.width() * ppp).floor().max(64.0), (r.height() * ppp).floor().max(64.0));
            let old = self.viewport;
            self.viewport = [(r.left() * ppp).round().max(0.0), (r.top() * ppp).round().max(0.0), w, h];
            // O rato é guardado relativo ao viewport: acompanha-o se mudar.
            self.cursor[0] += old[0] - self.viewport[0];
            self.cursor[1] += old[1] - self.viewport[1];
            let (side, old_side) = (w.min(h), old[2].min(old[3]));
            if old_side <= 0.0 {
                // Primeiro frame: o mundo inteiro à vista.
                // (A menos que o arranque tenha pedido outra câmara, para testes.)
                match std::env::var("RIBO_ZOOM").ok().and_then(|v| v.parse::<f32>().ok()) {
                    Some(z) => {
                        self.cam.zoom = z;
                        if let Some((x, y)) = std::env::var("RIBO_CENTER").ok().and_then(|v| v.split_once(',').and_then(|(x, y)| Some((x.trim().parse::<f32>().ok()?, y.trim().parse::<f32>().ok()?)))) {
                            self.cam.center = [x, y];
                        }
                    }
                    None => self.cam = Camera::fit(&self.world.cfg, [w, h]),
                }
            } else if old_side != side && std::env::var("RIBO_ZOOM").is_err() {
                // O mesmo pedaço de mundo continua à vista quando a área muda.
                self.cam.zoom *= side / old_side;
            }
        }
        let screen = self.screen();
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_cfg.width, self.surface_cfg.height],
            pixels_per_point: out.pixels_per_point,
        };
        for (id, deltas) in out.textures_delta.set.drain() {
            for delta in deltas {
                self.egui_renderer.update_texture(&self.gpu.device, &self.gpu.queue, id, &delta);
            }
        }
        self.view.uv_depth.set(self.world.params.uv_depth);
        // Restos de quem morre: duram tempo da SIMULAÇÃO (passos), como tudo
        // o resto; com menos passos por frame a animação vê-se mais devagar.
        self.view.epoch.set(self.world.params.epoch);
        self.view.ghost_steps.set(GHOST_STEPS);
        // Ao pintar fumarolas (ou a apagá-las), vê-se onde as há.
        self.view.show_vents.set(self.ui.paint_on && (4..=6).contains(&self.ui.paint_material));
        self.view.daylight.set(self.world.params.daylight(self.world.params.epoch));
        self.view.mark_organ.set(self.ui.mark_organ);
        self.view.coc_radius.set(self.ui.coc_radius);
        self.view.origin.set([self.viewport[0], self.viewport[1]]);
        self.view.update(
            &self.gpu.queue,
            &self.cam,
            screen,
            self.ui.view_mode,
            self.ui.monomer_brightness,
            self.ui.signal_view,
        );

        // Mapa genético: o genoma do selecionado passa a ser a referência;
        // a semelhança de todos recalcula-se de 15 em 15 frames.
        let mut want_kin = false;
        if self.ui.signal_view == 4 {
            let id = self.inspector.data.as_ref().map(|d| d.agent.id);
            if id != self.kin_id {
                let genome = self.inspector.data.as_ref().map(|d| d.genome.clone()).unwrap_or_default();
                self.world.set_kin_target(&self.gpu.queue, &genome);
                self.kin_id = id;
                self.kin_frames = 0;
            }
            want_kin = self.kin_frames == 0;
            self.kin_frames = (self.kin_frames + 1) % 15;
        }
        let epoch_now = self.world.params.epoch;
        if epoch_now.saturating_add(self.ui.history.every) < self.ui.history.next_epoch {
            // O epoch voltou atrás (mundo novo ou cena carregada).
            self.ui.history.next_epoch = epoch_now;
        }
        let want_stats = epoch_now >= self.ui.history.next_epoch;

        self.lineage_tick();
        if std::mem::take(&mut self.ui.report_now) {
            self.write_report();
        }
        self.photo_video();
        if let Some(action) = self.ui.agent_action.take() {
            self.agent_action(action);
        }
        let side = std::mem::take(&mut self.ui.big_shot);
        if side > 0 {
            let epoch = self.world.params.epoch;
            let rgb = ribossome::render::capture::world_mosaic(&self.gpu, &self.world, side, self.ui.view_mode, self.ui.monomer_brightness, self.ui.coc_radius);
            let dir = std::path::Path::new(SAVES_DIR).join("capturas");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("mundo_{}k_{epoch}.png", side / 1024));
            self.ui.scene_msg = format!("saving {} (in the background)", path.display());
            log::info!("{}", self.ui.scene_msg);
            // O PNG comprime-se noutra thread, para a simulação não ficar à espera.
            std::thread::spawn(move || match ribossome::render::capture::save_rgb_png(&rgb, side, &path) {
                Ok(()) => log::info!("capture saved to {}", path.display()),
                Err(e) => log::error!("capture {}: {e}", path.display()),
            });
        }
        if std::mem::take(&mut self.ui.tree_now) {
            let epoch = self.world.params.epoch;
            let pics = ribossome::tree_view::portraits(&self.gpu, &self.world, &self.lineages);
            let html = ribossome::tree_view::page(&self.lineages, &self.world, &format!("Ribossome: lineage tree at epoch {epoch}"), Some(&pics));
            self.write_page(&format!("arvore_{epoch}.html"), html, "tree");
        }
        let n_steps = self.adaptive_steps();
        // ZOOM SUAVE: em cada frame aplica-se uma parte do que a roda pediu.
        if self.zoom_pending.abs() > 1e-4 {
            let part = self.zoom_pending * 0.18;
            self.zoom_pending -= part;
            self.cam.zoom_at(part.exp(), self.zoom_anchor, screen);
        }
        let (vp, covered) = (self.viewport, self.covered);
        // MICROSCÓPIO 3D: a câmara do mapa continua a mandar (deslocar e
        // aproximar como sempre); o microscópio olha para o mesmo ponto com o
        // mesmo enquadramento em altura.
        // TRANSIÇÃO CONTÍNUA pelo zoom (píxeis por unidade do mundo): abaixo
        // de MICRO_Z0 é só o mapa; daí até MICRO_Z1 o microscópio entra em
        // fundido, visto de cima como o mapa, e depois inclina-se e ganha a
        // profundidade de campo até à câmara escolhida no painel.
        const MICRO_Z0: f32 = 0.6;
        const MICRO_Z1: f32 = 3.2;
        let smooth = |a: f32, b: f32, x: f32| {
            let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        self.micro_t = if self.ui.microscope && !covered && vp[2] >= 32.0 && vp[3] >= 32.0 { smooth(MICRO_Z0.ln(), MICRO_Z1.ln(), self.cam.zoom.max(1e-6).ln()) } else { 0.0 };
        let micro = self.micro_t > 0.0;
        let map_visible = self.micro_t < 0.75;
        if micro {
            use ribossome::microscope::{REF_TAN, REGION_PER_DIST, Scope};
            let dist = (0.5 * screen[1] / self.cam.zoom / REF_TAN).clamp(25.0, 5000.0);
            let region = (dist * REGION_PER_DIST).clamp(30.0, 4000.0);
            let scope = self.scope.get_or_insert_with(|| Scope::new(&self.gpu, &self.world, self.surface_cfg.format, [vp[2] as u32, vp[3] as u32], self.cam.center, region));
            // (Fundido LONGO, e a câmara só se começa a inclinar quando o
            // microscópio já pesa mais do que o mapa.)
            scope.opacity = smooth(0.0, 0.75, self.micro_t);
            scope.approach = smooth(0.5, 1.0, self.micro_t);
            // SEGUIR: é o microscópio que leva a câmara (a do mapa vai atrás).
            // MIRA: procura-se o agente do centro de vez em quando.
            if scope.follow {
                scope.orbit.centre = self.cam.center;
                scope.follow_step(&self.gpu, &self.world);
                self.cam.center = scope.orbit.centre;
            } else if scope.reticle && scope.subject.as_ref().is_none_or(|s| s.read_at.elapsed().as_secs_f32() > 1.0) {
                scope.find_subject(&self.gpu, &self.world);
            }
            scope.orbit.centre = self.cam.center;
            scope.orbit.dist = dist;
            scope.region = region;
            scope.monomers = if self.ui.monomer_brightness > 0.0 { 0.7 } else { 0.0 };
            if micro_asked.supersampling {
                scope.ss = if scope.ss >= 2 { 1 } else { 2 };
                let size = scope.size;
                scope.resize(&self.gpu.device, size);
            }
        }
        // Só se desenham os agentes à vista (aqui, mesmo antes de a lista
        // ser feita: um screenshot pelo MCP pode ter posto outro retângulo).
        let (hw, hh) = (0.5 * screen[0] / self.cam.zoom, 0.5 * screen[1] / self.cam.zoom);
        let c = self.cam.center;
        // (No microscópio, os da zona que as camadas dele cobrem.)
        let (c, hw, hh) = match self.scope.as_ref().filter(|_| micro) {
            Some(scope) => {
                let (zc, zr) = scope.zone();
                (zc, zr, zr)
            }
            None => (c, hw, hh),
        };
        self.world.set_draw_rect(&self.gpu.queue, Some(([c[0] - hw, c[1] - hh], [c[0] + hw, c[1] + hh])));
        let format = self.surface_cfg.format;
        let Running { gpu, world, view, egui_renderer, profiler, ui: st, inspector, msaa, depth, scope, .. } = self;
        let mut frame = profiler.begin(&gpu.device, &gpu.queue);
        if !st.paused {
            let n = n_steps;
            frame.segment("world", |enc| world.encode_steps(&gpu.queue, enc, n));
        }
        frame.segment("ledger", |enc| world.encode_ledger_readback(enc));
        if want_stats {
            let mut started = false;
            frame.segment("stats", |enc| started = world.encode_stats(enc));
            if started {
                st.history.next_epoch = epoch_now.saturating_add(st.history.every.max(100));
            }
        }
        if want_kin {
            frame.segment("kinship", |enc| world.encode_kinship(enc));
        }
        frame.segment("draw list", |enc| world.encode_draw_list(enc));
        frame.segment("inspect", |enc| {
            inspector.encode(world, enc);
            inspector.encode_preview(&gpu.queue, enc);
        });

        let mut egui_enc = gpu.device.create_command_encoder(&Default::default());
        let mut extra = egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut egui_enc, &jobs, &sd);
        extra.push(egui_enc.finish());
        frame.submit_now(extra);

        // O MUNDO vai para um alvo com 4 amostras por píxel, que se resolve
        // para a imagem da janela; a INTERFACE é desenhada por cima depois.
        let (sw, sh) = (sd.size_in_pixels[0], sd.size_in_pixels[1]);
        if msaa.as_ref().is_none_or(|t| t.width() != sw || t.height() != sh) {
            *msaa = Some(ribossome::render::msaa_texture(&gpu.device, format, sw, sh));
            *depth = Some(ribossome::render::depth_texture(&gpu.device, sw, sh));
        }
        let many = msaa.as_ref().unwrap().create_view(&Default::default());
        let depth_view = depth.as_ref().unwrap().create_view(&Default::default());
        frame.segment("render", |enc| {
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mundo"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &many,
                        depth_slice: None,
                        resolve_target: Some(&target),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Discard,
                        },
                    })],
                    depth_stencil_attachment: Some(ribossome::render::depth_attachment(&depth_view)),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                // A simulação só no espaço livre entre as barras.
                let (w, h) = (vp[2].min(sw as f32 - vp[0]), vp[3].min(sh as f32 - vp[1]));
                if !covered && w >= 1.0 && h >= 1.0 {
                    pass.set_viewport(vp[0], vp[1], w, h, 0.0, 1.0);
                    pass.set_scissor_rect(vp[0] as u32, vp[1] as u32, w as u32, h as u32);
                    if map_visible {
                        view.draw(&mut pass);
                    }
                }
            }
            // MICROSCÓPIO: a sua imagem por cima do retângulo da vista.
            if let (true, Some(scope)) = (micro, scope.as_mut()) {
                let (w, h) = (vp[2].min(sw as f32 - vp[0]).floor(), vp[3].min(sh as f32 - vp[1]).floor());
                scope.encode_embedded(gpu, world, enc, &target, [vp[0], vp[1], w, h], !st.paused);
            }
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("interface"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            egui_renderer.render(&mut pass, &jobs, &sd);
        });
        frame.finish();
        world.ledger_after_submit();
        world.stats_after_submit();
        inspector.after_submit();
        if let Some(w) = world.poll_stats(&gpu.device) {
            st.history.push(world.params.epoch, &w, st.ledger, world.last_counters);
        }

        self.runlog.after_frame(&self.world, &self.ui, &mut self.profiler);
        self.window.pre_present_notify();
        self.gpu.queue.present(tex);
        for id in out.textures_delta.free.drain() {
            self.egui_renderer.free_texture(&id);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let resp = self.egui_state.on_window_event(&self.window, &event);
        let ctx = self.egui_state.egui_ctx().clone();
        let egui_mouse = ctx.egui_wants_pointer_input() || ctx.is_pointer_over_egui();
        let egui_keys = ctx.egui_wants_keyboard_input();

        match event {
            WindowEvent::CloseRequested => {
                self.on_close();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.surface_cfg.width = size.width.max(1);
                self.surface_cfg.height = size.height.max(1);
                self.reconfigure();
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
                // OUTRO TAMANHO DE MUNDO: escolhe-se ao arrancar, por isso
                // grava-se, fecha-se e abre-se o programa de novo com ele.
                if let Some(grid) = self.ui.relaunch_grid.take() {
                    self.on_close();
                    // Fica escolhido para as próximas vezes que o programa abrir.
                    if let Err(e) = std::fs::write(WORLD_SIZE_FILE, format!("{grid}
")) {
                        log::warn!("world size: could not write {WORLD_SIZE_FILE}: {e}");
                    }
                    match std::env::current_exe().and_then(|exe| {
                        let mut cmd = std::process::Command::new(exe);
                        cmd.env("RIBO_NO_SPLASH", "1");
                        cmd.env_remove("RIBO_GRID");
                        cmd.spawn()
                    }) {
                        Ok(_) => event_loop.exit(),
                        Err(e) => self.ui.preset_msg = format!("could not restart the program: {e}"),
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = [position.x as f32 - self.viewport[0], position.y as f32 - self.viewport[1]];
                let d = [p[0] - self.cursor[0], p[1] - self.cursor[1]];
                // Botão direito no microscópio: roda e inclina a câmara.
                if let (true, Some(scope)) = (self.orbiting, self.scope.as_mut()) {
                    scope.orbit.yaw -= d[0] * 0.006;
                    scope.orbit.pitch = (scope.orbit.pitch + d[1] * 0.006).clamp(0.12, 1.55);
                }
                if self.dragging {
                    // (Deslocar à mão larga o agente que se estava a seguir.)
                    if let Some(scope) = self.scope.as_mut().filter(|_| d[0] != 0.0 || d[1] != 0.0) {
                        scope.follow = false;
                    }
                    // Com a câmara rodada, arrastar desloca no referencial dela.
                    let yaw = self.scope.as_ref().filter(|_| self.micro_t > 0.0).map_or(0.0, |s| s.effective().yaw);
                    if yaw.abs() > 1e-3 {
                        let (sy, cy) = yaw.sin_cos();
                        let k = 1.0 / self.cam.zoom;
                        self.cam.center[0] += (-d[0] * cy - d[1] * sy) * k;
                        self.cam.center[1] += (-d[0] * sy + d[1] * cy) * k;
                    } else {
                        self.cam.pan_pixels(d);
                    }
                    let (dx, dy) = (p[0] - self.press_pos[0], p[1] - self.press_pos[1]);
                    if dx * dx + dy * dy > 16.0 {
                        self.press_moved = true;
                    }
                }
                self.cursor = p;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pan_button = matches!(button, MouseButton::Left | MouseButton::Right | MouseButton::Middle);
                if button == MouseButton::Right && (self.micro_t > 0.5 || self.orbiting) {
                    self.orbiting = state == ElementState::Pressed && !egui_mouse;
                } else if button == MouseButton::Left && self.ui.paint_on {
                    // Pincel ligado: o botão esquerdo pinta (não arrasta nem seleciona).
                    self.painting = state == ElementState::Pressed && !egui_mouse;
                } else if pan_button {
                    let was_dragging = self.dragging;
                    self.dragging = state == ElementState::Pressed && !egui_mouse;
                    if self.dragging {
                        self.press_pos = self.cursor;
                        self.press_moved = false;
                    }
                    // Clique esquerdo sem arrastar: seleciona o organismo mais próximo.
                    if button == MouseButton::Left
                        && state == ElementState::Released
                        && was_dragging
                        && !self.press_moved
                    {
                        // No microscópio o ponto clicado vem da profundidade da
                        // imagem (a vista está inclinada), e o clique também foca.
                        let seen = self.scope.as_mut().filter(|_| self.micro_t > 0.5).and_then(|s| {
                            let w = s.world_at(&self.gpu, self.cursor);
                            s.focus_at(&self.gpu, self.cursor);
                            w
                        });
                        let w = seen.unwrap_or_else(|| self.cam.screen_to_world(self.cursor, self.screen()));
                        match self.loaded_agent.as_ref().filter(|_| self.ui.place_agent) {
                            // Agente carregado + "pôr com o rato": nasce uma cópia aqui.
                            Some(g) => self.world.request_seeds(&[ribossome::params::SpawnRequest::with_genome(w[0], w[1], g)]),
                            None => self.inspector.pick(&self.gpu, &self.world, w),
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_mouse => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                let screen = self.screen();
                let _ = screen;
                self.zoom_pending += lines * 1.15f32.ln();
                self.zoom_anchor = self.cursor;
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !egui_keys && !resp.consumed =>
            {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Space) => self.ui.paused = !self.ui.paused,
                    Key::Named(NamedKey::Home) => self.cam = Camera::fit(&self.world.cfg, self.screen()),
                    Key::Character(c) => {
                        if let Some(d) = c.chars().next().and_then(|c| c.to_digit(10))
                            && d <= 9
                        {
                            // A mesma tecla volta à vista normal.
                            self.ui.view_mode = if self.ui.view_mode == d { 0 } else { d };
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.run.is_none() {
            self.run = Some(Running::new(event_loop));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(r) = self.run.as_mut() {
            r.window_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = self.run.as_ref() {
            r.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,ribossome=info")).init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run_app");
}
