//! MICROSCÓPIO 3D (protótipo, modo fotográfico): uma zona de uma cena vista
//! como num microscópio eletrónico de varrimento, com câmara em perspetiva
//! que se pode rodar e imagem ACUMULADA no tempo (cada frame lança raios um
//! pouco diferentes e a média limpa o ruído; o que se mexe fica arrastado).
//!
//! A simulação é 2D: a profundidade é só desenho. Em cada frame a zona é
//! desenhada de cima, com o desenho normal do programa, para duas texturas:
//! a COR (as peças como as vemos, com as suas sombras) e a ALTURA de cada
//! peça (um troço é um cilindro deitado, um órgão uma cúpula sobre o seu
//! contorno, a rocha um planalto, o entulho seixos). Depois cada píxel do
//! ecrã lança um raio contra esse relevo. O brilho é o do microscópio: sem
//! luzes, mais claro nas arestas (superfícies de lado para o observador) e
//! mais escuro nas zonas encaixadas entre vizinhos.
//!
//! Uso: `cargo run --release --bin microscopio` (ou microscopio.bat).
//!   SCENE   cena a abrir (por omissão o autosave)
//!   REGION  meio lado da zona, em unidades do mundo (por omissão 420)
//!   STEPS   passos de simulação por frame (por omissão 2; 0 = parada)
//!   --foto ficheiro.png [amostras]   sem janela: acumula e grava a imagem
//! Rato: arrastar com o botão esquerdo roda a câmara, com o direito desloca a
//! zona, a roda aproxima. Começa PARADO (a imagem converge e fica nítida);
//! Espaço põe a simulação a correr e volta a parar. G/H fecham e abrem o
//! diafragma (profundidade de campo); Z/X alongam e encurtam a lente (longa =
//! quase axonométrica; por omissão 135 mm); E/R clareiam e escurecem a
//! exposição; C liga a cor (por omissão é a preto e branco); M esconde os
//! monómeros; S liga a superamostragem; Esc sai.
//!   LENS=mm, APERTURE, EXPOSURE  arrancam com esses valores
//! T liga a MIRA (cantos à volta do agente mais perto do centro, com o nome
//! da espécie, a linhagem, a geração, a idade e a energia).
//! P tira uma fotografia (saves/capturas) e V grava vídeo (saves/videos, com
//! o ffmpeg), os dois com a barra de dados e o título mas sem o painel.
//! Um CLIQUE (sem arrastar) foca no ponto clicado; Tab mostra e esconde o
//! painel de controlos; por baixo da imagem fica a barra de dados com a
//! escala em nanómetros (ver NM_PER_UNIT: é uma convenção).
//!   TERRAIN=1 centra numa zona com rocha, entulho e água (em vez de num agente)
//!   COLOR=1, MONOMERS=0.7  arrancam com cor / com monómeros

use std::sync::Arc;

use ribossome::gpu::Gpu;
use ribossome::microscope::{Asked, REGION_PER_DIST, Scope, interface, overlay};
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
    scope: Scope,
    cursor: [f32; 2],
    orbiting: bool,
    panning: bool,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    /// Painel de controlos à vista (Tab).
    panel: bool,
    /// Onde o botão esquerdo desceu (um clique sem arrastar foca ali).
    pressed_at: Option<[f32; 2]>,
    /// Último ponto focado e quando, para a marca.
    focused: Option<([f32; 2], std::time::Instant)>,
    /// FOTO E VÍDEO: a imagem com a barra de dados e o título (sem o painel)
    /// desenha-se à parte, com a sua própria interface, e lê-se de volta.
    shot_ctx: egui::Context,
    shot_renderer: egui_wgpu::Renderer,
    shot_tex: Option<wgpu::Texture>,
    photo_now: bool,
    rec: bool,
    rec_tx: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
    rec_size: [u32; 2],
    rec_frames: u32,
    rec_path: std::path::PathBuf,
    status: String,
}

impl Running {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Ribossome: microscope").with_inner_size(winit::dpi::LogicalSize::new(1280, 800)))
                .expect("criar janela"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = pollster::block_on(Gpu::new(instance, Some(&surface))).expect("GPU");
        let caps = surface.get_capabilities(&gpu.adapter);
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
        let (world, centre, region) = Scope::open_world(&gpu);
        let scope = Scope::new(&gpu, &world, format, [surface_cfg.width, surface_cfg.height], centre, region);
        let egui_state = egui_winit::State::new(egui::Context::default(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, Some(gpu.device.limits().max_texture_dimension_2d as usize));
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
        let shot_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
        Self {
            window,
            surface,
            surface_cfg,
            gpu,
            world,
            scope,
            cursor: [0.0; 2],
            orbiting: false,
            panning: false,
            egui_state,
            egui_renderer,
            panel: true,
            pressed_at: None,
            focused: None,
            shot_ctx: egui::Context::default(),
            shot_renderer,
            shot_tex: None,
            photo_now: false,
            rec: false,
            rec_tx: None,
            rec_size: [0; 2],
            rec_frames: 0,
            rec_path: Default::default(),
            status: String::new(),
        }
    }

    /// A imagem de `view` (a cena, já desenhada) leva a barra de dados e o
    /// título por cima, lê-se de volta e vai para um PNG e/ou para o vídeo.
    fn shoot(&mut self, view: &wgpu::TextureView, size: [u32; 2]) {
        let (w, h) = (size[0], size[1]);
        let ppp = self.window.scale_factor() as f32;
        let ctx = self.shot_ctx.clone();
        ctx.set_pixels_per_point(ppp);
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32 / ppp, h as f32 / ppp))), ..Default::default() };
        let scope = &self.scope;
        let mut out = ctx.run_ui(raw, |root| overlay(root.ctx(), root.max_rect(), scope, None, false));
        let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: size, pixels_per_point: out.pixels_per_point };
        for (id, deltas) in out.textures_delta.set.drain() {
            for delta in deltas {
                self.shot_renderer.update_texture(&self.gpu.device, &self.gpu.queue, id, &delta);
            }
        }
        let row = (w * 4).div_ceil(256) * 256;
        let readback = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shot"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let mut cmds = self.shot_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut enc, &jobs, &sd);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shot overlay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
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
            self.shot_renderer.render(&mut pass, &jobs, &sd);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: self.shot_tex.as_ref().unwrap(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        cmds.push(enc.finish());
        self.gpu.queue.submit(cmds);
        for id in out.textures_delta.free.drain() {
            self.shot_renderer.free_texture(&id);
        }
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        self.gpu.wait_idle();
        let bgra = matches!(self.surface_cfg.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = readback.get_mapped_range(..).expect("mapped");
            for y in 0..h as usize {
                let line = &data[y * row as usize..y * row as usize + (w * 4) as usize];
                for px in line.chunks_exact(4) {
                    rgba.extend_from_slice(&if bgra { [px[2], px[1], px[0], 255] } else { [px[0], px[1], px[2], 255] });
                }
            }
        }
        readback.unmap();
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        if std::mem::take(&mut self.photo_now) {
            let dir = std::path::Path::new("saves").join("capturas");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("microscopio_{stamp}_epoch{}.png", self.world.params.epoch));
            self.status = format!("photo saved: {}", path.display());
            let data = rgba.clone();
            // O PNG comprime-se noutra thread.
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
        // VÍDEO: as imagens vão cruas para um ffmpeg (como na aplicação principal).
        if !self.rec {
            if self.rec_tx.take().is_some() {
                self.status = format!("video saved: {} ({} images)", self.rec_path.display(), self.rec_frames);
            }
            return;
        }
        // (O x264 quer lados pares.)
        let even = [w & !1, h & !1];
        if self.rec_tx.is_none() {
            let dir = std::path::Path::new("saves").join("videos");
            let _ = std::fs::create_dir_all(&dir);
            self.rec_path = dir.join(format!("microscopio_{stamp}.mp4"));
            let child = std::process::Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
                .arg(format!("{}x{}", even[0], even[1]))
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
                    self.rec_size = even;
                    self.rec_frames = 0;
                }
                Err(e) => {
                    self.rec = false;
                    self.status = format!("could not start ffmpeg ({e}): it must be installed and on the PATH");
                    return;
                }
            }
        }
        if even != self.rec_size {
            // A janela mudou de tamanho: a gravação acaba aqui.
            self.rec = false;
            self.rec_tx = None;
            self.status = format!("window resized, video closed: {} ({} images)", self.rec_path.display(), self.rec_frames);
            return;
        }
        let mut img = Vec::with_capacity((even[0] * even[1] * 4) as usize);
        for y in 0..even[1] as usize {
            img.extend_from_slice(&rgba[y * w as usize * 4..y * w as usize * 4 + even[0] as usize * 4]);
        }
        match self.rec_tx.as_ref().map(|tx| tx.try_send(img)) {
            Some(Ok(())) => {
                self.rec_frames += 1;
                self.status = format!("recording: {} images ({:.1} s of video)", self.rec_frames, self.rec_frames as f32 / 60.0);
            }
            Some(Err(std::sync::mpsc::TrySendError::Disconnected(_))) => {
                self.rec_tx = None;
                self.rec = false;
                self.status = "ffmpeg stopped in the middle of the recording".into();
            }
            _ => {}
        }
    }

    fn redraw(&mut self) {
        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.surface.configure(&self.gpu.device, &self.surface_cfg);
                return;
            }
        };
        let target = tex.texture.create_view(&Default::default());
        // Fotografia ou imagem de vídeo neste frame: a cena desenha-se também
        // numa textura à parte (do tamanho da janela, ou do do arranque da
        // gravação: o ffmpeg precisa de imagens iguais).
        // (AUTOSHOT=n: fotografa sozinho ao frame n e sai 30 frames depois; para testes.)
        if let Some(n) = std::env::var("AUTOSHOT").ok().and_then(|v| v.parse::<u32>().ok()) {
            if self.scope.frame == n {
                self.photo_now = true;
            }
            if self.scope.frame == n + 30 {
                std::process::exit(0);
            }
        }
        let shooting = self.photo_now || self.rec || self.rec_tx.is_some();
        let size = [self.surface_cfg.width, self.surface_cfg.height];
        if shooting && self.shot_tex.as_ref().is_none_or(|t| [t.width(), t.height()] != size) {
            self.shot_tex = Some(self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shot"),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_cfg.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }));
        }
        // MIRA: procura-se de novo de vez em quando (o agente mexe-se, a câmara
        // também), nunca em todos os frames.
        // (A seguir, o agente é sempre o mesmo: não se procura outro.)
        self.scope.follow_step(&self.gpu, &self.world);
        if !self.scope.follow && self.scope.reticle && self.scope.subject.as_ref().is_none_or(|s| s.read_at.elapsed().as_secs_f32() > 1.0) {
            self.scope.find_subject(&self.gpu, &self.world);
        }
        let shot_view = self.shot_tex.as_ref().filter(|_| shooting).map(|t| t.create_view(&Default::default()));
        self.scope.frame(&self.gpu, &mut self.world, &target, shot_view.as_ref());
        if let Some(view) = shot_view.as_ref() {
            self.shoot(view, size);
        }
        if self.scope.frame % 30 == 0 {
            let s = &self.scope;
            self.window.set_title(&format!(
                "Ribossome: microscope   {} samples{}   {}   lens {:.0} mm   aperture {:.1}   exposure {:.2}   (click: focus, drag: orbit, right drag: move, wheel: zoom, space: run/pause, Tab: panel, Z/X: lens, G/H: aperture, E/R: exposure, F: follow, T: reticle, P: photo, V: video, C: colour, M: monomers, S: supersampling)",
                s.samples,
                if s.ss > 1 { " ×4 (supersampled)" } else { "" },
                if s.paused || s.steps == 0 { "paused" } else { "running" },
                12.0 / s.orbit.focal,
                s.orbit.aperture,
                s.exposure
            ));
        }
        // INTERFACE por cima da imagem.
        let raw = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();
        let marker = self.focused.filter(|(_, t)| t.elapsed().as_secs_f32() < 1.2).map(|(p, _)| p);
        let mut asked = Asked::default();
        let recording = self.rec_tx.is_some();
        let status = self.status.clone();
        let mut out = ctx.run_ui(raw, |root| asked = interface(root, &mut self.scope, &mut self.panel, marker, recording, &status));
        self.photo_now |= asked.photo;
        if asked.rec {
            self.rec = !self.rec;
        }
        if asked.supersampling {
            self.scope.ss = if self.scope.ss >= 2 { 1 } else { 2 };
            let size = self.scope.size;
            self.scope.resize(&self.gpu.device, size);
        }
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.surface_cfg.width, self.surface_cfg.height], pixels_per_point: out.pixels_per_point };
        for (id, deltas) in out.textures_delta.set.drain() {
            for delta in deltas {
                self.egui_renderer.update_texture(&self.gpu.device, &self.gpu.queue, id, &delta);
            }
        }
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let mut cmds = self.egui_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut enc, &jobs, &sd);
        {
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
            self.egui_renderer.render(&mut pass, &jobs, &sd);
        }
        cmds.push(enc.finish());
        self.gpu.queue.submit(cmds);
        self.window.pre_present_notify();
        self.gpu.queue.present(tex);
        for id in out.textures_delta.free.drain() {
            self.egui_renderer.free_texture(&id);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        // A interface vê primeiro: o que for dela (rato sobre o painel,
        // teclas num campo) não mexe na câmara.
        let _ = self.egui_state.on_window_event(&self.window, &event);
        let ctx = self.egui_state.egui_ctx().clone();
        let over_ui = ctx.egui_wants_pointer_input() || ctx.is_pointer_over_egui();
        let pressing = matches!(event, WindowEvent::MouseInput { state: ElementState::Pressed, .. }) || matches!(event, WindowEvent::MouseWheel { .. });
        let typing = matches!(event, WindowEvent::KeyboardInput { .. }) && ctx.egui_wants_keyboard_input();
        if (over_ui && pressing) || typing {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                self.surface_cfg.width = s.width.max(1);
                self.surface_cfg.height = s.height.max(1);
                self.surface.configure(&self.gpu.device, &self.surface_cfg);
                self.scope.resize(&self.gpu.device, [self.surface_cfg.width, self.surface_cfg.height]);
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                let p = [position.x as f32, position.y as f32];
                let (dx, dy) = (p[0] - self.cursor[0], p[1] - self.cursor[1]);
                let o = &mut self.scope.orbit;
                if self.orbiting {
                    o.yaw -= dx * 0.006;
                    o.pitch = (o.pitch + dy * 0.006).clamp(0.12, 1.55);
                }
                if self.panning && (dx != 0.0 || dy != 0.0) {
                    // (Deslocar à mão larga o agente.)
                    self.scope.follow = false;
                }
                if self.panning {
                    // Desloca a zona no plano do fundo, no referencial da câmara.
                    let k = o.dist / self.surface_cfg.height as f32;
                    let (sy, cy) = o.yaw.sin_cos();
                    // (Arrastar para baixo traz a cena para o observador.)
                    o.centre[0] += (-dx * cy - dy * sy) * k;
                    o.centre[1] += (-dx * sy + dy * cy) * k;
                }
                self.cursor = p;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.orbiting = down;
                        if down {
                            self.pressed_at = Some(self.cursor);
                        } else if let Some(p) = self.pressed_at.take() {
                            // Clique sem arrastar: AUTOFOCO nesse ponto.
                            let moved = (self.cursor[0] - p[0]).hypot(self.cursor[1] - p[1]);
                            if moved < 4.0 && self.scope.focus_at(&self.gpu, self.cursor) {
                                self.focused = Some((self.cursor, std::time::Instant::now()));
                            }
                        }
                    }
                    MouseButton::Right => self.panning = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                // ZOOM: a câmara afasta-se e a zona desenhada cresce com ela (a
                // de perto é pequena e detalhada, a de longe apanha mais mundo).
                let o = &mut self.scope.orbit;
                o.dist = (o.dist * 0.9f32.powf(lines)).clamp(25.0, 5000.0);
                self.scope.region = (o.dist * REGION_PER_DIST).clamp(30.0, 4000.0);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key.as_ref() {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Tab) => self.panel = !self.panel,
                Key::Named(NamedKey::Space) => {
                    self.scope.paused = !self.scope.paused;
                    self.scope.last_orbit = None;
                }
                // F: SEGUIR o agente mais perto do centro (outra vez: larga-o).
                Key::Character("f") => {
                    self.scope.follow = !self.scope.follow;
                    if self.scope.follow {
                        self.scope.find_subject(&self.gpu, &self.world);
                    }
                }
                Key::Character("g") => self.scope.orbit.aperture = (self.scope.orbit.aperture - 1.0).max(0.0),
                Key::Character("h") => self.scope.orbit.aperture = (self.scope.orbit.aperture + 1.0).min(30.0),
                Key::Character("c") => {
                    self.scope.colour = (self.scope.colour + 1) % 3;
                    self.scope.last_orbit = None;
                }
                // LENTE: Z alonga (mais axonométrica), X encurta (grande angular).
                Key::Character("z") => self.scope.orbit.focal = (self.scope.orbit.focal / 1.15).max(12.0 / 800.0),
                Key::Character("x") => self.scope.orbit.focal = (self.scope.orbit.focal * 1.15).min(12.0 / 14.0),
                // EXPOSIÇÃO: E clareia, R escurece (não reinicia a acumulação).
                Key::Character("e") => self.scope.exposure = (self.scope.exposure * 1.12).min(16.0),
                Key::Character("r") => self.scope.exposure = (self.scope.exposure / 1.12).max(0.05),
                Key::Character("s") => {
                    self.scope.ss = if self.scope.ss >= 2 { 1 } else { 2 };
                    let size = self.scope.size;
                    self.scope.resize(&self.gpu.device, size);
                }
                Key::Character("t") => {
                    self.scope.reticle = !self.scope.reticle;
                    self.scope.subject = None;
                }
                Key::Character("p") => self.photo_now = true,
                Key::Character("v") => self.rec = !self.rec,
                Key::Character("m") => {
                    self.scope.monomers = if self.scope.monomers > 0.0 { 0.0 } else { 0.7 };
                    self.scope.last_orbit = None;
                }
                _ => {}
            },
            _ => {}
        }
    }
}

#[derive(Default)]
struct App {
    run: Option<Running>,
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

/// Sem janela: acumula `samples` amostras com a simulação parada e grava a
/// imagem. YAW, PITCH (radianos), DIST e APERTURE mudam a câmara.
fn photo(path: &str, samples: u32) {
    let gpu = Gpu::new_headless().expect("GPU");
    let (w, h) = (1280u32, 768u32);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let (mut world, centre, region) = Scope::open_world(&gpu);
    let mut scope = Scope::new(&gpu, &world, format, [w, h], centre, region);
    scope.ss = (std::env::var("SS").ok().and_then(|v| v.parse().ok()).unwrap_or(2u32)).clamp(1, 3);
    scope.resize(&gpu.device, [w, h]);
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    scope.orbit.yaw = env("YAW", scope.orbit.yaw);
    scope.orbit.pitch = env("PITCH", scope.orbit.pitch);
    scope.orbit.dist = env("DIST", scope.orbit.dist);
    if std::env::var("REGION").is_err() {
        scope.region = (scope.orbit.dist * REGION_PER_DIST).clamp(30.0, 4000.0);
    }
    scope.orbit.aperture = env("APERTURE", scope.orbit.aperture);
    scope.orbit.focal = 12.0 / env("LENS", 12.0 / scope.orbit.focal);
    // Uns passos para a grelha de desenho e as poses assentarem, depois parada.
    scope.frame(&gpu, &mut world, &gpu.device.create_texture(&target_desc(w, h, format)).create_view(&Default::default()), None);
    scope.paused = true;
    scope.last_orbit = None;
    let target = gpu.device.create_texture(&target_desc(w, h, format));
    let view = target.create_view(&Default::default());
    for _ in 0..samples.max(1) {
        scope.frame(&gpu, &mut world, &view, None);
    }
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("photo"),
        size: (w * h * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    gpu.wait_idle();
    let data = readback.get_mapped_range(..).expect("mapped").to_vec();
    readback.unmap();
    let file = std::io::BufWriter::new(std::fs::File::create(path).expect("criar o PNG"));
    let mut e = png::Encoder::new(file, w, h);
    e.set_color(png::ColorType::Rgba);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&data).unwrap();
    println!("{path}: {samples} amostras, {w} × {h}");
}

fn target_desc(w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("photo"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,microscopio=info")).init();
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "--foto") {
        photo(args.get(2).map(String::as_str).unwrap_or("microscopio.png"), args.get(3).and_then(|v| v.parse().ok()).unwrap_or(64));
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run_app");
}
