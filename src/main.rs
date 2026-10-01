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
    cam: Camera,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    profiler: Profiler,
    ui: UiState,
    cursor: [f32; 2],
    dragging: bool,
    /// Onde o botão esquerdo foi premido e se o rato já se mexeu (clique vs arrastar).
    press_pos: [f32; 2],
    press_moved: bool,
    inspector: ui::inspector::Inspector,
    seed: u64,
    seed_rng: ribossome::life::SplitMix,
}

#[derive(Default)]
struct App {
    run: Option<Running>,
}

fn world_config_from_env() -> WorldConfig {
    let mut cfg = WorldConfig::DEFAULT;
    if let Some(g) = std::env::var("RIBO_GRID").ok().and_then(|v| v.parse::<u32>().ok()) {
        cfg.grid_size = g;
        cfg.fluid_size = (g / 2).max(16);
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
                        .with_title("Ribossome v4")
                        .with_inner_size(winit::dpi::LogicalSize::new(1400, 900)),
                )
                .expect("criar janela"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("criar surface");
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
        log::info!("mundo {}² células, {} unidades", cfg.grid_size, cfg.sim_size());
        let seed = 1;
        let mut world = World::new(&gpu, cfg, seed as u32);
        let baseline = world.seed_matter(&gpu, seed);
        let view = WorldView::new(&gpu.device, &world, format);
        let cam = Camera::fit(&cfg, [surface_cfg.width as f32, surface_cfg.height as f32]);

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
        let mut inspector = ui::inspector::Inspector::new(&gpu, &world);
        inspector.register(&gpu.device, &mut egui_renderer);

        Self {
            window,
            surface,
            surface_cfg,
            gpu,
            world,
            view,
            cam,
            egui_state,
            egui_renderer,
            profiler: Profiler::from_env(),
            ui: UiState::new(baseline),
            cursor: [0.0; 2],
            dragging: false,
            press_pos: [0.0; 2],
            press_moved: false,
            inspector,
            seed,
            seed_rng: ribossome::life::SplitMix(seed ^ 0x5EED),
        }
    }

    fn screen(&self) -> [f32; 2] {
        [self.surface_cfg.width as f32, self.surface_cfg.height as f32]
    }

    fn reconfigure(&mut self) {
        self.surface_cfg.present_mode =
            if self.ui.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        self.surface.configure(&self.gpu.device, &self.surface_cfg);
    }

    fn redraw(&mut self) {
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
        if self.ui.reseed {
            self.ui.reseed = false;
            self.seed += 1;
            self.ui.baseline = self.world.seed_matter(&self.gpu, self.seed);
            self.ui.ledger = None;
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
                log::warn!("sem textura da surface: {other:?}");
                return;
            }
        };
        let target = tex.texture.create_view(&Default::default());
        let screen = self.screen();

        // UI primeiro (pode mudar params e vista neste frame).
        let raw = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();
        let mut out = ctx.run_ui(raw, |root| {
            ui::draw(root.ctx(), &mut self.ui, &mut self.world, &mut self.profiler);
            ui::inspector::draw(root.ctx(), &mut self.inspector);
        });
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
        self.view.update(&self.gpu.queue, &self.cam, screen, self.ui.view_mode, self.ui.monomer_brightness);

        let Running { gpu, world, view, egui_renderer, profiler, ui: st, inspector, .. } = self;
        let mut frame = profiler.begin(&gpu.device, &gpu.queue);
        if !st.paused {
            let n = st.steps_per_frame;
            frame.segment("world", |enc| world.encode_steps(&gpu.queue, enc, n));
        }
        frame.segment("ledger", |enc| world.encode_ledger_readback(enc));
        frame.segment("draw list", |enc| world.encode_draw_list(enc));
        frame.segment("inspect", |enc| {
            inspector.encode(world, enc);
            inspector.encode_preview(&gpu.queue, enc);
        });

        let mut egui_enc = gpu.device.create_command_encoder(&Default::default());
        let mut extra = egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut egui_enc, &jobs, &sd);
        extra.push(egui_enc.finish());
        frame.submit_now(extra);

        frame.segment("render", |enc| {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            view.draw(&mut pass);
            egui_renderer.render(&mut pass, &jobs, &sd);
        });
        frame.finish();
        world.ledger_after_submit();
        inspector.after_submit();

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
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.surface_cfg.width = size.width.max(1);
                self.surface_cfg.height = size.height.max(1);
                self.reconfigure();
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                let p = [position.x as f32, position.y as f32];
                if self.dragging {
                    self.cam.pan_pixels([p[0] - self.cursor[0], p[1] - self.cursor[1]]);
                    let (dx, dy) = (p[0] - self.press_pos[0], p[1] - self.press_pos[1]);
                    if dx * dx + dy * dy > 16.0 {
                        self.press_moved = true;
                    }
                }
                self.cursor = p;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pan_button = matches!(button, MouseButton::Left | MouseButton::Right | MouseButton::Middle);
                if pan_button {
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
                        let w = self.cam.screen_to_world(self.cursor, self.screen());
                        self.inspector.pick(&self.gpu, &self.world, w);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_mouse => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                let screen = self.screen();
                self.cam.zoom_at(1.15f32.powf(lines), self.cursor, screen);
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
