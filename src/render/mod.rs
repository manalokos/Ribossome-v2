//! Render: vista do mundo e câmara. (Aminoácidos instanciados na fase 4.)

pub mod capture;

use crate::params::{ViewParams, WorldConfig};
use crate::shaders;
use crate::world::World;

/// Câmara em unidades do MUNDO. Cima no ecrã = +y no mundo.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub center: [f32; 2],
    /// Píxeis por unidade do mundo.
    pub zoom: f32,
}

impl Camera {
    pub fn fit(cfg: &WorldConfig, screen: [f32; 2]) -> Self {
        let s = cfg.sim_size();
        Self { center: [s * 0.5, s * 0.5], zoom: screen[0].min(screen[1]) / s * 0.95 }
    }

    pub fn screen_to_world(&self, p: [f32; 2], screen: [f32; 2]) -> [f32; 2] {
        [self.center[0] + (p[0] - 0.5 * screen[0]) / self.zoom, self.center[1] - (p[1] - 0.5 * screen[1]) / self.zoom]
    }

    /// Arrasta a vista por um deslocamento em píxeis.
    pub fn pan_pixels(&mut self, d: [f32; 2]) {
        self.center[0] -= d[0] / self.zoom;
        self.center[1] += d[1] / self.zoom;
    }

    /// Zoom à volta de um ponto do ecrã (que fica fixo).
    pub fn zoom_at(&mut self, factor: f32, p: [f32; 2], screen: [f32; 2]) {
        let before = self.screen_to_world(p, screen);
        self.zoom = (self.zoom * factor).clamp(1e-4, 100.0);
        let after = self.screen_to_world(p, screen);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }
}

pub struct WorldView {
    view_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    agents_bg: wgpu::BindGroup,
    agents_pipeline: wgpu::RenderPipeline,
    draw_args: wgpu::Buffer,
    /// Slot a desenhar sozinho (u32::MAX = todos).
    pub focus: std::cell::Cell<u32>,
    /// Profundidade ótica da água (copiada de SimParams antes de desenhar).
    pub uv_depth: std::cell::Cell<f32>,
    /// Fração do sol (escurece a vista de noite).
    pub daylight: std::cell::Cell<f32>,
    /// Órgão a marcar no mapa: tipo + 1 (0 = nenhum).
    pub mark_organ: std::cell::Cell<u32>,
    /// Raio do círculo de confusão dos monómeros (células; 0 = quadrados).
    pub coc_radius: std::cell::Cell<f32>,
    /// Canto do viewport no alvo (píxeis); 0,0 quando ocupa o alvo todo.
    pub origin: std::cell::Cell<[f32; 2]>,
}

impl WorldView {
    pub fn new(device: &wgpu::Device, world: &World, format: wgpu::TextureFormat) -> Self {
        let cfg = &world.cfg;
        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view params"),
            size: size_of::<ViewParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world view layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world view bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: world.chem_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: world.gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: world.light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: world.temp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: world.velocity_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: world.redox_buf.as_entire_binding() },
            ],
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world view pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let def = &shaders::WORLD_VIEW;
        let module = shaders::create(device, def, cfg);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("world view"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(shaders::entry(def, "vs_fullscreen")),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(shaders::entry(def, "fs_world")),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        // Agentes: quadrados instanciados (um por resíduo).
        let vertex_storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let agents_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agents view layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                vertex_storage(1),
                vertex_storage(2),
                vertex_storage(3),
                vertex_storage(4),
                vertex_storage(5),
                vertex_storage(6),
                vertex_storage(7),
                vertex_storage(8),
                vertex_storage(9),
                vertex_storage(10),
                vertex_storage(11),
                vertex_storage(12),
                vertex_storage(13),
            ],
        });
        let agents_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agents view bg"),
            layout: &agents_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: world.agents_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: world.bodies_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: world.body_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: world.draw_list_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: world.organs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: world.signals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: world.aa_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: world.genomes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: world.tail_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: world.variant_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: world.bonds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: world.kin_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: world.contact_disp_buf.as_entire_binding() },
            ],
        });
        let agents_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agents view pipeline layout"),
            bind_group_layouts: &[Some(&agents_layout)],
            immediate_size: 0,
        });
        let adef = &shaders::AGENTS_VIEW;
        let amodule = shaders::create(device, adef, cfg);
        let agents_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("agents view"),
            layout: Some(&agents_pl_layout),
            vertex: wgpu::VertexState {
                module: &amodule,
                entry_point: Some(shaders::entry(adef, "vs_agent")),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &amodule,
                entry_point: Some(shaders::entry(adef, "fs_agent")),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            view_buf,
            bind_group,
            pipeline,
            agents_bg,
            agents_pipeline,
            draw_args: world.draw_args_buf.clone(),
            focus: std::cell::Cell::new(u32::MAX),
            uv_depth: std::cell::Cell::new(11.0),
            daylight: std::cell::Cell::new(1.0),
            mark_organ: std::cell::Cell::new(0),
            origin: std::cell::Cell::new([0.0; 2]),
            coc_radius: std::cell::Cell::new(0.35),
        }
    }

    pub fn update(
        &self,
        queue: &wgpu::Queue,
        cam: &Camera,
        screen: [f32; 2],
        view_mode: u32,
        brightness: f32,
        signal_view: u32,
    ) {
        let p = ViewParams {
            center_x: cam.center[0],
            center_y: cam.center[1],
            zoom: cam.zoom,
            view_mode,
            screen_w: screen[0],
            screen_h: screen[1],
            monomer_brightness: brightness,
            focus_slot: self.focus.get(),
            signal_view,
            uv_depth: self.uv_depth.get(),
            daylight: self.daylight.get(),
            mark_organ: self.mark_organ.get(),
            coc_radius: self.coc_radius.get(),
            origin_x: self.origin.get()[0],
            origin_y: self.origin.get()[1],
            _pad_v2: 0,
        };
        queue.write_buffer(&self.view_buf, 0, bytemuck::bytes_of(&p));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.agents_pipeline);
        pass.set_bind_group(0, &self.agents_bg, &[]);
        // Só os vivos: a lista e o nº de instâncias vêm da GPU (build_draw_list).
        pass.draw_indirect(&self.draw_args, 0);
    }
}
