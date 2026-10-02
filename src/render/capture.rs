//! Captura sem janela: desenha a vista do mundo numa textura e grava um PNG.
//! Serve para depurar (e, mais tarde, para snapshots).

use crate::gpu::Gpu;
use crate::render::{Camera, WorldView};
use crate::world::World;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct Capture {
    pub view: WorldView,
    size: u32,
    texture: wgpu::Texture,
    readback: wgpu::Buffer,
    /// Cor dos agentes (0 química, 1 α, 2 β, 3 α e β).
    pub signal_view: std::cell::Cell<u32>,
}

impl Capture {
    pub fn new(gpu: &Gpu, world: &World, size: u32) -> Self {
        let view = WorldView::new(&gpu.device, world, FORMAT);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture readback"),
            size: (size * size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { view, size, texture, readback, signal_view: std::cell::Cell::new(0) }
    }

    /// Vista da textura (para mostrar no egui, p. ex. no inspetor).
    pub fn texture_view(&self) -> wgpu::TextureView {
        self.texture.create_view(&Default::default())
    }

    /// Grava o desenho na textura (sem ler de volta). A lista de desenho dos
    /// agentes tem de estar feita (`World::encode_draw_list`).
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        cam: &Camera,
        view_mode: u32,
        brightness: f32,
    ) {
        let s = self.size as f32;
        self.view.update(queue, cam, [s, s], view_mode, brightness, self.signal_view.get());
        let target = self.texture.create_view(&Default::default());
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("capture"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.view.draw(&mut pass);
    }

    /// Desenha com a câmara dada e devolve RGBA (linha de cima primeiro).
    pub fn render(&self, gpu: &Gpu, world: &World, cam: &Camera, view_mode: u32, brightness: f32) -> Vec<u8> {
        let s = self.size as f32;
        self.view.uv_depth.set(world.params.uv_depth);
        self.view.update(&gpu.queue, cam, [s, s], view_mode, brightness, self.signal_view.get());
        let target = self.texture.create_view(&Default::default());
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_draw_list(&mut enc);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("capture"),
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
            });
            self.view.draw(&mut pass);
        }
        // size*4 é múltiplo de 256 para os tamanhos usados (>= 64 e potência de 2).
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size * 4),
                    rows_per_image: Some(self.size),
                },
            },
            wgpu::Extent3d { width: self.size, height: self.size, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([enc.finish()]);
        self.readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        gpu.wait_idle();
        let data = self.readback.get_mapped_range(..).expect("mapped").to_vec();
        self.readback.unmap();
        data
    }

    pub fn save_png(&self, rgba: &[u8], path: &std::path::Path) -> std::io::Result<()> {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut enc = png::Encoder::new(file, self.size, self.size);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(std::io::Error::other)?;
        w.write_image_data(rgba).map_err(std::io::Error::other)?;
        Ok(())
    }
}
