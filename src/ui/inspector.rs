//! Inspetor: clicar num organismo mostra o genoma, a proteína e uma imagem
//! ampliada dele sozinho, ao vivo. Os dados chegam por leitura assíncrona
//! (um frame ou mais de atraso), como o livro-razão: não trava a simulação.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::Gpu;
use crate::life::amino::{AA_LETTERS, BASES};
use crate::life::organs::{ORGAN_SYMBOLS, describe};
use crate::params::Agent;
use crate::render::Camera;
use crate::render::capture::Capture;
use crate::world::World;

/// Bytes lidos por agente: Agent (64) + genoma (64) + corpo (64) + órgãos (128).
const READ_BYTES: u64 = 320;
const PREVIEW_SIZE: u32 = 256;

#[derive(Clone, Copy)]
struct Selected {
    slot: u32,
    id: u32,
}

pub struct InspectData {
    pub agent: Agent,
    pub genome: Vec<u8>,
    pub body: Vec<u8>,
    /// Código de órgão por resíduo (0 = nenhum; (tipo + 1) | (parâmetro << 5) | (intensidade << 8)).
    pub organs: Vec<u16>,
}

enum Readback {
    Idle,
    Encoded,
    Mapping(Arc<AtomicBool>),
}

pub struct Inspector {
    selected: Option<Selected>,
    staging: wgpu::Buffer,
    state: Readback,
    pub data: Option<InspectData>,
    /// O organismo selecionado morreu (o slot foi libertado ou reutilizado).
    pub dead: bool,
    pub follow: bool,
    pub open: bool,
    preview: Capture,
    pub preview_tex: Option<egui::TextureId>,
}

impl Inspector {
    pub fn new(gpu: &Gpu, world: &World) -> Self {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inspector staging"),
            size: READ_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            selected: None,
            staging,
            state: Readback::Idle,
            data: None,
            dead: false,
            follow: false,
            open: false,
            preview: Capture::new(gpu, world, PREVIEW_SIZE),
            preview_tex: None,
        }
    }

    /// Regista a imagem de pré-visualização no egui (uma vez).
    pub fn register(&mut self, device: &wgpu::Device, renderer: &mut egui_wgpu::Renderer) {
        if self.preview_tex.is_none() {
            let view = self.preview.texture_view();
            self.preview_tex = Some(renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear));
        }
    }

    /// Escolhe SEMPRE o organismo vivo mais próximo do ponto, a qualquer
    /// distância (leitura síncrona: só ao clicar). A distância é à borda do
    /// corpo (centro − raio), para um corpo grande ganhar a um pequeno ao lado.
    pub fn pick(&mut self, gpu: &Gpu, world: &World, p: [f32; 2]) {
        let agents = world.read_agents_blocking(gpu);
        let best = agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.alive != 0)
            .map(|(s, a)| (s, (((a.pos_x - p[0]).powi(2) + (a.pos_y - p[1]).powi(2)).sqrt() - a.radius).max(0.0), a))
            .min_by(|x, y| x.1.total_cmp(&y.1));
        self.selected = best.map(|(s, _, a)| Selected { slot: s as u32, id: a.id });
        self.data = None;
        self.dead = false;
        self.open = self.selected.is_some() || self.open;
    }

    pub fn deselect(&mut self) {
        self.selected = None;
        self.data = None;
    }

    pub fn focus_slot(&self) -> Option<u32> {
        self.selected.map(|s| s.slot)
    }

    /// Grava a cópia dos dados do selecionado para o staging (se não houver leitura em curso).
    pub fn encode(&mut self, world: &World, enc: &mut wgpu::CommandEncoder) {
        let (Some(sel), Readback::Idle) = (self.selected, &self.state) else { return };
        let s = sel.slot as u64;
        enc.copy_buffer_to_buffer(&world.agents_buf, s * 64, &self.staging, 0, 64);
        enc.copy_buffer_to_buffer(&world.genomes_buf, s * 64, &self.staging, 64, 64);
        enc.copy_buffer_to_buffer(&world.bodies_buf, s * 64, &self.staging, 128, 64);
        enc.copy_buffer_to_buffer(&world.organs_buf, s * 128, &self.staging, 192, 128);
        self.state = Readback::Encoded;
    }

    /// Imagem ampliada do selecionado, sozinho (câmara no centro de massa).
    pub fn encode_preview(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder) {
        let (Some(sel), Some(d)) = (self.selected, &self.data) else { return };
        let a = &d.agent;
        let r = a.radius.max(12.0);
        let cam = Camera { center: [a.pos_x, a.pos_y], zoom: PREVIEW_SIZE as f32 / (r * 3.2) };
        self.preview.view.focus.set(sel.slot);
        self.preview.encode(queue, enc, &cam, 0, 0.2);
    }

    pub fn after_submit(&mut self) {
        if let Readback::Encoded = self.state {
            let ready = Arc::new(AtomicBool::new(false));
            let flag = ready.clone();
            self.staging.map_async(wgpu::MapMode::Read, .., move |r| {
                if r.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
            self.state = Readback::Mapping(ready);
        }
    }

    pub fn poll(&mut self, device: &wgpu::Device) {
        let Readback::Mapping(ready) = &self.state else { return };
        device.poll(wgpu::PollType::Poll).ok();
        if !ready.load(Ordering::Acquire) {
            return;
        }
        let bytes = self.staging.get_mapped_range(..).map(|v| v.to_vec()).unwrap_or_default();
        self.staging.unmap();
        self.state = Readback::Idle;
        let (Some(sel), true) = (self.selected, bytes.len() == READ_BYTES as usize) else { return };
        let agent: Agent = *bytemuck::from_bytes(&bytes[0..64]);
        if agent.alive == 0 || agent.id != sel.id {
            self.dead = true;
            self.selected = None;
            return;
        }
        let words: &[u32] = bytemuck::cast_slice(&bytes[64..128]);
        let genome = (0..agent.gene_len as usize).map(|i| ((words[i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
        let bw: &[u32] = bytemuck::cast_slice(&bytes[128..192]);
        let body = (0..agent.body_len as usize).map(|i| ((bw[i / 4] >> ((i % 4) * 8)) & 0xFF) as u8).collect();
        let ow: &[u32] = bytemuck::cast_slice(&bytes[192..320]);
        let organs = (0..agent.body_len as usize).map(|i| ((ow[i / 2] >> ((i % 2) * 16)) & 0xFFFF) as u16).collect();
        self.data = Some(InspectData { agent, genome, body, organs });
    }
}

/// Cores dos nucleótidos (teclas 1–4 do v3) e das classes de aminoácidos (as do desenho).
fn base_color(b: u8) -> egui::Color32 {
    match b {
        0 => egui::Color32::from_rgb(255, 70, 60),
        1 => egui::Color32::from_rgb(255, 215, 40),
        2 => egui::Color32::from_rgb(60, 230, 80),
        _ => egui::Color32::from_rgb(70, 130, 255),
    }
}

fn aa_color(aa: u8) -> egui::Color32 {
    let c = match aa {
        0 | 7 | 9 | 10 | 17 => [184, 184, 158],
        4 | 18 | 19 => [178, 115, 242],
        15 | 16 | 11 | 13 => [102, 217, 115],
        1 => [242, 230, 77],
        8 | 14 | 6 => [89, 140, 255],
        2 | 3 => [255, 89, 77],
        5 => [242, 242, 242],
        _ => [255, 153, 51],
    };
    egui::Color32::from_rgb(c[0], c[1], c[2])
}

fn colored_seq(ui: &mut egui::Ui, items: impl Iterator<Item = (char, egui::Color32)>) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        for (ch, col) in items {
            ui.label(egui::RichText::new(ch.to_string()).monospace().color(col));
        }
    });
}

/// Volume médio dos 20 aminoácidos (igual a CAP_VOLUME_REF no shader).
const CAP_VOLUME_REF: f32 = 141.26;

pub fn draw(ctx: &egui::Context, ins: &mut Inspector, organ_table: &[crate::life::table::OrganRow], amino: &[crate::life::table::AminoRow]) {
    if !ins.open {
        return;
    }
    let mut open = ins.open;
    egui::Window::new("Inspetor").open(&mut open).default_pos([980.0, 12.0]).default_width(360.0).show(ctx, |ui| {
        if ins.dead {
            ui.colored_label(egui::Color32::LIGHT_RED, "o organismo selecionado morreu");
        }
        let Some(d) = &ins.data else {
            ui.label("clica num organismo (clique esquerdo sem arrastar)");
            return;
        };
        let a = &d.agent;
        if let Some(t) = ins.preview_tex {
            ui.image((t, egui::vec2(256.0, 256.0)));
        }
        ui.checkbox(&mut ins.follow, "câmara segue este organismo");
        egui::Grid::new("ins").striped(true).show(ui, |ui| {
            let mut row = |k: &str, v: String| {
                ui.label(k);
                ui.label(v);
                ui.end_row();
            };
            row(
                "id",
                format!("{}  (pai {})", a.id, if a.parent == u32::MAX { "—".into() } else { a.parent.to_string() }),
            );
            row("geração", a.generation.to_string());
            row("idade", format!("{} passos", a.age));
            // Capacidade: o volume dos aminoácidos do corpo (em média 1 por resíduo).
            let cap = (d.body.iter().map(|&aa| amino.get(aa as usize).map_or(0.0, |r| r.volume)).sum::<f32>() / CAP_VOLUME_REF).max(1.0);
            row("energia", format!("{:.2} / {:.1}", a.energy, cap));
            row("cópia", format!("{} / {} bases", a.pair_count, a.gene_len));
            row(
                "corpo",
                if a.body_len == 0 { "RNA nu (não codifica)".into() } else { format!("{} resíduos", a.body_len) },
            );
            row("raio", format!("{:.1}", a.radius));
        });
        ui.separator();
        ui.strong(format!("Genoma ({} bases)", d.genome.len()));
        colored_seq(ui, d.genome.iter().map(|&b| (BASES[b as usize], base_color(b))));
        if !d.body.is_empty() {
            ui.strong(format!("Proteína ({} resíduos; órgãos a branco)", d.body.len()));
            colored_seq(
                ui,
                d.body.iter().zip(&d.organs).map(|(&aa, &o)| {
                    if o != 0 {
                        (ORGAN_SYMBOLS[((o & 0x1F) - 1) as usize], egui::Color32::WHITE)
                    } else {
                        (AA_LETTERS[aa as usize], aa_color(aa))
                    }
                }),
            );
            let list: Vec<String> = d
                .organs
                .iter()
                .enumerate()
                .filter(|(_, o)| **o != 0)
                .map(|(k, &o)| {
                    let t = ((o & 0x1F) - 1) as u8;
                    format!(
                        "{}  posição {k}: {}",
                        ORGAN_SYMBOLS[t as usize],
                        describe(t, ((o >> 5) & 0x7) as u8, (o >> 8) as u8, organ_table)
                    )
                })
                .collect();
            if list.is_empty() {
                ui.label("sem órgãos");
            } else {
                ui.strong("Órgãos (símbolo na proteína, posição: o que faz)");
                for l in list {
                    ui.label(l);
                }
                ui.small("posição 0 = ponta N (início da proteína); esquerdo/direito = lados da cadeia de N para C");
            }
        }
    });
    ins.open = open;
    if !open {
        ins.deselect();
    }
}
