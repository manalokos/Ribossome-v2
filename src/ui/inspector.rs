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
/// agente 64 + genoma 64 + corpo 64 + órgãos 128 + posições 512 + ligações 64
/// + sinais 1024.
const READ_BYTES: u64 = 1920;
/// Candidatos (os de centro mais próximo) a que o clique mede a distância
/// resíduo a resíduo.
const PICK_CANDIDATES: usize = 48;
/// Folga à volta do corpo na imagem (órgãos com antenas, espessura), em unidades do mundo.
const PREVIEW_MARGIN: f32 = 40.0;
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
    /// Posição de cada resíduo no referencial do corpo.
    pub body_pos: Vec<[f32; 2]>,
    /// Ids dos agentes a que está ligado por âncoras (e se é de nascimento).
    pub bonds: Vec<(u32, bool)>,
    /// Sinais internos [α, β, γ, δ] de cada resíduo.
    pub signals: Vec<[f32; 4]>,
}

impl InspectData {
    /// Caixa do corpo no mundo: (centro, maior lado).
    fn bounds(&self) -> ([f32; 2], f32) {
        let a = &self.agent;
        let (s, c) = a.rot.sin_cos();
        let (mut lo, mut hi) = ([a.pos_x, a.pos_y], [a.pos_x, a.pos_y]);
        for (i, p) in self.body_pos.iter().enumerate() {
            let w = [a.pos_x + c * p[0] - s * p[1], a.pos_y + s * p[0] + c * p[1]];
            if i == 0 {
                (lo, hi) = (w, w);
            }
            lo = [lo[0].min(w[0]), lo[1].min(w[1])];
            hi = [hi[0].max(w[0]), hi[1].max(w[1])];
        }
        ([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5], (hi[0] - lo[0]).max(hi[1] - lo[1]))
    }
}

enum Readback {
    Idle,
    Encoded,
    Mapping(Arc<AtomicBool>),
}

/// Uma leitura pequena (só a posição) do agente selecionado. Há três, em
/// roda: pede-se uma por frame e cada uma chega um ou dois frames depois,
/// por isso há uma posição nova em todos os frames.
struct PosRead {
    buf: wgpu::Buffer,
    state: Readback,
    /// Frame e id do agente a que o pedido diz respeito.
    frame: u64,
    id: u32,
}

/// Última posição lida do selecionado e a sua velocidade (por frame).
#[derive(Clone, Copy)]
struct Track {
    id: u32,
    frame: u64,
    pos: [f32; 2],
    vel: [f32; 2],
}

pub struct Inspector {
    pos_reads: Vec<PosRead>,
    frame: u64,
    track: Option<Track>,
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
        let pos_reads = (0..3)
            .map(|_| PosRead {
                buf: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("inspector pos"),
                    size: 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Readback::Idle,
                frame: 0,
                id: 0,
            })
            .collect();
        Self {
            pos_reads,
            frame: 0,
            track: None,
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
    /// distância (leitura síncrona: só ao clicar). A distância é ao RESÍDUO
    /// mais próximo (clicar na cauda de um corpo comprido apanha-o a ele e
    /// não a um pequeno ao lado), medida nos PICK_CANDIDATES de centro mais
    /// próximo.
    pub fn pick(&mut self, gpu: &Gpu, world: &World, p: [f32; 2]) {
        let agents = world.read_agents_blocking(gpu);
        let mut near: Vec<(usize, f32)> = agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.alive != 0)
            .map(|(s, a)| (s, (a.pos_x - p[0]).powi(2) + (a.pos_y - p[1]).powi(2)))
            .collect();
        let n = near.len().min(PICK_CANDIDATES);
        if n > 0 && n < near.len() {
            near.select_nth_unstable_by(n - 1, |x, y| x.1.total_cmp(&y.1));
        }
        near.truncate(n);
        let ranges: Vec<(u64, u64)> = near.iter().map(|&(s, _)| (s as u64 * 512, 512)).collect();
        let raw = gpu.read_ranges_blocking(&world.body_pos_buf, &ranges);
        let pos: &[[f32; 2]] = bytemuck::cast_slice(&raw);
        let best = near
            .iter()
            .enumerate()
            .map(|(i, &(s, d2))| {
                let a = &agents[s];
                let (sn, cs) = a.rot.sin_cos();
                // Ponto do clique no referencial do corpo.
                let (dx, dy) = (p[0] - a.pos_x, p[1] - a.pos_y);
                let q = [cs * dx + sn * dy, -sn * dx + cs * dy];
                let body = &pos[i * 64..i * 64 + (a.body_len as usize).min(64)];
                let d = body.iter().map(|r| (r[0] - q[0]).powi(2) + (r[1] - q[1]).powi(2)).fold(d2, f32::min);
                (s, d, a)
            })
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
    /// Posição do selecionado para ESTE frame: a última lida, avançada pela
    /// velocidade o número de frames que a leitura demorou a chegar (no
    /// máximo 4). Para a mira não andar aos saltos.
    pub fn tracked_pos(&self) -> Option<[f32; 2]> {
        let (t, sel) = (self.track?, self.selected?);
        if t.id != sel.id {
            return None;
        }
        let lag = (self.frame.saturating_sub(t.frame)).min(4) as f32;
        Some([t.pos[0] + t.vel[0] * lag, t.pos[1] + t.vel[1] * lag])
    }

    pub fn encode(&mut self, world: &World, enc: &mut wgpu::CommandEncoder) {
        // Posição em todos os frames (8 bytes), numa das três leituras livres.
        self.frame += 1;
        if let Some(sel) = self.selected
            && let Some(r) = self.pos_reads.iter_mut().find(|r| matches!(r.state, Readback::Idle))
        {
            enc.copy_buffer_to_buffer(&world.agents_buf, sel.slot as u64 * 64, &r.buf, 0, 8);
            r.state = Readback::Encoded;
            r.frame = self.frame;
            r.id = sel.id;
        }
        let (Some(sel), Readback::Idle) = (self.selected, &self.state) else { return };
        let s = sel.slot as u64;
        enc.copy_buffer_to_buffer(&world.agents_buf, s * 64, &self.staging, 0, 64);
        enc.copy_buffer_to_buffer(&world.genomes_buf, s * 64, &self.staging, 64, 64);
        enc.copy_buffer_to_buffer(&world.bodies_buf, s * 64, &self.staging, 128, 64);
        enc.copy_buffer_to_buffer(&world.organs_buf, s * 128, &self.staging, 192, 128);
        enc.copy_buffer_to_buffer(&world.body_pos_buf, s * 512, &self.staging, 320, 512);
        // As 4 ligações (a 5.ª entrada do slot é a proposta do passo).
        enc.copy_buffer_to_buffer(&world.bonds_buf, s * crate::world::BOND_STRIDE * 16, &self.staging, 832, 64);
        enc.copy_buffer_to_buffer(&world.signals_buf, s * 1024, &self.staging, 896, 1024);
        self.state = Readback::Encoded;
    }

    /// Imagem ampliada do selecionado, sozinho (câmara no centro de massa).
    pub fn encode_preview(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder) {
        let (Some(sel), Some(d)) = (self.selected, &self.data) else { return };
        // Enquadra o corpo TODO como está agora (o raio do agente é o de
        // giração ao nascer: cortava as pontas dos corpos compridos).
        let (center, side) = d.bounds();
        let cam = Camera { center, zoom: PREVIEW_SIZE as f32 / (side + 2.0 * PREVIEW_MARGIN).max(40.0) };
        // O agente desenha-se na posição que tem AGORA na GPU; só o desvio
        // do centro da caixa do corpo vem desta leitura (atrasada).
        self.preview.view.focus_offset.set([center[0] - d.agent.pos_x, center[1] - d.agent.pos_y]);
        self.preview.view.focus.set(sel.slot);
        self.preview.encode(queue, enc, &cam, 0, 0.2);
    }

    pub fn after_submit(&mut self) {
        for r in &mut self.pos_reads {
            if let Readback::Encoded = r.state {
                let ready = Arc::new(AtomicBool::new(false));
                let flag = ready.clone();
                r.buf.map_async(wgpu::MapMode::Read, .., move |res| {
                    if res.is_ok() {
                        flag.store(true, Ordering::Release);
                    }
                });
                r.state = Readback::Mapping(ready);
            }
        }
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
        device.poll(wgpu::PollType::Poll).ok();
        for r in &mut self.pos_reads {
            let Readback::Mapping(ready) = &r.state else { continue };
            if !ready.load(Ordering::Acquire) {
                continue;
            }
            let bytes = r.buf.get_mapped_range(..).map(|v| v.to_vec()).unwrap_or_default();
            r.buf.unmap();
            r.state = Readback::Idle;
            if bytes.len() != 8 {
                continue;
            }
            let pos: [f32; 2] = *bytemuck::from_bytes(&bytes);
            // Só conta se for mais recente do que a que já se tem.
            match self.track {
                Some(t) if t.id == r.id && r.frame <= t.frame => {}
                Some(t) if t.id == r.id => {
                    let df = (r.frame - t.frame) as f32;
                    self.track = Some(Track { id: r.id, frame: r.frame, pos, vel: [(pos[0] - t.pos[0]) / df, (pos[1] - t.pos[1]) / df] });
                }
                _ => self.track = Some(Track { id: r.id, frame: r.frame, pos, vel: [0.0; 2] }),
            }
        }
        let Readback::Mapping(ready) = &self.state else { return };
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
        let pw: &[[f32; 2]] = bytemuck::cast_slice(&bytes[320..832]);
        let body_pos = pw[..(agent.body_len as usize).min(64)].to_vec();
        let lw: &[[u32; 4]] = bytemuck::cast_slice(&bytes[832..896]);
        let bonds = lw.iter().filter(|b| b[0] != u32::MAX).map(|b| (b[1], b[2] >> 16 != 0)).collect();
        let sw: &[[f32; 4]] = bytemuck::cast_slice(&bytes[896..1920]);
        let signals = sw[..(agent.body_len as usize).min(64)].to_vec();
        self.data = Some(InspectData { agent, genome, body, organs, body_pos, bonds, signals });
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

/// SINAIS INTERNOS: uma barra por canal, com um segmento por resíduo (ponta N
/// à esquerda). Cor cheia = sinal forte; as duas cores de cada canal são o
/// sinal positivo e o negativo (as mesmas da vista "sinal α" / "sinal β").
/// Por cima, um traço branco marca os resíduos com órgão.
fn signal_bars(ui: &mut egui::Ui, d: &InspectData) {
    let n = d.signals.len();
    if n == 0 {
        return;
    }
    const NAMES: [&str; 4] = ["α", "β", "γ", "δ"];
    const POS: [[f32; 3]; 4] = [[1.0, 0.45, 0.1], [0.3, 1.0, 0.3], [1.0, 0.25, 0.3], [0.3, 0.9, 0.9]];
    const NEG: [[f32; 3]; 4] = [[0.1, 0.6, 1.0], [0.95, 0.3, 0.9], [0.35, 0.45, 1.0], [0.9, 0.8, 0.2]];
    let (label_w, row_h, gap) = (16.0, 12.0, 3.0);
    let width = ui.available_width().min(340.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 6.0 + 4.0 * (row_h + gap)), egui::Sense::hover());
    let p = ui.painter_at(rect);
    let cell = (width - label_w) / n as f32;
    let x0 = rect.left() + label_w;
    for (k, o) in d.organs.iter().enumerate().take(n) {
        if *o != 0 {
            let x = x0 + k as f32 * cell;
            p.rect_filled(egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2((cell - 1.0).max(1.0), 3.0)), 0.0, egui::Color32::WHITE);
        }
    }
    for ch in 0..4 {
        let y = rect.top() + 6.0 + ch as f32 * (row_h + gap);
        p.text(egui::pos2(rect.left(), y + row_h * 0.5), egui::Align2::LEFT_CENTER, NAMES[ch], egui::FontId::proportional(12.0), egui::Color32::GRAY);
        for (k, s) in d.signals.iter().enumerate() {
            let v = s[ch];
            // Os modos 0 e 4 limitam o sinal a ±1; os outros a ±4: a tanh mostra bem os dois.
            let t = v.abs().tanh().clamp(0.0, 1.0);
            let c = if v >= 0.0 { POS[ch] } else { NEG[ch] };
            let base = 0.13;
            let col = egui::Color32::from_rgb(
                (255.0 * (base + (c[0] - base) * t)) as u8,
                (255.0 * (base + (c[1] - base) * t)) as u8,
                (255.0 * (base + (c[2] - base) * t)) as u8,
            );
            p.rect_filled(egui::Rect::from_min_size(egui::pos2(x0 + k as f32 * cell, y), egui::vec2((cell - 1.0).max(1.0), row_h)), 0.0, col);
        }
    }
    ui.small("signals per residue (N end on the left): α orange + / blue −, β green + / magenta −, γ red + / blue −, δ cyan + / yellow −; white mark = organ");
}

/// O inspetor, na barra fixa da direita (só existe com um organismo escolhido).
pub fn panel(ui: &mut egui::Ui, ins: &mut Inspector, organ_table: &[crate::life::table::OrganRow], amino: &[crate::life::table::AminoRow]) {
    let mut open = true;
    ui.horizontal(|ui| {
        ui.heading("Inspector");
        if ins.data.is_some() && ui.button("release").on_hover_text("stop following this organism").clicked() {
            open = false;
        }
    });
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        if ins.dead {
            ui.colored_label(egui::Color32::LIGHT_RED, "the selected organism died");
        }
        let Some(d) = &ins.data else {
            ui.label("click an organism (left click without dragging)");
            return;
        };
        let a = &d.agent;
        if let Some(t) = ins.preview_tex {
            ui.image((t, egui::vec2(256.0, 256.0)));
        }
        ui.checkbox(&mut ins.follow, "camera follows this organism");
        signal_bars(ui, d);
        egui::Grid::new("ins").striped(true).show(ui, |ui| {
            let mut row = |k: &str, v: String| {
                ui.label(k);
                ui.label(v);
                ui.end_row();
            };
            row(
                "id",
                format!("{}  (parent {})", a.id, if a.parent == u32::MAX { "—".into() } else { a.parent.to_string() }),
            );
            row("generation", a.generation.to_string());
            if !d.bonds.is_empty() {
                let list: Vec<String> = d.bonds.iter().map(|(id, birth)| format!("{id}{}", if *birth { " (birth)" } else { " (contact)" })).collect();
                row("bonded to", list.join(", "));
            }
            row("age", format!("{} steps", a.age));
            // Capacidade (energy_capacity em lifecycle.wgsl): o volume dos
            // aminoácidos do corpo (0,5 por resíduo médio) + os órgãos de
            // armazenamento (capacidade da variante × intensidade).
            let body_cap = 0.5 * d.body.iter().map(|&aa| amino.get(aa as usize).map_or(0.0, |r| r.volume)).sum::<f32>() / CAP_VOLUME_REF;
            let mut store = 0.0f32;
            let (mut run, mut run_cap) = (0.0f32, 0.0f32);
            let merge = |run: f32, cap: f32| if run > 0.5 { cap * (1.0 + 0.25 * (run.min(8.0) - 1.0)) } else { 0.0 };
            for &o in &d.organs {
                if o != 0 && (o & 0x1F) - 1 == 7 {
                    let v = organ_table.get(7).and_then(|r| r.variantes.get(((o >> 5) & 0x7) as usize)).and_then(|v| v.get("capacidade")).copied().unwrap_or(0.0);
                    run += 1.0;
                    run_cap += v.max(0.0) * crate::life::organs::organ_gain((o >> 8) as u8).clamp(0.5, 2.0);
                } else {
                    store += merge(run, run_cap);
                    (run, run_cap) = (0.0, 0.0);
                }
            }
            store += merge(run, run_cap);
            let cap = (body_cap + store).max(1.0);
            row("energy", format!("{:.2} / {:.1}", a.energy, cap));
            row("copy", format!("{} / {} bases", a.pair_count, a.gene_len));
            row(
                "body",
                if a.body_len == 0 { "naked RNA (non-coding)".into() } else { format!("{} residues", a.body_len) },
            );
            row("radius", format!("{:.1}", a.radius));
        });
        ui.separator();
        ui.strong(format!("Genome ({} bases)", d.genome.len()));
        colored_seq(ui, d.genome.iter().map(|&b| (BASES[b as usize], base_color(b))));
        if !d.body.is_empty() {
            ui.strong(format!("Protein ({} residues; organs in white)", d.body.len()));
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
                        "{}  position {k}: {}",
                        ORGAN_SYMBOLS[t as usize],
                        describe(t, ((o >> 5) & 0x7) as u8, (o >> 8) as u8, organ_table)
                    )
                })
                .collect();
            if list.is_empty() {
                ui.label("no organs");
            } else {
                ui.strong("Organs (symbol in the protein, position: what it does)");
                for l in list {
                    ui.label(l);
                }
                ui.small("position 0 = N end (start of the protein); left/right = sides of the chain from N to C");
            }
        }
    });
    ins.open = open;
    if !open {
        ins.deselect();
    }
}
