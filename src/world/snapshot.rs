//! CENAS GRAVADAS: o estado inteiro do mundo (matéria, terreno, fluido,
//! agentes vivos) e as definições, num ficheiro `.ribo`.
//!
//! Formato: "RIBOSCN1", u32 tamanho do cabeçalho, cabeçalho JSON, e blocos
//! (u16 tamanho do nome, nome, u64 tamanho, bytes lz4 com o tamanho à frente).
//! Os parâmetros vão por NOME: um parâmetro novo fica com o valor por omissão
//! e um que já não existe é ignorado (e dito no log).
//! As tabelas dos aminoácidos e dos órgãos NÃO se carregam: o programa usa
//! sempre as de `assets/` (a cena guarda uma cópia só para registo).
//! A luz, o declive e as grelhas de contacto refazem-se sozinhos.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::*;

const MAGIC: &[u8; 8] = b"RIBOSCN1";
const VERSION: u32 = 1;
/// Resíduos por slot nos buffers por resíduo.
const MAX_BODY: u64 = crate::life::amino::MAX_BODY as u64;
/// Palavras por agente nos buffers por slot.
const AGENT_WORDS: u64 = (size_of::<Agent>() / 4) as u64;
const TAIL_WORDS: u64 = 8;

/// Cópia de segmentos de u32 entre dois buffers (um fio por segmento):
/// compacta os slots vivos para gravar e espalha-os ao carregar.
const COPY_WGSL: &str = r#"
struct Seg { src: u32, dst: u32, len: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> segs: array<Seg>;
@group(0) @binding(1) var<storage, read> src_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_buf: array<u32>;

@compute @workgroup_size(64)
fn copy_segments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&segs)) { return; }
    let s = segs[i];
    for (var k = 0u; k < s.len; k++) {
        dst_buf[s.dst + k] = src_buf[s.src + k];
    }
}
"#;

pub(super) fn snap_buffers(
    device: &wgpu::Device,
    pressure: wgpu::Buffer,
    joint_angle: wgpu::Buffer,
    joint_base: wgpu::Buffer,
    joint_active: wgpu::Buffer,
    sensor_mem: wgpu::Buffer,
    bitten: wgpu::Buffer,
) -> SnapBuffers {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scene copy"),
        source: wgpu::ShaderSource::Wgsl(COPY_WGSL.into()),
    });
    let copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene copy"),
        entries: &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scene copy"),
        bind_group_layouts: &[Some(&copy_layout)],
        immediate_size: 0,
    });
    let copy_segments = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("copy_segments"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("copy_segments"),
        compilation_options: Default::default(),
        cache: None,
    });
    SnapBuffers { pressure, joint_angle, joint_base, joint_active, sensor_mem, bitten, copy_segments, copy_layout }
}

/// Um buffer por slot de agente: palavras por slot e se é por resíduo
/// (então só se gravam `body_len × por_residuo` palavras).
struct SlotBuf<'a> {
    name: &'static str,
    buf: &'a wgpu::Buffer,
    /// Palavras por slot (buffers por slot) ou por resíduo (por resíduo).
    words: u64,
    per_residue: bool,
}

/// Segmentos (origem, destino, tamanho) em palavras, para os slots dados.
/// `pack`: do buffer do mundo para o compacto; senão o contrário.
fn segments(b: &SlotBuf, slots: &[u32], body_len: &[u32], pack: bool) -> (Vec<[u32; 4]>, u64) {
    let mut segs = Vec::with_capacity(slots.len());
    let mut off = 0u64;
    for (i, &slot) in slots.iter().enumerate() {
        let (base, len) = if b.per_residue {
            (slot as u64 * MAX_BODY * b.words, body_len[i].min(MAX_BODY as u32) as u64 * b.words)
        } else {
            (slot as u64 * b.words, b.words)
        };
        if len > 0 {
            let (s, d) = if pack { (base, off) } else { (off, base) };
            segs.push([s as u32, d as u32, len as u32, 0]);
        }
        off += len;
    }
    (segs, off)
}

fn write_block(out: &mut Vec<u8>, name: &str, data: &[u8]) {
    let c = lz4_flex::compress_prepend_size(data);
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&(c.len() as u64).to_le_bytes());
    out.extend_from_slice(&c);
}

/// O que se lê do GPU antes de escrever (comprimir e escrever vai numa thread).
struct Raw {
    header: Value,
    blocks: Vec<(&'static str, Vec<u8>)>,
}

impl Raw {
    fn encode(self) -> Vec<u8> {
        let header = serde_json::to_vec_pretty(&self.header).unwrap();
        let mut out = Vec::with_capacity(64 << 20);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);
        for (name, data) in &self.blocks {
            write_block(&mut out, name, data);
        }
        out
    }
}

/// Uma cena lida do disco.
pub struct Scene {
    pub header: Value,
    blocks: std::collections::HashMap<String, Vec<u8>>,
}

impl Scene {
    pub fn read(path: &Path) -> Result<Self, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err(format!("{} não é uma cena do ribossome", path.display()));
        }
        let hl = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&bytes[12..12 + hl]).map_err(|e| format!("cabeçalho: {e}"))?;
        let mut blocks = std::collections::HashMap::new();
        let mut p = 12 + hl;
        while p < bytes.len() {
            let nl = u16::from_le_bytes(bytes[p..p + 2].try_into().unwrap()) as usize;
            let name = String::from_utf8_lossy(&bytes[p + 2..p + 2 + nl]).into_owned();
            p += 2 + nl;
            let cl = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
            p += 8;
            let data = lz4_flex::decompress_size_prepended(&bytes[p..p + cl]).map_err(|e| format!("bloco {name}: {e}"))?;
            p += cl;
            blocks.insert(name, data);
        }
        Ok(Self { header, blocks })
    }

    fn block(&self, name: &str) -> Result<&[u8], String> {
        self.blocks.get(name).map(|v| v.as_slice()).ok_or_else(|| format!("falta o bloco {name}"))
    }

    /// Um bloco binário de quem gravou (ver `save_scene`), se existir.
    pub fn extra_block(&self, name: &str) -> Option<&[u8]> {
        self.blocks.get(name).map(|v| v.as_slice())
    }

    pub fn epoch(&self) -> u32 {
        self.header["params"]["epoch"].as_f64().unwrap_or(0.0) as u32
    }
}

/// Configuração do mundo em JSON (tem de ser igual para carregar).
fn cfg_json(c: &WorldConfig) -> Value {
    json!({
        "grid_size": c.grid_size,
        "fluid_size": c.fluid_size,
        "world_units_per_cell": c.world_units_per_cell,
        "max_agents": c.max_agents,
    })
}

fn named_json(fields: Vec<(&'static str, f64)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect())
}

pub fn ledger_json(l: &Ledger) -> Value {
    json!({ "act": l.act, "spent": l.spent, "held": l.held })
}

pub fn ledger_from_json(v: &Value) -> Option<Ledger> {
    let arr = |k: &str| -> Option<[u32; 4]> {
        let a = v.get(k)?.as_array()?;
        let mut out = [0u32; 4];
        for (o, x) in out.iter_mut().zip(a) {
            *o = x.as_u64()? as u32;
        }
        Some(out)
    };
    Some(Ledger { act: arr("act")?, spent: arr("spent")?, held: arr("held")? })
}

impl World {
    fn slot_bufs(&self) -> Vec<SlotBuf<'_>> {
        let s = &self.snap;
        let per_slot = |name, buf, words| SlotBuf { name, buf, words, per_residue: false };
        let per_res = |name, buf, words| SlotBuf { name, buf, words, per_residue: true };
        vec![
            per_slot("agents", &self.agents_buf, AGENT_WORDS),
            per_slot("genomes", &self.genomes_buf, SLOT_WORDS),
            per_slot("bodies", &self.bodies_buf, SLOT_WORDS),
            per_slot("organs", &self.organs_buf, MAX_BODY / 2),
            per_slot("rna_tails", &self.tail_buf, TAIL_WORDS),
            per_slot("bitten", &s.bitten, 1),
            per_res("body_pos", &self.body_pos_buf, 2),
            per_res("signals", &self.signals_buf, 2),
            per_res("joint_angle", &s.joint_angle, 1),
            per_res("joint_base", &s.joint_base, 1),
            per_res("joint_state", &self.joint_state_buf, 1),
            per_res("joint_active", &s.joint_active, 1),
            per_res("sensor_mem", &s.sensor_mem, 1),
            // Ligações entre agentes por âncoras ("bonds2": o formato de
            // antes, por cargas, é ignorado; ver load_scene).
            per_slot("bonds2", &self.bonds_buf, super::BOND_STRIDE * 4),
        ]
    }

    /// Copia segmentos de `src` para `dst` na GPU (e espera).
    fn copy_segments(&self, gpu: &Gpu, segs: &[[u32; 4]], src: &wgpu::Buffer, dst: &wgpu::Buffer) {
        if segs.is_empty() {
            return;
        }
        let device = &gpu.device;
        let seg_buf = storage_buffer(device, "scene segments", (segs.len() * 16) as u64);
        gpu.queue.write_buffer(&seg_buf, 0, bytemuck::cast_slice(segs));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene copy"),
            layout: &self.snap.copy_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: seg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.snap.copy_segments);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups(segs.len() as u32, 64), 1, 1);
        }
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    }

    /// Lê da GPU tudo o que a cena precisa (bloqueia; ~1 s com 100 mil
    /// agentes). `extra` = estado da interface (vista, câmara, base do
    /// livro-razão). Devolve uma thread que comprime e escreve o ficheiro
    /// (primeiro num .tmp, depois troca: um ficheiro a meio nunca fica).
    /// `keep_previous`: o ficheiro que lá estava passa a `*.anterior.ribo`.
    /// `extra_blocks`: blocos binários de quem chama (p. ex. as estatísticas).
    pub fn save_scene(
        &self,
        gpu: &Gpu,
        path: PathBuf,
        extra: Value,
        extra_blocks: Vec<(&'static str, Vec<u8>)>,
        keep_previous: bool,
    ) -> std::thread::JoinHandle<Result<String, String>> {
        gpu.wait_idle();
        let agents = self.read_agents_blocking(gpu);
        let counters: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&self.life_counters_buf)).to_vec();
        let alive: Vec<(u32, u32)> =
            agents.iter().enumerate().filter(|(_, a)| a.alive != 0).map(|(i, a)| (i as u32, a.body_len)).collect();
        let slots: Vec<u32> = alive.iter().map(|a| a.0).collect();
        let body_len: Vec<u32> = alive.iter().map(|a| a.1).collect();
        drop(agents);

        let mut blocks: Vec<(&'static str, Vec<u8>)> = Vec::new();
        blocks.push(("slots", bytemuck::cast_slice(&slots).to_vec()));
        blocks.push(("life_counters", bytemuck::cast_slice(&counters).to_vec()));
        for b in self.slot_bufs() {
            let (segs, words) = segments(&b, &slots, &body_len, true);
            let data = if words == 0 {
                Vec::new()
            } else {
                let compact = storage_buffer(&gpu.device, "scene compact", words * 4);
                self.copy_segments(gpu, &segs, b.buf, &compact);
                gpu.read_buffer_blocking(&compact)
            };
            blocks.push((b.name, data));
        }
        blocks.push(("chem", gpu.read_buffer_blocking(&self.chem_buf)));
        blocks.push(("gamma", gpu.read_buffer_blocking(&self.gamma_buf)));
        blocks.push(("velocity", gpu.read_buffer_blocking(&self.velocity_buf)));
        blocks.push(("pressure", gpu.read_buffer_blocking(&self.snap.pressure)));
        blocks.push(("temperature", gpu.read_buffer_blocking(&self.temp_buf)));
        blocks.push(("redox", gpu.read_buffer_blocking(&self.redox_buf)));
        if let Some((g, h)) = &self.custom_terrain {
            blocks.push(("custom_gamma", bytemuck::cast_slice(g).to_vec()));
            blocks.push(("custom_heat", bytemuck::cast_slice(h).to_vec()));
        }
        if let Some(h) = &self.heat_image {
            blocks.push(("heat_image", bytemuck::cast_slice(h).to_vec()));
        }
        if let Some(c) = &self.custom_chem {
            blocks.push(("custom_chem", bytemuck::cast_slice(c).to_vec()));
        }
        if let Some(c) = &self.chem_image {
            blocks.push(("chem_image", bytemuck::cast_slice(c).to_vec()));
        }
        blocks.extend(extra_blocks);

        let header = json!({
            "versao": VERSION,
            "gravado": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            "agentes": slots.len(),
            "cfg": cfg_json(&self.cfg),
            "params": named_json(self.params.to_named()),
            "settings": serde_json::to_value(self.settings).unwrap(),
            "fumarolas": self.fumaroles.iter().map(|f| named_json(f.to_named())).collect::<Vec<_>>(),
            "ganho_fumarolas": self.fumarole_gain,
            "densidade_sementeira": self.seed_density,
            "fracao_ativada_sementeira": self.seed_active,
            // Só para registo: ao carregar usam-se sempre as de assets/.
            "tabela_aminoacidos": serde_json::to_value(&self.amino).unwrap(),
            "tabela_orgaos": serde_json::to_value(&self.organ_table).unwrap(),
            "interface": extra,
        });
        let raw = Raw { header, blocks };
        std::thread::spawn(move || {
            let t = std::time::Instant::now();
            let bytes = raw.encode();
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            }
            let tmp = path.with_extension("ribo.tmp");
            let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
            f.write_all(&bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
            drop(f);
            if keep_previous && path.exists() {
                let _ = std::fs::rename(&path, path.with_extension("anterior.ribo"));
            }
            std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(format!(
                "{} ({:.1} MB, {:.1} s a comprimir e escrever)",
                path.display(),
                bytes.len() as f64 / 1e6,
                t.elapsed().as_secs_f32()
            ))
        })
    }

    /// Carrega uma cena: matéria, terreno, fluido, agentes, parâmetros e
    /// definições. A configuração do mundo (tamanhos) tem de ser a mesma.
    /// Devolve o estado da interface gravado e avisos (para o log).
    pub fn load_scene(&mut self, gpu: &Gpu, scene: &Scene) -> Result<(Value, Vec<String>), String> {
        let h = &scene.header;
        if h["cfg"] != cfg_json(&self.cfg) {
            return Err(format!("a cena é de outro tamanho de mundo ({} ≠ {})", h["cfg"], cfg_json(&self.cfg)));
        }
        let mut notes = Vec::new();
        // Verifica os blocos antes de mudar o que quer que seja.
        let slots: Vec<u32> = bytemuck::pod_collect_to_vec(scene.block("slots")?);
        let counters: Vec<u32> = bytemuck::pod_collect_to_vec(scene.block("life_counters")?);
        let cells = self.cfg.cells() as usize;
        let fcells = self.cfg.fluid_cells() as usize;
        let expect = [
            ("chem", cells * 16),
            ("gamma", cells * 4),
            ("velocity", fcells * 8),
            ("pressure", fcells * 4),
            ("temperature", fcells * 4),
        ];
        for (name, size) in expect {
            if scene.block(name)?.len() != size {
                return Err(format!("bloco {name} com tamanho errado"));
            }
        }
        let agent_words = scene.block("agents")?;
        if agent_words.len() != slots.len() * size_of::<Agent>() || counters.len() != 8 {
            return Err("agentes com tamanho errado".into());
        }
        let agents: Vec<Agent> = bytemuck::pod_collect_to_vec(agent_words);
        if slots.iter().any(|&s| s >= self.cfg.max_agents) {
            return Err("slot fora do mundo".into());
        }
        let body_len: Vec<u32> = agents.iter().map(|a| a.body_len).collect();

        // Parâmetros e definições.
        let mut params = SimParams { seed: self.params.seed, ..Default::default() };
        if let Some(obj) = h["params"].as_object() {
            for (k, v) in obj {
                if !params.set_named(k, v.as_f64().unwrap_or(0.0)) {
                    notes.push(format!("parâmetro {k} já não existe (ignorado)"));
                }
            }
            let missing: Vec<&str> =
                params.to_named().iter().map(|(k, _)| *k).filter(|k| !obj.contains_key(*k)).collect();
            if !missing.is_empty() {
                notes.push(format!("parâmetros novos, com o valor por omissão: {}", missing.join(", ")));
            }
        }
        self.params = params;
        self.settings = serde_json::from_value(h["settings"].clone()).unwrap_or_default();
        self.fumaroles = h["fumarolas"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|f| {
                        let mut out = Fumarole::v3_default();
                        for (k, v) in f.as_object().into_iter().flatten() {
                            out.set_named(k, v.as_f64().unwrap_or(0.0));
                        }
                        out
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.fumarole_gain = h["ganho_fumarolas"].as_f64().unwrap_or(1.0) as f32;
        self.seed_density = h["densidade_sementeira"].as_f64().map_or(SEED_DENSITY_DEFAULT, |v| v as f32);
        self.seed_active = h["fracao_ativada_sementeira"].as_f64().map_or(super::SEED_ACTIVE_DEFAULT, |v| v as f32);
        // Compara depois de ler (o JSON arredonda os f32 de outra maneira).
        let same_amino = serde_json::from_value::<Vec<crate::life::table::AminoRow>>(h["tabela_aminoacidos"].clone())
            .is_ok_and(|t| serde_json::to_value(t).ok() == serde_json::to_value(&self.amino).ok());
        let same_organs = serde_json::from_value::<Vec<crate::life::table::OrganRow>>(h["tabela_orgaos"].clone())
            .is_ok_and(|t| serde_json::to_value(t).ok() == serde_json::to_value(&self.organ_table).ok());
        if !same_amino || !same_organs {
            notes.push("as tabelas de assets/ mudaram desde a gravação (uso as de assets/)".into());
        }
        let grid_u32 = |name: &str| -> Option<Vec<u32>> {
            scene.blocks.get(name).filter(|b| b.len() == cells * 4).map(|b| bytemuck::pod_collect_to_vec(b))
        };
        let grid_f32 = |name: &str| -> Option<Vec<f32>> {
            scene.blocks.get(name).filter(|b| b.len() == cells * 4).map(|b| bytemuck::pod_collect_to_vec(b))
        };
        self.custom_terrain = grid_u32("custom_gamma").zip(grid_f32("custom_heat"));
        self.heat_image = grid_f32("heat_image");
        // Cenas de antes do canal verde não têm estes blocos: a química segue o calor.
        self.custom_chem = grid_f32("custom_chem");
        self.chem_image = grid_f32("chem_image");
        self.heat_key.clear();

        // Grelhas.
        let q = &gpu.queue;
        q.write_buffer(&self.chem_buf, 0, scene.block("chem")?);
        q.write_buffer(&self.gamma_buf, 0, scene.block("gamma")?);
        q.write_buffer(&self.velocity_buf, 0, scene.block("velocity")?);
        q.write_buffer(&self.snap.pressure, 0, scene.block("pressure")?);
        q.write_buffer(&self.temp_buf, 0, scene.block("temperature")?);
        if let Some(r) = scene.blocks.get("redox").filter(|b| b.len() == fcells * 4) {
            q.write_buffer(&self.redox_buf, 0, r);
        }

        // Agentes: limpa, espalha os vivos pelos slots gravados e refaz a
        // pilha dos slots livres (o mais baixo sai primeiro, como no início).
        self.clear_agents(gpu);
        // Sem ligações por omissão (todas livres: 0xFFFFFFFF).
        q.write_buffer(&self.bonds_buf, 0, &vec![0xFFu8; self.bonds_buf.size() as usize]);
        for b in self.slot_bufs() {
            if b.name == "bonds2" && !scene.blocks.contains_key("bonds2") {
                notes.push("cena sem ligações por âncoras (gravada antes de existirem): agentes soltos".into());
                continue;
            }
            let data = scene.block(b.name)?;
            let (segs, words) = segments(&b, &slots, &body_len, false);
            if data.len() as u64 != words * 4 {
                return Err(format!("bloco {} com tamanho errado", b.name));
            }
            if words == 0 {
                continue;
            }
            let compact = storage_buffer(&gpu.device, "scene compact", words * 4);
            q.write_buffer(&compact, 0, data);
            self.copy_segments(gpu, &segs, &compact, b.buf);
        }
        let mut used = vec![false; self.cfg.max_agents as usize];
        for &s in &slots {
            used[s as usize] = true;
        }
        let free: Vec<u32> = (0..self.cfg.max_agents).rev().filter(|&s| !used[s as usize]).collect();
        q.write_buffer(&self.free_buf, 0, bytemuck::cast_slice(&free));
        let mut c = counters.clone();
        c[0] = free.len() as u32;
        q.write_buffer(&self.life_counters_buf, 0, bytemuck::cast_slice(&c));
        self.last_counters = None;
        self.light_dirty = true;
        // Os parâmetros na GPU já são os da cena (o livro-razão lê-os mesmo
        // antes do primeiro passo, p. ex. em pausa).
        self.params.max_agents = self.cfg.max_agents;
        q.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        gpu.wait_idle();
        Ok((h["interface"].clone(), notes))
    }
}
