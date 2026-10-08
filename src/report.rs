//! RELATÓRIO de uma cena numa página HTML (um só ficheiro, com as imagens
//! lá dentro): as espécies e as suas duas formas, onde vivem, quem pode
//! atacar quem, ataques e ligações a decorrer, a árvore de parentesco entre
//! as espécies vivas e a árvore das linhagens registadas durante a corrida.
//! É uma ferramenta do observador: só lê o mundo, não o altera.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::gpu::Gpu;
use crate::life::amino::AA_LETTERS;
use crate::life::organs::{ORGAN_SYMBOLS, Residue, describe, organ_gain, translate_organs};
use crate::lineage::Lineages;
use crate::params::Agent;
use crate::render::Camera;
use crate::render::capture::Capture;
use crate::species::{Species, cluster_members, distance, reverse_complement};
use crate::world::{BOND_STRIDE, World};

/// Espécies com ficha: as que têm pelo menos 1% dos agentes, até este número.
const MAX_CARDS: usize = 16;
const BANDS: usize = 8;
const PROTEASE: u8 = 11;
const PROLINE: u8 = 12;

struct Stats {
    n: u32,
    energy: f64,
    age: f64,
    generation: f64,
    copy: f64,
    residues: f64,
    linked: u32,
    bands: [u32; BANDS],
    dots: Vec<[f32; 2]>,
    /// Um agente exemplar de cada fita (slot), de preferência com o genoma do líder.
    rep: [Option<(u32, bool)>; 2],
}

impl Default for Stats {
    fn default() -> Self {
        Self { n: 0, energy: 0.0, age: 0.0, generation: 0.0, copy: 0.0, residues: 0.0, linked: 0, bands: [0; BANDS], dots: Vec::new(), rep: [None; 2] }
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn img(png: &[u8], w: u32, alt: &str) -> String {
    format!("<img width=\"{w}\" alt=\"{}\" src=\"data:image/png;base64,{}\">", esc(alt), crate::mcp::base64(png))
}

/// Proteína em letras: órgãos com o símbolo do inspetor, a branco.
fn protein_html(body: &[Residue]) -> String {
    let mut s = String::new();
    for r in body {
        match r.organ {
            Some((t, _, _)) => write!(s, "<b>{}</b>", ORGAN_SYMBOLS[t as usize]).unwrap(),
            None => s.push(AA_LETTERS[r.aa as usize]),
        }
    }
    s
}

/// Família que o bolso de um vizinho reconhece (pocket_family no shader):
/// 0 = nenhuma em especial.
fn pocket_family(aa: u8) -> usize {
    match aa {
        2 | 3 => 1,
        8 | 14 => 2,
        4 | 9 | 18 | 19 | 7 | 17 => 3,
        _ => 0,
    }
}

/// Pesos de uma protease no resíduo k: famílias 1, 2, 3 e GENERALISTA (o
/// vizinho decide; sem vizinho próprio é generalista: corta qualquer resíduo
/// a um terço da força). Como family_weights em contact.wgsl.
fn family_weights(body: &[Residue], k: usize) -> [f32; 4] {
    match body.get(k + 1).map_or(0, |r| pocket_family(r.aa)) {
        0 => [0.0, 0.0, 0.0, 1.0 / 3.0],
        f => {
            let mut w = [0.0; 4];
            w[f - 1] = 1.0;
            w
        }
    }
}

/// Força das proteases por família, todas ligadas e a tocar (o melhor caso).
fn protease_force(body: &[Residue], w: &World) -> [f32; 4] {
    let mut f = [0.0; 4];
    for (k, r) in body.iter().enumerate() {
        if let Some((t, p, g)) = r.organ
            && t == PROTEASE
            && let Some(v) = w.organ_table.get(t as usize).and_then(|o| o.variantes.get(p as usize))
        {
            let force = v.get("forca").copied().unwrap_or(0.0).max(0.0) * organ_gain(g);
            for (x, wt) in f.iter_mut().zip(family_weights(body, k)) {
                *x += force * wt;
            }
        }
    }
    f
}

/// (fração de resíduos-alvo de cada família já com a imunidade de quem tem
/// protease dessa família, fração de prolina).
fn defence(body: &[Residue], w: &World) -> ([f32; 4], f32) {
    let n = body.len().max(1) as f32;
    // O quarto "alvo" são todos os resíduos (o que a generalista corta).
    let mut t = [0.0, 0.0, 0.0, 1.0];
    let mut pro = 0.0;
    let mut own = [0.0f32; 4];
    for (k, r) in body.iter().enumerate() {
        let m = w.amino.get(r.aa as usize).map_or(0, |a| (a.protease_alvo.max(0.0) + 0.5) as u32);
        for (f, v) in t.iter_mut().take(3).enumerate() {
            if m & (1 << f) != 0 {
                *v += 1.0 / n;
            }
        }
        if r.aa == PROLINE {
            pro += 1.0 / n;
        }
        if r.organ.is_some_and(|o| o.0 == PROTEASE) {
            for (x, wt) in own.iter_mut().zip(family_weights(body, k)) {
                *x = x.max(if wt > 0.0 { 1.0 } else { 0.0 });
            }
        }
    }
    for (v, o) in t.iter_mut().zip(own) {
        *v *= 1.0 - 0.5 * o;
    }
    (t, pro)
}

/// Energia que o atacante tira à vítima por passo de contacto (as constantes
/// de contact.wgsl: PRED_DRAIN 0,2 × 10 × (1 − 0,9·prolina)).
fn drain(att: &[Residue], vic: &[Residue], w: &World) -> f32 {
    let f = protease_force(att, w);
    let (t, pro) = defence(vic, w);
    0.2 * w.params.protease_power * (f[0] * t[0] + f[1] * t[1] + f[2] * t[2] + f[3] * t[3]) * 10.0 * (1.0 - 0.9 * pro)
}

/// Retrato de um agente sozinho, enquadrado pelo corpo.
pub(crate) fn portrait(gpu: &Gpu, w: &World, cap: &Capture, slot: u32, a: &Agent, bright: f32) -> Vec<u8> {
    let raw = gpu.read_ranges_blocking(&w.body_pos_buf, &[(slot as u64 * 512, 512)]);
    let pos: &[[f32; 2]] = bytemuck::cast_slice(&raw);
    let (s, c) = a.rot.sin_cos();
    let (mut lo, mut hi) = ([a.pos_x, a.pos_y], [a.pos_x, a.pos_y]);
    for (i, p) in pos.iter().take((a.body_len as usize).min(64)).enumerate() {
        let q = [a.pos_x + c * p[0] - s * p[1], a.pos_y + s * p[0] + c * p[1]];
        if i == 0 {
            (lo, hi) = (q, q);
        }
        lo = [lo[0].min(q[0]), lo[1].min(q[1])];
        hi = [hi[0].max(q[0]), hi[1].max(q[1])];
    }
    let side = ((hi[0] - lo[0]).max(hi[1] - lo[1]) + 80.0).max(60.0);
    let cam = Camera { center: [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5], zoom: cap_size(cap) / side };
    cap.view.focus.set(slot);
    cap.view.focus_offset.set([cam.center[0] - a.pos_x, cam.center[1] - a.pos_y]);
    let rgba = cap.render(gpu, w, &cam, 0, bright);
    cap.view.focus.set(u32::MAX);
    cap.encode_png(&rgba).unwrap_or_default()
}

fn cap_size(cap: &Capture) -> f32 {
    cap.size() as f32
}

/// Árvore de parentesco entre as espécies vivas (UPGMA sobre a distância
/// entre os genomas dos líderes), desenhada de lado.
fn inferred_tree(species: &[&Species], total: usize) -> String {
    let n = species.len();
    if n < 2 {
        return "<p>There is only one species above 1%.</p>".into();
    }
    // Cada grupo: (folhas, altura, posição vertical média, svg já desenhado).
    struct Node {
        leaves: Vec<usize>,
        height: f32,
        y: f32,
    }
    let mut d = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..i {
            let v = distance(&species[i].leader, &species[j].leader);
            d[i][j] = v;
            d[j][i] = v;
        }
    }
    // Ordem das folhas: a da junção (calcula-se primeiro a árvore, depois as posições).
    let mut groups: Vec<(Vec<usize>, f32, usize)> = (0..n).map(|i| (vec![i], 0.0, i)).collect();
    let mut merges: Vec<(usize, usize, f32)> = Vec::new(); // ids dos nós juntos, altura
    let mut next_id = n;
    while groups.len() > 1 {
        let mut best = (0, 1, f32::MAX);
        for a in 0..groups.len() {
            for b in 0..a {
                let mut sum = 0.0;
                for &i in &groups[a].0 {
                    for &j in &groups[b].0 {
                        sum += d[i][j];
                    }
                }
                let avg = sum / (groups[a].0.len() * groups[b].0.len()) as f32;
                if avg < best.2 {
                    best = (b, a, avg);
                }
            }
        }
        let (b, a, h) = best;
        let ga = groups.remove(a);
        let gb = groups.remove(b);
        merges.push((gb.2, ga.2, h * 0.5));
        let mut leaves = gb.0;
        leaves.extend(ga.0);
        groups.push((leaves, h * 0.5, next_id));
        next_id += 1;
    }
    let order = &groups[0].0;
    let row = 22.0;
    let (left, width) = (20.0, 520.0);
    let max_h = merges.iter().map(|m| m.2).fold(0.01f32, f32::max);
    let x_of = |h: f32| left + width * (1.0 - h / max_h);
    let mut nodes: HashMap<usize, Node> = HashMap::new();
    for (r, &leaf) in order.iter().enumerate() {
        nodes.insert(leaf, Node { leaves: vec![leaf], height: 0.0, y: 14.0 + r as f32 * row });
    }
    let mut svg = String::new();
    for (k, &(a, b, h)) in merges.iter().enumerate() {
        let (na, nb) = (nodes.remove(&a).unwrap(), nodes.remove(&b).unwrap());
        let x = x_of(h);
        for c in [&na, &nb] {
            write!(svg, "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\"/>", x_of(c.height), c.y, c.y).unwrap();
        }
        write!(svg, "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\"/>", na.y, nb.y).unwrap();
        let mut leaves = na.leaves;
        leaves.extend(nb.leaves);
        nodes.insert(n + k, Node { leaves, height: h, y: (na.y + nb.y) * 0.5 });
    }
    for (r, &leaf) in order.iter().enumerate() {
        let s = species[leaf];
        write!(
            svg,
            "<text x=\"{:.0}\" y=\"{:.0}\">S{} · {:.1}% · {} bases</text>",
            left + width + 8.0,
            18.0 + r as f32 * row,
            leaf + 1,
            100.0 * s.count as f32 / total.max(1) as f32,
            s.leader.len()
        )
        .unwrap();
    }
    format!(
        "<svg class=\"tree\" width=\"760\" height=\"{:.0}\">{svg}</svg><p class=\"note\">Branch length = distance between genomes (the maximum drawn is {:.0}% difference). It is an estimate made from the living species only.</p>",
        28.0 + n as f32 * row,
        max_h * 200.0
    )
}

/// Barras de quantos vivem em cada faixa de altura (de cima para baixo).
fn bands_svg(b: &[u32; BANDS]) -> String {
    let max = b.iter().copied().max().unwrap_or(1).max(1) as f32;
    let mut s = String::from("<svg class=\"bands\" width=\"120\" height=\"104\">");
    for (i, &v) in b.iter().enumerate() {
        write!(s, "<rect x=\"0\" y=\"{}\" width=\"{:.0}\" height=\"11\"/>", i * 13, 118.0 * v as f32 / max).unwrap();
    }
    s + "</svg>"
}

fn map_svg(dots: &[[f32; 2]], sim: f32) -> String {
    let mut s = String::from("<svg class=\"map\" width=\"150\" height=\"150\"><rect width=\"150\" height=\"150\" class=\"bg\"/>");
    for d in dots {
        // Cima no ecrã = +y no mundo.
        write!(s, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"1.1\"/>", 150.0 * d[0] / sim, 150.0 * (1.0 - d[1] / sim)).unwrap();
    }
    s + "</svg>"
}

/// Gera a página. `lineages`: o registo da corrida, se houver.
pub fn generate(gpu: &Gpu, w: &World, lineages: Option<&Lineages>, title: &str) -> String {
    let cfg = w.cfg;
    let sim = cfg.sim_size();
    let agents = w.read_agents_blocking(gpu);
    let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
    let slots: Vec<u32> = (0..agents.len() as u32).filter(|&s| agents[s as usize].alive != 0).collect();
    let genomes: Vec<Vec<u8>> = slots
        .iter()
        .map(|&s| (0..agents[s as usize].gene_len as usize).map(|i| ((words[s as usize * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect())
        .collect();
    let total = genomes.len();
    let (species, members) = cluster_members(&genomes, 0.15);
    let mut species_of: Vec<u32> = vec![u32::MAX; agents.len()];
    for (k, &s) in slots.iter().enumerate() {
        species_of[s as usize] = members[k].0;
    }
    let cards: Vec<usize> = (0..species.len()).filter(|&i| species[i].count as f32 >= 0.01 * total as f32).take(MAX_CARDS).collect();
    let code = crate::life::table::code_to_gpu(&w.organ_code);
    let rs = w.params.require_start != 0;

    // Ligações vivas por agente (e com quem).
    let raw: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.bonds_buf)).to_vec();
    let mut pairs: HashMap<(u32, u32), u32> = HashMap::new();
    let mut linked = vec![false; agents.len()];
    for &s in &slots {
        for i in 0..4 {
            let o = (s as usize * BOND_STRIDE as usize + i) * 4;
            let (p, pid) = (raw[o], raw[o + 1]);
            if p == u32::MAX || agents[p as usize].alive == 0 || agents[p as usize].id != pid {
                continue;
            }
            linked[s as usize] = true;
            if p > s {
                let (a, b) = (species_of[s as usize], species_of[p as usize]);
                *pairs.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }

    let mut stats: Vec<Stats> = (0..species.len()).map(|_| Stats::default()).collect();
    for (k, &s) in slots.iter().enumerate() {
        let a = &agents[s as usize];
        let (si, same) = members[k];
        let st = &mut stats[si as usize];
        st.n += 1;
        st.energy += a.energy as f64;
        st.age += a.age as f64;
        st.generation += a.generation as f64;
        st.copy += a.pair_count as f64 / a.gene_len.max(1) as f64;
        st.residues += a.body_len as f64;
        st.linked += linked[s as usize] as u32;
        // Faixa 0 = superfície.
        st.bands[((1.0 - a.pos_y / sim).clamp(0.0, 0.999) * BANDS as f32) as usize] += 1;
        if st.dots.len() < 900 && (st.n % (species[si as usize].count / 900 + 1) == 0) {
            st.dots.push([a.pos_x, a.pos_y]);
        }
        let exact = if same { genomes[k] == species[si as usize].leader } else { genomes[k] == reverse_complement(&species[si as usize].leader) };
        let r = &mut st.rep[!same as usize];
        if a.body_len >= 2 && (r.is_none() || (exact && !r.unwrap().1)) {
            *r = Some((s, exact));
        }
    }

    let mut h = String::new();
    write!(
        h,
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>{t}</title><style>{CSS}{}</style><h1>{t}</h1>",
        crate::tree_view::CSS,
        t = esc(title)
    )
    .unwrap();
    let changed: Vec<String> = w.params.changed_from_default().iter().filter(|c| c.0 != "epoch").map(|(n, v, _)| format!("{n} = {}", (*v * 1e4).round() / 1e4)).collect();
    write!(
        h,
        "<p>Epoch <b>{}</b> · <b>{}</b> living agents · <b>{}</b> genetic groups (15% threshold), <b>{}</b> with at least 1% of the agents.</p><details><summary>Parameters differing from the default values ({})</summary><p class=\"note\">{}</p></details>",
        w.params.epoch,
        total,
        species.len(),
        cards.len(),
        changed.len(),
        esc(&changed.join(" · "))
    )
    .unwrap();

    // Mundo inteiro.
    let big = Capture::new(gpu, w, 768);
    let rgba = big.render(gpu, w, &Camera { center: [0.5 * sim, 0.5 * sim], zoom: 768.0 / sim }, 0, 0.5);
    write!(h, "<h2>The world</h2>{}", img(&big.encode_png(&rgba).unwrap_or_default(), 768, "world")).unwrap();

    // Árvores.
    h += "<h2>Recorded lineages (tree of life)</h2>";
    h += &match lineages {
        Some(l) => crate::tree_view::viewer(l, w, Some(&crate::tree_view::portraits(gpu, w, l))),
        None => "<p>No lineage record (scene opened outside the application).</p>".to_string(),
    };
    h += "<h2>Kinship between the living species (inferred tree)</h2>";
    let card_species: Vec<&Species> = cards.iter().map(|&i| &species[i]).collect();
    h += &inferred_tree(&card_species, total);

    // Corpos das duas fitas de cada espécie com ficha.
    let bodies: Vec<[Vec<Residue>; 2]> =
        cards.iter().map(|&i| [translate_organs(&species[i].leader, rs, &code), translate_organs(&reverse_complement(&species[i].leader), rs, &code)]).collect();

    // Quem pode atacar quem.
    h += "<h2>Who can attack whom</h2><p>Energy that one species (row) takes from another (column) per step of contact, by the rules: strength of the proteases of each family × fraction of target residues in the victim × proline defense. It is the best case between the two strands of each one; the diagonal is cannibalism.</p><table class=\"m\"><tr><th></th>";
    for k in 0..cards.len() {
        write!(h, "<th>S{}</th>", k + 1).unwrap();
    }
    h += "</tr>";
    for (a, ba) in bodies.iter().enumerate() {
        write!(h, "<tr><th>S{}</th>", a + 1).unwrap();
        for bv in &bodies {
            let mut v = 0.0f32;
            for x in ba {
                for y in bv {
                    v = v.max(drain(x, y, w));
                }
            }
            if v > 0.0 {
                write!(h, "<td style=\"background:rgba(230,60,50,{:.2})\">{v:.2}</td>", (v / 2.0).clamp(0.08, 0.9)).unwrap();
            } else {
                h += "<td>·</td>";
            }
        }
        h += "</tr>";
    }
    h += "</table>";

    // Ataques a decorrer (só há dados com a simulação a correr).
    let bite: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.contact_disp_buf)).to_vec();
    let mut victims: Vec<u32> = slots.iter().copied().filter(|&s| bite[s as usize][0] > 0.0).collect();
    victims.sort_by(|&a, &b| bite[b as usize][0].total_cmp(&bite[a as usize][0]));
    h += "<h2>Attacks in progress</h2>";
    if victims.is_empty() {
        h += "<p>No agent was being attacked on the last step (or the scene was opened without running any steps).</p>";
    } else {
        write!(h, "<p>{} agents were losing energy to a protease on the last step. The most attacked (the victim turns red, the attacking protease turns yellow):</p><div class=\"row\">", victims.len()).unwrap();
        let shot = Capture::new(gpu, w, 384);
        let mut seen: Vec<u32> = Vec::new();
        for &v in &victims {
            let sv = species_of[v as usize];
            if seen.iter().filter(|&&x| x == sv).count() >= 2 || seen.len() >= 6 {
                continue;
            }
            seen.push(sv);
            let a = &agents[v as usize];
            let rgba = shot.render(gpu, w, &Camera { center: [a.pos_x, a.pos_y], zoom: 384.0 / 420.0 }, 0, 0.25);
            write!(
                h,
                "<figure>{}<figcaption>victim: {} · loses {:.2} energy per step</figcaption></figure>",
                img(&shot.encode_png(&rgba).unwrap_or_default(), 300, "attack"),
                species_name(sv, &cards),
                bite[v as usize][0]
            )
            .unwrap();
        }
        h += "</div>";
    }

    // Ligações entre espécies.
    h += "<h2>Anchor bonds</h2>";
    if pairs.is_empty() {
        h += "<p>There are no bonded agents.</p>";
    } else {
        let mut list: Vec<(&(u32, u32), &u32)> = pairs.iter().collect();
        list.sort_by(|a, b| b.1.cmp(a.1));
        h += "<ul>";
        for ((a, b), n) in list.into_iter().take(10) {
            write!(h, "<li>{n} bonds between {} and {}</li>", species_name(*a, &cards), species_name(*b, &cards)).unwrap();
        }
        h += "</ul>";
    }

    // Fichas.
    h += "<h2>The species</h2><p>Each species has two forms: the child is read from the strand complementary to the parent's, so form A produces B and B produces A. The \"life cycle\" is that alternation.</p>";
    let cap = Capture::new(gpu, w, 256);
    for (k, &i) in cards.iter().enumerate() {
        let s = &species[i];
        let st = &stats[i];
        let n = st.n.max(1) as f64;
        write!(
            h,
            "<section><h3>S{} · {:.1}% of the agents ({}) · {} bases · {} distinct genomes</h3><div class=\"row\">",
            k + 1,
            100.0 * s.count as f32 / total.max(1) as f32,
            s.count,
            s.leader.len(),
            s.distinct
        )
        .unwrap();
        for strand in 0..2 {
            let body = &bodies[k][strand];
            let share = if strand == 0 { s.same_strand } else { s.count - s.same_strand };
            h += "<figure>";
            match st.rep[strand] {
                Some((slot, _)) => h += &img(&portrait(gpu, w, &cap, slot, &agents[slot as usize], 0.15), 256, "portrait"),
                None => h += "<div class=\"none\">none alive in this form</div>",
            }
            write!(
                h,
                "<figcaption><b>form {}</b> · {:.0}% of the group · {} residues<br><span class=\"seq\">{}</span></figcaption>",
                if strand == 0 { "A" } else { "B" },
                100.0 * share as f32 / s.count.max(1) as f32,
                body.len(),
                protein_html(body)
            )
            .unwrap();
            h += "<ul class=\"org\">";
            for (pos, r) in body.iter().enumerate() {
                if let Some((t, p, g)) = r.organ {
                    write!(h, "<li><b>{}</b> position {pos}: {}</li>", ORGAN_SYMBOLS[t as usize], esc(&describe(t, p, g, &w.organ_table))).unwrap();
                }
            }
            if body.iter().all(|r| r.organ.is_none()) {
                h += "<li>no organs</li>";
            }
            h += "</ul></figure>";
        }
        write!(
            h,
            "<div class=\"facts\"><table><tr><td>mean energy</td><td>{:.1}</td></tr><tr><td>mean age</td><td>{:.0} steps</td></tr><tr><td>mean generation</td><td>{:.0}</td></tr><tr><td>genome copy</td><td>{:.0}% done, on average</td></tr><tr><td>residues (mean)</td><td>{:.1}</td></tr><tr><td>bonded by anchor</td><td>{:.1}%</td></tr></table><div class=\"row\"><div>where they live{}</div><div>height (top → bottom){}</div></div></div></div></section>",
            st.energy / n,
            st.age / n,
            st.generation / n,
            100.0 * st.copy / n,
            st.residues / n,
            100.0 * st.linked as f64 / n,
            map_svg(&st.dots, sim),
            bands_svg(&st.bands)
        )
        .unwrap();
    }
    let rest = total as i64 - cards.iter().map(|&i| species[i].count as i64).sum::<i64>();
    write!(h, "<p class=\"note\">The remaining {} agents ({:.1}%) are in {} groups with less than 1% each.</p></html>", rest, 100.0 * rest as f32 / total.max(1) as f32, species.len() - cards.len()).unwrap();
    h
}

fn species_name(s: u32, cards: &[usize]) -> String {
    match cards.iter().position(|&i| i as u32 == s) {
        Some(k) => format!("S{}", k + 1),
        None => "a rare species".into(),
    }
}

const CSS: &str = "body{background:#12151a;color:#dde3ea;font:14px/1.45 system-ui,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px}\
h1{font-size:22px}h2{font-size:18px;margin-top:34px;border-bottom:1px solid #2c333b;padding-bottom:4px}h3{font-size:15px;margin:0 0 8px}\
section{background:#1a1f25;border:1px solid #2c333b;border-radius:8px;padding:12px;margin:14px 0}\
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}figure{margin:0;max-width:300px}figcaption{font-size:12px;color:#aab4c0;margin-top:4px}\
img{border-radius:6px;display:block}.seq{font-family:Consolas,monospace;font-size:12px;word-break:break-all;color:#8fa3b8}.seq b{color:#fff}\
ul.org{font-size:12px;color:#aab4c0;padding-left:16px;margin:6px 0}ul.org b{color:#fff}\
.facts{font-size:12px}.facts td{padding:1px 10px 1px 0;color:#aab4c0}.facts td+td{color:#dde3ea}\
.note{font-size:12px;color:#8b96a3}.none{width:256px;height:256px;display:flex;align-items:center;justify-content:center;background:#0e1014;color:#666;border-radius:6px;font-size:12px}\
table.m{border-collapse:collapse;font-size:12px}table.m td,table.m th{border:1px solid #2c333b;padding:3px 7px;text-align:center}\
svg.tree line{stroke:#9fb0c3;stroke-width:1.3}svg.tree text{fill:#dde3ea;font-size:11px}svg.tree line.alive{stroke:#6fcf7f}svg.tree line.dead{stroke:#6b7480}\
svg.tree line.link{stroke:#55606c;stroke-dasharray:2 2}svg.tree line.axis{stroke:#262c34}svg.tree text.axis{fill:#8b96a3;text-anchor:middle}svg.tree text.dead{fill:#7d8792}\
.facts .row{margin-top:10px}.facts .row div{display:flex;flex-direction:column;gap:4px;color:#aab4c0}svg.bands rect{fill:#e8b84a}svg.map .bg{fill:#0e1014}svg.map circle{fill:#6fcf7f}details{margin:8px 0}";
