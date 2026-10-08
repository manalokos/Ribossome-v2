//! NOMES em latim de fantasia (ferramenta do observador; a simulação não os
//! conhece). Dois usos:
//! - CENAS: um nome, um adjetivo e um número ("Abyssus_lucidus_417"). O
//!   adjetivo concorda em género com o nome e, metade das vezes, vem do
//!   estado do mundo (corrente forte, muita luz, fumarolas, mundo cheio...);
//! - ORGANISMOS: um binome "científico" ("Sulfovibrio tenax"). As duas formas
//!   de uma linhagem (o genoma e o seu complemento reverso) têm o MESMO nome;
//!   a segunda leva "_B".
//!
//! O género diz o modo de vida: a primeira metade é de onde vem a energia
//! (luz, quimiossíntese, boca, protease), a segunda é o hábito (nada, sente,
//! agarra-se, liga-se a outros, dorme), ambas tiradas dos órgãos do corpo da
//! fita canónica. O epíteto vem só do genoma. É determinista: o mesmo genoma
//! dá sempre o mesmo nome. Uma mutação muda o epíteto (e o género, se mudar
//! o modo de vida), como muda a espécie; para um nome estável de um grupo
//! usa-se o genoma do seu líder.

use crate::life::organs::{Organ, Residue, translate_organs};
use crate::species::reverse_complement;

/// FNV-1a de 64 bits, com uma mistura final (o FNV sozinho espalha mal os
/// bits baixos de entradas curtas, e os nomes tiram-se por módulo).
fn hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

/// Escolha independente n.º `k` a partir do mesmo hash.
fn pick<'a>(list: &[&'a str], h: u64, k: u64) -> &'a str {
    list[(hash(&(h ^ k.wrapping_mul(0x9e3779b97f4a7c15)).to_le_bytes()) % list.len() as u64) as usize]
}

// ---------------------------------------------------------------- CENAS

/// Nomes de lugares, por género (para o adjetivo concordar). Lista do v3
/// (`naming.rs`) limpa de repetições e de palavras que não eram nomes, mais
/// alguns de água. ("Abyssus" é feminino em latim clássico; aqui vai com os
/// masculinos, como no latim tardio.)
const NOUNS_M: [&str; 44] = [
    "Abyssus", "Fons", "Lacus", "Gurges", "Fluctus", "Limus", "Fumus", "Vapor", "Nimbus", "Rivus", "Scopulus", "Hortus", "Campus", "Nidus", "Flos", "Sal", "Oceanus", "Mons", "Collis", "Lapis", "Ventus", "Turbo",
    "Imber", "Ignis", "Focus", "Aestus", "Calor", "Sol", "Aether", "Orbis", "Mundus", "Ager", "Fundus", "Locus", "Saltus", "Lucus", "Vortex", "Sinus", "Amnis", "Puteus", "Ramus", "Arcus", "Fructus", "Vulcanus",
];
const NOUNS_F: [&str; 48] = [
    "Umbra", "Caligo", "Aurora", "Silva", "Radix", "Aqua", "Palus", "Vallis", "Rupes", "Arbor", "Herba", "Aura", "Procella", "Tempestas", "Pluvia", "Flamma", "Luna", "Stella", "Nubes", "Iris", "Terra", "Tellus",
    "Natura", "Vita", "Gens", "Turba", "Vastitas", "Solitudo", "Regio", "Planities", "Fauna", "Flora", "Arcadia", "Seges", "Messis", "Unda", "Lacuna", "Spuma", "Arena", "Ripa", "Insula", "Caverna", "Fossa", "Origo",
    "Materia", "Tenebra", "Vinea", "Abundantia",
];
const NOUNS_N: [&str; 36] = [
    "Aequor", "Stagnum", "Saxum", "Lumen", "Ostium", "Vadum", "Antrum", "Semen", "Germen", "Sulphur", "Pelagus", "Mare", "Flumen", "Cavum", "Folium", "Caelum", "Pratum", "Desertum", "Barathrum", "Chaos", "Vacuum",
    "Elysium", "Viridarium", "Pomarium", "Nemus", "Frigus", "Gelu", "Litus", "Profundum", "Fretum", "Marmor", "Sidus", "Ovum", "Regnum", "Territorium", "Spatium",
];

/// Adjetivos sem tema (forma masculina; `agree` dá as outras). Só de três
/// tipos: -us/-a/-um, -is/-is/-e e os de uma só forma (-ns, -x, -rs).
const ADJ_ANY: [&str; 72] = [
    "profundus", "tacitus", "vivus", "antiquus", "novus", "amarus", "dulcis", "limpidus", "vagus", "mitis", "viridis", "rubens", "caeruleus", "altus", "imus", "primus", "ultimus", "avidus", "lentus", "aequus", "stabilis",
    "constans", "vivax", "vitalis", "vigens", "acutus", "vividus", "vastus", "horridus", "florens", "frondosus", "nemorosus", "silvestris", "aquosus", "montanus", "maritimus", "fluvialis", "lacustris", "palustris",
    "aridus", "siccus", "tepidus", "durus", "severus", "austerus", "gravis", "levis", "suavis", "amoenus", "iucundus", "laetus", "nebulosus", "magnificus", "grandis", "immensus", "infinitus", "aetherius", "caelestis",
    "sublimis", "humilis", "modestus", "tenuis", "fragilis", "delicatus", "borealis", "australis", "cavernosus", "salsus", "pellucidus", "abditus", "arcanus", "primaevus",
];

/// Estado do mundo que pode dar o adjetivo de uma cena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mood {
    /// Água muito agitada.
    Current,
    /// Água parada (fluido desligado).
    Still,
    /// Muito sol.
    Bright,
    /// Sem sol.
    Dark,
    /// Com fumarolas.
    Hot,
    /// Sem fumarolas.
    Cold,
    /// Muita matéria semeada.
    Crowded,
    /// Pouca matéria semeada.
    Sparse,
}

fn mood_adjectives(m: Mood) -> &'static [&'static str] {
    match m {
        Mood::Current => &["turbidus", "procellosus", "turbulentus", "rapidus", "vehemens", "tumultuosus", "fluens", "vorticosus", "ventosus", "ferox", "mobilis", "agilis"],
        Mood::Still => &["tranquillus", "placidus", "quietus", "immobilis", "stagnans", "serenus", "silens", "lenis", "languidus", "torpidus", "iners"],
        Mood::Bright => &["lucidus", "clarus", "splendidus", "aureus", "fulgens", "apricus", "candidus", "radians"],
        Mood::Dark => &["obscurus", "tenebrosus", "caliginosus", "opacus", "nocturnus", "umbrosus", "caecus"],
        Mood::Hot => &["fervens", "calidus", "ardens", "igneus", "sulphureus", "torridus", "fumidus", "vulcanius"],
        Mood::Cold => &["gelidus", "frigidus", "algidus", "glacialis", "nivalis"],
        Mood::Crowded => &["densus", "fecundus", "frequens", "plenus", "opimus", "fertilis", "abundans"],
        Mood::Sparse => &["rarus", "vacuus", "sterilis", "desertus", "inanis", "exiguus", "solitarius"],
    }
}

/// Concordância do adjetivo com o género do nome (0 m, 1 f, 2 n).
fn agree(adj: &str, gender: usize) -> String {
    match (gender, adj.strip_suffix("us"), adj.strip_suffix("is")) {
        (1, Some(stem), _) => format!("{stem}a"),
        (2, Some(stem), _) => format!("{stem}um"),
        (2, _, Some(stem)) => format!("{stem}e"),
        _ => adj.to_string(),
    }
}

/// Os estados que se reconhecem num mundo (pode não haver nenhum).
pub fn world_moods(w: &crate::world::World) -> Vec<Mood> {
    let d = crate::params::SimParams::default();
    let mut m = Vec::new();
    if !w.settings.fluid_enabled {
        m.push(Mood::Still);
    } else if w.params.fluid_vorticity > 1.3 * d.fluid_vorticity {
        m.push(Mood::Current);
    }
    if w.params.uv_strength <= 0.0 {
        m.push(Mood::Dark);
    } else if w.params.uv_strength > 1.3 * d.uv_strength {
        m.push(Mood::Bright);
    }
    m.push(if w.fumaroles.iter().any(|f| f.enabled != 0 && f.strength > 0.0) { Mood::Hot } else { Mood::Cold });
    if w.seed_density > 1.3 * crate::world::SEED_DENSITY_DEFAULT {
        m.push(Mood::Crowded);
    } else if w.seed_density < 0.6 * crate::world::SEED_DENSITY_DEFAULT {
        m.push(Mood::Sparse);
    }
    m
}

/// Nome de uma cena a partir de uma semente qualquer (por exemplo a semente
/// do mundo misturada com a hora) e dos estados do mundo: metade das vezes o
/// adjetivo é de um desses estados, na outra metade é um qualquer. O mesmo
/// (semente, estados) dá sempre o mesmo nome.
pub fn scene_name_with(seed: u64, moods: &[Mood]) -> String {
    let h = hash(&seed.to_le_bytes());
    let total = (NOUNS_M.len() + NOUNS_F.len() + NOUNS_N.len()) as u64;
    let i = (hash(&(h ^ 1).to_le_bytes()) % total) as usize;
    let (noun, gender) = if i < NOUNS_M.len() {
        (NOUNS_M[i], 0)
    } else if i < NOUNS_M.len() + NOUNS_F.len() {
        (NOUNS_F[i - NOUNS_M.len()], 1)
    } else {
        (NOUNS_N[i - NOUNS_M.len() - NOUNS_F.len()], 2)
    };
    let themed = !moods.is_empty() && hash(&(h ^ 2).to_le_bytes()) % 2 == 0;
    let adj = if themed {
        let mood = moods[(hash(&(h ^ 3).to_le_bytes()) % moods.len() as u64) as usize];
        pick(mood_adjectives(mood), h, 4)
    } else {
        pick(&ADJ_ANY, h, 4)
    };
    format!("{noun}_{}_{:03}", agree(adj, gender), hash(&(h ^ 5).to_le_bytes()) % 1000)
}

/// Nome de uma cena só pela semente (adjetivo ao acaso).
pub fn scene_name(seed: u64) -> String {
    scene_name_with(seed, &[])
}

/// Nome novo para o mundo `w`, único na prática: a semente do mundo
/// misturada com a hora (em milissegundos).
pub fn new_scene_name(w: &crate::world::World, world_seed: u64) -> String {
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    scene_name_with(world_seed.wrapping_mul(0x9e3779b97f4a7c15) ^ ms, &world_moods(w))
}

/// Se o nome de um ficheiro de cena (sem extensão) parece um nome destes,
/// devolve-o sem o sufixo da epoch ("Abyssus_lucidus_417_e52000" →
/// "Abyssus_lucidus_417"). "autosave", "cena_epoch1000" e afins não parecem.
pub fn name_from_stem(stem: &str) -> Option<String> {
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() < 3 || parts.len() > 4 {
        return None;
    }
    let word = |s: &str, upper: bool| s.len() >= 2 && s.chars().all(|c| c.is_ascii_alphabetic()) && s.chars().next().is_some_and(|c| c.is_ascii_uppercase() == upper) && s.chars().skip(1).all(|c| c.is_ascii_lowercase());
    let number = parts[2].len() == 3 && parts[2].chars().all(|c| c.is_ascii_digit());
    let epoch = parts.get(3).is_none_or(|e| e.len() > 1 && e.starts_with('e') && e[1..].chars().all(|c| c.is_ascii_digit()));
    (word(parts[0], true) && word(parts[1], false) && number && epoch).then(|| parts[..3].join("_"))
}

// ----------------------------------------------------------- ORGANISMOS

const EPITHET_ROOT: [&str; 64] = [
    "ten", "vag", "min", "long", "brev", "lat", "grac", "robust", "pall", "nigr", "alb", "flav", "rubr", "vir", "prim", "sol", "noct", "aquat", "ign", "sals", "amar", "mit", "fer", "rap", "tard", "cel", "mult",
    "pauc", "simpl", "dupl", "curv", "rect", "fug", "aud", "tac", "viv", "plac", "luc", "umbr", "spin", "glabr", "pil", "cili", "torqu", "flex", "rigid", "moll", "dur", "magn", "parv", "crass", "acut", "obtus",
    "vari", "commun", "rar", "mir", "dubi", "obscur", "clar", "argent", "aur", "ferr", "vitr",
];
const EPITHET_END: [&str; 10] = ["ax", "us", "is", "ans", "ens", "osus", "atus", "ellus", "inus", "icus"];

/// Tipos de órgão que um corpo tem (sem o fio que liga dois genes: não é
/// um órgão que a criatura use).
pub fn organ_types(body: &[Residue]) -> Vec<u8> {
    let mut t: Vec<u8> = body.iter().filter_map(|r| r.organ.map(|o| o.0)).filter(|&t| t != Organ::Linker as u8).collect();
    t.sort_unstable();
    t.dedup();
    t
}

/// Primeira metade do género: de onde vem a energia, por ordem de
/// prioridade. Cada modo tem algumas raízes; o genoma escolhe uma.
fn energy_roots(organs: &[u8]) -> &'static [&'static str] {
    let has = |o: Organ| organs.contains(&(o as u8));
    let producer = has(Organ::Photosystem) || has(Organ::Chemosynthesis);
    if has(Organ::Protease) && producer {
        &["Spino", "Acantho", "Armi"] // produtor armado
    } else if has(Organ::Protease) {
        &["Rapto", "Lyso", "Preda"] // protease
    } else if has(Organ::Photosystem) && has(Organ::Chemosynthesis) {
        &["Ambi", "Amphi"] // as duas fontes
    } else if has(Organ::Photosystem) {
        &["Lumi", "Photo", "Helio"] // fotossistema
    } else if has(Organ::Chemosynthesis) {
        &["Sulfo", "Thermo", "Pyro"] // quimiossíntese
    } else if has(Organ::Mouth) {
        &["Vora", "Phago", "Gulo"] // boca
    } else if organs.is_empty() {
        &["Nudi", "Gymno"] // sem órgãos (ou RNA nu)
    } else {
        &["Proto", "Archaeo", "Primi"] // órgãos, mas nenhuma fonte própria
    }
}

/// Segunda metade do género: o hábito, por ordem de prioridade.
fn habit_stems(organs: &[u8]) -> &'static [&'static str] {
    let has = |o: Organ| organs.contains(&(o as u8));
    let senses = has(Organ::FoodSensor) || has(Organ::LightSensor) || has(Organ::EnergySensor) || has(Organ::FoodSensorDirectional) || has(Organ::LightSensorDirectional);
    if has(Organ::Holdfast) {
        &["petra", "saxa", "haerens"] // agarra-se ao terreno
    } else if has(Organ::Anchor) {
        &["nexa", "socia", "catena", "desmus"] // liga-se a outros
    } else if has(Organ::Muscle) && senses {
        &["venator", "cursor", "nauta"] // move-se e sente: procura
    } else if has(Organ::Muscle) || has(Organ::Clock) {
        &["vibrio", "natans", "spira", "remus"] // nadador (músculo ou relógio)
    } else if senses {
        &["vigil", "sensor", "specta"] // só sente
    } else if has(Organ::Dormancy) {
        &["somnus", "spora", "cystis"] // forma de resistência
    } else if has(Organ::Storage) {
        &["theca", "cella", "saccus"] // depósito
    } else {
        &["bacter", "monas", "coccus", "plasma", "zoon", "forma", "idium", "ella"]
    }
}

/// A fita canónica de uma linhagem: a menor (por ordem das bases) entre o
/// genoma e o seu complemento reverso. É a forma "A" do nome.
pub fn canonical(genome: &[u8]) -> (Vec<u8>, bool) {
    let rc = reverse_complement(genome);
    if rc.as_slice() < genome { (rc, true) } else { (genome.to_vec(), false) }
}

/// Nome da LINHAGEM (as duas fitas, sem sufixo). `organs_canonical` = tipos
/// de órgão do corpo da fita canónica (ver `organ_types`).
pub fn lineage_name(genome: &[u8], organs_canonical: &[u8]) -> String {
    let h = hash(&canonical(genome).0);
    format!("{}{} {}{}", pick(energy_roots(organs_canonical), h, 1), pick(habit_stems(organs_canonical), h, 2), pick(&EPITHET_ROOT, h, 3), pick(&EPITHET_END, h, 4))
}

/// Binome do organismo com este genoma: o nome da linhagem e, se esta fita
/// for a complementar da canónica, "_B".
pub fn organism_name(genome: &[u8], organs_canonical: &[u8]) -> String {
    format!("{}{}", lineage_name(genome, organs_canonical), if canonical(genome).1 { "_B" } else { "" })
}

/// Órgãos do corpo da fita canónica de `genome`, com o código de órgãos do
/// mundo (`table::code_to_gpu`).
fn canonical_organs(genome: &[u8], require_start: bool, code: &[u32]) -> Vec<u8> {
    organ_types(&translate_organs(&canonical(genome).0, require_start, code))
}

/// `lineage_name` a traduzir a fita canónica com o código do mundo.
pub fn lineage_name_in(genome: &[u8], require_start: bool, code: &[u32]) -> String {
    lineage_name(genome, &canonical_organs(genome, require_start, code))
}

/// `organism_name` a traduzir a fita canónica com o código do mundo.
pub fn organism_name_in(genome: &[u8], require_start: bool, code: &[u32]) -> String {
    organism_name(genome, &canonical_organs(genome, require_start, code))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARMED: [u8; 2] = [Organ::Protease as u8, Organ::Chemosynthesis as u8];

    #[test]
    fn both_strands_share_the_name() {
        let g: Vec<u8> = vec![0, 1, 2, 3, 3, 2, 1, 0, 2, 2, 1, 3, 0, 0, 1];
        let a = organism_name(&g, &ARMED);
        let b = organism_name(&reverse_complement(&g), &ARMED);
        assert_eq!(a.trim_end_matches("_B"), b.trim_end_matches("_B"));
        assert!(a.ends_with("_B") != b.ends_with("_B"), "{a} / {b}");
        assert_eq!(lineage_name(&g, &ARMED), a.trim_end_matches("_B"));
        assert_eq!(a, organism_name(&g, &ARMED), "determinista");
    }

    #[test]
    fn a_palindrome_has_one_form() {
        // AUAU é o seu próprio complemento reverso: não há forma B.
        let g = [0u8, 1, 0, 1];
        assert_eq!(reverse_complement(&g), g);
        assert!(!organism_name(&g, &[]).ends_with("_B"));
    }

    #[test]
    fn genus_follows_the_way_of_life() {
        let g: Vec<u8> = vec![0, 1, 2, 3, 3, 2, 1, 0, 2, 2, 1, 3, 0, 0, 1];
        let genus = |organs: &[u8]| lineage_name(&g, organs).split(' ').next().unwrap().to_string();
        let starts = |organs: &[u8], roots: &[&str]| roots.iter().any(|r| genus(organs).starts_with(r));
        assert!(starts(&ARMED, &["Spino", "Acantho", "Armi"]), "{}", genus(&ARMED));
        assert!(starts(&[Organ::Photosystem as u8], &["Lumi", "Photo", "Helio"]));
        assert!(starts(&[Organ::Mouth as u8, Organ::Muscle as u8], &["Vora", "Phago", "Gulo"]));
        assert!(starts(&[], &["Nudi", "Gymno"]));
        // O hábito: uma ventosa passa à frente de tudo.
        let held = genus(&[Organ::Photosystem as u8, Organ::Holdfast as u8, Organ::Muscle as u8]);
        assert!(["petra", "saxa", "haerens"].iter().any(|s| held.ends_with(s)), "{held}");
        // O epíteto não depende dos órgãos.
        let epithet = |organs: &[u8]| lineage_name(&g, organs).split(' ').nth(1).unwrap().to_string();
        assert_eq!(epithet(&ARMED), epithet(&[]));
    }

    #[test]
    fn the_linker_is_not_a_way_of_life() {
        let body = [Residue { aa: 0, organ: Some((Organ::Linker as u8, 0, 0)) }, Residue { aa: 1, organ: Some((Organ::Mouth as u8, 0, 0)) }, Residue { aa: 2, organ: None }];
        assert_eq!(organ_types(&body), vec![Organ::Mouth as u8]);
    }

    #[test]
    fn organism_names_are_varied() {
        // 300 genomas ao acaso com o mesmo modo de vida: quase todos com nome diferente.
        let mut x = 12345u64;
        let mut names = std::collections::HashSet::new();
        for _ in 0..300 {
            let g: Vec<u8> = (0..40)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (x >> 62) as u8
                })
                .collect();
            names.insert(lineage_name(&g, &[Organ::Mouth as u8]));
        }
        assert!(names.len() > 285, "{} nomes distintos em 300", names.len());
    }

    #[test]
    fn scene_names_are_stable_and_varied() {
        assert_eq!(scene_name(42), scene_name(42));
        assert_eq!(scene_name_with(42, &[Mood::Hot]), scene_name_with(42, &[Mood::Hot]));
        let n: std::collections::HashSet<String> = (0..500).map(scene_name).collect();
        assert!(n.len() > 480, "{} nomes distintos em 500", n.len());
        for name in &n {
            assert_eq!(name_from_stem(name).as_deref(), Some(name.as_str()), "{name}");
            assert_eq!(name_from_stem(&format!("{name}_e1234")).as_deref(), Some(name.as_str()));
        }
    }

    #[test]
    fn the_adjective_follows_the_world_and_the_gender() {
        let hot = mood_adjectives(Mood::Hot);
        let themed = (0..400u64)
            .filter(|&s| {
                let name = scene_name_with(s, &[Mood::Hot]);
                let adj = name.split('_').nth(1).unwrap().to_string();
                hot.iter().any(|a| (0..3).any(|g| agree(a, g) == adj))
            })
            .count();
        assert!((120..280).contains(&themed), "{themed} em 400 com adjetivo de calor");
        assert_eq!(agree("lucidus", 1), "lucida");
        assert_eq!(agree("lucidus", 2), "lucidum");
        assert_eq!(agree("dulcis", 2), "dulce");
        assert_eq!(agree("dulcis", 1), "dulcis");
        assert_eq!(agree("fervens", 2), "fervens");
    }

    #[test]
    fn word_lists_have_no_repeats() {
        let mut all: Vec<&str> = NOUNS_M.iter().chain(&NOUNS_F).chain(&NOUNS_N).copied().collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "nomes repetidos");
        let moods = [Mood::Current, Mood::Still, Mood::Bright, Mood::Dark, Mood::Hot, Mood::Cold, Mood::Crowded, Mood::Sparse];
        let mut adj: Vec<&str> = ADJ_ANY.iter().copied().chain(moods.iter().flat_map(|&m| mood_adjectives(m).iter().copied())).collect();
        let n = adj.len();
        adj.sort_unstable();
        adj.dedup();
        assert_eq!(adj.len(), n, "adjetivos repetidos");
    }

    #[test]
    fn file_stems_that_are_not_names() {
        for s in ["autosave", "cena_epoch1000", "Abyssus_lucidus", "abyssus_lucidus_417", "Abyssus_lucidus_41", "Abyssus_lucidus_417_final", "Abyssus_Lucidus_417"] {
            assert_eq!(name_from_stem(s), None, "{s}");
        }
        assert_eq!(name_from_stem("Abyssus_lucidus_417").as_deref(), Some("Abyssus_lucidus_417"));
    }
}
