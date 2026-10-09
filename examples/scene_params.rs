//! Parâmetros de uma cena que diferem dos valores por omissão do código, e
//! as outras definições do cabeçalho. SCENE = caminho da cena.
use ribossome::params::SimParams;
use ribossome::world::Scene;

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let h = &scene.header;
    let mut p = SimParams::default();
    let mut unknown = Vec::new();
    if let Some(obj) = h["params"].as_object() {
        for (k, v) in obj {
            if !p.set_named(k, v.as_f64().unwrap_or(0.0)) {
                unknown.push(k.clone());
            }
        }
    }
    println!("epoch {}", scene.epoch());
    for (k, v, d) in p.changed_from_default() {
        println!("{k} = {v} (código {d})");
    }
    println!("desconhecidos: {unknown:?}");
    for (k, v) in h.as_object().unwrap() {
        if k != "params" {
            let t = v.to_string();
            println!("{k}: {}", if t.len() > 400 { format!("{}…", &t[..400]) } else { t });
        }
    }
}
