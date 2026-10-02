//! Imprime a tabela dos aminoácidos tal como a simulação a usa
//! (assets/aminoacidos.json).
use ribossome::life::table;

fn main() {
    let (rows, src) = table::load();
    println!("fonte: {src}");
    println!("aa | massa | repouso(rad) | dobra máx | sens α | sens β | cond αN αC βN βC | substrato A U G C | catálise | flex | UV");
    for r in &rows {
        println!(
            "{} | {:.0} | {:+.3} | {:.2} | {:+.2} | {:+.2} | {:+.2} {:+.2} {:+.2} {:+.2} | {:.2} {:.2} {:.2} {:.2} | {:.2} | {:.3} | {:.2}",
            r.letra, r.massa, r.angulo_repouso, r.dobra_max, r.sens_alfa, r.sens_beta, r.cond_alfa_n, r.cond_alfa_c,
            r.cond_beta_n, r.cond_beta_c, r.substrato_a, r.substrato_u, r.substrato_g, r.substrato_c, r.catalise,
            r.flexibilidade, r.absorcao_uv
        );
    }
}
