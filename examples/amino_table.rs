//! Imprime todas as propriedades dos aminoácidos tal como o simulador as usa.
use ribossome::life::amino::*;

fn main() {
    let flex = flexibility();
    println!("aa | massa | repouso(rad) | dobra máx | sens α | sens β | cond αN αC βN βC | substrato A U G C | catálise | flex | sombra");
    for i in 0..20 {
        let a = &AMINO[i];
        let (sa, sb) = SIGNAL_SENSITIVITY[i];
        let c = CONDUCTANCE[i];
        let s = SUBSTRATE[i];
        let shade = match a.letter { 'W' => "1", 'Y' => "0.27", 'F' => "0.04", _ => "-" };
        println!(
            "{} | {:.0} | {:+.3} | {:.2} | {:+.2} | {:+.2} | {:+.2} {:+.2} {:+.2} {:+.2} | {:.2} {:.2} {:.2} {:.2} | {:.2} | {:.3} | {}",
            a.letter, a.mass, REST_ANGLE[i], MAX_BEND[i], sa, sb, c.0, c.1, c.2, c.3, s[0], s[1], s[2], s[3], a.catalytic, flex[i], shade
        );
    }
}
