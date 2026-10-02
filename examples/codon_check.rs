//! Confere o código genético do simulador contra a tabela standard.
use ribossome::life::amino::{AA_LETTERS, STOP, codon};

fn main() {
    let b = |c: char| "AUGC".find(c).unwrap() as u8;
    let cases = [
        ("AUG", 'M'), ("UGG", 'W'), ("UAA", '*'), ("UAG", '*'), ("UGA", '*'), ("UUU", 'F'), ("UUA", 'L'),
        ("CUG", 'L'), ("AUA", 'I'), ("GUC", 'V'), ("UCA", 'S'), ("AGU", 'S'), ("AGA", 'R'), ("CGG", 'R'),
        ("CCU", 'P'), ("ACG", 'T'), ("GCA", 'A'), ("UAU", 'Y'), ("CAU", 'H'), ("CAA", 'Q'), ("AAU", 'N'),
        ("AAA", 'K'), ("GAU", 'D'), ("GAG", 'E'), ("UGU", 'C'), ("GGA", 'G'),
    ];
    let mut bad = 0;
    for (c, want) in cases {
        let ch: Vec<char> = c.chars().collect();
        let aa = codon(b(ch[0]), b(ch[1]), b(ch[2]));
        let got = if aa == STOP { '*' } else { AA_LETTERS[aa as usize] };
        if got != want { bad += 1; println!("{c}: {got} (esperado {want})"); }
    }
    // Contagem: 61 codões de aminoácido + 3 stops.
    let stops = (0..64u8).filter(|&i| codon(i >> 4, (i >> 2) & 3, i & 3) == STOP).count();
    println!("{} casos conferidos, {bad} errados; stops: {stops}", cases.len());
}
