//! Recursos do executável no Windows: ícone e dados da versão (o que o
//! Explorador mostra nas propriedades do ficheiro).
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "Ribossome");
        res.set("FileDescription", "Ribossome: artificial life on the GPU");
        res.set("LegalCopyright", "Copyright (c) 2026 Filipe da Veiga Ventura Alves");
        res.compile().expect("recursos do Windows (ícone)");
    }
}
