# Ribossome v4

Simulador de vida artificial a correr na placa gráfica (Rust + wgpu + WGSL).
Um mundo aquático 2D, fechado, onde organismos feitos de RNA e proteína
evoluem por química e não por regras desenhadas.

*Artificial-life simulator on the GPU. A closed 2D water world where
RNA/protein organisms evolve through chemistry: genomes are translated into
chains of amino acids and organs, matter is conserved exactly, and nothing
compares genomes. The interface and the code comments are in Portuguese.*

## A ideia

- **Matéria exata.** O mundo é fechado. Os monómeros (A, U, G, C) são
  quantidades inteiras que só mudam de sítio; nada os cria nem destrói. Há
  um teste que o verifica.
- **Química igual para todos.** O que um organismo consegue fazer vem de
  regras químicas globais. A sequência só decide onde e quanto.
- **O organismo não conhece genomas.** Nenhuma regra compara o genoma de um
  agente com o de outro.
- **Replicação por emparelhamento.** Um agente copia-se base a base com
  monómeros ativados apanhados à volta. O filho é o complemento reverso do
  pai, por isso cada linhagem tem duas formas que alternam.
- **Baixo Reynolds.** Movimento sobreamortecido, sem inércia; nadar é
  mudar de forma.

## Como é um organismo

O genoma (até 256 bases) lê-se a partir do primeiro `AUG` até um codão de
paragem, três bases por aminoácido. O resultado é uma cadeia de até 64
resíduos que se dobra conforme os ângulos de cada aminoácido.

- **Órgãos.** Certos pares de aminoácidos seguidos (promotor + modificador,
  mais um codão de intensidade) dão um órgão em vez de dois resíduos: boca,
  músculo, sensores (comida, luz, energia, corpos; totais ou de um só lado),
  relógio, relé, fotossistema, quimiossíntese, protease, âncora, ventosa,
  armazenamento, dormência, revisão, quiral, bias. As tabelas estão em
  `assets/` e editam-se com a simulação a correr.
- **Sinais.** Quatro canais internos percorrem a cadeia. Dois (α, β) dobram
  as juntas; dois (γ, δ) ligam e desligam órgãos.
- **Segundo gene.** Outro `AUG` depois da paragem dá um segundo corpo, preso
  ao primeiro por um fio mole.
- **Energia.** Vem de comer monómeros ativados, da luz (fotossistema), do
  redutor das fumarolas (quimiossíntese) ou de outros agentes (protease).

## O mundo

Grelha de 2048 × 2048 células com terreno de grãos (rocha e entulho), um
fluido, luz que entra por cima com ciclo de dia e noite, fumarolas no fundo
que dão calor e redutor, e uma sopa de monómeros em dois estados (ativado e
gasto). Suporta 400 000 agentes.

## Requisitos

- Windows com uma placa gráfica recente (desenvolvido numa NVIDIA; usa wgpu
  30). Não foi testado noutros sistemas.
- Rust 1.99 (a versão está fixada em `rust-toolchain.toml`; o `rustup`
  instala-a sozinho).
- Opcional: `ffmpeg` no PATH, para gravar vídeo.

## Como correr

```
run.bat
```

ou `cargo run --release`. A primeira compilação demora alguns minutos.

O mundo arranca vazio de vida: no painel da esquerda, **semear** põe genomas
ao acaso. Ao fechar, o estado fica em `saves/autosave.ribo` e é retomado no
arranque seguinte. **Mundo novo com os valores por omissão** recomeça do zero.

Há também `lab.bat` (piscina pequena de laboratório, sem terreno).

## Interface

- **Rato:** arrastar move, roda aproxima, clique seleciona um agente (abre o
  inspetor à direita, com genoma, corpo, órgãos e sinais).
- **Painel esquerdo:** parâmetros da simulação, terreno e pincel, vistas
  (monómeros, luz, temperatura, corrente...), cenas, gráficos.
- **Relatório:** gera uma página HTML com as espécies, quem pode atacar
  quem e a árvore das linhagens, interativa.
- **Capturas:** o mundo inteiro em 8k ou 16k; fotografia e vídeo (MP4) do
  enquadramento, com mira.
- **Agentes guardados:** gravar o genoma do selecionado, carregar um,
  espalhá-lo ou pô-lo com o rato.

A app abre um servidor MCP local (porta 8788) para ler o estado e mudar
parâmetros a partir de fora.

## Estrutura

| pasta | conteúdo |
|---|---|
| `src/` | aplicação: mundo, vida, desenho, interface, relatório |
| `shaders/` | simulação e desenho em WGSL (`world/`, `life/`, `render/`) |
| `assets/` | tabelas dos aminoácidos, dos órgãos e do código dos órgãos |
| `examples/` | sondas: programas pequenos que medem uma regra sem janela |
| `tests/` | testes, incluindo a conservação da matéria |
| `docs/` | arquitetura e notas da passagem do v3 para o v4 |

Para correr os testes: `cargo test --release` (precisam da placa).
Uma sonda: `cargo run --release --example probe_reach`.

## Estado

Projeto de investigação pessoal, em mudança constante. As regras e as
tabelas mudam com frequência e as cenas gravadas com uma versão podem
comportar-se de outra forma na seguinte.

## Licença

MIT (ver `LICENSE`).
