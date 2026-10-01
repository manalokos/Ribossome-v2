# Mapa de portagem: onde está cada sistema no v3

Raiz do v3: `C:\Filipe\ALsimulatorv3` (branch `rna-world`). Os números de
linha mudam; procura pelos nomes das funções.

## Ficheiros
| Ficheiro | Linhas | Conteúdo |
|---|---|---|
| `src/main.rs` | ~18 000 | tudo do lado CPU: GPU, UI egui, ciclo, snapshots, profiler, livro-razão |
| `shaders/shared.wgsl` | ~2 100 | tipos (Agent, BodyPart, SimParams), bindings, acessores de `chem`, tabela `AMINO_DATA`, utilitários |
| `shaders/simulation.wgsl` | ~5 000 | agentes, química, terreno, luz, reduções |
| `shaders/fluid.wgsl` | ~2 300 | fluido, temperatura, ativação térmica |
| `shaders/reproduction.wgsl` | ~700 | nascimentos e reconciliação de matéria |
| `shaders/compact_merge_minimal.wgsl` | ~120 | compactação estável e merge |
| `shaders/render.wgsl` | ~2 900 | desenho por compute (antigo), inspector, emissão de instâncias |
| `shaders/amino_render.wgsl` | ~190 | **render novo dos aminoácidos (portar tal e qual)** |
| `shaders/composite.wgsl` | ~450 | paleta dos monómeros, vistas de debug, composição |
| `shaders/microswim.wgsl` | ~290 | natação por deformação |
| `config/part_properties.json` | | **sobrepõe-se aos valores do shader ao arrancar** (no v4: uma só fonte) |

## Kernels a portar
| Sistema | Kernel / função | Ficheiro |
|---|---|---|
| Transporte + reações | `transport_quanta`, `parcel_hop_p` | simulation.wgsl |
| Acessores de monómeros | `chem_act_count`, `chem_spent_count`, `chem_spend_one`, `chem_activate_one`, `chem_take_state_one`, `chem_add_state`, `chem_open_cell`, `chem_capacity`, `chem_cell_total` | shared.wgsl / simulation.wgsl |
| Luz UV | `compute_uv_light`, `uv_light_at_cell` | simulation.wgsl |
| Terreno | `diffuse_grids_stage1/2` (relaxação de grãos), `gamma_move_one`, `compute_gamma_slope` | simulation.wgsl |
| Fluido | `advect_velocity`, `compute_divergence`, `jacobi_pressure`, `subtract_gradient`, `vorticity_confinement`, `enforce_boundaries`, `add_forces` | fluid.wgsl |
| Temperatura | `update_temperature`, `copy_temperature`, `temp_ambient_at`; flutuação em `inject_fumarole_force_vector` | fluid.wgsl |
| Ativação térmica | `inject_fumarole_dye` (o nome é histórico) | fluid.wgsl |
| Replicação | emparelhamento em `process_agents`; nascimento em `reproduce_agents` | simulation / reproduction |
| Compactação | `compact_agents`, `merge_agents_cooperative` (devolve a cadeia se a população estiver cheia) | compact_merge_minimal.wgsl |
| Livro-razão | `reduce_matter_chem` (8 contadores) + `measure_free_quanta` (Rust) | simulation.wgsl / main.rs |
| Espécies | `compute_species_hash` (FNV-1a do conjunto de partes) | simulation.wgsl |
| Paleta | `composite_agents` (tom por fração de ativados, opacidade, teclas 1–8) | composite.wgsl |
| Enzimas | bloco "CATALYTIC TRIPEPTIDES" em `process_agents` | simulation.wgsl |
| Difusioforese | bloco "DIFFUSIOPHORESIS" + `ate_here` na alimentação | simulation.wgsl |

## Constantes afinadas (valores no momento da passagem)

### Química (simulation.wgsl)
| Constante | Valor | Nota |
|---|---|---|
| CHEM_CELL_CAP | 48 | 0 em células com gamma |
| CHEM_SQUEEZE_P | 0.25 | expulsão quando acima da capacidade |
| DIFF_HOP_P | 0.01 | × slider de difusão |
| DIFF_HOP_FLOOR / DIFF_AGITATION_SPEED | 0.15 / 0.5 | difusão depende da agitação |
| MONOMER_ADV_CAP | 0.9 | |
| MONOMER_SETTLE_P | 0.002 | × slider de gravidade |
| LIGHT_ACT_P | 0.006 | fotoativação |
| CHEM_DECAY_P | 0.0002 | |
| CHEM_SENSITIZE / CHEM_SHIELD / CHEM_COHESION | 0.5 / 0.5 / 0.6 | |
| CHEM_INTERCONV_P | 0.0 | retirado (enzimas no lugar) |
| UPTAKE_P_PER_QUANTUM | 0.01 | |
| ENZYME_RATE / ENZYME_COST | 0.1 / 0.05 | motivos H-W-Y (A↔G), C-M-F (U↔C), Q-W-N (A↔U), Y-M-P (G↔C) |
| PAIRING_REACH | 2.5 | |

### Luz, temperatura, fluido
| Constante | Valor | Ficheiro |
|---|---|---|
| UV_SHADOW_ABSORB / UV_SPREAD | 0.6 / 4 | simulation |
| TEMP_HEAT_RATE / TEMP_COOL_RATE / TEMP_MAX | 0.01 / 0.12 / 12 | fluid |
| TEMP_ACT_THRESHOLD / FUMAROLE_ACT_P | 2.0 / 0.15 | fluid |
| TEMP_BUOYANCY | 12 | fluid |
| TEMP_AMBIENT_SURFACE / TEMP_AMBIENT_ATTEN | 0 / 4 | fluid |
| SUN_HEAT_RATE / SUN_WATER_ABSORB | 0.15 / 0.05 | fluid |
| TEMP_DIFFUSE | 0.08 | fluid |
| Vorticidade | limitada a 10 (3–5 recomendado) | main.rs |
| Jacobi | par, 10–40 | main.rs |

### Terreno
| Constante | Valor |
|---|---|
| GAMMA_SOLID_THRESHOLD | 3 |
| SANDPILE_DIFF / GAMMA_RELAX_P | 3 / 0.12 |
| GAMMA_COHESION_P / GAMMA_SHED_P / GAMMA_STRAY_WALK_P | 0.01 / 0.03 / 0.05 |
| GAMMA_PILE_MAX | 6 |
| DIG_P_PER_FRAME | 0.3 |
| GAMMA_EJECT_FORCE / GAMMA_SOLID_BRAKE | 30 / 1.0 |

### Organismos
| Constante | Valor |
|---|---|
| AGENT_SINK_RATE | 0.05 |
| PHORETIC_GAIN / PHORETIC_MAX_VEL | 100 / 3 |
| UPTAKE_ENABLER_BOOST | 3 (desaparece com os enablers) |
| COLD_DEATH_MULT / HOT_DEATH_MULT | 0.1 / 10 (T ≥ 8) |
| UV_HAZARD_SCALE | 0.001 |
| VEL_MAX / MORPHOLOGY_SWIM_MAX_FRAME_VEL | 24 / 2 |

### Mundo
| Grandeza | Valor |
|---|---|
| SIM_SIZE | 61440 unidades (para ambiente 2048) |
| Ambiente / fluido / grelha espacial | 2048² / 512² / 1024² |
| Máximo de agentes | 60 000 (com ambiente 2048) |
| MAX_BODY_PARTS / genoma | 64 / 256 bases |

## Dados reais dos aminoácidos (já usados no v3)
Massa do resíduo (Da), volume da cadeia lateral (Å³), Chou-Fasman Pa/Pb/Pt:

```
A  71.08  88.6 1.42 0.83 0.66   C 103.14 108.5 0.70 1.19 1.19
D 115.09 111.1 1.01 0.54 1.46   E 129.12 138.4 1.51 0.37 0.74
F 147.18 189.9 1.13 1.38 0.60   G  57.05  60.1 0.57 0.75 1.56
H 137.14 153.2 1.00 0.87 0.95   I 113.16 166.7 1.08 1.60 0.47
K 128.17 168.6 1.16 0.74 1.01   L 113.16 166.7 1.21 1.30 0.59
M 131.19 162.9 1.45 1.05 0.60   N 114.10 114.1 0.67 0.89 1.56
P  97.12 112.7 0.57 0.55 1.52   Q 128.13 143.8 1.11 1.10 0.98
R 156.19 173.4 0.98 0.93 0.95   S  87.08  89.0 0.77 0.75 1.43
T 101.10 116.1 0.83 1.19 0.96   V  99.13 140.0 1.06 1.70 0.50
W 186.21 227.8 1.08 1.37 0.96   Y 163.18 193.6 0.69 1.47 1.14
```

Mapeamento usado no v3:
- massa = 0.02·Da/118
- espessura = 4·√(vol/130)
- comprimento = 11 (cadeia principal constante)
- dobra = clamp(0.55·(Pt−0.6) + 0.25·max(Pa−1, 0), 0, 0.6) rad, sempre com o mesmo sinal

Classes: alifáticos A I L M V, aromáticos F W Y, polares S T N Q (C à parte),
positivos K R H, negativos D E, G e P especiais.

Para o v4 ainda falta: hidrofobicidade (Kyte-Doolittle), carga a pH 7 e a
matriz de Miyazawa–Jernigan (20×20; procurar a tabela publicada de 1996).
