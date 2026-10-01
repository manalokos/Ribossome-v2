# Estado do v3 e pendentes (2026-10-01)

Último trabalho no branch `rna-world`, com backup na tag
`backup-pre-amino-redesign-2026-10-01` (no GitHub `Manalokosdev/Ribossome`).

## Feito na última sessão
- Difusioforese com fluxo de consumo suave e rasto de dipolo no fluido.
- Microswimming só em baixo Reynolds.
- Dano UV que não depende da energia.
- Varredores desligados (`SWEEPERS_ENABLED = false`); passe de drain retirado;
  desenho dos cílios retirado.
- Aminoácidos com propriedades reais (massa, volume, Chou-Fasman, cor por
  classe), no shader e no `config/part_properties.json`.
- Render por hardware dos aminoácidos (`amino_render.wgsl`): instâncias
  emitidas em `render_agents` (grupo 1, bindings 2/3), passe de render para
  `amino_texture` e composição por baixo dos órgãos. **Ainda não confirmado
  visualmente pelo Filipe.**
- Profiler: segmento "backlog" separado e "post" partido em partes.

## Tabela de pendentes
| Frente | Estado | Onde fazer |
|---|---|---|
| Confirmar o render por hardware (visual e tempo) | à espera do Filipe | v3 |
| Medir o "post" partido (`--release`) | à espera do Filipe | v3 |
| Chuva que não conserva (`rain.wgsl` escreve por cima da célula) | decidir: desligar de vez | v3 (barato) |
| Render por hardware no inspector | por fazer | v4 |
| Dobragem ao nascer (Miyazawa–Jernigan) | por fazer | v4 |
| Ângulos que respondem ao ambiente (no lugar dos sensores) | por fazer | v4 |
| Propagação mecânica (no lugar dos sinais) | por fazer | v4 |
| Retirar os órgãos antigos | por fazer | v4 (não há no v4) |
| Rever o microswimming com RFT (perpendicular ≈ 2× paralelo) | por fazer | v4 |
| Quadrante 1024² diferente dos outros | desapareceu sozinho, causa por encontrar | ver abaixo |
| Mutações de inserção que criam bases | por rever | v4 (reconciliar desde o início) |
| Telemetria exportável (CSV por corrida) | combinado, por fazer | v4 |
| Limpeza do painel de controlo | parcial | v4 (UI nova) |

## O mistério do quadrante
Numa corrida a 2048², o quadrante inferior esquerdo (1024×1024) passou a ter
mais monómeros gastos do que o resto, com uma fronteira perfeitamente reta a
meio dos dois eixos. Desapareceu sem nenhuma mudança de código. Já foi
descartado: tamanhos dos dispatches, hash, ativação térmica, luz UV,
aquecimento solar, chuva, merge e composite. Hipóteses: estado antigo de uma
resolução de 1024 (snapshot ou autosave) ou o plano de luz desatualizado. No
v4, se aparecer: ver logo as teclas 7 e 8 e anotar o que se fez antes.

## Sugestão de passagem
Os passos grandes do organismo fazem-se no v4. No v3 só vale a pena fechar o
que é barato e útil: confirmar o render novo, medir o "post" e desligar a chuva.
