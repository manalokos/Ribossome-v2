# Ribossome v4 — Documento de arquitetura

Estado: **proposta para discussão** (2026-10-01). Nada aqui está implementado.

---

## 1. Porquê um simulador novo

O Ribossome atual (v3, branch `rna-world`) chegou a um ponto em que cada
mudança custa mais do que devia:

- `src/main.rs` tem mais de 18 000 linhas (GPU, UI, simulação, snapshots,
  gravação e definições, tudo junto).
- O bind group principal tem as **20/20 bindings ocupadas**. Os sistemas novos
  só cabem com grupos secundários improvisados.
- Há várias camadas de sistemas retirados ou meio retirados: dye, vampiros,
  espinhos, varredores, chuva que não conserva matéria, overrides de
  propriedades das partes, 27 órgãos com comportamento escrito à mão, sinais
  alfa/beta abstratos.
- O modelo de organismo que queremos (aminoácidos reais, dobragem, forma que
  responde ao ambiente) substitui quase todo o `process_agents` atual.

O v3 fica congelado como referência (tag `backup-pre-amino-redesign-2026-10-01`)
e serve para comparar resultados.

## 2. Princípios (não negociáveis)

1. **Mundo fechado, matéria exata.** Todas as mudanças de matéria são trocas
   inteiras e atómicas. A matéria total é constante ao quantum. Há um teste
   automático que o verifica.
2. **Emergência acima de engenharia.** As capacidades são **química global**,
   igual para todos. A sequência só decide *onde* e *quanto* de cada motivo
   existe. Não há parâmetros evolutivos por agente nem regras "este órgão
   faz X".
3. **A criatura não conhece o seu genoma.** Nenhuma regra lê o genoma de
   outro agente nem compara genomas.
4. **Sem sexo nem crossover desenhados.** A reprodução é replicação por
   emparelhamento de complementos.
5. **Física de baixo Reynolds.** Movimento sobreamortecido (a velocidade é
   proporcional à força) e nenhuma propulsão por inércia.
6. **Medir antes de otimizar.** O profiler por kernel faz parte da arquitetura
   desde o primeiro dia.

## 3. O que portar do v3 e o que deixar

| Portar (afinado e funciona) | Deixar |
|---|---|
| Grelha de monómeros quantizada (ativado/gasto, 4 canais), transporte, capacidade por célula, *squeeze* | Dye e os seus passes |
| Reações: fotoativação, sensibilização, blindagem, decaimento, coesão | Chuva (não conserva) e mapas de chuva |
| Fluido Stable Fluids: pressão com arranque quente, vorticidade, temperatura, convecção | Vampiros, espinhos, varredores, drain |
| Terreno gamma quantizado (rocha/entulho sem monómeros), relaxação estocástica | 27 órgãos escritos à mão, promotores/modificadores |
| Luz UV por varrimento de linhas, com sombras difusas | Sinais alfa/beta abstratos, enablers |
| Replicação por emparelhamento e reconciliação de matéria; morte devolve matéria | Overrides de propriedades/ângulos (CSV/JSON) |
| Tripéptidos catalíticos (como primeiro "domínio químico") | Microswim de alto Reynolds |
| Render por hardware dos aminoácidos (SDF instanciado) | Desenho de agentes por compute (uma thread por agente) |
| Validador naga 22, profiler por segmento | `main.rs` monolítico |

## 4. Estrutura do código

Workspace Rust com módulos pequenos e responsabilidades claras:

```
ribossome4/
  src/
    main.rs            arranque, loop de eventos (curto)
    gpu/               device, buffers, layouts, pipelines, profiler
    world/
      chem.rs          grelha de monómeros, transporte, reações
      fluid.rs         Stable Fluids + temperatura
      terrain.rs       gamma, relaxação
      light.rs         UV
    life/
      genome.rs        genoma, mutação, tradução
      amino.rs         tabela de aminoácidos (dados reais, um ficheiro)
      fold.rs          dobragem ao nascer (Miyazawa–Jernigan)
      body.rs          física do corpo, conformação, alosteria
      metabolism.rs    alimentação, energia, morte
      replicate.rs     emparelhamento, nascimento
    render/            textura do mundo, aminoácidos (SDF), composite, inspector
    ui/                painéis egui (um ficheiro por painel)
    io/                snapshots, definições, gravação
  shaders/
    common/            tipos partilhados (gerados ou incluídos UMA vez)
    world/  life/  render/
  tests/
    conservation.rs    corre N frames sem UI e verifica matéria total
```

**Shaders:** os tipos partilhados (Agent, Params) existem num único sítio. O
v3 tinha 3 cópias de SimParams que tinham de ser mantidas em sincronia à mão;
no v4 são gerados a partir de Rust (ou incluídos uma vez).

## 5. GPU: buffers e bind groups

Um bind group por sistema, em vez de um grupo gigante partilhado:

| Grupo | Conteúdo | Quem usa |
|---|---|---|
| 0 — frame | Params do frame (uniform), câmara | todos |
| 1 — mundo | chem (u32 ×4/célula), gamma (altura), luz, temperatura | química, organismos, render |
| 2 — fluido | velocidade, pressão, divergência, forças | fluido, organismos |
| 3 — organismos | agentes A/B (ping-pong), pedidos de nascimento, grelha espacial | organismos, render |

Notas:
- O limite por grupo deixa de ser problema, porque cada pipeline só liga os
  grupos de que precisa.
- O **agente** separa os dados quentes (posição, velocidade, energia, contagem
  de partes; lidos todos os frames) dos frios (genoma; lido só ao nascer e na
  replicação). No v3, cada agente tem 2192 bytes e a compactação copia tudo.
- **Compactação estável** em 3 passes (contar por bloco → prefixo → espalhar),
  em vez de um único workgroup a copiar tudo.
- `chem` pode ficar com 2×u32 por célula (4 canais × (8 bits ativados + 8 bits
  gastos)), porque a capacidade é 48. Isto reduz a largura de banda para
  metade. A decidir depois de medir.

## 6. Ciclo de um frame

```
1. world:  luz (de 100 em 100 frames) → temperatura → fluido (forças, advecção, pressão)
2. world:  transporte de monómeros + reações (incl. ativação térmica)
3. world:  terreno (de 4 em 4 frames)
4. life:   grelha espacial → corpo (conformação + forças + colisões)
5. life:   metabolismo (alimentação, catálise, custos, morte → devolve matéria)
6. life:   replicação (emparelhamento, nascimentos; dobragem dos recém-nascidos)
7. life:   compactação (3 passes)
8. render: mundo → textura; aminoácidos (instanciados) → textura; composite
9. ui
```

A telemetria (conservação, população, espécies) é lida de forma assíncrona,
com um frame de atraso e sem bloquear.

## 7. O mundo

Igual ao v3 no comportamento (já validado), reorganizado:

- **Química:** 4 nucleótidos, cada um ativado ou gasto. A capacidade por célula
  é 48 e é zero dentro de rocha ou entulho. Transporte por saltos quantizados:
  advecção pelo fluido (norma L1), difusão, assentamento (gravidade leve).
  Reações locais: fotoativação, sensibilização, blindagem, decaimento, coesão.
- **Fluido:** Stable Fluids com arranque quente, número par de iterações de
  Jacobi, vorticidade limitada, temperatura advectada, convecção por desvio
  térmico, fumarolas e aquecimento solar leve por absorção.
- **Luz UV:** vem de cima. Média de 3 tomadas espaçadas por linha, atenuação da
  água e absorção pelo gamma.
- **Terreno:** gamma quantizado com relaxação estocástica. Quando o terreno se
  move, desloca monómeros (troca exata).

## 8. O organismo (a parte nova)

### 8.1 Genoma e tradução
- Genoma de RNA (A/U/G/C, 2 bits por base), tradução por codões a partir do
  AUG, código genético padrão.
- A cadeia de aminoácidos é o corpo, até 64 resíduos.
- Mutação pontual e indels, com o custo de matéria reconciliado (nada é criado
  nem destruído).

### 8.2 Aminoácidos: só dados reais
Um ficheiro `amino.rs`, com uma linha por aminoácido e propriedades medidas:
massa (Da), volume da cadeia lateral, carga a pH 7, hidrofobicidade
(Kyte-Doolittle), tendência para hélice, folha ou volta (Chou-Fasman) e
flexibilidade. **Nenhum outro número por aminoácido.**

### 8.3 Dobragem ao nascer

> **Retirada (out. 2026).** Custava O(n²) por agente nos primeiros passos de
> vida e pouco acrescentava num modelo 2D com um segmento por resíduo. O
> corpo nasce com o ângulo de repouso de cada aminoácido (`REST_ANGLE`,
> valores do v3), como no v3. O texto abaixo fica como registo.

- Energia de contacto da **matriz de Miyazawa–Jernigan** (20×20, valores
  medidos), mais uma penalização de ângulo pela tendência local.
- Cerca de 50 iterações de relaxação quando o agente nasce, num kernel que
  corre só para os recém-nascidos. O resultado é guardado como **ângulo base
  de cada junta**.
- Como a sequência não muda durante a vida, o custo por frame é zero.

### 8.4 Conformação que responde ao ambiente (substitui os sensores)
Cada junta tem um ângulo atual que relaxa para um alvo:
`alvo = base + Σ efeitos locais`. Os efeitos são regras químicas globais:
- **Ligação de monómeros ativados** a resíduos carregados ou polares, conforme
  a concentração local.
- **Temperatura:** o calor aumenta a flexibilidade e puxa para o ângulo neutro
  (desnaturação); o frio torna a junta mais rígida.
- **UV:** os aromáticos (W, Y, F) absorvem e mudam de conformação.
- **pH ou carga local**, se vier a existir um campo para isso.

### 8.5 Alosteria (substitui os sinais internos)
- A deformação de uma junta propaga-se às vizinhas com um acoplamento que vem
  de uma tabela 20×20 (por par de aminoácidos), com atenuação.
- Os "circuitos" emergem da sequência: um estímulo numa ponta pode fazer
  bater a cauda.

### 8.6 Movimento
- Sobreamortecido, por teoria das forças resistivas (RFT): arrasto anisotrópico
  por segmento, com coeficiente perpendicular cerca de 2× o paralelo. A
  propulsão vem da deformação do corpo, e o teorema da vieira aplica-se (um
  movimento recíproco não avança).
- Difusioforese a partir da assimetria do consumo (já existe no v3).
- O corpo é arrastado pelo fluido (deslizador 0–1) e afunda levemente com a
  gravidade.

### 8.7 Metabolismo
- A energia é ativação colhida. O monómero gasto fica onde estava, por isso a
  matéria nunca sai da célula.
- A alimentação é feita pelas partes do corpo e regulada pela fome.
- **Catálise por motivos de 3 resíduos** (já existe): a ordem dos resíduos
  define o sentido da reação, e há um custo de energia.
- Mais tarde, outros "domínios químicos" com a mesma lógica (a sequência ativa
  uma regra global): transporte, ligação a superfícies, etc.
- Mortalidade térmica e por UV (o dano UV não depende da energia).

### 8.8 Replicação
- Captura de complementos do meio para emparelhar o genoma. Quando o
  emparelhamento termina, nasce a cópia e a matéria é reconciliada.
- Quando não cabe mais nenhum agente, a cadeia do recém-nascido é devolvida
  ao meio.

## 9. Render

- **Mundo:** um compute por píxel (paleta ativado/gasto, vistas de debug 1–8).
- **Aminoácidos:** quadrados instanciados com glifos SDF (já existe no v3). O
  inspector usa o mesmo pipeline com outra câmara.
- **Composite:** junta as camadas. Os rastos dos agentes são opcionais.
- **Sem desenho por agente em compute.** O custo depende dos píxeis cobertos e
  não do número de agentes.

## 10. Ferramentas e qualidade

- **Validação de shaders** (naga com a versão do wgpu) no `cargo test`, e
  verificação de que todos os `entry_point` existem.
- **Teste de conservação sem UI:** mundo pequeno, N frames, matéria total
  constante ao quantum.
- **Profiler por kernel**, com a espera pelo trabalho pendente medida à parte.
- **Snapshots** com versão do formato. Os do v3 não são compatíveis; vale a
  pena só um conversor de genomas.

## 11. Fases

1. **Esqueleto:** janela, câmara, profiler, validação de shaders, teste de
   conservação.
2. **Mundo:** química, fluido, temperatura, luz e terreno portados e validados
   (conservação exata, desempenho igual ou melhor que o v3).
3. **Organismo mínimo:** genoma, tradução, corpo rígido, alimentação,
   replicação, morte. Primeira população que se mantém.
4. **Render:** aminoácidos instanciados, inspector, gráficos de população e
   espécies.
5. **Dobragem ao nascer** (Miyazawa–Jernigan).
6. **Movimento RFT** e difusioforese.
7. **Conformação ambiental**, que substitui os sensores.
8. **Alosteria**, que substitui os sinais.
9. **Domínios químicos** além da catálise.

Cada fase termina com o teste de conservação e uma medição de desempenho.

## 12. Questões em aberto

- Nome e repositório: projeto novo, ou pasta nova no mesmo repositório?
- Resolução por omissão (2048² de ambiente, 512² de fluido) e número máximo de
  agentes.
- Mundo 2D com dobragem 2D: aceitamos que as hélices viram curvas
  homoquirais (sempre para o mesmo lado)?
- A escala de tempo da dobragem é instantânea ao nascer ou visível
  (o recém-nascido a dobrar-se durante alguns frames)?
- Interface: manter egui, e que painéis são essenciais no início?
- Linguagem: ver §13 (recomendado: Rust + wgpu atualizado).

## 13. Linguagem e stack

Opções consideradas:

| Opção | Prós | Contras |
|---|---|---|
| **Rust + wgpu (WGSL), versão atual do wgpu** | todos os shaders do v3 portam diretamente; corre em qualquer GPU (Vulkan/DX12/Metal); egui já conhecido; segurança de memória | compilação lenta; WGSL é mais limitado que CUDA |
| C++ / CUDA | compute mais maduro, ferramentas de profiling excelentes (Nsight) | só NVIDIA; recomeçar render/UI; perde os shaders do v3 |
| Python + Taichi / NVIDIA Warp | protótipos muito rápidos de kernels | interface e render em tempo real mais pobres; desempenho final inferior; tudo por reescrever |
| Motor de jogo (Unity / Godot compute) | editor, UI e render prontos | camada pesada por cima; controlo fino da GPU mais difícil |

**Recomendação: manter Rust + wgpu, mas atualizar para a versão atual do
wgpu/naga.** Os problemas do v3 vieram da arquitetura (monólito, bindings
esgotadas, sistemas mortos) e não da linguagem. Atualizar o wgpu deve também
levantar restrições do naga 22, como a de indexar arrays locais (confirmar ao
começar). Se, mais tarde, a investigação pedir iteração rápida de regras
químicas, um protótipo em Taichi/Warp ao lado é uma opção, mas não como base.

## 14. Ponto de situação do v3 na passagem (2026-10-01)

| Frente | Estado | Onde fazer |
|---|---|---|
| Aminoácidos com propriedades reais | ✅ feito | — |
| Render por hardware dos aminoácidos | ✅ feito, falta confirmação visual | v3 |
| Render por hardware no inspector | por fazer | v4 |
| Dobragem ao nascer (Miyazawa–Jernigan) | por fazer | v4 |
| Ângulos que respondem ao ambiente | por fazer | v4 |
| Propagação mecânica (no lugar dos sinais) | por fazer | v4 |
| Retirar os órgãos antigos | por fazer | v4 (não existem) |
| Medir o "post" partido em partes | à espera de corrida `--release` | v3 |
| Chuva que não conserva matéria | decidir (desligar de vez) | v3 |
| Rever o microswimming com RFT | por fazer | v4 |
| Quadrante 1024² diferente dos outros | desapareceu, causa por encontrar | — |

Os passos grandes do organismo ficam para o v4: fazê-los no v3 seria
construir sobre a estrutura que vai ser substituída. Os detalhes da passagem
estão em `v4_kickoff/`.
