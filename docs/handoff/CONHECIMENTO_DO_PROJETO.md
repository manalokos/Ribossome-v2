# Conhecimento do projeto Ribossome (passagem v3 → v4)

Escrito a 2026-10-01 pela sessão que levou o v3 até aqui. A ideia é que
quem pegar no v4 não tenha de redescobrir nada.

---

## 1. A visão

O Ribossome é um **aquário de mundo de RNA**: uma "sopa primordial" 2D
fechada, onde moléculas replicantes (genomas de RNA traduzidos em cadeias
de aminoácidos) competem por monómeros. O Filipe quer **realismo científico
honesto** e **emergência**: quer ver comportamentos surgir de regras químicas
simples e globais, e não de mecanismos desenhados.

Frases do Filipe que resumem a filosofia:
- "a criatura não sabe o seu genoma"
- "isso acontece na realidade?"
- "não podem existir monómeros em rocha ou rubble"
- "o dye nem deveria existir" (tudo o que se vê deve ser matéria real)
- "a solução que temos agora é um Frankenstein entre RNA world e proteínas"

## 2. O mundo físico (v3, validado)

### Monómeros (a matéria)
- 4 nucleótidos A, U, G, C. Cada célula da grelha (2048²) guarda, por canal,
  um u32: **16 bits baixos = ativados** (com energia, servem para comer e
  emparelhar), **16 bits altos = gastos** (inertes).
- **Capacidade de 48 por célula**, e **zero se houver gamma** (rocha ou
  entulho). Isto foi decidido depois de uma longa saga: a rocha porosa
  acumulava matéria e "comia" o mundo.
- Comer = ativado → gasto **no lugar** (a matéria nunca sai). A energia do
  agente é ativação colhida e evapora na morte. Converter energia em matéria
  criaria massa.
- Ativação: UV vindo de cima (com sombras do gamma) e calor (água acima de
  T=2,0). Decaimento lento de ativado para gasto.
- Reações locais (regras de autómato celular): fotosensibilização (ativados
  ajudam a ativar vizinhos), blindagem (ativados juntos decaem menos) e
  coesão (difusão enviesada para vizinhos ativados do mesmo tipo). A
  interconversão A↔G foi retirada por ser irrealista e substituída por
  **enzimas** (tripéptidos).
- Transporte: saltos quantizados. Advecção pelo fluido (norma L1, velocidade
  bilinear), difusão, assentamento por gravidade leve e *squeeze* quando uma
  célula está acima da capacidade.
- Depósitos (morte, reconciliação) vão sempre para a célula livre de gamma
  mais próxima (raio 12).

### Lições de conservação (custaram muitas horas)
- "Estou a perder monómeros" teve **cinco causas diferentes**, uma de cada vez:
  rocha a acumular, terreno a enterrar monómeros, depósitos diretamente em
  gamma, o merge a descartar os nascimentos 2000–2047 (limite 2000 contra
  2048 pedidos), e mutações de inserção/remoção a criar ou destruir bases.
- O instrumento certo: um **livro-razão** (total livre + total preso em
  agentes) mostrado em permanência. Variação aceitável: ~0,05% (ruído de leitura).
- Mundos fechados morrem de **entropia**: numa corrida de 9 milhões de epochs
  a difusão espalhou tudo de forma uniforme e tudo morreu. Os mecanismos
  anti-entropia (difusão que depende da agitação, coesão, gravidade,
  convecção) são essenciais.

### Fluido
- Stable Fluids em 512². Operadores consistentes (divergência central,
  Jacobi de 5 pontos, gradiente central); operadores inconsistentes deixavam
  vórtices permanentes em malha.
- **Pressão com arranque quente** (não limpar entre frames) e número par de
  iterações de Jacobi. Arrancar do zero deixava a pressão por convergir e
  criava "bolhas sem monómeros" nas plumas, um bug que o Filipe perseguiu
  muito tempo.
- Temperatura: campo próprio advectado de forma bilinear, flutuação pelo
  desvio ao ambiente local, arrefecimento. O ambiente é mais frio em baixo e
  junto às bordas. Fumarolas como fontes de calor (gizmos arrastáveis e
  redimensionáveis). Aquecimento solar leve por absorção, em que as rochas
  iluminadas aquecem mais e geram convecção.
- Agentes levados pelo fluido: deslizador de 0 a 1, onde 1 = exatamente à
  velocidade da água e 0 = imune. Já houve várias versões erradas.
- Jacobi 40 contra 10 iterações: cerca de 20 fps de diferença. O fluido era
  o maior custo antes da otimização.

### Luz UV
- Varrimento por linhas a partir de cima. Cada célula recebe a **média de 3
  células acima** (espaçadas 4), × atenuação da água × absorção do gamma
  (exp(−0,6·g)). Dá sombras difusas. O Filipe pediu explicitamente "a média
  das 3 de cima".
- Calculada de 100 em 100 epochs.

### Terreno (gamma)
- Quanta inteiros de "rocha". ≥3 por célula = sólido; 1–2 = entulho. O fluido
  trata a célula como parede pela maioria dos vizinhos.
- Física de grãos: relaxação de declive, coesão (o fino migra para o maior),
  escavação por pressão, e o corpo empurra grãos para trás de si.
- Quando um grão se move para uma célula com monómeros, estes são trocados
  para a origem (`gamma_move_one`), para não ficarem presos.

## 3. Os organismos (v3, a substituir)
- Genoma: 256 bases (2 bits cada), tradução por codões a partir de AUG.
  Corpo de até 64 partes: 20 aminoácidos + 27 "órgãos" formados por pares
  promotor+modificador.
- **Replicação:** o corpo inteiro captura monómeros complementares ativados
  (alcance de 2,5). Quando o emparelhamento termina, nasce um filho. As
  mutações são reconciliadas: o excedente é depositado e o défice retirado
  da vizinhança.
- Morte: devolve a cadeia e os complementos já emparelhados. As sementes da
  geração 0 não devolvem a cadeia, porque essa matéria foi criada do nada.
- Mortalidade: base dividida pela energia, × térmica (frio ×0,1, quente
  ×10), + risco de UV que não depende da energia (antes era multiplicado e os
  agentes bem alimentados à superfície ficavam imunes).
- Movimento: microswimming por deformação (agora só de baixo Reynolds),
  propulsores, difusioforese (deriva pela assimetria do consumo, ganho 100,
  limite 3/frame) e "pesca" pelo fluido. Os agentes afundam levemente.
- **Retirados no v3:** vampiros, espinhos, predação, varredores (desligados
  porque exploravam todas as falhas: juntavam comida à superfície, etc.), dye
  e chuva (por desligar).

### Comportamentos emergentes e falhas observadas
- As populações tendem a concentrar-se onde a luz é forte (superfície).
  Contrapesos: dano UV que não depende da energia, e água fria segura em baixo.
- Os varredores com custo quase nulo tornaram-se acumuladores. **Qualquer
  ação que mova matéria tem de ter um custo proporcional ao que rende.**
- Extinção com comida por todo o lado: aconteceu por desequilíbrio da
  composição (poucos monómeros livres de certos tipos). Mesmo com reprodução
  assexuada, os complementos têm de existir.

## 4. Desempenho (medido no v3, release, cerca de 7k agentes)
- GPU por frame ≈ 7–9 ms: post (compactação, merge, desenho) 2,3–3,4 ms;
  `process_agents` 2,2 ms; fluido 0,2 ms (2,3 ms quando resolve a pressão);
  transporte de monómeros só 0,3 ms.
- A compactação estável corria num único workgroup a copiar agentes de
  2192 bytes. No v4: compactação em 3 passes e agente com dados quentes e
  frios separados.
- O desenho por compute (uma thread por agente a desenhar píxeis) era lento
  com zoom e dava "pintinhas" com o zoom afastado. Já está substituído no v3
  por quadrados instanciados com SDF para os aminoácidos.

## 5. Direção do v4 (decidida com o Filipe)
- Caminho **B**: mundo de RNA com tradução para proteínas (o nome é
  Ribossome), mas **sem órgãos escritos à mão**.
- Aminoácidos com **propriedades reais**: massa (Da), volume, carga,
  hidrofobicidade, Chou-Fasman (hélice/folha/volta), flexibilidade.
- **Dobragem ao nascer** com a matriz de contactos de **Miyazawa–Jernigan**
  (o Filipe perguntou por "uma lookup table": é esta).
- **Sensores substituídos por mudança de conformação** (alosteria
  ambiental): ligação de monómeros, temperatura, UV nos aromáticos.
- **Sinais internos substituídos por propagação mecânica** ao longo da
  cadeia (acoplamento por par de aminoácidos).
- **Domínios químicos** (como os tripéptidos catalíticos) em vez de órgãos.
- Natação por **RFT** (teoria das forças resistivas), respeitando o teorema
  da vieira.
- Render: cada aminoácido desenhado pela sua química (cadeia principal +
  cadeia lateral; aromáticos como anel, carregados com + ou −, prolina como
  anel na cadeia principal, glicina sem cadeia lateral). Feito por hardware.
- Homoquiralidade: todas as dobras para o mesmo lado (2D). Fica para
  confirmar com o Filipe se lhe agrada.

## 6. Como o Filipe trabalha
- Corre sessões longas e observa visualmente. Manda screenshots com
  descrições curtas ("bolsas de vácuo", "cascatas", "loophole").
- Gosta de **sliders** para afinar ao vivo e de **vistas de debug** por tecla
  (1–4 nucleótidos ativados, 6 gamma, 7 temperatura, 8 UV).
- Gosta de propostas com opções e uma recomendação clara. Quando diz "ok",
  "avança" ou "vamos a isso", quer implementação completa.
- Odeia regressões silenciosas. Prefere "não sei a causa" a uma explicação
  inventada.
- Pede backups antes de mudanças grandes (git tag e push).
- Corre com `cargo run` e às vezes esquece o `--release`. Lembrar quando os
  números de desempenho vierem de uma versão debug.

## 7. Ferramentas que existiam no v3 (vale a pena recriar)
- Validador naga exato (projeto cargo com `naga = "=22.1.0"` e `wgsl-in`) que
  valida os módulos concatenados com as constantes reais.
- Profiler por segmento (`ALSIM_PROFILE=1 ALSIM_PROFILE_DISPATCH_TIMING=1
  ALSIM_PROFILE_DISPATCH_TIMING_DETAIL=1`). Os timestamps da GPU
  (`ALSIM_GPU_TIMESTAMPS`) davam *device lost* no driver dele.
- Painel de conservação com Δ% em relação a uma base.
- Registo do agente selecionado em CSV, para depurar física.
- Snapshots como PNG com os dados embutidos, e autosave.
