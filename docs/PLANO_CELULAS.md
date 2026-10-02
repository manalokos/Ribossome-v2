# Plano: protocélulas (variante "células" do Ribossome)

Estado: **proposta, por aprovar.** Nada disto está implementado.

## 1. Porquê

O modelo atual é um mundo de RNA nu: cada agente é um genoma com uma
proteína colada, e tudo o que faz beneficia-o diretamente. Falta-lhe a
**compartimentação**, o passo que a biologia considera essencial na origem
da vida. Sem membrana, uma molécula que só se copia e não contribui nada
ganha sempre (o paradoxo de Eigen, o problema dos parasitas). Dentro de uma
vesícula, os genes que trabalham juntos ficam juntos e a seleção passa a
atuar sobre o conjunto.

Objetivo: bolhas de membrana com um genoma de RNA lá dentro, que crescem,
se dividem por física, herdam o conteúdo do seu lado e evoluem. Nenhum
comportamento é programado: quimiotaxia, controlo da divisão e
especialização têm de emergir.

## 2. A ideia central

**Cada agente é uma célula: tem a sua membrana e o seu conteúdo é contado
dentro dele.**

- **Membrana:** um anel de N nós (começa em 32) que reaproveita os arrays por
  resíduo que já existem (64 posições por slot). É sobreamortecido, como as
  juntas de hoje.
- **Conteúdo:** contagens INTEIRAS por agente (nucleótidos ativados e gastos
  por canal, lípido livre, etc.). Contam no livro-razão como matéria
  "presa", como hoje o genoma.
- **Genoma:** o RNA atual. As cópias são partículas pontuais dentro da
  célula (posição local).
- **Divisão por toque:** quando dois nós **não vizinhos** da membrana (a mais
  de N/4 posições ao longo do anel) se tocam, o anel parte-se ali em dois
  anéis e nasce um filho num slot novo. Não há regra "divide-te agora": a
  divisão acontece quando a forma da bolha a fecha sobre si própria.
- **Herança pelo lado:**
  - **moléculas abundantes:** repartem-se pela área de cada lado (binomial,
    ao quantum);
  - **proteínas de membrana:** ficam com os nós onde estão (podem acabar
    todas numa filha);
  - **cópias do genoma:** vão para o lado onde estão. Uma filha sem genoma
    vive do que herdou e acaba por morrer.

### O que faz a bolha fechar-se sobre si própria

É a pergunta física mais importante.

- Uma bolha 2D com **perímetro em excesso** para a sua área (cresceu a
  membrana mais depressa que o conteúdo) fica mole e alongada.
- As flutuações (agitação térmica, corrente, empurrões das vizinhas) acabam
  por juntar dois lados opostos.
- **Área-alvo:** vem do conteúdo (pressão osmótica: mais moléculas dentro
  dá mais área). **Perímetro:** vem dos lípidos incorporados.
- Crescer membrana sem crescer conteúdo empurra para a divisão; crescer
  conteúdo sem membrana empurra para a rutura (morte).

Isto é física real: as vesículas de ácidos gordos alongam-se e partem-se ao
crescer (Zhu & Szostak 2009). Mais tarde, uma proteína de constrição pode
evoluir para puxar dois lados um para o outro, o que torna a divisão
controlada; ver a etapa 7.

## 3. O que se reaproveita

| Peça existente | Uso nas células |
|---|---|
| Grelha química, transporte, livro-razão, teste de conservação | Meio exterior; os **lípidos** entram como mais um canal de quanta |
| Fluido, terreno, luz | Iguais (ambiente e pressões seletivas) |
| Slots, pilha livre, nascimentos, cenas, editor | Iguais |
| Arrays por resíduo (64 por slot) | Nós da membrana |
| Juntas sobreamortecidas, RFT | Mecânica da membrana; os flagelos empurram pelo RFT a partir do nó onde estão |
| Genoma, tradução, código dos órgãos, mutações | Cada ORF (AUG…stop) é um **gene**; o "órgão" diz que proteína é e a variante dá os parâmetros |
| Replicação por emparelhamento | Igual, mas com os nucleótidos de **dentro** |
| Sinais α/β | Sinais internos (sensores, reguladores, flagelos) |
| Contacto por grelha, âncoras | Contacto membrana–membrana; colónias por adesão |

## 4. Tensões com os princípios do projeto

1. **Matéria exata.** Lípidos, nucleótidos internos e o que mais houver são
   quanta inteiros. As trocas membrana ↔ meio são somas atómicas com as
   células da grelha por baixo dos nós. A divisão reparte ao quantum. O
   teste de conservação passa a incluir o conteúdo das células. A "moeda
   energética" pode continuar a ser a ativação dos nucleótidos (não é
   matéria).
2. **Emergência > engenharia.** A lista de proteínas da proposta original é,
   em parte, uma lista de órgãos com funções desenhadas. O caso mais
   delicado é o **inibidor da divisão "enquanto houver só uma cópia do
   genoma"**: conta genomas, é um coordenador escondido. Versão química: a
   constrição precisa de uma proteína que a própria replicase produz ao
   trabalhar; só depois de copiar há muita, e o acoplamento pode evoluir em
   vez de vir feito.
3. **A célula não conhece o genoma.** A expressão por taxas está bem; não
   pode haver regras que leiam a sequência fora da tradução.
4. **Baixo Reynolds.** A membrana é sobreamortecida e o flagelo funciona por
   arrasto anisotrópico (RFT), sem inércia.

## 5. Etapas

Cada etapa compila, tem uma sonda curta e é verificável na janela.

| # | Etapa | O que se vê |
|---|---|---|
| 0 | **Modo "células"** (como o modo laboratório): o mesmo mundo, outro conjunto de kernels de vida | o mundo de sempre, sem agentes |
| 1 | **Bolhas deformáveis**: anel de nós com tensão, rigidez à flexão e pressão de área; contacto membrana–membrana; arrasto do fluido e do entulho | bolhas a espremer-se entre rochas e a empurrar-se |
| 2 | **Lípidos e divisão por toque**: canal de lípido; os nós incorporam lípido e a membrana cresce (inserção de nós); excesso de perímetro dá alongamento; toque de lados opostos parte o anel e nasce o filho; conteúdo repartido pelo lado | bolhas que crescem perto de fontes de lípido e se partem sozinhas |
| 3 | **Química interna**: contagens por célula, permeabilidade da membrana, trocas ao quantum | vista com a concentração interna em cores |
| 4 | **Genoma dentro**: RNA replicado com nucleótidos internos; cópias como partículas; filha sem genoma morre mais tarde | linhagens; filhas vazias a apagar-se |
| 5 | **Genes e proteínas**: ORFs traduzidas em contagens de proteína; enzimas e replicase internas; diluição pelo crescimento | barras de composição no inspetor |
| 6 | **Proteínas de membrana posicionadas**: transportadores e flagelos presos a nós; sensores e reguladores | quimiotaxia a emergir num gradiente |
| 7 | **Ciclo celular**: proteína de constrição que puxa lados opostos; acoplamento químico à replicação | divisões mais regulares nas linhagens evoluídas |

As etapas 1 e 2 já dão um resultado forte (bolhas que crescem e se dividem
sozinhas) e permitem decidir se vale a pena continuar.

## 6. Escolhas em aberto

| Escolha | Opções | Recomendação |
|---|---|---|
| Membrana | anel de nós · Potts celular na grelha | **anel**: o Potts trata a divisão de borla, mas à nossa resolução (30 unidades por célula da grelha) as células teriam poucos píxeis; o anel encaixa no sobreamortecido e no RFT |
| Nós por célula | fixo · crescente | começa em **32**, cresce por inserção quando a membrana incorpora lípido, até 64; a divisão reparte os nós |
| Critério do toque | distância entre nós não vizinhos | nós a mais de **N/4** posições no anel e a menos de ~1 espessura de membrana; só conta se cada lado ficar com área e nós mínimos |
| Proteínas | contagens por gene · cadeias físicas | **contagens**: dentro da célula interessa quantas há e onde estão, não a forma |
| Espécies químicas | muitas abstratas · poucas reais | só **nucleótidos (ativados/gastos) + lípido** ao princípio; um "metabolito" apenas se for preciso |
| Relação com os agentes atuais | modo separado · coexistência | **modo separado** primeiro; mais tarde, vesículas que se formam sozinhas a partir de lípidos e aprisionam agentes de RNA (a via mais emergente, mas a mais difícil) |
| Rutura | nunca · por tensão | **por tensão**: área-alvo muito acima da que o perímetro aguenta = rutura, e o conteúdo volta ao meio ao quantum |

## 7. Custos e riscos

- **Desempenho:** uma célula de 32 nós custa mais ou menos o mesmo que um
  agente atual. O contacto membrana–membrana nó a nó é mais caro que os
  discos de hoje. Estimativa: ~30–50 mil células com fluidez; a medir na
  etapa 1.
- **Partir o anel na GPU:** é o ponto técnico mais delicado da etapa 2:
  reindexar nós, alocar o slot do filho, repartir o conteúdo ao quantum sem
  corridas entre células.
- **A divisão por toque pode não acontecer sozinha** se as flutuações
  forem fracas; a etapa 2 tem de medir com que frequência os lados se
  tocam. Se for raro, a primeira afinação é a rigidez à flexão e a pressão,
  não uma regra.
- **Tamanho do trabalho:** várias semanas, por etapas; cada etapa vale por
  si.
