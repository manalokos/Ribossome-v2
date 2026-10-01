# Ribossome v4 — Desenho do organismo

Estado: **decidido com o Filipe em 2026-10-01** (fase 3 em curso). Este
documento regista os princípios e as escolhas; o detalhe do código vive no
código.

---

## 1. Princípio: sem órgãos desenhados

Um organismo não tem "órgãos que fazem X". Tem uma **cadeia de aminoácidos
reais** (o corpo) e o mundo tem uma **lista curta de reações químicas
globais**, iguais para todos. A sequência só decide *onde* e *quanto* cada
reação é catalisada à volta do corpo. Um "órgão" é aquilo que emerge: um
bolso de resíduos catalíticos que a sequência (e, a partir da fase 5, a
dobragem) junta.

O que continua a ser artificial (assumido):
- a fórmula que converte propriedades de um resíduo em taxa catalítica
  (os dados são reais; a função que os combina é nossa);
- "contacto" num mundo 2D;
- a lista de reações fecha o espaço do possível. Para abrir mais, a lista
  cresce, sempre com reações reais.

## 2. Matéria e energia

- O genoma é feito de **monómeros reais** tirados da grelha; enquanto o
  agente vive, essa matéria conta no livro-razão como "presa em agentes".
- A energia do agente é **ativação colhida**. O monómero gasto fica onde
  estava. A energia evapora na morte (convertê-la em matéria criaria massa).
- Na morte, o genoma (e os complementos já capturados) voltam à grelha como
  monómeros gastos, na célula livre de terreno mais próxima.
- O teste de conservação cobre o ciclo completo: nascer, viver, replicar,
  morrer.

## 3. Geração 0

- Escolhem-se pontos aleatórios do espaço. Em cada ponto, o genoma
  **monta-se com os monómeros ativados mais próximos, pela ordem da
  distância**: a sequência espelha a composição local da sopa. O
  comprimento é aleatório. Se não houver bases suficientes, não nasce.
- Opção (desligável): **AUG injetado** no início, com A, U e G tirados
  também da vizinhança (só se escolhe a ordem; não se cria matéria).
- Nenhuma semente é criada do nada (no v3 as sementes da geração 0 eram
  matéria nova e por isso não devolviam a cadeia ao morrer).

## 4. Tradução

Genoma de RNA (A/U/G/C, 2 bits por base), leitura a partir do primeiro AUG,
**código genético padrão**, até ao primeiro codão stop ou 64 resíduos. A
cadeia traduzida é o corpo. Os aminoácidos só têm propriedades medidas
(`MAPA_DE_PORTAGEM.md`): massa, volume, carga a pH 7, hidrofobicidade
(Kyte-Doolittle), Chou-Fasman, flexibilidade, e a propensão catalítica.

## 5. Reações globais

Todas conservam matéria; todas têm uma taxa espontânea (sem catálise) baixa.

| Reação | O que é | Fase |
|---|---|---|
| Hidrólise de um ativado → gasto + energia para o agente | **comer** | 3 |
| Ligação dirigida por molde (emparelhamento) | **copiar** | 3 |
| Interconversão de bases | metabolismo | mais tarde |
| Adesão de resíduos hidrofóbicos a rocha / outros corpos | fixar-se, agregar | mais tarde |

### 5.1 Comer = hidrólise catalisada
Cada resíduo e os seus vizinhos em contacto formam um sítio. A força
catalítica do sítio vem da **propensão catalítica medida** de cada
aminoácido (razão entre a frequência entre resíduos catalíticos de enzimas
reais e a frequência geral). Dados: Bartlett et al. 2002 (J Mol Biol 324:105)
e Ribeiro et al. 2020 / M-CSA (PMC6956550): His ×8,2, Cys ×4,6, Asp/Glu/
Lys/Arg ×1,7–3,0, hidrofóbicos muito baixos. **A tabela completa dos 20
aminoácidos vai ser calculada a partir dos dados públicos do M-CSA** (não
inventada); o script e a fonte ficam no repositório.

Na fase 3 "contacto" = vizinhos na cadeia (i−1, i, i+1). Na fase 5 passa a
ser contacto após a dobragem.

### 5.2 Copiar = emparelhamento espontâneo (como no v3)
Mantém-se o mecanismo do v3: o corpo captura monómeros ativados
complementares da vizinhança até o emparelhamento estar completo; então
nasce o descendente complementar. Base real: a cópia dirigida por molde sem
enzimas existe (Orgel), é lenta. Mais tarde a catálise pode acelerá-la
(uma "replicase" passa a ser algo que evolui), sem a substituir.

Mutações (pontuais, inserções, remoções) com a matéria **reconciliada desde
o início**: o que sobra é depositado, o que falta é tirado da vizinhança;
se não houver, a mutação não acontece.

## 6. Movimento (sem relógios nem sinais desenhados)

1. **Passivo** (fase 3): levado pelo fluido, assenta devagar.
2. **Difusioforese** (fase 3): um corpo que consome mais de um lado é
   empurrado pelo gradiente que ele próprio cria (partículas catalíticas
   "Janus", Golestanian 2005). Emerge da posição dos resíduos catalíticos.
3. **Ciclos quimiomecânicos** (fases 6–7): uma junta catalítica passa por
   vazia → ligada a ativado → hidrolisa → solta, e cada estado tem um ângulo
   diferente. O ciclo é irreversível (gasta energia) e por isso não
   recíproco: é o que o teorema da vieira exige a baixo Reynolds (como a
   miosina e a cinesina). Se as taxas dependem da concentração local, a
   quimiotaxia emerge sem sensores.
4. **Coordenação** (fase 8): acoplamento mecânico entre juntas vizinhas
   (alosteria) pode dar ondas sem relógio central, como as ondas
   metacrónicas dos cílios.

Relógios e sinais internos do v3 não existem no v4. Osciladores químicos
reais (p. ex. KaiABC) só se emergirem.

## 7. Aleatoriedade

Todo o sorteio vem de um gerador baseado em contador (PCG4D, Jarzynski &
Olano 2020) com entradas separadas (chave, passo, fluxo, semente). A
simulação do mundo é reprodutível bit a bit e não depende de quantos passos
correm por frame (teste `simulation_is_deterministic`).
