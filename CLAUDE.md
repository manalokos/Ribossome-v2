# Ribossome v4 — instruções para o Claude

Simulador de vida artificial em GPU (Rust + wgpu + WGSL). Mundo aquático 2D
fechado onde organismos de RNA/proteína evoluem por química, não por regras
desenhadas. Reescrita do v3 (`C:\Filipe\ALsimulatorv3`, branch `rna-world`).

Documentos: `docs/ARQUITETURA_V4.md`, `docs/handoff/*.md`.

## Quem é o utilizador
- Filipe. Fala português de Portugal e escreve depressa, informal, muitas vezes
  sem acentos. Responde-lhe em PT-PT.
- Pensa como cientista/artista: quer realismo físico e química honesta, e
  repara visualmente em tudo (artefactos, cores, movimento estranho).
- Testa ele próprio e cola logs ou screenshots. Diz-lhe sempre o que
  verificaste (compila, valida, correu X segundos) e o que **não** viste.
- Quando ele propõe uma ideia, avalia-a com honestidade ("isso acontece na
  realidade?" é uma pergunta que ele faz muito). Se uma regra for artificial,
  diz.

## Princípios do projeto (não negociáveis)
1. **Matéria exata.** Mundo fechado: monómeros são quanta inteiros,
   movidos por trocas atómicas. Nada cria nem destrói matéria. Há um teste.
2. **Química global.** Capacidades = regras químicas iguais para todos. A
   sequência só decide onde e quanto. Nada de parâmetros evolutivos por agente.
3. **A criatura não conhece o genoma** (seu ou alheio). Nenhuma regra compara
   genomas. Predação/varrimento "inteligente" foi rejeitado por isto.
4. **Sem sexo/crossover desenhados.** Replicação por emparelhamento.
5. **Baixo Reynolds.** Sobreamortecido; nada de inércia nem vórtices de propulsão.
6. **Emergência > engenharia.** Se uma funcionalidade precisa de um "órgão
   que faz X", procura antes a versão química.

## Armadilhas técnicas (aprendidas no v3)
- **naga do wgpu 30 aceita indexar `const` arrays diretamente**
  (`AA_MJ[i]`). NUNCA copiar uma tabela para `var` dentro de uma função
  (era a regra do wgpu 22): cada chamada copia a tabela inteira para
  memória privada. Com `mj()` na dobragem O(n²), isto tornava o passo 10×
  mais lento (12 ms -> 1,2 ms com ~4500 agentes).
- Valida todos os shaders com a versão exata do naga do wgpu em uso, com as
  constantes injetadas reais (os módulos são concatenados em Rust).
- Verifica que todos os `entry_point` referidos em Rust existem nos shaders
  (um pipeline órfão crasha no arranque).
- `from` e `target` são palavras reservadas em WGSL.
- naga (wgpu 30): uma função com valor de retorno não pode acabar só dentro
  de um `loop` (com `return` lá dentro). Usa `var` + `break` + `return` final.
- Unidades: `SIM_SIZE` (unidades do mundo, 61440) ≠ `GRID_SIZE` (células,
  2048) ≠ fluido (512). Converter sempre explicitamente; já houve bugs de
  monómeros congelados por passar coordenadas de célula a funções de mundo.
- **Cima no ecrã = +y no mundo** (determinado empiricamente no v3).
- Async readbacks atrasam: nunca usar o `agent_count` lido do CPU para
  limitar dispatches (causou agentes invisíveis e fugas de matéria).
- `submit + poll(Wait)` mede também o trabalho pendente do frame anterior:
  mede o "backlog" à parte.
- Windows: o shell do Bash tool é Git Bash; heredocs com aspas simples dentro
  de python podem partir — para scripts grandes, escreve um ficheiro .py.
  Em PowerShell, mensagens de commit sem aspas duplas.
- Ficheiros do v3 com LF; mantém LF.
- O v3 carregava automaticamente o autosave e um JSON de propriedades por
  cima dos defaults do shader — no v4, uma só fonte de verdade.

## Forma de trabalhar
- Commits pequenos, mensagem em inglês, terminar com a linha Co-Authored-By
  pedida pela sessão. Push só quando pedido.
- Antes de mudar comportamento visível, explica a proposta em 3–6 linhas e
  espera "ok" se a mudança for grande.
- Constantes afinadas do v3 estão em `docs/handoff/MAPA_DE_PORTAGEM.md`; não
  as inventes de novo.
