Olá. Vamos começar o **Ribossome v4**, um simulador de vida artificial em GPU
que reescreve do zero o Ribossome v3 (que está em `C:\Filipe\ALsimulatorv3`,
branch `rna-world`).

Antes de escreveres código, lê por esta ordem:

1. `CLAUDE.md` (raiz deste projeto): regras e armadilhas.
2. `docs/handoff/CONHECIMENTO_DO_PROJETO.md`: filosofia, história, lições e
   como eu trabalho.
3. `docs/ARQUITETURA_V4.md`: a arquitetura proposta.
4. `docs/handoff/MAPA_DE_PORTAGEM.md`: onde está cada sistema no v3 e os
   valores afinados.
5. `docs/handoff/ESTADO_E_PENDENTES.md`: o que ficou a meio no v3.

Depois:

- Resume-me em poucas linhas o que percebeste do projeto e da filosofia, para
  eu confirmar que estamos alinhados.
- Faz-me as perguntas que faltam decidir (ARQUITETURA_V4 §12), com uma
  recomendação para cada.
- Só depois começa pela **Fase 1 (esqueleto)**: janela, câmara, profiler por
  segmento, validação de shaders no `cargo test`, teste de conservação sem UI.

Regras de trabalho:
- Fala comigo em português de Portugal.
- Commits pequenos e frequentes; nunca push sem eu pedir.
- Quando portares um sistema do v3, lê o código original de lá; não o
  reinventes de memória.
- Mede antes de otimizar, e diz-me sempre o que verificaste e o que não
  verificaste.
