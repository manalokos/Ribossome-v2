# v4_kickoff — como arrancar o Ribossome v4 numa sessão nova

Esta pasta tem tudo o que uma sessão nova do Claude Code precisa para
começar a construir o Ribossome v4 sem perder o que se aprendeu no v3.

## Ficheiros

| Ficheiro | Para quê |
|---|---|
| `KICKOFF_PROMPT.md` | O texto a colar na primeira mensagem da sessão nova |
| `CLAUDE.md` | Instruções permanentes do projeto. Copiar para a raiz do repositório novo |
| `CONHECIMENTO_DO_PROJETO.md` | Tudo o que se sabe: filosofia, história, lições, armadilhas, como o Filipe trabalha |
| `MAPA_DE_PORTAGEM.md` | Onde vive cada sistema no v3 (ficheiros, kernels, constantes afinadas) |
| `ESTADO_E_PENDENTES.md` | Ponto de situação do v3 no momento da passagem e o que ficou a meio |
| `../docs/ARQUITETURA_V4.md` | O documento de arquitetura (a proposta) |

## Passos

1. Criar a pasta do projeto novo, por exemplo `C:\Filipe\Ribossome4`. A decisão
   sobre ser um repositório novo ou um branch está em aberto (ver ARQUITETURA_V4 §12).
2. Copiar para lá `CLAUDE.md`. Copiar também esta pasta inteira para
   `C:\Filipe\Ribossome4\docs\handoff\`, e o `docs/ARQUITETURA_V4.md`.
3. Abrir o Claude Code nessa pasta.
4. Colar o conteúdo de `KICKOFF_PROMPT.md` como primeira mensagem.
5. O v3 continua em `C:\Filipe\ALsimulatorv3` (branch `rna-world`, tag
   `backup-pre-amino-redesign-2026-10-01`) e serve de referência: a sessão
   nova pode e deve ler o código de lá.
