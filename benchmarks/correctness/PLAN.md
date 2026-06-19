# Correctness bench — plan

## Problème
Le headline "68 % tokens saved" (`benchmarks/session_token_economics.py`) ne mesure **que des tokens**
(`1 - sent/raw`), jamais si la réponse reste correcte une fois le contexte compressé. Un acheteur
sceptique demande : *« et l'info survit-elle ? »* — aucun bench n'y répond aujourd'hui.

## But (critères d'acceptation)
1. Pour ≥60 tâches sur **données réelles du repo**, mesurer le **recall de faits-clés** (0-100 %) d'une
   réponse produite par un LLM répondeur, dans **3 conditions à budget de tokens comparable** :
   - `oracle`   : contexte **complet** (plafond de qualité)
   - `sophon`   : contexte **compressé par Sophon** au budget B
   - `truncate` : contexte **tronqué bêtement** au **même budget B que Sophon a réellement produit**
2. Le verdict central : **`recall(sophon) − recall(truncate)` à budget égal**, avec intervalle de
   confiance (bootstrap apparié). C'est ça qui prouve que la compression est *intelligente*, pas juste
   *agressive*. Comparé aussi à `oracle` pour mesurer la perte absolue vs contexte complet.
3. Rapport reproductible : `results/correctness.json` + `results/REPORT.md` avec breakdown par op et par
   type de tâche, taux d'économie de tokens **et** recall côte à côte.

## Décisions verrouillées
- **Périmètre** : `compress_prompt` + `compress_history` + `compress_output` (deltas exclus : sans perte par construction).
- **Répondeur** : `claude -p --model haiku`. **Juge** : `claude -p --model sonnet` (indépendant, plus fort).
- **Métrique** : recall de faits-clés. Faits **gelés** dans `tasks.json`, dérivés du contexte COMPLET
  avant toute compression → identiques pour les 3 bras (pas de circularité).
- **N≈60** : ~25 prompt (vrais fichiers sources) · ~20 history (vraies sessions/commits) · ~15 output (vraies sorties shell).

## Rigueur méthodologique
- **Parité de budget** : on lance Sophon d'abord, on mesure ses tokens de sortie *réels*, puis on tronque
  le contexte brut à ce même nombre. La troncature n'est jamais désavantagée sur le budget.
- **Juge aveugle** : le juge voit (question, faits-clés, réponse) — **jamais** la condition ni le contexte.
  Réponses étiquetées neutrement, ordre mélangé.
- **Faits gelés** : générés une fois par sonnet depuis le contexte complet, écrits dans `tasks.json`
  (committé), inspectables. Le run ne les régénère jamais.
- **Tokenizer unique** : `count_tokens` de Sophon partout (même mesure pour tous les bras).
- **Reproductible & résumable** : cache disque par (task_id, condition) ; un crash ne reperd pas les appels LLM.
- **Honnêteté** : on logge tout cas où Sophon perd (recall < truncate), N réel, et les désaccords.

## Phases
- **P1 — Squelette & lib** : `lib.py` (RPC sophon, count_tokens, appel LLM via llm_cli, cache disque).
- **P2 — Dataset** : `build_tasks.py` mine les données réelles + extraction de faits (sonnet) → `tasks.json` gelé.
- **P3 — Run** : `run_correctness.py` → 3 bras × N, répondeur haiku, cache, résumable.
- **P4 — Juge & scoring** : juge sonnet aveugle → recall par fait ; agrégation + bootstrap CI apparié.
- **P5 — Rapport** : `REPORT.md` (table tokens×recall par op, delta sophon−truncate ±CI, cas d'échec).
- **P6 — Gate** : smoke run (N petit) vert, puis run complet ; mise à jour README/BENCHMARK avec le chiffre correctness.
