# Audit Sophon — 2026-06-18

Diagnostic complet de `mcp-sophon` croisant **mesures** (bench de correctness end-to-end,
N=51 tâches données réelles, juge LLM aveugle) et **audit de code adversarial** (findings
vérifiés, faux positifs élagués). Objectif : situer les forces/faiblesses/angles morts et
classer les prochains leviers par impact × coût.

Artefacts liés :
- Bench : [`benchmarks/correctness/`](benchmarks/correctness/) — `REPORT.md`, `COMPARISON.md`, `PLAN.md`, `correctness.json`
- Fix livré ce jour : summariser extractif déterministe (`compress_history`)

---

## TL;DR

Le projet est **sain et honnête** (build/tests verts, MCP solide, pas d'injection shell,
méthodo de bench transparente). Mais :

1. Sa promesse "compression sémantique" est **sous-tenue par défaut** : `compress_prompt`
   sous-remplit le budget et le scoring sémantique est creux (embedder ML absent du binaire publié).
2. Deux bugs **delta** (corruption multi-edit, path traversal) doivent être colmatés avant tout
   positionnement "prod".
3. Le **query-aware history** — pressenti comme prochain levier — est légitime mais arrive
   **dernier**, pas premier.

---

## Méthode

- **Bench correctness** : pour 51 tâches réelles (24 docs/sources, 20 historiques git→conversation,
  7 sorties shell), recall de faits-clés gelés, répondeur `haiku`, juge aveugle `sonnet`, 3 bras à
  budget égal : `oracle` (contexte complet) / `sophon` / `truncate` (troncature naïve au même budget).
- **Audit code** : chaque finding reproduit (harnais Rust) ou vérifié par lecture du chemin d'appel
  réel ; faux positifs des sous-agents écartés explicitement.

---

## ✅ Forces (vérifiées)

- **Socle sain** : 440+ tests verts, `cargo build/clippy` propres (seulement ~5 warnings cosmétiques).
- **MCP JSON-RPC fait-main solide** : zéro panic dans `jsonrpc.rs`, mapping d'erreurs correct,
  parse-error ne tue pas la boucle, panics de tool isolés (`JoinSet`/`spawn_blocking`).
- **Pas d'injection shell** : `SOPHON_LLM_CMD` est exec direct (`Command::new`), pas de `sh -c`.
  Tokens secrets gitignored, non versionnés, perms `-rw-------`.
- **`compress_prompt` gagne en moyenne** (+3.5 pts vs troncature) et reste déterministe.
- **Fix `compress_history` (ce jour)** : summariser extractif déterministe → recall history ×24
  (voir `COMPARISON.md`).

## ❌ Faiblesses mesurées (bench)

Résultats globaux (défaut actuel, post-fix history) : oracle **68.7%** · sophon **28.1%** ·
truncate **23.9%** · Sophon−truncate **+4.2 pts** (IC95 [−3.6, +11.6]) · Sophon−oracle **−40.6 pts**.

| Outil | Mesure | Cause racine vérifiée |
|---|---|---|
| **compress_prompt** | 24 tâches, nombreux **zéros** (prompt-002 : 9 tokens émis sur ~700 autorisés, oracle 83%) | `min_tokens: 0` par défaut (`compressor.rs:29`) → `backfill_to_minimum` ne se déclenche jamais → **budget jamais rempli vers `max_tokens`**. ET scoring sémantique opt-in (retriever off par défaut, `server.rs:75-80`) → repli sur dictionnaire de mots-clés figé (`analyzer.rs` `TOPIC_KEYWORDS`) qui rate les termes spécifiques |
| **compress_output** | **−16.3 pts vs troncature** (valeur négative) | filtres génériques sur-filtrent sur commandes inconnues (output-050 : sophon 0% vs troncature 100%) |
| **compress_history** | 19.1% post-fix, plafonné | résumé **aveugle à la question** (résume à l'ingest, `summarizer.rs`) |

> Caveat : `output` n'a que 7 tâches — signal directionnel, pas concluant statistiquement.

## 🕳️ Angles morts (audit code, vérifiés adversarialement)

| # | Sévérité | Zone | Preuve | Impact | Fix |
|---|---|---|---|---|---|
| F5/F6 | **Majeur** | embedder ML creux + `bert` mort | `server.rs:119-135` (`_ => HashEmbedder`), `release.yml` build sans `--features`, `tools.rs:35` annonce `bert` non câblé | `SOPHON_EMBEDDER=bge` retombe **silencieusement** sur le hash ; le "sémantique" vendu est absent du binaire publié + description outil mensongère envoyée au LLM | Bas — `warn!`/erreur si embedder non compilé ; câbler ou supprimer `bert` |
| F2 | **Majeur** | `write_file_delta` multi-edit | `patcher.rs:101-116` — reproduit : `Replace(1,2)`+`Replace(2,3)` sur `[a,b,c]` → `[X]` (perte de données) | **Corruption silencieuse** de fichier sur plages adjacentes/chevauchantes, zéro garde | Moyen — détecter overlap → `EditError` |
| F1 | **Majeur** | path traversal delta | `delta-streamer/lib.rs:152,181` + `handlers.rs:393,365` — `PathBuf::from(path)` brut, aucun confinement | Lecture/écriture de fichier arbitraire via prompt-injection (`../../.ssh/...`) | Bas — canonicaliser + exiger un root (`SOPHON_FS_ROOT`) |
| F3 | **Majeur** | roundtrip CRLF | `differ.rs:115` + `patcher.rs:226` (`.lines()`) — reproduit : `\r\n` → `\n` | Tout fichier Windows réécrit converti en LF silencieusement ; 0 test CRLF | Moyen — capturer/restituer le style EOL |
| F4 | **Mineur** | collision placeholder fragment | `decoder.rs:13,24` — contenu contenant `[FRAGMENT:…]` → `Err(FragmentNotFound)` dur | Decode échoue sur du contenu légitime (docs/logs sur Sophon) ; pas de fallback | Moyen — délimiteur non-collisionnable / échappement |
| F7-F10 | Mineur / *à confirmer* | LRU delta faux mismatch (F7), navigate avale non-UTF8 (F9), méthodes JS de classe ratées (F10, non reproduit) | voir rapport audit | Dégradations, pas corruption | Bas-Moyen |

**Robustesse / sécurité — points à signaler mais non bloquants :**
- `panic = "abort"` (release) + **110 `unwrap`/`expect`** dans les outils hors tests → un unwrap qui
  pète dans un outil **abort le serveur entier** au lieu d'une erreur JSON-RPC. *À auditer* (slices
  `[..16]`, `partial_cmp().unwrap()`).
- 2 tokens en clair sur disque (publish rights) : gitignored + non versionnés → pas de fuite, mais à
  rotationner par principe.
- `next_line()` bufferise une ligne entière (DoS théorique sur input multi-Go sans `\n`) — mineur (stdio local).

---

## 🎯 Classement des leviers

### Tier 1 — crédibilité + valeur, coût bas (faire d'abord)
1. **`compress_prompt` par défaut** : backfill vers `max_tokens` (au lieu de `min_tokens:0`) +
   vrai scoring lexical par défaut (BM25 / substring / overlap d'identifiants) au lieu du dictionnaire.
   → tue les zéros catastrophiques, touche le plus de tâches. **Effort bas, impact le plus large.**
2. **Honnêteté embedder (F5/F6)** : warning explicite si embedder ML non compilé, corriger/supprimer
   `bert`. Même racine que #1, et arrête de mentir au modèle.

### Tier 2 — correctness/sécurité, avant tout positionnement "prod"
3. **F2 (corruption multi-edit)** + **F1 (path traversal)** : bugs de perte de données / sécurité,
   pas d'optimisation. Un outil qui écrit des fichiers ne doit ni corrompre ni sortir du projet.
4. **`compress_output` : plancher de sécurité** — ne jamais faire pire que la troncature. Retire l'anti-valeur.

### Tier 3 — réel mais le plus cher
5. **query-aware `compress_history`** : gain réel (plafond ~50 pts) mais nécessite de re-résumer
   au query-time avec la question, ou de stocker la query à l'ingest. **Après** les tiers 1-2.

---

## Reproduire le bench

```bash
cd benchmarks/correctness
python3 build_tasks.py                       # (re)génère tasks.json depuis le repo réel
python3 run_correctness.py                   # extractif (défaut)  → results/REPORT.md
python3 run_correctness.py --summary-mode llm     # LLM abstractif → results/REPORT.llm.md
python3 run_correctness.py --summary-mode legacy  # ancien défaut  → results/REPORT.legacy.md
```
Cache LLM sur disque (`results/cache/`, gitignored) → re-runs quasi instantanés.
