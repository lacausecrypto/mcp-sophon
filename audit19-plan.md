# Plan v0.7 — Puissance + économie de tokens

Suite de [`audit18.md`](audit18.md) (clôturé par v0.6.0). Cet audit cible **où Sophon
laisse des tokens sur la table** et **où la qualité de compression plafonne**, croisant
3 audits de code vérifiés adversarialement (chemin prompt, chemin output/history,
comptabilité de tokens). Chaque levier cite `fichier:ligne` et précise *reproduit* vs *supposé*.

---

## État d'exécution (2026-06-19)

| Item | État | Note |
|---|---|---|
| T0.1 double payload | **Non fait (requalifié)** | Aucun `outputSchema` ⇒ `structuredContent` non injecté au modèle par un client conforme ; consommé en interne (benchs/tests). Le retirer casserait du code pour un gain non vérifié. |
| T0.2 comptage delta JSON | ✅ `fad9a57` | |
| T0.3 honnêteté fragments | ✅ `fad9a57` | |
| T1.1 strip ANSI | ✅ `fad9a57` | prouvé par test coloré (0.98→<0.5) |
| T1.2 hijack parser XML | ✅ `fad9a57` | |
| T1.3 troncature query-aware | ✅ `fad9a57` | prompt recall 48.8→51.3 |
| T2.1 collapse `\r` | ✅ `61d591e` | |
| T2.2 filtre build | ✅ `61d591e` | cargo build 107→18 tok |
| T2.3 repli stack traces | ✅ `61d591e` | |
| T3.1 index conditionnel | ✅ `4299e5d` | perf (index jeté plus construit) |
| T3.2 rolling récursif | **Déféré (raison)** | Re-résumer le brut est le choix *fidèle* en extractif déterministe ; le seul vrai gain (incrémental LLM) est hors-bench, opt-in, risqué. |
| T3.3 relevance composants | ✅ `4299e5d` | history 32.1→32.7 |
| T4.2 dédup sections | ✅ `d053bc8` | neutre sur le bench (corpus sans redite), gain hors-corpus |
| T4.1 embedder/retriever défaut | **Déféré (mesure-d'abord)** | Retriever hors-chemin ; exige une **nouvelle infra de bench** sinon optimisation à l'aveugle. |
| T4.3 prompt-caching | **Hors-scope serveur** | `cache_control` est posé par l'orchestrateur/client sur ses appels API ; un serveur MCP ne le contrôle pas. Reste une *guidance d'intégration*. |
| T4.4 tokenizer Claude | **Déféré (faible valeur)** | Pas de tokenizer Claude open ; cl100k cohérent avec lui-même ; surtout cosmétique. |

Bench final N=51 : oracle 68.7 · **sophon 43.7** · truncate 23.7 · **sophon−truncate +20.0 pts** (CI [+10.9,+29.5]). Détail : prompt 51.3 (+15.7 vs trunc) · history 32.7 (+32.7) · output 49.1 (−2.0). 480+ tests verts, clippy clean.

---

## Recadrage (à lire avant d'agir)

Trois faits vérifiés changent les priorités par rapport à l'intuition de départ :

1. **Les chiffres ont bougé.** Le `28.1%` d'avant-v0.6.0 est périmé. `REPORT.md` post-v0.6.0,
   reproduit via le binaire release :

   | op | n | oracle | sophon | truncate | récup. vs oracle |
   |---|---|---|---|---|---|
   | prompt | 24 | 72.4% | **48.8%** | 31.1% | 67% |
   | history | 20 | 69.4% | **32.1%** | 0.0% | 46% |
   | output | 7 | 54.0% | **49.1%** | 51.2% | 91% |

2. **Le `semantic-retriever` est hors-chemin.** chunker / BM25-retriever / reranker / MMR /
   entity_graph ne s'activent que si `SOPHON_RETRIEVER_PATH` est défini
   (`mcp-integration/src/server.rs:80`), ce que le bench ne fait jamais. Le recall `prompt`
   ne dépend QUE de `prompt-compressor/{parser,analyzer,compressor}`. Optimiser le retriever
   ne bougera aucun chiffre tant qu'il n'est pas branché → c'est un chantier "mesure-d'abord".

3. **Le plus gros gain de tokens n'est pas dans la compression, c'est l'enveloppe.** Chaque
   réponse d'outil est sérialisée **deux fois** dans la même enveloppe MCP.

Conséquence : les meilleurs ratios impact×coût sont des fixes **bas-coût** déjà localisés,
pas un gros refactor sémantique.

---

## Tier 0 — Économie de tokens pure (enveloppe & comptabilité)

> Ces items réduisent les tokens *facturés* sans toucher à la qualité. Coût bas, à faire en premier.

### T0.1 — Supprimer le double envoi du payload  ⭐ levier n°1
- **Constat (prouvé, `server.rs:360-372`)** : chaque résultat d'outil part en `content[].text`
  (= `serde_json::to_string(result)`) **ET** en `structuredContent` (= le même `result`).
  Un client MCP qui lit les deux paie ~+100 % sur **chaque** sortie d'outil — ce qui peut
  annuler le gain de compression.
- **Fix** : n'émettre le payload qu'une fois. Mettre `structuredContent` = result complet et
  `text` = forme minimale (résumé / pointeur), OU l'inverse selon ce que lit le client.
- **⚠️ Garde-fou** : le spec MCP recommande que `text` reflète `structuredContent` pour la
  rétro-compat. **Vérifier d'abord ce que lit le client cible (Claude Code)** via un vrai
  round-trip avant de trancher. Ne pas casser les clients qui exigent `content[].text`.
- **Effort** : bas (1 bloc). **Impact** : jusqu'à −50 % sur toute sortie d'outil. **Mesure** :
  round-trip MCP réel avant/après.

### T0.2 — Compter le delta sur la sérialisation réelle, pas sur `{:?}`
- **Constat (reproduit, `differ.rs:100`, `lib.rs:152`)** : `token_count`/`savings_percent` du
  delta sont comptés sur `format!("{:?}", operations)` (Debug Rust) alors que le serveur envoie
  du JSON (`handlers.rs:389`). Reproduit : Debug 157 chars vs JSON 182 chars (**+16 %**, plus
  avec l'escaping des guillemets/backslashes). On **sous-estime** le coût réel → dépassement de
  budget possible et `savings_percent` surévalué.
- **Fix** : compter sur `serde_json::to_string(operations)`. **Effort** : bas (2 lignes).

### T0.3 — Arrêter de mentir sur `tokens_saved` d'`encode_fragments` (honnêteté)
- **Constat (reproduit, `encoder.rs:41-106`, `handlers.rs:412-418`)** : la réponse renvoie
  `new_fragments` = le Fragment **complet re-sérialisé** (content intégral + hash + 2 timestamps
  + token_count + use_count + tags). Le contenu "économisé" repart donc en entier dans la même
  réponse → premier passage **net négatif** (doc 340 chars → réponse 725 chars, ×2,1), alors que
  `tokens_saved` annonce +48 en ignorant ce coût.
- **Fix** : ne renvoyer qu'`id`+`hash` (pas le content) dans la réponse `encode` ; recalculer
  `tokens_saved` en soustrayant le payload réellement renvoyé. Le gain n'existe qu'aux
  réutilisations ultérieures — le documenter ainsi. **Effort** : moyen. **Impact** : honnêteté
  de métrique + premier passage non-pénalisant.

---

## Tier 1 — Qualité de compression, coût bas (le meilleur ratio)

### T1.1 — Strip ANSI/escape en pré-passe `compress_output`  ⭐ cause racine du "output ≈ troncature"
- **Constat (reproduit, grep nul sur `output-compressor/src`)** : aucun strip ANSI. Les filtres
  reposent sur des regex **ancrées** (`^test … ok$`, `^✓`, `^--- PASS`…). Sur sortie colorée
  réelle, chaque ligne commence par `\u{1b}[…m` → aucune ancre ne matche → le filtre dégénère en
  no-op. Reproduit : `cargo test` **avec** ANSI 344→338 (ratio 0.98) vs **sans** ANSI 198→12 (0.06).
- **Fix** : 1 regex `\x1b\[[0-9;?]*[a-zA-Z]` (+ variantes OSC) appliquée une fois en tête de
  `run_pipeline`, avant tout filtre. **Effort** : ~10 lignes. **Impact** : débloque TOUS les
  filtres ancrés (cargo/pytest/npm/vitest/docker). **Gate** : sortie colorée doit passer de
  ratio ~0.98 à ~0.1. C'est le facteur dominant du `−2.0 pts` mesuré.

### T1.2 — Corriger le détournement du parser par un `<tag>` inline
- **Constat (reproduit, `parser.rs:52-62`)** : `contains_xml_sections` est testé **en premier**.
  Deux occurrences triviales `<rust>?: operator</rust>` dans un README font basculer tout le
  prompt en branche XML, qui ignore les **13 vrais headers markdown**. Reproduit : prompt-002
  émet **17 tokens / 700 (fill 2 %)**, réponse perdue. Le backfill ne peut rien : `sections` ne
  contient que les 2 micro-sections.
- **Fix** : ne déclencher la branche XML que si elle couvre l'essentiel du prompt (≥2 sections
  couvrant >X % du texte) **ou** si aucun `##` markdown présent. **Effort** : bas (1 fonction).
  **Impact** : élevé sur tout prompt markdown contenant un `<…>`.

### T1.3 — Troncature de section query-aware
- **Constat (reproduit, `compressor.rs:543-563`)** : quand une section dépasse le budget, on
  garde le **préfixe** (`chars().take(cut)`). Si la réponse est en milieu/fin de section, elle
  disparaît. Cause directe de prompt-011 (17 % vs 33 % troncature — Sophon **perd**).
- **Fix** : tronquer en gardant les phrases à plus haut score BM25 vs query, pas le préfixe
  (la troncature-milieu existe déjà côté output `truncate.rs:30` — s'en inspirer). **Effort** :
  bas-moyen. **Impact** : moyen-élevé sur sections longues.

---

## Tier 2 — Couverture des filtres output (économie sans perte)

### T2.1 — Collapse des progress bars / retours chariot `\r`
- **Constat (reproduit)** : `pip download` 474→474 tokens (`has_CR=true`) — aucune compression.
- **Fix** : split sur `\r`/`\b`, ne garder que le dernier segment par run. **Effort** : bas.

### T2.2 — Filtre `build` manquant
- **Constat (reproduit, `detector.rs:41` définit `BUILD_RE` mais `filters/mod.rs:30-65` n'enregistre
  aucun filtre build)** : `cargo build`, `cargo clippy`, `npm run build`, `tsc`, `eslint` tombent
  tous sur `generic`. Familles ultra-fréquentes.
- **Fix** : `FilterConfig` dédié build (garder erreurs/warnings, drop lignes de progression /
  "Compiling …" répétitives). **Effort** : moyen.

### T2.3 — Repli de stack traces (Python/Node/Rust/Java)
- **Constat (reproduit)** : 3× traceback identique 150→71 (dédup exacte) mais frames internes
  jamais repliées ; aucune logique trace-aware.
- **Fix** : filtre `^\s+File "…"` / `at …(…)` → garder type+message+1-2 frames applicatives,
  replier les frames de stdlib/deps. **Effort** : moyen.

### T2.4 — Whitespace intra-ligne + logs level-aware (mineurs)
- `generic` ne retire que les lignes vides, jamais l'indentation/espaces multiples
  (`generic.rs:27`). Drop INFO/DEBUG répétés en gardant WARN/ERROR (aujourd'hui ça marche "par
  chance" via `normalize()`, pas par garde sémantique). **Effort** : bas-moyen.

---

## Tier 3 — Historique (plafonne à 32 %)

### T3.1 — Exploiter fact_cards & SemanticIndex au query-time (calcul déjà fait, puis jeté)
- **Constat (prouvé, `handlers.rs:218-246`, `summarizer.rs:167`)** : `compress_history`
  **construit** un `SemanticIndex` (`build_index`) puis le **jette** (strippé sauf
  `include_index`). Les `fact_cards` ne sont extraites que si `SOPHON_FACT_CARDS=1`, et **la query
  est ignorée** dans `extract_fact_cards(&messages)`. Du calcul payé puis jeté.
- **Fix** : soit se servir de l'index pour ranker le tail au query-time, soit ne pas le construire.
  Passer la `query` à l'extraction de fact_cards. **Effort** : moyen. **Impact** : levier "gratuit"
  (le calcul est déjà fait).

### T3.2 — Rolling summary récursif en mode déterministe
- **Constat (prouvé, `summarizer.rs:718-781`)** : `refresh_rolling_summary` re-résume les
  **messages bruts** (`to_summarize = &history[..cap_until]`) à chaque seuil — il ne "résume
  jamais le résumé". Le 2e pass `llm_summarize` n'existe **qu'en mode LLM** (`summarizer.rs:588`) ;
  en mode déterministe (celui du bench) le résumé du tail est reconstruit from scratch.
- **Fix** : repli récursif déterministe (résumer le résumé précédent + nouveau delta). **Effort** :
  moyen. **Impact** : réel sur le 32 %.

### T3.3 — Pertinence : remplacer le substring nu
- `relevance_score` (`summarizer.rs:521`) est un substring : `age` matche `page`, `storage`…
  Passer à un match par token/identifiant (réutiliser le scorer lexical). **Effort** : bas.

---

## Tier 4 — Gros chantiers / mesure-d'abord (ne pas capitaliser sans bench)

### T4.1 — Activer un embedder de scoring par défaut + brancher le retriever dans un bench
- Le scoring sémantique (`compute_section_scores`, seuil 0.55) est **mort dans le bench** car
  `retriever = None`. Forcer au moins le `HashEmbedder` (gratuit, sans store disque) pour donner
  un 2e signal query-aware. **Mais impact non mesuré** → créer d'abord un bench qui active
  `SOPHON_RETRIEVER_PATH`, sinon on optimise à l'aveugle. **Effort** : bas (embedder) + moyen (bench).

### T4.2 — Dédup exacte de sections + MMR (diversité)
- Le compressor n'a **aucune** dédup/normalisation/MMR (grep vide) ; backfill purement glouton
  par score décroissant (`compressor.rs:435-454`). Sur doc à sections quasi-redondantes, le budget
  se remplit de variantes. **Fix** : dédup exacte (bas) puis MMR (moyen). Récupère du budget gratuit.

### T4.3 — Prompt-caching Anthropic (`cache_control: ephemeral`)
- Aucun `cache_control` dans tout le crate. Préfixes stables (system, schémas de tools, fichiers
  relus) → jusqu'à −90 % sur les tokens en cache, cumulable avec la compression. **Dépend de
  l'orchestrateur** (Sophon est MCP) → à cadrer côté intégration client. **Effort** : moyen, **impact
  potentiel très élevé**.

### T4.4 — Écart tokenizer cl100k ↔ Claude
- `count_tokens` = vrai BPE `cl100k_base` (OpenAI), pas une heuristique char/4 (sain). Mais la cible
  est Claude → écart typique ±10-20 % (code/unicode). Tous les "% saved" sont en tokens cl100k, pas
  facturés Anthropic. **Fix réaliste** : facteur de correction empirique par modèle + le documenter.
  **Effort** : élevé (pas de tokenizer Claude open). Faible priorité, surtout cosmétique/honnêteté.

---

## Discipline de mesure (transversal)

Le bench LLM (`benchmarks/correctness/run_correctness.py`) coûte des appels haiku/sonnet mais a un
cache disque (`results/cache/`). **Règle** : après chaque item Tier 1-3, relancer le bench de l'op
concernée et consigner le delta réel **avant** de capitaliser sur un gain. Aucun chiffre annoncé
sans mesure (cf. mémoire projet "hype vs preuve"). Les impacts ci-dessus sont des estimations
fondées sur des pertes reproduites, **pas des deltas mesurés**.

---

## Ordre d'exécution recommandé

1. **T0.1** (double payload) — plus gros gain de tokens, coût minimal, mais *vérifier le client d'abord*.
2. **T1.1** (strip ANSI) — débloque mécaniquement output, ~10 lignes.
3. **T1.2** (parser XML hijack) — tue les fill-2 % catastrophiques.
4. **T0.2 + T0.3** (comptabilité delta + honnêteté fragments) — bas coût, honnêteté.
5. **T1.3** (troncature query-aware) — attaque les pertes nettes prompt.
6. **T2.x** (filtres build/progress/traces) — couverture output.
7. **T3.1 → T3.2** (history : exploiter l'existant avant d'inventer).
8. **T4.x** — seulement après avoir branché un bench qui les exerce.

Chaque tier est shippable indépendamment ; aucun ne dépend du suivant.
