# Guide de test du repli PubTator3 → BERN2 — Scénarios détaillés

> **Objectif :** vérifier que le pipeline bascule correctement de **PubTator3**
> vers **BERN2** dans chaque situation de défaillance, **sans toucher au code
> de production** et **sans casser les exécutions réelles**.
>
> **Fichier testé :** `src/utils/annotation_client.py`
> **Fonction testée :** `collect_annotations()`

---

## 0. Règles d'or (à lire avant de commencer)

Pour tester sans rien casser, on respecte **3 principes** :

1. **On n'édite JAMAIS `src/utils/annotation_client.py`.** Les tests vivent
   dans un nouveau dossier `tests/` séparé. Ajouter ce dossier ne modifie
   aucune ligne du code réel.
2. **On simule les pannes (mocking)** au lieu de vraiment casser le réseau.
   On remplace temporairement les fonctions réseau par de fausses fonctions
   *uniquement pendant le test*. Le vrai code reste intact.
3. **Toute modification de config réelle est interdite.** Si un scénario a
   besoin d'une config différente (ex. repli désactivé), on crée un
   **nouveau fichier de config de test**, on ne modifie pas
   `config/collect_pubtator.json` (utilisé en production).

> 💡 **Pourquoi le mocking ?** Casser réellement le réseau (couper le Wi-Fi,
> bloquer NCBI dans le fichier `hosts`) est risqué : on oublie souvent de
> remettre les choses en état, et les exécutions suivantes échouent. Le
> mocking remplace les appels réseau **seulement le temps du test**, puis
> Python restaure tout automatiquement à la fin.

---

## 1. Mise en place de l'environnement de test (fichiers à AJOUTER)

> ⚠️ Tous les fichiers de cette section sont **nouveaux**. On n'édite que
> `requirements.txt`, et seulement en **ajoutant** des lignes (rien n'est
> retiré ni modifié → aucune exécution réelle n'est affectée).

### 1.1 Fichier à AJOUTER : `requirements-dev.txt`

On crée un fichier séparé pour les dépendances de test, afin de **ne pas
polluer** `requirements.txt` (utilisé pour le déploiement réel).

**Chemin :** `C:\Users\ghof\pubmed_classifier\requirements-dev.txt`

```text
# Dépendances UNIQUEMENT pour les tests (ne pas installer en production)
pytest            # framework de test
pytest-mock       # fixture `mocker` pour simuler les pannes proprement
pytest-cov        # mesure de couverture de code (optionnel)
```

Installation (dans le venv du projet) :

```bash
# Depuis C:\Users\ghof\pubmed_classifier
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 1.2 Fichier à AJOUTER : `tests/conftest.py`

Ce fichier rend le module importable dans les tests **exactement comme le
fait le pipeline réel** (les scripts ajoutent `src/` au `sys.path` puis font
`from utils import annotation_client`). On reproduit ce comportement pour ne
rien changer à la structure existante.

**Chemin :** `C:\Users\ghof\pubmed_classifier\tests\conftest.py`

```python
"""
Configuration partagée par tous les tests.
Ajoute le dossier `src/` au chemin d'import, comme le font les scripts du
pipeline (scripts/collect_annotations.py). Ainsi `from utils import
annotation_client` fonctionne sans installer le projet et SANS modifier le
code de production.
"""
import os
import sys

# Chemin absolu vers .../pubmed_classifier/src
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# On l'ajoute en tête du sys.path s'il n'y est pas déjà.
if SRC not in sys.path:
    sys.path.insert(0, SRC)
```

### 1.3 Fichier à AJOUTER : `pytest.ini`

Configuration minimale de pytest. Déclare un marqueur `integration` pour
pouvoir séparer les tests réseau réels des tests rapides simulés.

**Chemin :** `C:\Users\ghof\pubmed_classifier\pytest.ini`

```ini
[pytest]
# Où chercher les tests
testpaths = tests
# Convention de nommage des fichiers de test
python_files = test_*.py
# Marqueur personnalisé pour les tests qui touchent le vrai réseau
markers =
    integration: tests nécessitant une connexion réelle à PubTator3 / BERN2
```

### 1.4 Arborescence finale (rien d'existant n'est modifié)

```
pubmed_classifier/
├── src/utils/annotation_client.py     ← INCHANGÉ (code de production)
├── config/collect_pubtator.json       ← INCHANGÉ (config de production)
├── requirements.txt                   ← INCHANGÉ
├── requirements-dev.txt               ← NOUVEAU
├── pytest.ini                         ← NOUVEAU
└── tests/                             ← NOUVEAU DOSSIER
    ├── conftest.py                    ← NOUVEAU
    ├── fixtures.py                    ← NOUVEAU (faux documents partagés)
    ├── test_scenario1_pubtator_down.py
    ├── test_scenario2_partial_coverage.py
    ├── test_scenario3_both_down.py
    ├── test_scenario4_fallback_disabled.py
    ├── test_scenario5_rate_limit_429.py
    ├── test_scenario6_logs.py
    └── test_scenario7_integration_live.py
```

### 1.5 Fichier à AJOUTER : `tests/fixtures.py` (faux documents réutilisés)

Pour éviter de répéter les mêmes faux documents dans chaque test.

**Chemin :** `C:\Users\ghof\pubmed_classifier\tests\fixtures.py`

```python
"""Faux documents partagés par les scénarios de test."""

# --- Document BERN2 déjà au format interne (ce que renvoie _request_bern2) ---
def make_bern2_doc(pmid):
    """Fabrique un faux document BERN2 minimal mais valide."""
    return {
        "_id": str(pmid),
        "pmid": str(pmid),
        "text": "BRCA1 mutation and breast cancer risk.",
        "source": "bern2",                       # provenance = BERN2
        "annotations": [{
            "id": ["672"],                       # identifiant du gène BRCA1
            "mention": "BRCA1",
            "obj": "gene",
            "prob": 0.99,
            "span": {"begin": 0, "end": 5},
        }],
    }

# --- Document BioC brut (ce que renvoie le RÉSEAU PubTator3, avant normalisation) ---
def make_pubtator_bioc_doc(pmid):
    """Fabrique un faux document BioC PubTator3 (format réseau brut)."""
    return {
        "pmid": str(pmid),
        "passages": [{
            "offset": 0,
            "text": "TP53 and colorectal cancer.",
            "annotations": [{
                "infons": {"type": "Gene", "identifier": "7157"},  # gène TP53
                "text": "TP53",
                "locations": [{"offset": 0, "length": 4}],
            }],
        }],
    }
```

---

## SCÉNARIO 1 — PubTator3 totalement en panne (réseau coupé / serveur down)

### 🌍 Situation réelle
Le serveur NCBI est inaccessible : coupure réseau, DNS qui ne résout pas,
ou NCBI renvoie une erreur 500/503. **Aucun** PMID ne peut être annoté par
PubTator3.

### 🔧 Chemin de code déclenché
Dans `collect_annotations()` :

```python
try:
    normalized = fetch_pubtator3(chunk, timeout=timeout)
except Exception as exc:               # ← l'exception réseau est attrapée ICI
    logger.warning("PubTator3 request failed for a chunk: %s", exc)
    normalized = {}                    # ← dict vide → TOUT le chunk part en repli
```
Puis `missing = [tous les PMIDs]` → tous passent à BERN2.

### 📄 Fichier à AJOUTER
**Chemin :** `tests\test_scenario1_pubtator_down.py`

```python
"""
SCÉNARIO 1 : PubTator3 lève une exception (panne totale).
Attendu : BERN2 annote TOUS les PMIDs, aucun échec.
"""
from utils import annotation_client          # import comme dans le pipeline réel
from fixtures import make_bern2_doc

def test_pubtator_totally_down_falls_back_to_bern2(mocker):
    # 1) On SIMULE PubTator3 en panne : son appel réseau lève une exception.
    #    `mocker.patch` remplace temporairement la fonction ; Python la
    #    restaure automatiquement à la fin du test.
    mocker.patch(
        "utils.annotation_client._request_pubtator3",
        side_effect=ConnectionError("PubTator3 injoignable"),   # panne simulée
    )

    # 2) On SIMULE BERN2 qui fonctionne : il renvoie un document valide.
    mocker.patch(
        "utils.annotation_client._request_bern2",
        return_value=[make_bern2_doc("11111111")],
    )

    # 3) On lance la collecte avec le repli activé.
    annotations, failed = annotation_client.collect_annotations(
        ["11111111"], use_bern2_fallback=True,
    )

    # 4) Vérifications :
    assert len(annotations) == 1                       # 1 document récupéré
    assert annotations[0]["source"] == "bern2"         # il vient bien de BERN2
    assert failed == []                                # aucun PMID perdu
```

### ▶️ Lancer
```bash
pytest tests/test_scenario1_pubtator_down.py -v
```

### ✅ Résultat attendu
```
tests/test_scenario1_pubtator_down.py::test_pubtator_totally_down_falls_back_to_bern2 PASSED
```

### 🧹 Nettoyage
**Aucun.** Le mock est automatiquement annulé à la fin du test. Le réseau réel
et le code ne sont jamais touchés.

---

## SCÉNARIO 2 — PubTator3 répond mais ne connaît pas certains PMIDs

### 🌍 Situation réelle
PubTator3 répond normalement (HTTP 200) mais ne couvre pas tous les articles :
les très récents ou non indexés sont absents de la réponse. Exemple : sur 2
PMIDs demandés, PubTator en connaît 1, l'autre doit basculer sur BERN2.

### 🔧 Chemin de code déclenché
```python
missing = [p for p in chunk if p not in normalized]   # ← détecte les absents
annotations.extend(normalized.values())               # ← garde les trouvés PubTator3
if missing and use_bern2_fallback:                    # ← bascule les absents sur BERN2
    ...
```

### 📄 Fichier à AJOUTER
**Chemin :** `tests\test_scenario2_partial_coverage.py`

```python
"""
SCÉNARIO 2 : couverture partielle de PubTator3.
PubTator connaît le PMID 22222222 mais PAS le 33333333.
Attendu : 22222222 vient de PubTator3, 33333333 bascule sur BERN2.
"""
from utils import annotation_client
from fixtures import make_bern2_doc, make_pubtator_bioc_doc

def test_partial_pubtator_coverage(mocker):
    # 1) On simule le RÉSEAU PubTator3 : il ne renvoie QUE le doc 22222222.
    #    On patche _request_pubtator3 (couche réseau) pour que la vraie
    #    fonction fetch_pubtator3() fasse la normalisation réelle par-dessus.
    mocker.patch(
        "utils.annotation_client._request_pubtator3",
        return_value=[make_pubtator_bioc_doc("22222222")],   # un seul doc BioC
    )

    # 2) On simule BERN2 : il sait répondre pour le PMID manquant 33333333.
    mocker.patch(
        "utils.annotation_client._request_bern2",
        return_value=[make_bern2_doc("33333333")],
    )

    # 3) On demande les DEUX PMIDs.
    annotations, failed = annotation_client.collect_annotations(
        ["22222222", "33333333"], use_bern2_fallback=True,
    )

    # 4) On indexe par PMID pour vérifier la provenance de chacun.
    by_pmid = {d["pmid"]: d["source"] for d in annotations}

    assert len(annotations) == 2
    assert by_pmid["22222222"] == "pubtator3"   # trouvé par PubTator3
    assert by_pmid["33333333"] == "bern2"       # basculé sur BERN2
    assert failed == []                          # aucun perdu
```

### ▶️ Lancer
```bash
pytest tests/test_scenario2_partial_coverage.py -v
```

### ✅ Résultat attendu
```
tests/test_scenario2_partial_coverage.py::test_partial_pubtator_coverage PASSED
```

### 🧹 Nettoyage
Aucun (mock auto-annulé).

---

## SCÉNARIO 3 — Les DEUX APIs sont en panne

### 🌍 Situation réelle
Coupure réseau totale, ou les deux serveurs sont down en même temps. Le
pipeline ne doit **pas planter** : il doit simplement signaler les PMIDs en
échec dans la liste `failed` (qui sera écrite dans `failed_pmids.txt` par le
script `collect_annotations.py`).

### 🔧 Chemin de code déclenché
```python
for pmid in missing:
    try:
        annotations.extend(_request_bern2([pmid], timeout=timeout))
    except Exception as exc:
        logger.warning("BERN2 fallback failed for %s: %s", pmid, exc)
        failed.append(pmid)          # ← PMID signalé comme échec total
        time.sleep(sleep)
```

### 📄 Fichier à AJOUTER
**Chemin :** `tests\test_scenario3_both_down.py`

```python
"""
SCÉNARIO 3 : PubTator3 ET BERN2 en panne.
Attendu : aucune annotation, le PMID atterrit dans `failed`, PAS de crash.
"""
from utils import annotation_client

def test_both_apis_down(mocker):
    # 1) PubTator3 en panne.
    mocker.patch(
        "utils.annotation_client._request_pubtator3",
        side_effect=ConnectionError("PubTator3 down"),
    )
    # 2) BERN2 en panne aussi.
    mocker.patch(
        "utils.annotation_client._request_bern2",
        side_effect=ConnectionError("BERN2 down"),
    )

    # 3) Collecte.
    annotations, failed = annotation_client.collect_annotations(
        ["99999999"], use_bern2_fallback=True,
    )

    # 4) Le pipeline survit : pas d'exception, et l'échec est tracé.
    assert annotations == []
    assert "99999999" in failed
```

### ▶️ Lancer
```bash
pytest tests/test_scenario3_both_down.py -v
```

### ✅ Résultat attendu
```
tests/test_scenario3_both_down.py::test_both_apis_down PASSED
```
👉 Le point clé démontré : **le pipeline ne lève jamais d'exception**, il
isole les échecs dans `failed`.

### 🧹 Nettoyage
Aucun.

---

## SCÉNARIO 4 — Repli BERN2 désactivé (`use_bern2_fallback=False`)

### 🌍 Situation réelle
On sait que BERN2 est indisponible (maintenance prolongée) et on veut gagner
du temps : on désactive le repli. Les PMIDs non couverts par PubTator3 doivent
aller directement dans `failed`, **sans aucun appel à BERN2**.

### 🔧 Chemin de code déclenché
```python
elif missing:
    failed.extend(missing)       # ← pas de repli → manquants directement en échec
```

### 📄 Comment activer ce mode SANS casser la prod

Il y a **deux façons**. La 2ᵉ est la plus sûre pour un vrai run.

#### Option A — en test (mock, recommandé pour vérifier le comportement)

**Chemin :** `tests\test_scenario4_fallback_disabled.py`

```python
"""
SCÉNARIO 4 : repli désactivé.
PubTator connaît 44444444 mais pas 55555555.
Attendu : 55555555 va dans `failed`, et BERN2 n'est JAMAIS appelé.
"""
from utils import annotation_client
from fixtures import make_pubtator_bioc_doc

def test_no_fallback_when_disabled(mocker):
    # PubTator3 ne renvoie qu'un seul des deux PMIDs demandés.
    mocker.patch(
        "utils.annotation_client._request_pubtator3",
        return_value=[make_pubtator_bioc_doc("44444444")],
    )
    # On espionne BERN2 pour PROUVER qu'il n'est jamais appelé.
    spy_bern2 = mocker.patch("utils.annotation_client._request_bern2")

    annotations, failed = annotation_client.collect_annotations(
        ["44444444", "55555555"],
        use_bern2_fallback=False,     # ← repli DÉSACTIVÉ
    )

    spy_bern2.assert_not_called()      # BERN2 n'a jamais été appelé
    assert len(annotations) == 1       # seul 44444444 est annoté
    assert "55555555" in failed        # 55555555 directement en échec
```

#### Option B — pour un VRAI run, ne touchez PAS `collect_pubtator.json`

Le fichier de production `config/collect_pubtator.json` contient :
```json
{ "use_bern2_fallback": true }
```
**Ne le modifiez pas.** Créez plutôt un **nouveau** fichier de config :

**Chemin à AJOUTER :** `config/collect_pubtator_nofallback.json`
```json
{
  "input": "../pipline_output/pmids.txt",
  "output_dir": "../pipline_output/annotations_nofallback",
  "use_bern2_fallback": false
}
```
Puis lancez le flux en pointant sur CETTE config (selon votre lanceur
Metaflow / script), ce qui laisse la config de prod intacte.

> ✅ **Avantage :** après le test, vous supprimez simplement
> `collect_pubtator_nofallback.json`. Le run de production continue d'utiliser
> `collect_pubtator.json` avec le repli activé, sans aucune intervention.

### ▶️ Lancer (option A)
```bash
pytest tests/test_scenario4_fallback_disabled.py -v
```

### ✅ Résultat attendu
```
tests/test_scenario4_fallback_disabled.py::test_no_fallback_when_disabled PASSED
```

### 🧹 Nettoyage
- Option A : aucun.
- Option B : supprimer `config/collect_pubtator_nofallback.json` après usage.

---

## SCÉNARIO 5 — PubTator3 renvoie HTTP 429 (trop de requêtes)

### 🌍 Situation réelle
On dépasse la limite de débit de NCBI (> 3 requêtes/seconde) : PubTator3
répond **429 Too Many Requests**. `resp.raise_for_status()` lève alors une
`requests.HTTPError`, traitée comme n'importe quelle panne → repli BERN2.

### 🔧 Chemin de code déclenché
Dans `_request_pubtator3()` :
```python
resp.raise_for_status()    # ← HTTP 429 → lève requests.HTTPError
```
…qui remonte jusqu'au `except Exception` de `collect_annotations()` → repli.

### 📄 Fichier à AJOUTER
**Chemin :** `tests\test_scenario5_rate_limit_429.py`

```python
"""
SCÉNARIO 5 : PubTator3 renvoie HTTP 429 (rate-limit).
Attendu : l'HTTPError est attrapée → repli BERN2.
"""
import requests
from utils import annotation_client
from fixtures import make_bern2_doc

def test_pubtator_rate_limited_falls_back(mocker):
    # On simule un 429 en faisant lever une HTTPError par la couche réseau
    # PubTator3 (c'est exactement ce que produit resp.raise_for_status()
    # sur un code 429 réel).
    mocker.patch(
        "utils.annotation_client._request_pubtator3",
        side_effect=requests.HTTPError("429 Too Many Requests"),
    )
    # BERN2 prend le relais correctement.
    mocker.patch(
        "utils.annotation_client._request_bern2",
        return_value=[make_bern2_doc("66666666")],
    )

    annotations, failed = annotation_client.collect_annotations(
        ["66666666"], use_bern2_fallback=True,
    )

    assert len(annotations) == 1
    assert annotations[0]["source"] == "bern2"   # le 429 a bien déclenché le repli
    assert failed == []
```

### ▶️ Lancer
```bash
pytest tests/test_scenario5_rate_limit_429.py -v
```

### ✅ Résultat attendu
```
tests/test_scenario5_rate_limit_429.py::test_pubtator_rate_limited_falls_back PASSED
```

### 🧹 Nettoyage
Aucun.

---

## SCÉNARIO 6 — Vérifier que les bons messages de log sont émis

### 🌍 Situation réelle
En production, on diagnostique les bascules grâce aux logs. On veut prouver
que le code émet bien :
- un **WARNING** quand PubTator3 échoue,
- un **INFO** « Falling back to BERN2 for N PMIDs » quand le repli s'active.

### 🔧 Chemin de code déclenché
```python
logger.warning("PubTator3 request failed for a chunk: %s", exc)   # WARNING
logger.info("Falling back to BERN2 for %d PMIDs", len(missing))    # INFO
```

### 📄 Fichier à AJOUTER
**Chemin :** `tests\test_scenario6_logs.py`

```python
"""
SCÉNARIO 6 : vérification des logs de bascule.
`caplog` est une fixture pytest qui capture les messages de log émis.
"""
import logging
from utils import annotation_client
from fixtures import make_bern2_doc

def test_warning_when_pubtator_fails(mocker, caplog):
    mocker.patch(
        "utils.annotation_client._request_pubtator3",
        side_effect=ConnectionError("down"),
    )
    mocker.patch(
        "utils.annotation_client._request_bern2",
        return_value=[make_bern2_doc("77777777")],
    )

    # On capture les logs de niveau WARNING émis par le module.
    with caplog.at_level(logging.WARNING, logger="utils.annotation_client"):
        annotation_client.collect_annotations(["77777777"])

    # Le message d'échec PubTator3 doit apparaître.
    assert any("PubTator3 request failed" in r.message for r in caplog.records)


def test_info_when_falling_back(mocker, caplog):
    # PubTator3 répond mais ne connaît pas le PMID → repli déclenché.
    mocker.patch(
        "utils.annotation_client._request_pubtator3",
        return_value=[],                    # aucun document → tout est "missing"
    )
    mocker.patch(
        "utils.annotation_client._request_bern2",
        return_value=[make_bern2_doc("88888888")],
    )

    with caplog.at_level(logging.INFO, logger="utils.annotation_client"):
        annotation_client.collect_annotations(["88888888"])

    # Le message de bascule doit apparaître.
    assert any("Falling back to BERN2" in r.message for r in caplog.records)
```

### ▶️ Lancer
```bash
pytest tests/test_scenario6_logs.py -v
```

### ✅ Résultat attendu
```
tests/test_scenario6_logs.py::test_warning_when_pubtator_fails PASSED
tests/test_scenario6_logs.py::test_info_when_falling_back       PASSED
```

### 🧹 Nettoyage
Aucun.

---

## SCÉNARIO 7 — Test d'intégration avec le VRAI réseau (optionnel)

### 🌍 Situation réelle
On veut confirmer, de temps en temps, que les deux APIs réelles répondent
encore (les URLs n'ont pas changé, NCBI fonctionne). Ce test **utilise
vraiment Internet** ; il est isolé sous le marqueur `integration` pour ne pas
ralentir / casser les tests rapides simulés.

### ⚠️ Précautions
- Nécessite une connexion Internet.
- Peut échouer si une API est temporairement down — c'est **normal**, ce
  n'est pas un bug du code.
- On le lance **séparément**, à la demande, jamais dans la suite rapide.

### 📄 Fichier à AJOUTER
**Chemin :** `tests\test_scenario7_integration_live.py`

```python
"""
SCÉNARIO 7 : intégration réseau réelle (marqueur `integration`).
Lancer explicitement : pytest -m integration
"""
import pytest
from utils import annotation_client

# PMIDs stables et anciens, bien indexés dans PubTator3.
KNOWN_PMIDS = ["25359038", "28212749"]

@pytest.mark.integration
def test_live_pubtator_returns_annotations():
    """PubTator3 réel doit annoter au moins un des PMIDs connus."""
    annotations, failed = annotation_client.collect_annotations(
        KNOWN_PMIDS, use_bern2_fallback=False,   # on isole PubTator3 seul
    )
    pubtator = [d for d in annotations if d["source"] == "pubtator3"]
    assert len(pubtator) > 0, "PubTator3 n'a retourné aucune annotation"

@pytest.mark.integration
def test_live_fallback_does_not_crash():
    """
    Un PMID volontairement invalide ne doit pas faire planter le pipeline :
    il finit soit annoté par BERN2, soit dans `failed`, sans exception.
    """
    annotations, failed = annotation_client.collect_annotations(
        ["999999999999"], use_bern2_fallback=True,
    )
    assert isinstance(annotations, list)
    assert isinstance(failed, list)
```

### ▶️ Lancer (uniquement les tests d'intégration)
```bash
pytest -m integration tests/test_scenario7_integration_live.py -v
```

### ✅ Résultat attendu (si le réseau et les APIs répondent)
```
tests/test_scenario7_integration_live.py::test_live_pubtator_returns_annotations PASSED
tests/test_scenario7_integration_live.py::test_live_fallback_does_not_crash      PASSED
```

### 🧹 Nettoyage
Aucun (lecture seule sur les APIs ; aucune écriture).

---

## 2. Lancer toute la suite

### Tests rapides (simulés, sans réseau) — à lancer souvent
```bash
# Depuis C:\Users\ghof\pubmed_classifier (venv activé)
pytest -v -m "not integration"
```

### Tests d'intégration (réseau réel) — à lancer ponctuellement
```bash
pytest -v -m integration
```

### Avec couverture de code du module testé
```bash
pytest -m "not integration" --cov=utils.annotation_client --cov-report=term-missing
```

---

## 3. Tableau récapitulatif des scénarios

| # | Scénario | PubTator3 | BERN2 | `failed` | Fichier de test |
|---|----------|-----------|-------|----------|-----------------|
| 1 | Panne totale PubTator3 | ❌ exception | ✅ | vide | `test_scenario1_pubtator_down.py` |
| 2 | Couverture partielle | ✅ partiel | ✅ | vide | `test_scenario2_partial_coverage.py` |
| 3 | Les deux en panne | ❌ | ❌ | rempli | `test_scenario3_both_down.py` |
| 4 | Repli désactivé | ✅ partiel | non appelé | rempli | `test_scenario4_fallback_disabled.py` |
| 5 | Rate-limit 429 | ❌ HTTP 429 | ✅ | vide | `test_scenario5_rate_limit_429.py` |
| 6 | Vérification des logs | ❌ / partiel | ✅ | — | `test_scenario6_logs.py` |
| 7 | Intégration réseau réel | réel | réel | — | `test_scenario7_integration_live.py` |

---

## 4. Récapitulatif : impact sur le code de production

| Action | Fichier | Type | Risque pour la prod |
|--------|---------|------|---------------------|
| Ajouter dépendances de test | `requirements-dev.txt` | **Nouveau** | Aucun |
| Configurer pytest | `pytest.ini` | **Nouveau** | Aucun |
| Rendre le module importable | `tests/conftest.py` | **Nouveau** | Aucun |
| Faux documents partagés | `tests/fixtures.py` | **Nouveau** | Aucun |
| 7 fichiers de scénarios | `tests/test_*.py` | **Nouveau** | Aucun |
| Config repli désactivé (scénario 4, option B) | `config/collect_pubtator_nofallback.json` | **Nouveau** | Aucun (à supprimer après usage) |
| **Code métier** | `src/utils/annotation_client.py` | **JAMAIS modifié** | — |
| **Config de prod** | `config/collect_pubtator.json` | **JAMAIS modifié** | — |

> ✅ **Conclusion :** tous les tests sont **purement additifs**. On ne modifie
> aucune ligne du code métier ni de la config de production. Supprimer le
> dossier `tests/` et les fichiers `*-dev.txt` / `pytest.ini` ramène le projet
> exactement à son état initial.
