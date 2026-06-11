# Mécanisme de repli PubTator3 → BERN2 — Explication détaillée (code commenté)

> **Fichier source concerné :** `src/utils/annotation_client.py`
> **Fonction principale :** `collect_annotations()`
>
> Dans ce document, **chaque ligne de code est commentée** pour que vous
> puissiez expliquer le fonctionnement précisément lors de la soutenance.

---

## 1. Vue d'ensemble : pourquoi deux sources d'annotation ?

Le pipeline a besoin d'annotations biomédicales (gènes, maladies, mutations,
médicaments…) pour chaque article PubMed identifié par son PMID.
Deux APIs publiques peuvent fournir ces annotations :

| API | Fournisseur | Format retourné | Disponibilité |
|-----|------------|-----------------|---------------|
| **PubTator3** | NCBI (NIH) | BioC-JSON | Très stable, mais ne couvre pas 100 % des PMIDs |
| **BERN2** | Korea University | JSON "plat" (format interne) | Couverture plus large, serveur parfois surchargé |

La stratégie choisie est **PubTator3 en priorité, BERN2 en repli**.
Cela maximise la stabilité (NCBI est plus fiable) tout en maximisant la
couverture (BERN2 comble les trous).

---

## 2. Les constantes et la configuration du module (commenté)

```python
import logging          # journalisation (WARNING / INFO) au lieu de print()
import time             # time.sleep() pour respecter la limite de débit NCBI
from typing import Dict, List, Tuple   # annotations de type (lisibilité)

import requests         # bibliothèque HTTP pour interroger les deux APIs

# Crée un logger propre à ce module. __name__ vaut "src.utils.annotation_client".
# Tous les messages porteront ce nom → on peut filtrer/capturer ces logs précisément.
logger = logging.getLogger(__name__)

# URL de l'API PubTator3 (NCBI). Endpoint "export/biocjson" = sortie BioC-JSON.
PUBTATOR3_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/export/biocjson"
)

# URL de l'API BERN2 (Korea University). On y ajoutera /<pmid> à la requête.
BERN2_URL = "http://bern2.korea.ac.kr/pubmed"

# PubTator3 ne fournit PAS de score de confiance par annotation.
# Le pipeline aval filtre les annotations sur prob >= seuil ; on donne donc
# le poids maximum (1.0) à toute entité PubTator3 pour qu'aucune ne soit rejetée.
DEFAULT_PROB = 1.0

# Table de correspondance : convertit le vocabulaire de types PubTator3
# vers le vocabulaire "obj" attendu par le reste du pipeline (issu de BERN2).
# Clé = type PubTator (en minuscules) ; Valeur = type interne.
_TYPE_MAP = {
    "gene":             "gene",       # gène → gène
    "disease":          "disease",    # maladie → maladie
    "chemical":         "drug",       # PubTator dit "chemical", le pipeline veut "drug"
    "species":          "species",    # espèce → espèce
    "variant":          "mutation",   # toutes les variantes deviennent "mutation"
    "mutation":         "mutation",
    "dnamutation":      "mutation",
    "proteinmutation":  "mutation",
    "snp":              "mutation",
    "cellline":         "cell_line",  # lignée cellulaire → cell_line
}
```

---

## 3. Fonctions utilitaires (commentées)

### 3.1 `chunk_list()` — découper la liste de PMIDs en lots

```python
def chunk_list(items: List, chunk_size: int = 100) -> List[List]:
    """Découpe une liste en lots consécutifs d'au plus `chunk_size` éléments."""
    # range(0, len, chunk_size) → indices de départ : 0, 100, 200, ...
    # items[i:i+chunk_size]     → tranche de 100 éléments à partir de i.
    # PubTator3 limite ~100 PMIDs par requête : d'où le découpage.
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
```

### 3.2 `_map_obj()` — traduire un type d'entité

```python
def _map_obj(pubtator_type: str) -> str:
    # Si le type est vide / None, on renvoie "unknown" pour ne jamais planter.
    if not pubtator_type:
        return "unknown"
    # .lower() → normalise la casse ("Gene" → "gene").
    # _TYPE_MAP.get(clé, défaut) → si le type n'est pas dans la table,
    # on garde le type tel quel (en minuscules) au lieu de lever une erreur.
    return _TYPE_MAP.get(pubtator_type.lower(), pubtator_type.lower())
```

### 3.3 `_split_identifier()` — convertir l'identifiant en liste

```python
def _split_identifier(identifier, obj: str) -> List[str]:
    """Transforme l'identifiant PubTator (chaîne) en liste d'ids façon BERN2."""
    # Cas "pas d'identifiant" : PubTator peut renvoyer None, "", "-", "None".
    # On renvoie ["CUI-less"] = convention du pipeline pour "entité non normalisée".
    if identifier in (None, "", "-", "None"):
        return ["CUI-less"]

    # Un identifiant peut contenir plusieurs ids séparés par "," ou ";".
    # 1) .replace(",", ";")  → on uniformise tout en ";".
    # 2) .split(";")          → on découpe.
    # 3) p.strip()            → on enlève les espaces autour de chaque id.
    # 4) if p.strip()         → on ignore les morceaux vides.
    parts = [p.strip() for p in str(identifier).replace(",", ";").split(";") if p.strip()]

    # Si après nettoyage il ne reste rien, on retombe sur "CUI-less".
    if not parts:
        return ["CUI-less"]

    # Cas particulier des espèces : BERN2 écrit l'humain "NCBITaxon:9606",
    # mais PubTator donne juste "9606". On préfixe "NCBITaxon:" si absent,
    # sinon le filtre "espèce humaine" en aval ne reconnaîtrait pas l'entité.
    if obj == "species":
        parts = [p if p.startswith("NCBITaxon:") else f"NCBITaxon:{p}" for p in parts]

    return parts
```

### 3.4 `_reconstruct_text()` — reconstruire le texte complet

```python
def _reconstruct_text(passages: List[dict]) -> str:
    """
    Reconstruit le texte complet du document à partir des 'passages' BioC,
    de sorte que les offsets ABSOLUS des annotations correspondent aux bonnes
    positions de caractères dans le texte renvoyé.
    """
    # Aucun passage → texte vide (article sans contenu).
    if not passages:
        return ""

    # Longueur totale = position de fin la plus lointaine parmi tous les passages.
    # offset = position de début du passage ; len(text) = sa longueur.
    total = max(p.get("offset", 0) + len(p.get("text", "")) for p in passages)

    # On crée un "buffer" rempli d'espaces, de la bonne longueur totale.
    # On va y replacer chaque passage à sa position absolue exacte.
    buffer = [" "] * total

    # Pour chaque passage (titre, résumé...), on recopie son texte
    # caractère par caractère à la position absolue (offset + i).
    for passage in passages:
        offset = passage.get("offset", 0)
        text = passage.get("text", "")
        for i, char in enumerate(text):
            buffer[offset + i] = char

    # On recolle le buffer en une seule chaîne de caractères.
    return "".join(buffer)
```

---

## 4. La normalisation PubTator3 → format interne (commentée)

C'est la fonction clé : elle rend une annotation PubTator3 **indiscernable**
d'une annotation BERN2 pour le reste du pipeline.

```python
def normalize_pubtator_doc(doc: dict) -> dict:
    """Convertit UN document BioC PubTator3 dans le format interne (façon BERN2)."""

    # Le PMID peut se trouver sous différentes clés selon la réponse :
    # on essaie "pmid", puis "id", puis "_id". str(...) pour forcer le texte.
    pmid = str(doc.get("pmid") or doc.get("id") or doc.get("_id") or "")

    # Les passages = morceaux du document (titre, résumé...).
    passages = doc.get("passages", [])

    # On reconstruit le texte complet pour que les offsets restent cohérents.
    text = _reconstruct_text(passages)

    annotations = []                       # liste finale d'annotations normalisées
    for passage in passages:               # on parcourt chaque passage...
        for ann in passage.get("annotations", []):   # ...et chaque annotation dedans

            # infons = métadonnées BioC de l'annotation (type, identifiant...).
            infons = ann.get("infons", {})

            # On traduit le type ("Gene" → "gene"). On tente "type" puis "biotype".
            obj = _map_obj(infons.get("type") or infons.get("biotype"))

            # locations = positions de l'entité dans le texte.
            locations = ann.get("locations", [])
            if not locations:              # pas de position → annotation inutilisable
                continue                   # on l'ignore

            loc = locations[0]             # on prend la première position
            begin = loc.get("offset", 0)              # début = offset absolu
            end = begin + loc.get("length", 0)        # fin = début + longueur

            # On assemble l'annotation au FORMAT INTERNE (identique à BERN2) :
            annotations.append({
                "id":      _split_identifier(infons.get("identifier"), obj),  # liste d'ids
                "mention": ann.get("text", ""),    # texte exact de l'entité ("BRCA1")
                "obj":     obj,                     # type traduit ("gene")
                "prob":    DEFAULT_PROB,            # 1.0 (PubTator n'a pas de score)
                "span":    {"begin": begin, "end": end},   # positions dans le texte
            })

    # On renvoie le document complet au format interne.
    # Le champ "source" = "pubtator3" sert UNIQUEMENT à la traçabilité.
    return {
        "_id":         pmid,
        "pmid":        pmid,
        "text":        text,
        "source":      "pubtator3",
        "annotations": annotations,
    }
```

---

## 5. Les appels réseau bas niveau (commentés)

### 5.1 `_request_pubtator3()` — interroger PubTator3

```python
def _request_pubtator3(pmids: List[str], timeout: int = 30) -> List[dict]:
    """Interroge l'endpoint d'export PubTator3 pour un lot de PMIDs."""

    # Requête HTTP GET vers PubTator3.
    resp = requests.get(
        PUBTATOR3_URL,
        # params : ?pmids=12345,67890,... (PMIDs séparés par des virgules)
        params={"pmids": ",".join(str(p) for p in pmids)},
        timeout=timeout,    # abandonne après `timeout` secondes (évite de bloquer)
    )

    # Si le serveur renvoie une erreur HTTP (4xx/5xx), lève une HTTPError.
    # ⇒ C'EST CETTE LIGNE qui déclenche le repli en cas de 429/500/503 etc.
    resp.raise_for_status()

    # Convertit la réponse JSON en objet Python (dict ou liste).
    payload = resp.json()

    # L'endpoint enveloppe les documents sous la clé "PubTator3"
    # (ou parfois "documents"). On gère les deux cas + le cas liste brute.
    if isinstance(payload, dict):
        return payload.get("PubTator3", payload.get("documents", []))
    if isinstance(payload, list):
        return payload
    return []      # format inattendu → liste vide (aucun document)
```

### 5.2 `_request_bern2()` — interroger BERN2 (le repli)

```python
def _request_bern2(pmids: List[str], timeout: int = 30) -> List[dict]:
    """Interroge BERN2 pour un lot de PMIDs (déjà au format interne)."""

    # BERN2 attend les PMIDs dans l'URL : .../pubmed/12345,67890
    url = f"{BERN2_URL}/{','.join(str(p) for p in pmids)}"

    resp = requests.get(url, timeout=timeout)   # requête GET
    resp.raise_for_status()                     # lève une erreur si HTTP 4xx/5xx
    data = resp.json()                          # JSON → Python

    # BERN2 peut renvoyer un dict unique (un seul PMID) au lieu d'une liste.
    # On l'enveloppe dans une liste pour traiter tous les cas uniformément.
    if isinstance(data, dict):
        data = [data]

    # On garantit les champs attendus par le pipeline :
    for doc in data:
        doc.setdefault("source", "bern2")        # marque la provenance "bern2"
        # Si "_id" manque mais "pmid" existe, on recopie pmid dans _id.
        if "_id" not in doc and "pmid" in doc:
            doc["_id"] = str(doc["pmid"])

    return data
```

### 5.3 `fetch_pubtator3()` — appeler + normaliser en une étape

```python
def fetch_pubtator3(pmids: List[str], timeout: int = 30) -> Dict[str, dict]:
    """Récupère ET normalise les annotations PubTator3, indexées par PMID."""

    # 1) Appel réseau brut (peut lever une exception → repli).
    docs = _request_pubtator3(pmids, timeout=timeout)

    normalized = {}                # dictionnaire { pmid: document_normalisé }
    for doc in docs:
        norm = normalize_pubtator_doc(doc)   # BioC → format interne
        if norm["pmid"]:                     # on ignore les docs sans PMID
            normalized[norm["pmid"]] = norm  # indexé par PMID pour recherche rapide

    # Renvoie un DICT (pas une liste) : permet de tester `if pmid in normalized`
    # très efficacement à l'étape de détection des PMIDs manquants.
    return normalized
```

---

## 6. LE CŒUR : `collect_annotations()` (commenté ligne par ligne)

```python
def collect_annotations(
    pmids: List[str],            # liste des PMIDs à annoter
    chunk_size: int = 100,       # taille des lots (limite PubTator3)
    sleep: float = 0.34,         # pause entre requêtes (~3 req/s, limite NCBI)
    timeout: int = 30,           # délai max par requête HTTP
    use_bern2_fallback: bool = True,   # active/désactive le repli BERN2
) -> Tuple[List[dict], List[str]]:
    """
    Collecte les annotations pour `pmids` via PubTator3, avec repli sur BERN2.

    Renvoie un tuple (annotations, failed_pmids) :
      - annotations  = liste de docs au format interne (1 par PMID annoté)
      - failed_pmids = PMIDs qu'AUCUNE des deux sources n'a pu annoter
    """
    annotations: List[dict] = []   # accumulateur des documents annotés
    failed: List[str] = []         # accumulateur des PMIDs en échec total

    # Nettoyage : on convertit en str, on enlève les espaces, on retire les vides.
    pmids = [str(p).strip() for p in pmids if str(p).strip()]

    # On traite les PMIDs par lots de `chunk_size`.
    for chunk in chunk_list(pmids, chunk_size):

        # ───────────────────────────────────────────────────────────────
        # ÉTAPE 1 — Tentative PubTator3 (source prioritaire)
        # ───────────────────────────────────────────────────────────────
        try:
            # Appel + normalisation. Renvoie { pmid: doc } pour les PMIDs trouvés.
            normalized = fetch_pubtator3(chunk, timeout=timeout)
        except Exception as exc:
            # ⚠️ DÉCLENCHEUR DE REPLI #1 (panne totale) :
            # réseau coupé, timeout, HTTP 4xx/5xx, JSON invalide... → on log
            # un WARNING et on repart avec un dict VIDE : tout le chunk
            # sera considéré "manquant" et envoyé à BERN2.
            logger.warning("PubTator3 request failed for a chunk: %s", exc)
            normalized = {}

        # ───────────────────────────────────────────────────────────────
        # ÉTAPE 2 — Quels PMIDs manquent à l'appel ?
        # ───────────────────────────────────────────────────────────────
        # ⚠️ DÉCLENCHEUR DE REPLI #2 (couverture partielle) :
        # même si PubTator3 a répondu, certains PMIDs peuvent être absents
        # de la réponse (articles trop récents, non indexés...).
        missing = [p for p in chunk if p not in normalized]

        # On accepte tout de suite les annotations PubTator3 réussies.
        # .values() = les documents (on n'a plus besoin des clés ici).
        annotations.extend(normalized.values())

        # ───────────────────────────────────────────────────────────────
        # ÉTAPE 3 — Repli BERN2 pour les PMIDs manquants
        # ───────────────────────────────────────────────────────────────
        if missing and use_bern2_fallback:
            # Il y a des manquants ET le repli est activé.
            logger.info("Falling back to BERN2 for %d PMIDs", len(missing))

            # On interroge BERN2 PMID PAR PMID (granularité fine : un échec
            # sur un PMID n'empêche pas les autres d'être récupérés).
            for pmid in missing:
                try:
                    # Appel BERN2 ; déjà au format interne → on ajoute directement.
                    annotations.extend(_request_bern2([pmid], timeout=timeout))
                except Exception as exc:
                    # BERN2 échoue aussi pour ce PMID → échec total.
                    logger.warning("BERN2 fallback failed for %s: %s", pmid, exc)
                    failed.append(pmid)        # on le note dans `failed`
                    time.sleep(sleep)          # petite pause même en cas d'échec

        elif missing:
            # Il y a des manquants MAIS le repli est désactivé
            # (use_bern2_fallback=False) → ces PMIDs sont directement perdus.
            failed.extend(missing)

        # Pause de politesse entre deux lots (respect de la limite NCBI).
        time.sleep(sleep)

    # Bilan final dans les logs : combien annotés, combien échoués.
    logger.info(
        "Collected %d annotated documents (%d failed)",
        len(annotations), len(failed),
    )

    # On renvoie les deux listes à l'appelant.
    return annotations, failed
```

---

## 7. Arbre décisionnel (résumé)

```
Pour chaque PMID dans la liste :
│
├── PubTator3 répond ET connaît le PMID ?
│   └── OUI → annotation PubTator3 acceptée ✅
│
├── PubTator3 répond MAIS ne connaît pas le PMID ? (DÉCLENCHEUR #2)
│   └── use_bern2_fallback = True ?
│       ├── OUI → essai BERN2
│       │   ├── BERN2 répond → annotation BERN2 acceptée ✅
│       │   └── BERN2 échoue → PMID ajouté à `failed` ❌
│       └── NON → PMID ajouté à `failed` ❌
│
└── PubTator3 lève une exception (panne, timeout, 429, 500…) ? (DÉCLENCHEUR #1)
    └── tout le chunk → BERN2 (même sous-arbre que ci-dessus)
```

---

## 8. Points à retenir pour une soutenance

1. **Le repli est automatique et transparent** : aucun redémarrage ni
   reconfiguration nécessaire en cas de panne PubTator3.
2. **Deux déclencheurs distincts** :
   - #1 = exception réseau/HTTP (`except Exception` → `normalized = {}`)
   - #2 = couverture partielle (`missing = [p for p in chunk if p not in normalized]`)
3. **Granularité fine** : le repli opère PMID par PMID, pas en bloc.
4. **Format de sortie unique** : `normalize_pubtator_doc()` rend toute la
   chaîne aval (NER, TF-IDF, classifieur) indépendante de la source.
5. **Repli désactivable** : `use_bern2_fallback=False` dans
   `config/collect_pubtator.json`.
6. **Politesse réseau** : `time.sleep(0.34)` ≈ 3 requêtes/s (limite NCBI).
