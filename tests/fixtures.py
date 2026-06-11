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
