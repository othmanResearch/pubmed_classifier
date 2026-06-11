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
