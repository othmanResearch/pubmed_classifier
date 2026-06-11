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
