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
