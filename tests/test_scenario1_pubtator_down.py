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
