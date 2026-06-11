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
