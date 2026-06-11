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
