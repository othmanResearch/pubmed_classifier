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
