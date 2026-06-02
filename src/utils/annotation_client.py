"""
Annotation collection layer.

Fetches biomedical entity annotations for a list of PubMed IDs from
PubTator3 (primary) and falls back to BERN2 when PubTator3 is unavailable or
returns no document for a given PMID.

PubTator3 returns BioC-JSON, which is normalized here into the same dict shape
that BERN2 produces and that the rest of this repo already consumes:

    {
        "_id":   "<pmid>",
        "pmid":  "<pmid>",
        "text":  "<title + abstract>",
        "source": "pubtator3" | "bern2",
        "annotations": [
            {
                "id":      ["MESH:D003550", ...],
                "mention": "cystic fibrosis",
                "obj":     "disease",
                "prob":    1.0,
                "span":    {"begin": 161, "end": 176}
            },
            ...
        ]
    }
"""

import logging
import time
from typing import Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

PUBTATOR3_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/export/biocjson"
)
BERN2_URL = "http://bern2.korea.ac.kr/pubmed"

# PubTator3 does not expose a per-annotation confidence; downstream probability
# filtering keeps anything >= cutoff, so PubTator entities are given full weight.
DEFAULT_PROB = 1.0

# Map PubTator entity types to the BERN2 'obj' vocabulary used downstream.
_TYPE_MAP = {
    "gene": "gene",
    "disease": "disease",
    "chemical": "drug",
    "species": "species",
    "variant": "mutation",
    "mutation": "mutation",
    "dnamutation": "mutation",
    "proteinmutation": "mutation",
    "snp": "mutation",
    "cellline": "cell_line",
}


def chunk_list(items: List, chunk_size: int = 100) -> List[List]:
    """Split a list into consecutive chunks of at most ``chunk_size`` items."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _map_obj(pubtator_type: str) -> str:
    if not pubtator_type:
        return "unknown"
    return _TYPE_MAP.get(pubtator_type.lower(), pubtator_type.lower())


def _split_identifier(identifier, obj: str) -> List[str]:
    """Turn a PubTator identifier string into a BERN2-style list of ids."""
    if identifier in (None, "", "-", "None"):
        return ["CUI-less"]
    parts = [p.strip() for p in str(identifier).replace(",", ";").split(";") if p.strip()]
    if not parts:
        return ["CUI-less"]
    # BERN2 represents the human species as 'NCBITaxon:9606'; PubTator gives the
    # bare taxonomy id, so prefix it to keep the human-species filter working.
    if obj == "species":
        parts = [p if p.startswith("NCBITaxon:") else f"NCBITaxon:{p}" for p in parts]
    return parts


def _reconstruct_text(passages: List[dict]) -> str:
    """
    Rebuild the full document text from BioC passages so that the absolute
    annotation offsets line up with character positions in the returned text.
    """
    if not passages:
        return ""
    total = max(p.get("offset", 0) + len(p.get("text", "")) for p in passages)
    buffer = [" "] * total
    for passage in passages:
        offset = passage.get("offset", 0)
        text = passage.get("text", "")
        for i, char in enumerate(text):
            buffer[offset + i] = char
    return "".join(buffer)


def normalize_pubtator_doc(doc: dict) -> dict:
    """Convert one PubTator3 BioC document into the internal BERN2-like shape."""
    pmid = str(doc.get("pmid") or doc.get("id") or doc.get("_id") or "")
    passages = doc.get("passages", [])
    text = _reconstruct_text(passages)

    annotations = []
    for passage in passages:
        for ann in passage.get("annotations", []):
            infons = ann.get("infons", {})
            obj = _map_obj(infons.get("type") or infons.get("biotype"))
            locations = ann.get("locations", [])
            if not locations:
                continue
            loc = locations[0]
            begin = loc.get("offset", 0)
            end = begin + loc.get("length", 0)
            annotations.append({
                "id": _split_identifier(infons.get("identifier"), obj),
                "mention": ann.get("text", ""),
                "obj": obj,
                "prob": DEFAULT_PROB,
                "span": {"begin": begin, "end": end},
            })

    return {
        "_id": pmid,
        "pmid": pmid,
        "text": text,
        "source": "pubtator3",
        "annotations": annotations,
    }


def _request_pubtator3(pmids: List[str], timeout: int = 30) -> List[dict]:
    """Query the PubTator3 export endpoint for a batch of PMIDs."""
    resp = requests.get(
        PUBTATOR3_URL,
        params={"pmids": ",".join(str(p) for p in pmids)},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    # The export endpoint wraps documents under the 'PubTator3' key.
    if isinstance(payload, dict):
        return payload.get("PubTator3", payload.get("documents", []))
    if isinstance(payload, list):
        return payload
    return []


def _request_bern2(pmids: List[str], timeout: int = 30) -> List[dict]:
    """Query BERN2 for a batch of PMIDs (already in the internal shape)."""
    url = f"{BERN2_URL}/{','.join(str(p) for p in pmids)}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        data = [data]
    for doc in data:
        doc.setdefault("source", "bern2")
        if "_id" not in doc and "pmid" in doc:
            doc["_id"] = str(doc["pmid"])
    return data


def fetch_pubtator3(pmids: List[str], timeout: int = 30) -> Dict[str, dict]:
    """Fetch and normalize PubTator3 annotations, keyed by PMID."""
    docs = _request_pubtator3(pmids, timeout=timeout)
    normalized = {}
    for doc in docs:
        norm = normalize_pubtator_doc(doc)
        if norm["pmid"]:
            normalized[norm["pmid"]] = norm
    return normalized


def collect_annotations(
    pmids: List[str],
    chunk_size: int = 100,
    sleep: float = 0.34,
    timeout: int = 30,
    use_bern2_fallback: bool = True,
) -> Tuple[List[dict], List[str]]:
    """
    Collect annotations for ``pmids`` using PubTator3, falling back to BERN2.

    Returns a tuple ``(annotations, failed_pmids)`` where ``annotations`` is a
    list of internal-shape dicts (one per successfully annotated PMID) and
    ``failed_pmids`` lists the PMIDs that neither source could annotate.
    """
    annotations: List[dict] = []
    failed: List[str] = []
    pmids = [str(p).strip() for p in pmids if str(p).strip()]

    for chunk in chunk_list(pmids, chunk_size):
        try:
            normalized = fetch_pubtator3(chunk, timeout=timeout)
        except Exception as exc:  # network / HTTP / JSON errors
            logger.warning("PubTator3 request failed for a chunk: %s", exc)
            normalized = {}

        missing = [p for p in chunk if p not in normalized]
        annotations.extend(normalized.values())

        if missing and use_bern2_fallback:
            logger.info("Falling back to BERN2 for %d PMIDs", len(missing))
            for pmid in missing:
                try:
                    annotations.extend(_request_bern2([pmid], timeout=timeout))
                except Exception as exc:
                    logger.warning("BERN2 fallback failed for %s: %s", pmid, exc)
                    failed.append(pmid)
                    time.sleep(sleep)
        elif missing:
            failed.extend(missing)

        time.sleep(sleep)

    logger.info(
        "Collected %d annotated documents (%d failed)",
        len(annotations), len(failed),
    )
    return annotations, failed
