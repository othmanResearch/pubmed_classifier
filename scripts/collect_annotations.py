from metaflow import FlowSpec, step, Config, Parameter
import sys
import os
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import annotation_client

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)


def _read_pmids(path):
    """Read a PMID list: one id per line, blanks and '#' comments ignored."""
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        pmids = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pmids.append(line)
    return pmids


class collectAnnotations(FlowSpec):
    """
    Collect biomedical annotations for a list of PMIDs from PubTator3, falling
    back to BERN2, and write downstream-ready JSON chunks (BERN2 shape).
    """

    config = Config("collect", required=True)
    chunk_size = Parameter("chunk_size", default=100,
                           help="Number of PMIDs per PubTator3 request")
    files_per_chunk = Parameter("files_per_chunk", default=50,
                                help="Number of documents written per output JSON file")

    @step
    def start(self):
        self.pmids = _read_pmids(self.config.input)
        logging.info("%d PMIDs will be annotated", len(self.pmids))
        self.next(self.collect)

    @step
    def collect(self):
        """Fetch annotations (PubTator3 primary, BERN2 fallback)."""
        use_fallback = bool(getattr(self.config, "use_bern2_fallback", True))
        self.annotations, self.failed_pmids = annotation_client.collect_annotations(
            self.pmids,
            chunk_size=self.chunk_size,
            use_bern2_fallback=use_fallback,
        )
        logging.info(
            "%d documents annotated, %d failed",
            len(self.annotations), len(self.failed_pmids),
        )
        self.next(self.store)

    @step
    def store(self):
        """Persist annotations as JSON chunk files consumable by preprocess_ners."""
        output_dir = os.path.expanduser(self.config.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        batches = annotation_client.chunk_list(self.annotations, self.files_per_chunk)
        for i, batch in enumerate(batches):
            out_path = os.path.join(output_dir, f"chunk{i}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %d JSON files to %s", len(batches), output_dir)

        if self.failed_pmids:
            failed_path = os.path.join(output_dir, "failed_pmids.txt")
            with open(failed_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.failed_pmids))
            logging.info("Logged %d failed PMIDs to %s",
                         len(self.failed_pmids), failed_path)
        self.next(self.end)

    @step
    def end(self):
        logging.info("Annotation collection complete")


if __name__ == "__main__":
    collectAnnotations()
