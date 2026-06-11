"""
Configuration partagée par tous les tests.
Ajoute le dossier `src/` au chemin d'import, comme le font les scripts du
pipeline (scripts/collect_annotations.py). Ainsi `from utils import
annotation_client` fonctionne sans installer le projet et SANS modifier le
code de production.
"""
import os
import sys

# Chemin absolu vers .../pubmed_classifier/src
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# On l'ajoute en tête du sys.path s'il n'y est pas déjà.
if SRC not in sys.path:
    sys.path.insert(0, SRC)
