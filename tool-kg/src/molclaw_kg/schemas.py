from __future__ import annotations

from .edge_ontology import (
    build_adjudication_schema,
    default_edge_ontology_path,
    load_edge_ontology,
)


# Compatibility export for code that does not carry ProjectConfig. The schema
# is generated from the ontology file rather than maintained independently.
ADJUDICATION_SCHEMA = build_adjudication_schema(
    load_edge_ontology(default_edge_ontology_path())
)
