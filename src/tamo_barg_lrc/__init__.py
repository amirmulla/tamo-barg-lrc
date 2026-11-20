"""Optimal Locally Recoverable Codes (single-set and two-set Tamo-Barg codes)."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tamo_barg_lrc._ff import (
        GF,
        HAS_CYTHON,
        DecodingError,
        MatrixSpace,
        PolynomialRing,
        RSCodec,
        VectorSpace,
        matrix,
        set_random_seed,
        vector,
        zero_vector,
    )
    from tamo_barg_lrc.expander import (
        compute_expansion_metrics,
        create_bipartite_coset_graph,
        get_error_position_subgraph,
        save_bipartite_graph_plot,
    )
    from tamo_barg_lrc.subgroups import (
        field_element_to_int,
        find_additive_subgroups,
        find_additive_subgroups_bruteforce,
        find_subspaces,
        is_closed_under_addition,
        list_canonical_additive_subgroups,
        list_multiplicative_subgroups,
        list_subgroup_cosets,
        shift_additive_subgroup,
        sort_field_elements,
    )
    from tamo_barg_lrc.tamo_barg import (
        TamoBargCode,
        TamoBargErasureDecoder,
        TamoBargErrorDecoder,
        TamoBargVectorEncoder,
    )
    from tamo_barg_lrc.tamo_barg_two_sets import (
        TamoBargCodeTwoSets,
        TamoBargGlobalDecoder,
        TamoBargGlobalEEDecoder,
        TamoBargIterativeDecoder,
        TamoBargIterativeEEDecoder,
        TamoBargTwoSetsVectorEncoder,
        TamoBargTwoStepDecoder,
        TamoBargTwoStepEEDecoder,
    )

__version__ = "0.2.0"

__all__ = [
    "GF",
    "VectorSpace",
    "MatrixSpace",
    "PolynomialRing",
    "DecodingError",
    "vector",
    "zero_vector",
    "matrix",
    "RSCodec",
    "HAS_CYTHON",
    "set_random_seed",
    "TamoBargCode",
    "TamoBargVectorEncoder",
    "TamoBargErasureDecoder",
    "TamoBargErrorDecoder",
    "TamoBargCodeTwoSets",
    "TamoBargTwoSetsVectorEncoder",
    "TamoBargIterativeDecoder",
    "TamoBargIterativeEEDecoder",
    "TamoBargGlobalDecoder",
    "TamoBargGlobalEEDecoder",
    "TamoBargTwoStepDecoder",
    "TamoBargTwoStepEEDecoder",
    "field_element_to_int",
    "sort_field_elements",
    "is_closed_under_addition",
    "find_subspaces",
    "find_additive_subgroups",
    "find_additive_subgroups_bruteforce",
    "list_multiplicative_subgroups",
    "list_canonical_additive_subgroups",
    "shift_additive_subgroup",
    "list_subgroup_cosets",
    "create_bipartite_coset_graph",
    "get_error_position_subgraph",
    "compute_expansion_metrics",
    "save_bipartite_graph_plot",
]

_EXPORT_TO_MODULE: dict[str, str] = {
    # tamo_barg_lrc._ff
    "GF": "tamo_barg_lrc._ff",
    "VectorSpace": "tamo_barg_lrc._ff",
    "MatrixSpace": "tamo_barg_lrc._ff",
    "PolynomialRing": "tamo_barg_lrc._ff",
    "DecodingError": "tamo_barg_lrc._ff",
    "vector": "tamo_barg_lrc._ff",
    "zero_vector": "tamo_barg_lrc._ff",
    "matrix": "tamo_barg_lrc._ff",
    "RSCodec": "tamo_barg_lrc._ff",
    "HAS_CYTHON": "tamo_barg_lrc._ff",
    "set_random_seed": "tamo_barg_lrc._ff",
    # tamo_barg_lrc.tamo_barg
    "TamoBargCode": "tamo_barg_lrc.tamo_barg",
    "TamoBargVectorEncoder": "tamo_barg_lrc.tamo_barg",
    "TamoBargErasureDecoder": "tamo_barg_lrc.tamo_barg",
    "TamoBargErrorDecoder": "tamo_barg_lrc.tamo_barg",
    # tamo_barg_lrc.tamo_barg_two_sets
    "TamoBargCodeTwoSets": "tamo_barg_lrc.tamo_barg_two_sets",
    "TamoBargTwoSetsVectorEncoder": "tamo_barg_lrc.tamo_barg_two_sets",
    "TamoBargIterativeDecoder": "tamo_barg_lrc.tamo_barg_two_sets",
    "TamoBargIterativeEEDecoder": "tamo_barg_lrc.tamo_barg_two_sets",
    "TamoBargGlobalDecoder": "tamo_barg_lrc.tamo_barg_two_sets",
    "TamoBargGlobalEEDecoder": "tamo_barg_lrc.tamo_barg_two_sets",
    "TamoBargTwoStepDecoder": "tamo_barg_lrc.tamo_barg_two_sets",
    "TamoBargTwoStepEEDecoder": "tamo_barg_lrc.tamo_barg_two_sets",
    # tamo_barg_lrc.subgroups
    "field_element_to_int": "tamo_barg_lrc.subgroups",
    "sort_field_elements": "tamo_barg_lrc.subgroups",
    "is_closed_under_addition": "tamo_barg_lrc.subgroups",
    "find_subspaces": "tamo_barg_lrc.subgroups",
    "find_additive_subgroups": "tamo_barg_lrc.subgroups",
    "find_additive_subgroups_bruteforce": "tamo_barg_lrc.subgroups",
    "list_multiplicative_subgroups": "tamo_barg_lrc.subgroups",
    "list_canonical_additive_subgroups": "tamo_barg_lrc.subgroups",
    "shift_additive_subgroup": "tamo_barg_lrc.subgroups",
    "list_subgroup_cosets": "tamo_barg_lrc.subgroups",
    # tamo_barg_lrc.expander
    "create_bipartite_coset_graph": "tamo_barg_lrc.expander",
    "get_error_position_subgraph": "tamo_barg_lrc.expander",
    "compute_expansion_metrics": "tamo_barg_lrc.expander",
    "save_bipartite_graph_plot": "tamo_barg_lrc.expander",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve and cache public package exports per PEP 562."""
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return sorted module attributes including lazy PEP 562 exports."""
    return sorted({*globals().keys(), *__all__})
