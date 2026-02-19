#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use it except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Build Elasticsearch request body for retriever-based hybrid search (ES 8.14+).

Minimal dependencies so unit tests can import without loading the ES connection or settings.
"""

from elasticsearch_dsl import Q

from common.constants import PAGERANK_FLD, TAG_FLD
from common.doc_store.doc_store_base import MatchTextExpr, MatchDenseExpr, OrderByExpr


def rank_feature_field_name(fld: str) -> str:
    """Return the ES field name for a rank feature (pagerank as-is, others under TAG_FLD)."""
    return fld if fld == PAGERANK_FLD else f"{TAG_FLD}.{fld}"


def build_retriever_body(
    match_text: MatchTextExpr,
    match_dense: MatchDenseExpr,
    bool_query: Q,
    vector_similarity_weight: float,
    offset: int,
    limit: int,
    highlight_fields: list[str],
    order_by: OrderByExpr,
    agg_fields: list[str] | None,
    rank_feature: dict | None = None,
) -> dict:
    """Build the Elasticsearch request body for retriever-based hybrid search (ES 8.14+)."""
    size = limit if limit > 0 else match_dense.topn
    rank_window_size = max(size, match_dense.topn)

    # Text (keyword) part of the hybrid query
    filter_clauses = [q.to_dict() for q in bool_query.filter]
    minimum_should_match = match_text.extra_options.get("minimum_should_match", 0.0)
    if isinstance(minimum_should_match, float):
        minimum_should_match = str(int(minimum_should_match * 100)) + "%"
    text_query = Q(
        "query_string",
        fields=match_text.fields,
        type="best_fields",
        query=match_text.matching_text,
        minimum_should_match=minimum_should_match,
    ).to_dict()
    standard_query_bool: dict = {"must": [text_query], "filter": filter_clauses}
    if rank_feature:
        for fld, sc in rank_feature.items():
            field_name = rank_feature_field_name(fld)
            standard_query_bool.setdefault("should", []).append(
                Q("rank_feature", field=field_name, linear={}, boost=sc).to_dict()
            )
    standard_query = {"bool": standard_query_bool}

    # This does the vector search part using knn
    knn_filter = {"bool": {"filter": filter_clauses}} if filter_clauses else None
    knn_spec = {
        "field": match_dense.vector_column_name,
        "query_vector": list(match_dense.embedding_data),
        "k": match_dense.topn,
        "num_candidates": match_dense.topn * 2,
    }
    if knn_filter:
        knn_spec["filter"] = knn_filter

    use_linear = 0.0 < vector_similarity_weight < 1.0
    keyword_weight = 1.0 - vector_similarity_weight
    vector_weight = vector_similarity_weight

    if use_linear:
        retriever = {
            "linear": {
                "retrievers": [
                    {"retriever": {"standard": {"query": standard_query}}, "weight": keyword_weight},
                    {"retriever": {"knn": knn_spec}, "weight": vector_weight},
                ],
                "rank_window_size": rank_window_size,
            }
        }
    else:
        retriever = {
            "rrf": {
                "retrievers": [
                    {"retriever": {"standard": {"query": standard_query}}},
                    {"retriever": {"knn": knn_spec}},
                ],
                "rank_window_size": rank_window_size,
            }
        }

    # This specifies that we want to use the retriever defined above and then adds optional elements (min_score, etc...)
    body = {
        "retriever": retriever,
        "size": size,
        "from": offset,
        "track_total_hits": True,
        "_source": True,
    }
    if match_dense.extra_options.get("similarity") is not None:
        body["min_score"] = match_dense.extra_options["similarity"]

    if highlight_fields:
        body["highlight"] = {"fields": {f: {} for f in highlight_fields}}
    if agg_fields:
        body["aggs"] = {f"aggs_{fld}": {"terms": {"field": fld, "size": 1000000}} for fld in agg_fields}
    if order_by and order_by.fields:
        orders = []
        for field, order in order_by.fields:
            order_str = "asc" if order == 0 else "desc"
            if field in ["page_num_int", "top_int"]:
                orders.append({field: {"order": order_str, "unmapped_type": "float", "mode": "avg", "numeric_type": "double"}})
            elif field.endswith("_int") or field.endswith("_flt"):
                orders.append({field: {"order": order_str, "unmapped_type": "float"}})
            else:
                orders.append({field: {"order": order_str, "unmapped_type": "text"}})
        body["sort"] = orders

    return body
