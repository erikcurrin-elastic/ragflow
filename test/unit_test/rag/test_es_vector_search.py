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
Unit tests for Elasticsearch vector search: retriever (hybrid) and vector-only paths.

Tests build_retriever_body (rag.utils.es_retriever_body) so the test does not load the ES connection. Covers:
- Option A: Hybrid search via ES retriever API (RRF / linear)
- Option B: Vector-only request shape when match_expressions contains only MatchDenseExpr (legacy kNN path).

Run from repo root or from ragflow:
  cd ragflow && uv run pytest test/unit_test/rag/test_es_vector_search.py -v
  # or with PYTHONPATH so imports resolve:
  cd ragflow && PYTHONPATH=. pytest test/unit_test/rag/test_es_vector_search.py -v
"""

import sys
from pathlib import Path

# Ensure ragflow root is on sys.path when run via system pytest or from repo root
# parents[3] = repo root when this file lives at test/unit_test/rag/test_es_vector_search.py
_ragflow_root = Path(__file__).resolve().parents[3]
if str(_ragflow_root) not in sys.path:
    sys.path.insert(0, str(_ragflow_root))

import pytest
from elasticsearch_dsl import Q, Search

from common.constants import PAGERANK_FLD, TAG_FLD
from common.doc_store.doc_store_base import MatchTextExpr, MatchDenseExpr, OrderByExpr
from rag.utils.es_retriever_body import build_retriever_body


def _make_match_text():
    return MatchTextExpr(
        fields=["content_ltks", "title_tks"],
        matching_text="test query",
        topn=100,
        extra_options={"minimum_should_match": 0.3},
    )


def _make_match_dense(vec_dim=768):
    return MatchDenseExpr(
        vector_column_name=f"q_{vec_dim}_vec",
        embedding_data=[0.1] * vec_dim,
        embedding_data_type="float",
        distance_type="cosine",
        topn=20,
        extra_options={"similarity": 0.2},
    )


def _make_bool_query_with_filter():
    """Bool query with only filter (e.g. kb_id) as used in search."""
    q = Q("bool", must=[])
    q.filter.append(Q("terms", kb_id=["kb1"]))
    return q


class TestBuildRetrieverBody:
    """Test the structure of the retriever body produced for hybrid search."""

    def test_body_has_retriever_key(self):
        body = build_retriever_body(
            match_text=_make_match_text(),
            match_dense=_make_match_dense(),
            bool_query=_make_bool_query_with_filter(),
            vector_similarity_weight=0.5,
            offset=0,
            limit=10,
            highlight_fields=["content_ltks"],
            order_by=OrderByExpr(),
            agg_fields=None,
        )
        assert "retriever" in body
        assert body["size"] == 10
        assert body["from"] == 0
        assert body["track_total_hits"] is True
        assert body["_source"] is True

    def test_linear_retriever_when_weight_between_0_and_1(self):
        vector_weight = 0.7
        body = build_retriever_body(
            match_text=_make_match_text(),
            match_dense=_make_match_dense(),
            bool_query=_make_bool_query_with_filter(),
            vector_similarity_weight=vector_weight,
            offset=0,
            limit=20,
            highlight_fields=[],
            order_by=OrderByExpr(),
            agg_fields=None,
        )
        retriever = body["retriever"]
        linear = retriever["linear"]
        assert len(linear["retrievers"]) == 2
        assert "standard" in linear["retrievers"][0]["retriever"]
        assert "knn" in linear["retrievers"][1]["retriever"]

    def test_rrf_retriever_when_weight_0_or_1(self):
        body = build_retriever_body(
            match_text=_make_match_text(),
            match_dense=_make_match_dense(),
            bool_query=_make_bool_query_with_filter(),
            vector_similarity_weight=1.0,
            offset=0,
            limit=10,
            highlight_fields=[],
            order_by=OrderByExpr(),
            agg_fields=None,
        )
        retriever = body["retriever"]
        assert "rrf" in retriever
        rrf = retriever["rrf"]
        assert "retrievers" in rrf
        assert len(rrf["retrievers"]) == 2
        assert "rank_window_size" in rrf

    def test_standard_sub_retriever_has_bool_query(self):
        match_text = _make_match_text()
        match_dense = _make_match_dense()
        bool_query = _make_bool_query_with_filter()
        body = build_retriever_body(
            match_text=match_text,
            match_dense=match_dense,
            bool_query=bool_query,
            vector_similarity_weight=0.5,
            offset=0,
            limit=10,
            highlight_fields=[],
            order_by=OrderByExpr(),
            agg_fields=None,
        )
        retriever = body["retriever"]
        linear_or_rrf = retriever["linear"] if "linear" in retriever else retriever["rrf"]
        first = linear_or_rrf["retrievers"][0]
        standard_spec = first.get("retriever", first)
        assert "standard" in standard_spec
        query = standard_spec["standard"]["query"]
        assert "bool" in query
        assert "must" in query["bool"]
        assert "filter" in query["bool"]

    def test_rank_feature_in_standard_query_when_provided(self):
        rank_feature = {PAGERANK_FLD: 10.0, "some_tag": 2.0}
        body = build_retriever_body(
            match_text=_make_match_text(),
            match_dense=_make_match_dense(),
            bool_query=_make_bool_query_with_filter(),
            vector_similarity_weight=0.5,
            offset=0,
            limit=10,
            highlight_fields=[],
            order_by=OrderByExpr(),
            agg_fields=None,
            rank_feature=rank_feature,
        )
        retriever = body["retriever"]
        linear_or_rrf = retriever["linear"] if "linear" in retriever else retriever["rrf"]
        first = linear_or_rrf["retrievers"][0]
        standard_spec = first.get("retriever", first)
        query = standard_spec["standard"]["query"]
        assert "should" in query["bool"]
        should = query["bool"]["should"]
        assert len(should) == 2

    def test_knn_sub_retriever_has_field_and_query_vector(self):
        body = build_retriever_body(
            match_text=_make_match_text(),
            match_dense=_make_match_dense(vec_dim=512),
            bool_query=_make_bool_query_with_filter(),
            vector_similarity_weight=0.5,
            offset=0,
            limit=10,
            highlight_fields=[],
            order_by=OrderByExpr(),
            agg_fields=None,
        )
        retriever = body["retriever"]
        linear_or_rrf = retriever["linear"] if "linear" in retriever else retriever["rrf"]
        second = linear_or_rrf["retrievers"][1]
        knn_spec = second.get("retriever", second)
        assert "knn" in knn_spec
        knn = knn_spec["knn"]
        assert knn["field"] == "q_512_vec"
        assert "filter" in knn

    def test_min_score_when_similarity_in_extra_options(self):
        match_dense = _make_match_dense()
        match_dense.extra_options["similarity"] = 0.15
        body = build_retriever_body(
            match_text=_make_match_text(),
            match_dense=match_dense,
            bool_query=_make_bool_query_with_filter(),
            vector_similarity_weight=0.5,
            offset=0,
            limit=10,
            highlight_fields=[],
            order_by=OrderByExpr(),
            agg_fields=None,
        )
        assert body["min_score"] == 0.15

    def test_highlight_and_aggs_in_body(self):
        body = build_retriever_body(
            match_text=_make_match_text(),
            match_dense=_make_match_dense(),
            bool_query=_make_bool_query_with_filter(),
            vector_similarity_weight=0.5,
            offset=0,
            limit=10,
            highlight_fields=["content_ltks", "title_tks"],
            order_by=OrderByExpr(),
            agg_fields=["docnm_kwd"],
        )
        assert "highlight" in body
        assert "fields" in body["highlight"]
        assert "aggs" in body
        assert "aggs_docnm_kwd" in body["aggs"]


class TestVectorOnlyLegacyPath:
    """
    Vector-only case: match_expressions = [MatchDenseExpr] only (no text, no fusion).
    Exercises the same request shape built by ESConnection.search() Option B / legacy path.
    Uses elasticsearch_dsl Search + kNN only (no retriever API, no query_string).
    """

    def test_vector_only_request_has_knn_and_no_text_query(self):
        # Same pattern as es_conn.search() when match_expressions contains only MatchDenseExpr
        bool_query = _make_bool_query_with_filter()
        match_dense = _make_match_dense(vec_dim=768)
        similarity = match_dense.extra_options.get("similarity", 0.0)

        s = Search()
        s = s.knn(
            match_dense.vector_column_name,
            match_dense.topn,
            match_dense.topn * 2,
            query_vector=list(match_dense.embedding_data),
            filter=bool_query.to_dict(),
            similarity=similarity,
        )
        s = s.query(bool_query)
        s = s[0:10]
        body = s.to_dict()

        assert "knn" in body
        knn = body["knn"]
        assert knn["field"] == "q_768_vec"
        assert len(knn["query_vector"]) >= 64  # reasonable min embedding dimension
        assert knn["k"] >= 1
        assert knn["num_candidates"] >= knn["k"]  # num_candidates must be at least k
        assert "filter" in knn
        assert knn["filter"]["bool"]["filter"]

        assert "query" in body
        query_bool = body["query"]["bool"]
        assert "filter" in query_bool
        assert query_bool.get("must", []) == []  # vector-only: no text query
