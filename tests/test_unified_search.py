import json

from nudemo.explorer.app import _looks_like_structured_search_text, build_explorer_html
from nudemo.storage.elasticsearch_store import ElasticsearchBackend


def test_build_explorer_html_uses_single_search_surface() -> None:
    html = build_explorer_html()

    assert 'label for="q">Search</label>' in html
    assert "Search one field for scene names, token prefixes, locations, categories, and simple natural-language concepts." in html
    assert "Try: <code>car</code>, <code>truck</code>, <code>human</code>, <code>tree</code>, <code>duck</code>" in html
    assert "Mining workspace" not in html
    assert "Positive examples" not in html
    assert "Negative examples" not in html
    assert "Saved cohorts" not in html
    assert "Matching tracks" not in html
    assert "/api/search?" in html
    assert "/api/mining/search" not in html
    assert 'id="es_status"' not in html
    assert 'id="es-results-section"' not in html
    assert "Annotation search" not in html


def test_structured_search_text_classifier_prefers_location_and_identifier_style_queries() -> None:
    assert _looks_like_structured_search_text("scene-0001")
    assert _looks_like_structured_search_text("73030fb67d3c")
    assert _looks_like_structured_search_text("vehicle.car")
    assert not _looks_like_structured_search_text("car")
    assert not _looks_like_structured_search_text("duck")


def test_elasticsearch_search_supports_scene_and_token_lookups(monkeypatch) -> None:
    backend = ElasticsearchBackend(url="http://example.invalid")
    captured: dict[str, object] = {}

    def fake_req(self, method: str, path: str, body=None, *, ndjson: bool = False):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = body
        return {
            "hits": {"total": {"value": 0}, "hits": []},
            "aggregations": {
                "top_categories": {"cats": {"buckets": []}},
                "locations": {"buckets": []},
                "scenes": {"buckets": []},
            },
        }

    monkeypatch.setattr(ElasticsearchBackend, "_req", fake_req)

    backend.search(
        q="73030FB67D3C",
        scene_token="73030fb67d3c46cfb5e590168088ae39",
        location="singapore-onenorth",
        category="vehicle.car",
        min_annotations=4,
        size=12,
        from_=24,
    )

    body = captured["body"]
    assert isinstance(body, dict)

    filters = body["query"]["bool"]["filter"]
    assert {"term": {"scene_token": "73030fb67d3c46cfb5e590168088ae39"}} in filters
    assert {"term": {"location": "singapore-onenorth"}} in filters
    assert {"range": {"num_annotations": {"gte": 4}}} in filters
    assert {
        "nested": {
            "path": "annotations",
            "query": {"term": {"annotations.category": "vehicle.car"}},
        }
    } in filters

    should = body["query"]["bool"]["must"][0]["bool"]["should"]
    assert {"term": {"scene_token": {"value": "73030FB67D3C", "boost": 12.0}}} in should
    assert {"term": {"token": {"value": "73030FB67D3C", "boost": 12.0}}} in should
    assert {
        "wildcard": {
            "scene_token": {
                "value": "73030FB67D3C*",
                "case_insensitive": True,
                "boost": 6.0,
            }
        }
    } in should
    assert {
        "wildcard": {
            "token": {
                "value": "73030FB67D3C*",
                "case_insensitive": True,
                "boost": 6.0,
            }
        }
    } in should


def test_elasticsearch_search_avoids_annotation_substring_wildcards_for_short_terms(monkeypatch) -> None:
    backend = ElasticsearchBackend(url="http://example.invalid")
    captured: dict[str, object] = {}

    def fake_req(self, method: str, path: str, body=None, *, ndjson: bool = False):
        captured["body"] = body
        return {
            "hits": {"total": {"value": 0}, "hits": []},
            "aggregations": {
                "top_categories": {"cats": {"buckets": []}},
                "locations": {"buckets": []},
                "scenes": {"buckets": []},
            },
        }

    monkeypatch.setattr(ElasticsearchBackend, "_req", fake_req)

    backend.search(q="man", size=8, from_=0)

    body = captured["body"]
    assert isinstance(body, dict)
    nested_query = body["query"]["bool"]["must"][0]["bool"]["should"][-1]["nested"]["query"]["bool"]["should"]
    assert not any(
        "wildcard" in clause and "annotations.category" in clause["wildcard"]
        for clause in nested_query
    )
    assert any("match" in clause for clause in nested_query)


def test_elasticsearch_search_adds_human_aliases_without_group_wildcard(monkeypatch) -> None:
    backend = ElasticsearchBackend(url="http://example.invalid")
    captured: dict[str, object] = {}

    def fake_req(self, method: str, path: str, body=None, *, ndjson: bool = False):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = body
        return {
            "hits": {"total": {"value": 0}, "hits": []},
            "aggregations": {
                "top_categories": {"cats": {"buckets": []}},
                "locations": {"buckets": []},
                "scenes": {"buckets": []},
            },
        }

    monkeypatch.setattr(ElasticsearchBackend, "_req", fake_req)

    backend.search(q="a man", size=12)

    body = captured["body"]
    assert isinstance(body, dict)
    serialized = json.dumps(body)
    assert '"annotations.category_group"' not in serialized or '"*a man*"' not in serialized
    assert "human.pedestrian.adult" in serialized
