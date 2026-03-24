from nudemo.explorer.app import build_explorer_html
from nudemo.storage.elasticsearch_store import ElasticsearchBackend


def test_build_explorer_html_uses_single_search_surface() -> None:
    html = build_explorer_html()

    assert 'label for="q">Search</label>' in html
    assert "Search scenes, scene tokens, sample tokens, locations, and annotation categories from one field." in html
    assert 'id="es_status"' not in html
    assert 'id="es-results-section"' not in html
    assert "Annotation search" not in html


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
