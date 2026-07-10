"""Manual FEA run from the web UI: POST /api/run needs no LLM."""

import pytest
from fastapi.testclient import TestClient

from assistant.app import main, wiki_logic


@pytest.fixture()
def tmp_wiki(monkeypatch, tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    index = wiki / "index.md"
    index.write_text("# Wiki Index\n", encoding="utf-8")
    monkeypatch.setattr(wiki_logic, "WIKI_DIR", wiki)
    monkeypatch.setattr(wiki_logic, "INDEX_FILE", index)
    return wiki


def test_api_run_executes_and_ingests(tmp_wiki):
    with TestClient(main.app) as tc:
        resp = tc.post("/api/run", json={
            "speed": {"start_rad_s": 10, "stop_rad_s": 800, "step_rad_s": 20},
        })
        assert resp.status_code == 200
        result = resp.json()
        assert result["critical_speeds_rad_s"]
        assert 100 < result["critical_speeds_rad_s"][0] < 260
        page_id = result["report_slug"].rsplit("/", 1)[-1]
        assert (tmp_wiki / "runs" / f"{page_id}.md").exists()

        # the page the UI opens after a run is served
        assert tc.get(f"/api/pages/runs/{page_id}").status_code == 200


def test_api_run_rejects_bad_params(tmp_wiki):
    with TestClient(main.app) as tc:
        assert tc.post("/api/run", json={"shaft": {"diameter_m": -1}}).status_code == 422
        assert tc.post("/api/run", json={"bearing": {"kind": "magnetic"}}).status_code == 422
