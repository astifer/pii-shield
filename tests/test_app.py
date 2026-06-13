"""API tests for the FastAPI backend (pii_ner.web.server) + label-tag coverage.

Run with:  uv run pytest tests/
"""

from fastapi.testclient import TestClient

from pii_ner.export_onnx import write_labels_js
from pii_ner.labels import ENTITY_LABELS
from pii_ner.web.server import app

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/static/app.js" in r.text
    assert "htmx" in r.text


def test_chat_echoes_obfuscated_message():
    msg = "Меня зовут [ФИО], адрес [ПОЛНЫЙ_АДРЕС]"
    r = client.post("/chat", data={"message": msg})
    assert r.status_code == 200
    assert msg in r.text
    assert 'class="msg assistant"' in r.text


def test_chat_requires_message_field():
    r = client.post("/chat", data={})
    assert r.status_code == 422


def test_chat_escapes_html_in_reply():
    r = client.post("/chat", data={"message": "<script>alert(1)</script>"})
    assert r.status_code == 200
    assert "<script>alert(1)</script>" not in r.text
    assert "&lt;script&gt;" in r.text


def test_static_module_files_served():
    for path in ("/static/app.js", "/static/obfuscate.js", "/static/styles.css"):
        r = client.get(path)
        assert r.status_code == 200, path


def test_label_tags_cover_every_entity(tmp_path):
    """The generated frontend tag map gives every trained PII label a bracketed tag."""
    tags = write_labels_js(tmp_path / "labels.generated.js")
    assert set(tags) == set(ENTITY_LABELS)
    assert all(v.startswith("[") and v.endswith("]") for v in tags.values())
