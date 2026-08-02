import json
import shutil
import subprocess
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


CHAT_JS = Path(__file__).parent / "static" / "js" / "chat.js"


def extract_js_function(source: str, name: str) -> str:
    marker = f"function {name}"
    start = source.index(marker)
    signature_end = source.index(")", start)
    brace_start = source.index("{", signature_end)
    depth = 0
    for index in range(brace_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"Could not extract {name}")


def _node_required(func):
    if pytest is not None:
        return pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")(func)
    return func


@_node_required
def test_answer_action_policy_by_intent():
    if shutil.which("node") is None:
        return
    source = CHAT_JS.read_text(encoding="utf-8")
    function_source = "\n".join([
        extract_js_function(source, "isDocumentTaskPrompt"),
        extract_js_function(source, "getAnswerActionPolicy"),
    ])
    cases = [
        {"intent": "identity_help"},
        {"intent": "clarify"},
        {"intent": "form_template_export"},
        {"intent": "spreadsheet_transform"},
        {"intent": "knowledge_qa"},
        {"intent": "doc_drafting"},
        {"intent": "doc_formatting"},
        {"intent": "knowledge_qa", "hasDocumentContent": True},
        {"intent": "knowledge_qa", "originalPrompt": "帮我写一份会议通知"},
        {"intent": "knowledge_qa", "originalPrompt": "公文格式要求有哪些？"},
    ]
    script = (
        function_source
        + "\nconst cases = "
        + json.dumps(cases)
        + ";\nconsole.log(JSON.stringify(cases.map(getAnswerActionPolicy)));"
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    policies = json.loads(result.stdout)

    assert policies[0] == {"showEditorActions": False, "showEvidenceAction": False}
    assert policies[1] == {"showEditorActions": False, "showEvidenceAction": False}
    assert policies[2] == {"showEditorActions": False, "showEvidenceAction": False}
    assert policies[3] == {"showEditorActions": False, "showEvidenceAction": False}
    assert policies[4] == {"showEditorActions": False, "showEvidenceAction": True}
    assert policies[5] == {"showEditorActions": True, "showEvidenceAction": True}
    assert policies[6] == {"showEditorActions": True, "showEvidenceAction": True}
    assert policies[7] == {"showEditorActions": True, "showEvidenceAction": True}
    assert policies[8] == {"showEditorActions": True, "showEvidenceAction": True}
    assert policies[9] == {"showEditorActions": False, "showEvidenceAction": True}


@_node_required
def test_sse_chunk_parser_handles_partial_and_invalid_events():
    if shutil.which("node") is None:
        return
    source = CHAT_JS.read_text(encoding="utf-8")
    function_source = extract_js_function(source, "parseSseStreamChunk")
    script = (
        function_source
        + "\nconst first = parseSseStreamChunk('', 'data: {\"type\":\"start\"}\\n\\ndata: {\"type\"');"
        + "\nconst second = parseSseStreamChunk(first.buffer, ':\"done\",\"answer\":\"ok\"}\\n\\ndata: nope\\n\\n');"
        + "\nconsole.log(JSON.stringify({first, second}));"
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    parsed = json.loads(result.stdout)

    assert parsed["first"]["events"] == [{"type": "start"}]
    assert parsed["first"]["buffer"] == 'data: {"type"'
    assert parsed["second"]["events"] == [{"type": "done", "answer": "ok"}]
    assert parsed["second"]["errors"][0]["line"] == "data: nope"


@_node_required
def test_done_payload_resolution_keeps_document_and_qa_boundaries():
    if shutil.which("node") is None:
        return
    source = CHAT_JS.read_text(encoding="utf-8")
    function_source = extract_js_function(source, "resolveChatDonePayload")
    cases = [
        [{"intent": "knowledge_qa", "answer": "问答"}, ""],
        [{"intent": "doc_drafting", "answer": "正文", "export_template": "default"}, ""],
        [{"intent": "knowledge_qa", "document": "正文"}, "流式正文"],
    ]
    script = (
        function_source
        + "\nconst cases = "
        + json.dumps(cases)
        + ";\nconsole.log(JSON.stringify(cases.map(item => resolveChatDonePayload(item[0], item[1]))));"
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    resolved = json.loads(result.stdout)

    assert resolved[0]["isDocumentIntent"] is False
    assert resolved[0]["finalDoc"] == "问答"
    assert resolved[0]["documentTemplate"] == "auto"
    assert resolved[1]["isDocumentIntent"] is True
    assert resolved[1]["finalDoc"] == "正文"
    assert resolved[1]["documentTemplate"] == "default"
    assert resolved[2]["isDocumentIntent"] is True
    assert resolved[2]["finalDoc"] == "正文"


@_node_required
def test_editor_template_preview_tracks_manual_and_auto_selection():
    if shutil.which("node") is None:
        return
    source = CHAT_JS.read_text(encoding="utf-8")
    function_source = "\n".join([
        extract_js_function(source, "normalizeDocumentTemplate"),
        extract_js_function(source, "detectEditorTemplate"),
        extract_js_function(source, "escapeHtml"),
        extract_js_function(source, "renderEditorDocument"),
    ])
    script = (
        "global.document = { createElement() { return { innerHTML: '', "
        "set textContent(value) { this.innerHTML = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); } }; } };\n"
        + function_source
        + "\nconst proposal = '关于审议测试事项的议案\\n一、背景';"
        + "\nconst normal = '普通通知\\n一、背景';"
        + "\nconsole.log(JSON.stringify({"
        + "autoProposal: detectEditorTemplate(proposal, 'auto'),"
        + "manualDefault: detectEditorTemplate(proposal, 'default'),"
        + "unknown: normalizeDocumentTemplate('missing'),"
        + "reviewHtml: renderEditorDocument(proposal, 'review_proposal'),"
        + "normalHtml: renderEditorDocument(normal, 'default')"
        + "}));"
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    parsed = json.loads(result.stdout)

    assert parsed["autoProposal"] == "review_proposal"
    assert parsed["manualDefault"] == "default"
    assert parsed["unknown"] == "auto"
    assert "document-review-title" in parsed["reviewHtml"]
    assert "document-review-title" not in parsed["normalHtml"]


if __name__ == "__main__":
    test_answer_action_policy_by_intent()
