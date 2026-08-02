import json
import os
import urllib.request

import pytest

from chat_architecture import INTENT_KNOWLEDGE_QA, IntentRouter
from agents.knowledge_agent import KnowledgeAgent
from test_chat_architecture import reimbursement_detector


FORBIDDEN_SOURCE_MARKERS = ("chunk", "片段", "Sheet", "行号", "source_path")

RAG_CENTER_FILE_CASES = [
    {
        "filename": "示例单位存储服务器运营方案-2025.4.28.docx",
        "questions": [
            ("NAS服务器如何使用？", ["192.0.2.10"]),
            ("存储服务器账号和访问地址是什么？", ["192.0.2.10"]),
            ("如何使用网盘呢", ["192.0.2.10"]),
            ("网盘地址是什么", ["192.0.2.10"]),
            ("nas服务器里面应该有地址吧", ["192.0.2.10"]),
        ],
    },
    {
        "filename": "差旅费.xlsx",
        "questions": [
            ("差旅费报销需要填哪些信息？", ["差旅"]),
            ("差旅费报销表里有哪些字段？", ["报销"]),
        ],
    },
    {
        "filename": "会议费.xlsx",
        "questions": [
            ("会议费报销需要准备什么材料？", ["会议"]),
            ("会议费报销表需要填写哪些内容？", ["会议"]),
        ],
    },
    {
        "filename": "劳务费&专家咨询费.xlsx",
        "questions": [
            ("劳务费报销有什么要求？", ["劳务"]),
            ("专家咨询费报销需要哪些信息？", ["专家"]),
        ],
    },
    {
        "filename": "其他费用报销.xlsx",
        "questions": [
            ("其他费用报销表怎么填？", ["费用"]),
            ("其他费用报销需要提供什么信息？", ["费用"]),
        ],
    },
    {
        "filename": "示例单位场地管理办法.docx",
        "questions": [
            ("场地管理办法主要规定了什么？", ["场地"]),
            ("使用场地需要遵守哪些管理要求？", ["场地"]),
        ],
    },
    {
        "filename": "附件：场地使用收费表.xlsx",
        "questions": [
            ("场地使用收费标准在哪里看？", ["场地"]),
            ("场地收费表包含哪些场地收费信息？", ["收费"]),
        ],
    },
    {
        "filename": "示例单位办公场地出租出借管理办法.docx",
        "questions": [
            ("办公场地出租出借怎么申请？", ["出租", "出借"]),
            ("办公场地出租出借管理办法有哪些要求？", ["出租", "出借"]),
        ],
    },
]


@pytest.mark.parametrize("case", RAG_CENTER_FILE_CASES)
def test_center_file_questions_route_to_knowledge_qa(case):
    router = IntentRouter(reimbursement_detector)
    for question, _keywords in case["questions"]:
        result = router.route(message=question, display_message=question, mode="quick")
        assert result.intent == INTENT_KNOWLEDGE_QA, question
        assert result.actions == []


def test_fee_standard_lookup_prefers_spreadsheet_metadata():
    question = "场地使用收费标准在哪里看？"
    spreadsheet_score = KnowledgeAgent._metadata_match_score(question, {
        "source_type": "spreadsheet",
        "filename": "附件：场地使用收费表.xlsx",
        "sheet_name": "场地收费表",
        "column_headers": ["计费方式", "金额", "备注"],
        "text": "",
    })
    document_score = KnowledgeAgent._metadata_match_score(question, {
        "source_type": "document",
        "filename": "示例单位场地管理办法.docx",
        "heading_path": ["第四章 场地收费标准与财务管理"],
        "section_title": "第四章 场地收费标准与财务管理",
        "text": "",
    })

    assert spreadsheet_score > document_score


def _read_sse_chat(base_url: str, token: str, question: str):
    body = json.dumps({"message": question, "mode": "quick"}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    events = []
    with urllib.request.urlopen(request, timeout=90) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line[6:]))
    return events


@pytest.mark.skipif(os.getenv("RUN_RAG_EVAL") != "1", reason="set RUN_RAG_EVAL=1 to call a running app")
@pytest.mark.parametrize("case", RAG_CENTER_FILE_CASES)
def test_center_file_rag_answer_contract(case):
    base_url = os.getenv("RAG_EVAL_BASE_URL", "http://127.0.0.1:5000")
    token = os.getenv("RAG_EVAL_TOKEN")
    if not token:
        pytest.skip("RAG_EVAL_TOKEN is required for authenticated /api/chat")

    for question, keywords in case["questions"]:
        events = _read_sse_chat(base_url, token, question)
        done = next(event for event in reversed(events) if event.get("type") == "done")
        answer = done.get("answer") or done.get("document") or ""
        sources = done.get("source_details") or []
        filenames = [source.get("filename") for source in sources]

        assert done["intent"] == INTENT_KNOWLEDGE_QA
        assert done.get("document", "") == ""
        assert done.get("export_template", "") == ""
        assert done.get("export_spreadsheet_template", "") == ""
        assert case["filename"] in filenames
        assert any(keyword in answer for keyword in keywords), question
        assert not any(marker in answer for marker in FORBIDDEN_SOURCE_MARKERS)
        assert all(set(source.keys()) == {"filename"} for source in sources)
