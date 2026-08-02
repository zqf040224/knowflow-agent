from agents.writer_agent import WriterAgent


def test_writer_prompt_prevents_unsupported_specific_facts():
    writer = WriterAgent.__new__(WriterAgent)

    prompt = writer._build_prompt(
        user_request="帮我写一份培训通知，时间9:30，地点会议室",
        search_context="",
        knowledge_context="",
        document_type="通知",
        key_points=[],
    )

    assert "不得编造用户未提供、知识库未明确支持的具体事实" in prompt
    assert "具体门牌号" in prompt
    assert "调休/加班政策" in prompt
    assert "审查建议中的“如……”示例只能作为方向" in prompt


def test_writer_revision_prompt_keeps_examples_from_becoming_facts():
    writer = WriterAgent.__new__(WriterAgent)

    prompt = writer._build_revision_prompt(
        user_request="帮我写一份培训通知",
        draft_document="关于召开培训会的通知",
        document_type="通知",
        task_type="公文生成",
        revision_history=[{
            "round": 1,
            "suggestions": ["注明地点细节：如示例单位A栋3楼301会议室。"],
        }],
        evidence_items=[],
        user_constraints=[],
    )

    assert "修订意见中的示例、建议或假设不得直接写成确定事实" in prompt
    assert "不得新增用户未提供、证据未支持的具体门牌号" in prompt
