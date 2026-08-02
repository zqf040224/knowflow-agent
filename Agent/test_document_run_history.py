from agents.document_run_history import reflection_history_entry, review_history_entry


def test_review_history_entry_keeps_audit_and_issue_fields():
    entry = review_history_entry({
        "needs_revision": True,
        "revision_focus": ["结构"],
        "suggestions": ["补标题"],
        "format_check": {"issues": ["格式"]},
        "content_check": {"issues": ["内容"]},
        "logic_check": {"issues": ["逻辑"]},
        "language_check": {"issues": ["语言"]},
        "fact_check": {"issues": ["事实"]},
        "spreadsheet_audit": {"ok": True},
        "confidence": 0.61,
    }, 1)

    assert entry == {
        "round": 2,
        "needs_revision": True,
        "revision_focus": ["结构"],
        "suggestions": ["补标题"],
        "format_issues": ["格式"],
        "content_issues": ["内容"],
        "logic_issues": ["逻辑"],
        "language_issues": ["语言"],
        "fact_issues": ["事实"],
        "spreadsheet_audit": {"ok": True},
        "confidence": 0.61,
    }


def test_reflection_history_entry_marks_source():
    entry = reflection_history_entry({
        "needs_revision": True,
        "revision_suggestions": ["加依据"],
        "weaknesses": ["依据不足"],
        "counter_arguments": ["缺反例"],
        "logic_score": 0.7,
    }, 0)

    assert entry == {
        "round": 1,
        "needs_revision": True,
        "revision_focus": ["加依据"],
        "suggestions": ["加依据"],
        "format_issues": [],
        "content_issues": ["依据不足"],
        "logic_issues": ["缺反例"],
        "language_issues": [],
        "confidence": 0.7,
        "source": "reflection",
    }
