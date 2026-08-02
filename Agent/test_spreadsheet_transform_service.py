from spreadsheet_transform_service import (
    XLSX_MIMETYPE,
    SpreadsheetTransformDependencies,
    SpreadsheetTransformService,
)


class FakeUploadManager:
    def __init__(self):
        self.info = {
            "file_1": {
                "file_path": "/tmp/table.csv",
                "filename": "表格.csv",
            },
            "doc_1": {
                "file_path": "/tmp/doc.txt",
                "filename": "文档.txt",
            },
        }

    def get_temp_file_info(self, file_id, user_id):
        return self.info.get(file_id)


def build_service(
    *,
    upload_manager=None,
    is_spreadsheet_file=lambda value: str(value).endswith((".csv", ".xlsx", ".xls")),
    transform_spreadsheet_file=None,
    deepseek_api_key="",
    llm_client_factory=None,
):
    calls = []

    def default_transform(file_path, filename, instruction, client=None, model=""):
        calls.append((file_path, filename, instruction, client, model))
        return {
            "success": True,
            "filename": "处理结果.xlsx",
            "content": b"xlsx",
            "summary": {"output_count": 2},
        }

    service = SpreadsheetTransformService(SpreadsheetTransformDependencies(
        upload_manager=upload_manager or FakeUploadManager(),
        is_spreadsheet_file=is_spreadsheet_file,
        transform_spreadsheet_file=transform_spreadsheet_file or default_transform,
        deepseek_api_key=deepseek_api_key,
        llm_client_factory=llm_client_factory,
    ))
    return service, calls


def test_spreadsheet_transform_service_validates_request():
    service, _ = build_service()

    missing_file = service.transform({"instruction": "排序"}, user_id="user_1")
    missing_instruction = service.transform({"file_id": "file_1"}, user_id="user_1")
    missing_temp = service.transform({"file_id": "missing", "instruction": "排序"}, user_id="user_1")
    non_spreadsheet = service.transform({"file_id": "doc_1", "instruction": "排序"}, user_id="user_1")

    assert missing_file.status == 400
    assert missing_file.error == {"success": False, "message": "缺少表格文件"}
    assert missing_instruction.status == 400
    assert missing_instruction.error == {"success": False, "message": "请填写筛选或排序规则"}
    assert missing_temp.status == 404
    assert missing_temp.error == {"success": False, "message": "文件不存在或已过期，请重新上传"}
    assert non_spreadsheet.status == 400
    assert non_spreadsheet.error == {"success": False, "message": "当前文件不是 Excel/CSV 表格"}


def test_spreadsheet_transform_service_success_contract():
    service, calls = build_service()

    result = service.transform({"file_id": "file_1", "instruction": "按预算排序"}, user_id="user_1")

    assert result.success
    assert result.filename == "处理结果.xlsx"
    assert result.payload == b"xlsx"
    assert result.summary == {"output_count": 2}
    assert result.mimetype == XLSX_MIMETYPE
    assert calls == [("/tmp/table.csv", "表格.csv", "按预算排序", None, "deepseek-v4-flash")]


def test_spreadsheet_transform_service_returns_transformer_failure_as_422():
    def fail_transform(*args, **kwargs):
        return {"success": False, "message": "没有识别到明确规则"}

    service, _ = build_service(transform_spreadsheet_file=fail_transform)

    result = service.transform({"file_id": "file_1", "instruction": "帮我处理"}, user_id="user_1")

    assert not result.success
    assert result.status == 422
    assert result.error == {"success": False, "message": "没有识别到明确规则"}


def test_spreadsheet_transform_service_records_transformer_exception():
    def exploding_transform(*args, **kwargs):
        raise RuntimeError("boom")

    service, _ = build_service(transform_spreadsheet_file=exploding_transform)

    result = service.transform({"file_id": "file_1", "instruction": "排序"}, user_id="user_1")

    assert not result.success
    assert result.status == 500
    assert result.error == {"success": False, "message": "表格处理失败: boom"}


def test_spreadsheet_transform_service_builds_optional_llm_client():
    clients = []

    def factory(**kwargs):
        clients.append(kwargs)
        return "client"

    service, calls = build_service(deepseek_api_key="key", llm_client_factory=factory)

    result = service.transform({"file_id": "file_1", "instruction": "排序"}, user_id="user_1")

    assert result.success
    assert clients == [{
        "api_key": "key",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 120,
    }]
    assert calls[0][3] == "client"
