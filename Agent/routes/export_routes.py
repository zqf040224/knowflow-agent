from flask import request

from auth import login_required


def register_export_routes(app, context):
    @app.route("/api/export_docx", methods=["POST"])
    @login_required
    def export_docx():
        """导出 Word 文档"""
        return context.send_export_result(context.export_service().export_docx(request.json or {}))

    @app.route("/api/export_xlsx", methods=["POST"])
    @login_required
    def export_xlsx():
        """导出 Excel 表格。优先识别 Markdown 表格，否则按文本行导出。"""
        return context.send_export_result(context.export_service().export_xlsx(request.json or {}))

    @app.route("/api/export_reimbursement_xlsx", methods=["POST"])
    @login_required
    def export_reimbursement_xlsx():
        """按公共资料中的报销模板直接导出 Excel 原表。"""
        return context.send_export_result(context.export_service().export_reimbursement_xlsx(request.json or {}))
