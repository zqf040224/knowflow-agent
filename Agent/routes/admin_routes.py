import logging
from io import BytesIO

from flask import g, jsonify, request, send_file

from auth import admin_required, login_required


logger = logging.getLogger(__name__)


def register_admin_routes(app, context):
    def _json_payload():
        return request.get_json(silent=True) or {}

    @app.route("/api/admin/users", methods=["GET", "POST"])
    @login_required
    @admin_required
    def admin_users():
        """用户管理"""
        return jsonify(context.account_admin_service.users(
            g.user_id,
            _json_payload(),
            method=request.method,
        ))

    @app.route("/api/admin/departments", methods=["GET"])
    @login_required
    @admin_required
    def admin_departments():
        """获取可分配给账号的部门白名单"""
        return jsonify(context.account_admin_service.departments_payload())

    @app.route("/api/admin/users/<user_id>/status", methods=["PUT"])
    @login_required
    @admin_required
    def admin_toggle_user(user_id):
        """启用/禁用用户"""
        return jsonify(context.account_admin_service.toggle_user_status(g.user_id, user_id, _json_payload()))

    @app.route("/api/admin/users/<user_id>/reset-password", methods=["POST"])
    @login_required
    @admin_required
    def admin_reset_password(user_id):
        """重置密码"""
        return jsonify(context.account_admin_service.reset_password(g.user_id, user_id, _json_payload()))

    @app.route("/api/admin/logs", methods=["GET"])
    @login_required
    @admin_required
    def admin_logs():
        """获取登录日志"""
        return jsonify(context.account_admin_service.login_logs(g.user_id))

    @app.route("/api/admin/stats", methods=["GET"])
    @login_required
    @admin_required
    def admin_stats():
        """获取系统统计"""
        return jsonify(context.account_admin_service.stats(g.user_id))

    @app.route("/api/feedback", methods=["POST"])
    @login_required
    def submit_feedback():
        """Collect internal-beta user feedback from the workbench."""
        result = context.record_beta_feedback(_json_payload())
        status = 200 if result.get("success") else 400
        return jsonify(result), status

    @app.route("/api/admin/beta-data", methods=["GET"])
    @login_required
    @admin_required
    def admin_beta_data():
        """Internal beta usage and feedback dashboard."""
        return jsonify(context.get_beta_dashboard(request.args.get("limit", 80)))

    @app.route("/api/admin/beta-feedback/<int:feedback_id>/status", methods=["PUT"])
    @login_required
    @admin_required
    def admin_update_beta_feedback_status(feedback_id):
        """Update internal beta feedback triage status."""
        result = context.update_beta_feedback_status(feedback_id, _json_payload())
        status_code = 200 if result.get("success") else 400
        return jsonify(result), status_code

    @app.route("/api/admin/token-usage", methods=["GET"])
    @login_required
    @admin_required
    def admin_token_usage():
        """Estimated LLM token usage dashboard for internal beta monitoring."""
        return jsonify(context.get_token_usage_dashboard(request.args.get("limit", 120)))

    @app.route("/api/admin/knowledge-health", methods=["GET"])
    @login_required
    @admin_required
    def admin_knowledge_health():
        """获取知识库索引和权限过滤健康状态"""
        return jsonify(context.runtime_query_service().knowledge_health(context.get_user_info()))

    @app.route("/api/admin/storage-health", methods=["GET"])
    @login_required
    @admin_required
    def admin_storage_health():
        """检查本地/NAS 存储目录是否可读写。"""
        return jsonify(context.runtime_query_service().storage_health_payload())

    @app.route("/api/admin/vector-map", methods=["GET"])
    @login_required
    @admin_required
    def admin_vector_map():
        """返回知识库向量的 3D PCA 可视化坐标。"""
        payload, status = context.runtime_query_service().vector_map(limit=request.args.get("limit", 1200))
        if not payload.get("success"):
            logger.warning("向量地图生成失败: %s", payload.get("message", ""))
        return jsonify(payload), status

    @app.route("/api/admin/spreadsheets", methods=["GET"])
    @login_required
    @admin_required
    def admin_spreadsheets():
        """管理员查看已结构化入库的表格文件。"""
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
        return jsonify(context.knowledge_admin_read_service().spreadsheets(limit=limit))

    @app.route("/api/admin/spreadsheets/<content_hash>/rows", methods=["GET"])
    @login_required
    @admin_required
    def admin_spreadsheet_rows(content_hash):
        """管理员查看某个表格文件的结构化行。"""
        sheet_name = request.args.get("sheet_name") or None
        row_start = request.args.get("row_start", type=int)
        row_end = request.args.get("row_end", type=int)
        return jsonify(context.knowledge_admin_read_service().spreadsheet_rows(
            content_hash,
            sheet_name=sheet_name,
            row_start=row_start,
            row_end=row_end,
        ))

    @app.route("/api/admin/spreadsheets/<content_hash>/export", methods=["GET"])
    @login_required
    @admin_required
    def admin_export_spreadsheet(content_hash):
        """管理员导出某个结构化表格文件的全部入库行。"""
        result = context.knowledge_admin_read_service().export_spreadsheet(content_hash)
        if not result.success:
            return jsonify({"success": False, "message": result.error}), result.status
        return send_file(
            BytesIO(result.payload or b""),
            mimetype=result.mimetype,
            as_attachment=True,
            download_name=result.filename,
        )

    @app.route("/api/admin/knowledge-files", methods=["GET"])
    @login_required
    @admin_required
    def admin_knowledge_files():
        """管理员查看知识库文件级状态。"""
        limit = min(max(int(request.args.get("limit", 200)), 1), 1000)
        return jsonify(context.knowledge_admin_read_service().knowledge_files(limit=limit, filters={
            "q": request.args.get("q", ""),
            "status": request.args.get("status", ""),
            "source_type": request.args.get("source_type", ""),
            "access_level": request.args.get("access_level", ""),
        }))

    @app.route("/api/admin/knowledge-audit", methods=["GET"])
    @login_required
    @admin_required
    def admin_knowledge_audit():
        """管理员查看知识库管理操作审计。"""
        limit = min(max(int(request.args.get("limit", 80)), 1), 300)
        content_hash = request.args.get("content_hash") or None
        return jsonify(context.knowledge_admin_read_service().knowledge_audit(limit=limit, content_hash=content_hash))

    @app.route("/api/admin/knowledge-files/<content_hash>", methods=["DELETE"])
    @login_required
    @admin_required
    def admin_delete_knowledge_file(content_hash):
        """软删除知识库文件：移出检索、表格库，并归档原文件。"""
        payload, status = context.knowledge_admin_write_service().delete_knowledge_file(
            content_hash,
            _json_payload(),
        )
        return jsonify(payload), status

    @app.route("/api/admin/knowledge-files/<content_hash>/metadata", methods=["PUT"])
    @login_required
    @admin_required
    def admin_update_knowledge_file_metadata(content_hash):
        """同步更新知识库文件的分类、部门和访问级别。"""
        payload, status = context.knowledge_admin_write_service().update_knowledge_file_metadata(
            content_hash,
            request.get_json(silent=True) or {},
        )
        return jsonify(payload), status

    @app.route("/api/admin/knowledge-files/<content_hash>/reindex", methods=["POST"])
    @login_required
    @admin_required
    def admin_reindex_knowledge_file(content_hash):
        """对单个文件重新解析并替换向量/结构化表。"""
        if not context.knowledge_manifest.get_record(content_hash):
            return jsonify({"success": False, "message": "未找到该知识库文件"}), 404

        user_id = g.user_id
        job = context.job_service().submit(
            "knowledge_reindex",
            user_id,
            {"content_hash": content_hash},
            lambda: context.knowledge_admin_write_service().reindex_knowledge_file(
                content_hash,
                user_id=user_id,
            ),
            message="重建任务已提交",
        )
        return jsonify(job), 202
