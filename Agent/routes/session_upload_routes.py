import json
import logging
from io import BytesIO
from urllib.parse import quote

from flask import g, jsonify, request, send_file

from auth import login_required


logger = logging.getLogger(__name__)


def register_session_upload_routes(app, context):
    @app.route("/api/sessions", methods=["GET"])
    @login_required
    def list_sessions():
        """获取用户的所有会话"""
        return jsonify(context.session_service().list_sessions(g.user_id, limit=20))

    @app.route("/api/sessions", methods=["POST"])
    @login_required
    def create_new_session():
        """创建新会话"""
        return jsonify(context.session_service().create_session(g.user_id, request.json or {}))

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    @login_required
    def get_session(session_id):
        """获取会话详情"""
        payload, status = context.session_service().get_session(g.user_id, session_id)
        return jsonify(payload), status

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    @login_required
    def delete_session_route(session_id):
        """删除会话"""
        payload, status = context.session_service().delete_session(g.user_id, session_id)
        return jsonify(payload), status

    @app.route("/api/sessions/<session_id>/messages", methods=["GET"])
    @login_required
    def get_session_messages(session_id):
        """获取会话消息历史"""
        limit = request.args.get("limit", 20, type=int)
        payload, status = context.session_service().get_session_messages(g.user_id, session_id, limit=limit)
        return jsonify(payload), status

    @app.route("/api/user/profile", methods=["GET", "PUT"])
    @login_required
    def user_profile_with_memory():
        """获取/更新用户画像（包含记忆系统）"""
        if request.method == "GET":
            return jsonify(context.session_service().get_user_profile(g.user_id))
        return jsonify(context.session_service().update_user_profile(g.user_id, request.json or {}))

    @app.route("/api/user/memories", methods=["GET"])
    @login_required
    def list_long_term_memories():
        """列出当前用户显式保存或设置生成的长期记忆。"""
        return jsonify(context.session_service().list_long_term_memories(g.user_id))

    @app.route("/api/user/memories/<int:memory_id>", methods=["DELETE"])
    @login_required
    def forget_long_term_memory(memory_id):
        """忘记一条用户显式保存的长期记忆。"""
        payload, status = context.session_service().forget_long_term_memory(g.user_id, memory_id)
        return jsonify(payload), status

    @app.route("/api/user/stats", methods=["GET"])
    @login_required
    def user_stats():
        """获取用户统计"""
        return jsonify(context.session_service().user_stats(g.user_id))

    @app.route("/api/search-history", methods=["POST"])
    @login_required
    def search_history():
        """搜索历史会话"""
        return jsonify(context.session_service().search_history(g.user_id, request.json or {}))

    @app.route("/api/upload", methods=["POST"])
    @login_required
    def upload_file():
        """
        文件上传接口

        支持模式：
        1. knowledge: 添加到团队知识库（长期共享）
        2. temp: 仅本次使用（30分钟后自动删除）
        """
        if "file" not in request.files:
            return jsonify({"success": False, "message": "请选择文件"}), 400

        mode = request.form.get("mode", "temp")
        if mode == "knowledge":
            prepared, status = context.upload_service().prepare_knowledge_upload(
                request.files["file"],
                user_id=g.user_id,
                user_role=getattr(g, "role", "user"),
                user_department=getattr(g, "department", ""),
                category=request.form.get("category", ""),
            )
            if status != 200:
                return jsonify(prepared), status

            try:
                job = context.job_service().submit(
                    "knowledge_upload",
                    g.user_id,
                    {
                        "filename": prepared.get("filename", ""),
                        "category": prepared.get("category", ""),
                        "department": prepared.get("department", ""),
                    },
                    lambda: context.upload_service().process_prepared_knowledge_upload(prepared),
                    message="上传入库任务已提交",
                )
            except Exception as exc:
                context.upload_service().cleanup_prepared_upload(prepared)
                logger.exception("上传入库任务提交失败: %s", exc)
                return jsonify({"success": False, "message": "上传入库任务提交失败，请重试"}), 500
            return jsonify(job), 202

        result, status = context.upload_service().upload_file(
            request.files["file"],
            user_id=g.user_id,
            user_role=getattr(g, "role", "user"),
            user_department=getattr(g, "department", ""),
            mode=mode,
            category=request.form.get("category", ""),
        )
        return jsonify(result), status

    @app.route("/api/upload/temp/<file_id>", methods=["GET"])
    @login_required
    def get_temp_file_content(file_id):
        """获取临时文件内容（用于在对话中引用）"""
        result, status = context.upload_service().get_temp_file_content(file_id, user_id=g.user_id)
        return jsonify(result), status

    @app.route("/api/spreadsheets/transform", methods=["POST"])
    @login_required
    def transform_uploaded_spreadsheet():
        """根据自然语言规则处理临时上传的表格，并导出 Excel。"""
        result = context.spreadsheet_transform_service().transform(request.json or {}, user_id=g.user_id)
        if not result.success:
            return jsonify(result.error), result.status

        response = send_file(
            BytesIO(result.payload or b""),
            mimetype=result.mimetype,
            as_attachment=True,
            download_name=result.filename,
        )
        response.headers["X-Spreadsheet-Transform-Summary"] = quote(
            json.dumps(result.summary or {}, ensure_ascii=False)
        )
        return response

    @app.route("/api/upload/categories", methods=["GET"])
    @login_required
    def get_upload_categories():
        """获取知识库分类列表（按用户权限过滤）"""
        return jsonify(context.upload_service().upload_categories(
            user_role=getattr(g, "role", "user"),
            user_department=getattr(g, "department", ""),
        ))

    @app.route("/api/upload/stats", methods=["GET"])
    @login_required
    def get_upload_stats():
        """获取上传统计（仅管理员）"""
        if g.role != "admin":
            return jsonify({"success": False, "message": "权限不足"}), 403

        return jsonify(context.upload_service().upload_stats())
