from flask import Response, g, jsonify, request

from auth import login_required
from chat_runtime import ChatRunConflictError, ChatRunNotFoundError


def register_chat_routes(app, context):
    @app.route("/api/spreadsheets/query", methods=["POST"])
    @login_required
    def query_spreadsheets():
        """按权限查询结构化表格行，用于精确数据问答和人工核对。"""
        return jsonify(context.runtime_query_service().spreadsheet_query(
            request.json or {},
            user_info=context.get_user_info(),
        ))

    @app.route("/api/search", methods=["POST"])
    @login_required
    def search():
        """知识库搜索"""
        return jsonify(context.runtime_query_service().search(
            request.json or {},
            user_info=context.get_user_info(),
        ))

    @app.route("/api/chat", methods=["POST"])
    @login_required
    @context.rate_limit
    def chat():
        """聊天主入口。

        Production chat requests must enter through ChatGraphRuntime so request
        preparation, intent routing, and SSE contracts stay centralized.
        """
        try:
            runtime = context.chat_runtime()
            stream_http = getattr(runtime, "stream_http", runtime.stream)
            stream = stream_http(
                request.json or {},
                user_id=g.user_id,
                user_info=context.get_user_info(),
            )
        except ChatRunConflictError as exc:
            return jsonify({"success": False, "error": str(exc)}), 409
        return Response(stream, mimetype="text/event-stream")

    @app.route("/api/chat/runs/<run_id>", methods=["GET"])
    @login_required
    def chat_run_status(run_id):
        try:
            payload = context.chat_runtime().run_status(run_id, user_id=g.user_id)
        except ChatRunNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404
        return jsonify({"success": True, "run": payload})

    @app.route("/api/chat/runs/<run_id>/resume", methods=["POST"])
    @login_required
    @context.rate_limit
    def resume_chat_run(run_id):
        data = request.get_json(silent=True) or {}
        resume_value = data.get("resume", data.get("decision", data))
        try:
            stream = context.chat_runtime().resume(
                run_id,
                resume_value,
                user_id=g.user_id,
                user_info=context.get_user_info(),
                interrupt_id=str(data.get("interrupt_id") or ""),
            )
        except ChatRunNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404
        except ChatRunConflictError as exc:
            return jsonify({"success": False, "error": str(exc)}), 409
        return Response(stream, mimetype="text/event-stream")

    @app.route("/api/chat/runs/<run_id>/recover", methods=["POST"])
    @login_required
    @context.rate_limit
    def recover_chat_run(run_id):
        try:
            stream = context.chat_runtime().recover(
                run_id,
                user_id=g.user_id,
                user_info=context.get_user_info(),
            )
        except ChatRunNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404
        except ChatRunConflictError as exc:
            return jsonify({"success": False, "error": str(exc)}), 409
        return Response(stream, mimetype="text/event-stream")

    @app.route("/api/agent/generate", methods=["POST"])
    @login_required
    @context.rate_limit
    def agent_generate():
        """Agent 生成"""
        result = context.agent_generate_service().generate(
            request.json or {},
            user_id=g.user_id,
            user_info=context.get_user_info(),
        )
        return jsonify(result)
