from flask import Response, g, jsonify, request

from auth import login_required


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
        stream = context.chat_runtime().stream(
            request.json or {},
            user_id=g.user_id,
            user_info=context.get_user_info(),
        )
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
