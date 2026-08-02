from flask import g, jsonify, request


def register_request_hooks(app, context):
    @app.before_request
    def before_request():
        """请求前处理，检查登录状态和请求体大小"""
        g.user_id = None
        g.username = None
        g.role = None
        g.token = None

        content_length = request.content_length
        if content_length and content_length > context.max_request_size:
            return jsonify({"error": "请求体过大，最大支持 10MB"}), 413

        if request.endpoint in [
            "index", "login_page", "login_page_v3",
            "api_login", "api_public_register", "api_health", "static",
        ]:
            return

        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            payload = context.auth_manager.verify_token(token)
            if payload:
                g.user_id = payload["user_id"]
                g.username = payload["username"]
                g.role = payload["role"]
                g.token = token
