from flask import g, jsonify, request

from auth import admin_required, login_required


def register_auth_routes(app, context):
    def _json_payload():
        return request.get_json(silent=True) or {}

    @app.route("/api/health", methods=["GET"])
    def api_health():
        """健康检查"""
        return jsonify(context.runtime_query_service().health())

    @app.route("/api/auth/login", methods=["POST"])
    @context.rate_limit
    def api_login():
        """用户登录"""
        result = context.auth_route_service.login(
            _json_payload(),
            ip_address=request.remote_addr or "unknown",
            user_agent=request.headers.get("User-Agent", ""),
        )
        response = jsonify(result.payload)
        if result.set_cookie_token:
            response.set_cookie(
                "token",
                result.set_cookie_token,
                httponly=True,
                secure=context.cookie_secure,
                samesite="Lax",
                max_age=24 * 60 * 60,
            )
        return response, result.status

    @app.route("/api/auth/logout", methods=["POST"])
    def api_logout():
        """用户登出"""
        result = context.auth_route_service.logout()
        response = jsonify(result.payload)
        if result.clear_cookie:
            response.set_cookie("token", "", expires=0)
        return response

    @app.route("/api/auth/public-register", methods=["POST"])
    def api_public_register():
        """公开注册（仅允许注册普通用户）"""
        result = context.auth_route_service.public_register(_json_payload())
        return jsonify(result.payload), result.status

    @app.route("/api/auth/register", methods=["POST"])
    @login_required
    @admin_required
    def api_register():
        """注册新用户（仅管理员）"""
        return jsonify(context.register_user_from_admin_payload(_json_payload()))

    @app.route("/api/auth/change-password", methods=["POST"])
    @login_required
    def api_change_password():
        """修改密码"""
        result = context.auth_route_service.change_password(g.user_id, _json_payload())
        return jsonify(result.payload), result.status

    @app.route("/api/auth/profile", methods=["GET", "PUT"])
    @login_required
    def api_profile():
        """获取/更新用户信息"""
        result = context.auth_route_service.profile(
            method=request.method,
            user_id=g.user_id,
            data=_json_payload(),
        )
        return jsonify(result.payload), result.status
