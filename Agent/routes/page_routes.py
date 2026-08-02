from flask import redirect, render_template, request

from auth import admin_required, login_required


def register_page_routes(app, context):
    @app.route("/")
    def index():
        """首页 - 已登录跳转到聊天页，未登录显示登录页"""
        token = request.cookies.get("token")
        if token:
            payload = context.auth_manager.verify_token(token)
            if payload:
                return redirect("/chat")
        return render_template("login_v3.html")

    @app.route("/login")
    def login_page():
        """登录页 - 双栏设计"""
        return render_template("login_v3.html")

    @app.route("/login/v3")
    def login_page_v3():
        """登录页 - v3 现代化设计"""
        return render_template("login_v3.html")

    @app.route("/chat")
    @login_required
    def chat_page():
        """聊天页"""
        return render_template("chat.html")

    @app.route("/admin")
    @login_required
    @admin_required
    def admin_page():
        """管理后台"""
        return render_template("admin.html")
