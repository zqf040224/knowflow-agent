from flask import g, jsonify

from auth import login_required


def register_job_routes(app, context):
    @app.route("/api/jobs/<job_id>", methods=["GET"])
    @login_required
    def get_job(job_id):
        """查询后台任务状态。"""
        payload, status = context.job_service().get_job(
            job_id,
            user_id=g.user_id,
            role=getattr(g, "role", "user"),
        )
        return jsonify(payload), status
