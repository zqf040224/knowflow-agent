import argparse
import logging
import os

from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from logging.handlers import RotatingFileHandler


# 加载环境变量必须早于本地模块导入，auth.py 会在导入时读取 JWT_SECRET
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("✓ 从项目目录加载环境变量")
else:
    load_dotenv()
    print("⚠️ 使用系统环境变量")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("agent.log", maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ],
)
logging.getLogger("agents").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

from app_dependencies import create_app_context  # noqa: E402
from routes import register_routes  # noqa: E402


app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    CORS(app, origins=cors_origins, supports_credentials=True)
else:
    CORS(app, supports_credentials=True)


@app.after_request
def disable_frontend_cache(response):
    """Avoid stale template/CSS mixes while iterating on the LAN frontend."""
    if request.endpoint in {"chat", "index", "login_page"} or request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


context = create_app_context()
register_routes(app, context)
app.config["APP_CONTEXT"] = context

# Compatibility aliases for tests and ad-hoc diagnostics that import app.py.
auth_manager = context.auth_manager
memory = context.memory
upload_manager = context.upload_manager
knowledge_agent = context.knowledge_agent
kb_instance = context.kb_instance
knowledge_manifest = context.knowledge_manifest
spreadsheet_store = context.spreadsheet_store
_auth_route_service = context.auth_route_service
_account_admin_service = context.account_admin_service
_get_user_info = context.get_user_info
rate_limit = context.rate_limit
reimbursement_template_path = context.reimbursement_template_path
_explicit_document_request = context.explicit_document_request
create_docx = context.create_docx
_record_beta_feedback = context.record_beta_feedback
_get_beta_dashboard = context.get_beta_dashboard
_update_beta_feedback_status = context.update_beta_feedback_status
_record_token_usage = context.record_token_usage
_record_agent_run_token_usage = context.record_agent_run_token_usage
_get_token_usage_dashboard = context.get_token_usage_dashboard
_get_existing_departments = context.get_existing_departments
_validate_account_department = context.validate_account_department
_register_user_from_admin_payload = context.register_user_from_admin_payload
_knowledge_actor = context.knowledge_actor
_record_knowledge_upload_audit = context.record_knowledge_upload_audit
_llm_intent_classifier = context.llm_intent_classifier
_writer_factory = context.writer_factory
_runtime_query_service = context.runtime_query_service
_chat_container = context.chat_container
_chat_runtime = context.chat_runtime
_agent_generate_service = context.agent_generate_service
_export_service = context.export_service
_session_service = context.session_service
_upload_service = context.upload_service
_spreadsheet_transform_service = context.spreadsheet_transform_service
_knowledge_admin_read_service = context.knowledge_admin_read_service
_knowledge_admin_audit_service = context.knowledge_admin_audit_service
_knowledge_admin_write_service = context.knowledge_admin_write_service
_job_service = context.job_service
_request_orchestrator = context.request_orchestrator

for _endpoint in (
    "api_login",
    "search",
    "query_spreadsheets",
):
    if _endpoint in app.view_functions:
        globals()[_endpoint] = app.view_functions[_endpoint]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能知识库平台 Web 服务")
    parser.add_argument("--port", type=int, default=5003, help="服务端口")
    args = parser.parse_args()

    print("=" * 60)
    print("智能知识库平台 Web 服务")
    print(f"开发模式: http://localhost:{args.port}")
    print(f"生产部署: gunicorn -w 4 -b 0.0.0.0:{args.port} app:app")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=args.port, threaded=True)
