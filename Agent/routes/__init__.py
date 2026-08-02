from routes.admin_routes import register_admin_routes
from routes.auth_routes import register_auth_routes
from routes.chat_routes import register_chat_routes
from routes.export_routes import register_export_routes
from routes.job_routes import register_job_routes
from routes.page_routes import register_page_routes
from routes.request_hooks import register_request_hooks
from routes.session_upload_routes import register_session_upload_routes


def register_routes(app, context):
    register_request_hooks(app, context)
    register_page_routes(app, context)
    register_auth_routes(app, context)
    register_admin_routes(app, context)
    register_chat_routes(app, context)
    register_export_routes(app, context)
    register_job_routes(app, context)
    register_session_upload_routes(app, context)
