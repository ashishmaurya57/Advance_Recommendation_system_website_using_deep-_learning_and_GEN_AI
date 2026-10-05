import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from app.admin.views import setup_admin
from app.api.router import api_router
from app.core.config import settings
from app.core.deps import DB
from app.services import ml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=ml.warm_up, daemon=True).start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="BookTown API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="booktown_session",
        max_age=settings.session_max_age,
        same_site="lax",
        https_only=settings.is_production,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.state.admin = setup_admin(app)

    @app.get("/api/health", tags=["meta"])
    def health():
        """Liveness check used by Render. Doesn't touch the database."""
        return {"status": "ok"}

    @app.get("/api/health/db", tags=["meta"])
    def health_db(db: DB):
        """Pinged by the keep-alive cron: keeps Render awake and Supabase from pausing."""
        try:
            db.execute(text("select 1"))
        except Exception:
            logging.getLogger(__name__).exception("Database health check failed")
            return JSONResponse({"status": "error", "database": "unreachable"}, status_code=503)
        return {"status": "ok", "database": "ok"}

    return app


app = create_app()
