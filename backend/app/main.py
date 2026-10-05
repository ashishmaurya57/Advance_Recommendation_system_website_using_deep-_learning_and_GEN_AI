import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.admin.views import setup_admin
from app.api.router import api_router
from app.core.config import settings
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
        https_only=False,
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
        return {"status": "ok"}

    return app


app = create_app()
