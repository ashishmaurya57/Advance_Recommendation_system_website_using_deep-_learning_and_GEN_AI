from fastapi import APIRouter

from app.api.routes import auth, catalog, contact, discover, profile, reader, shop

api_router = APIRouter(prefix="/api")
for module in (auth, profile, catalog, shop, discover, reader, contact):
    api_router.include_router(module.router)
