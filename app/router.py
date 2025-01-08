from fastapi import APIRouter

from app.health.router import router as health_router
from app.photobooth.router import router as photobooth_router

router = APIRouter()

# /health
router.include_router(health_router)

# /photobooth
router.include_router(photobooth_router)
