"""
REST API Routes
"""
from fastapi import APIRouter
from . import health, upload, artifacts

router = APIRouter()

router.include_router(health.router, tags=["health"])
router.include_router(upload.router, prefix="/upload", tags=["upload"])
router.include_router(artifacts.router, prefix="/artifacts", tags=["artifacts"])
