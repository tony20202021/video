"""Сервис логики — точка входа FastAPI."""

from fastapi import FastAPI

from backend.routers import status

app = FastAPI(title="Video Surveillance Backend", version="0.1.0")

app.include_router(status.router)
