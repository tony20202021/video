"""Сервис логики — точка входа FastAPI."""

from fastapi import FastAPI

from backend.routers import cameras, events, ingest, persons, status, training, unclassified, web
from common.version import get_version

app = FastAPI(title="Video Surveillance Backend", version=get_version())

# Web UI — должен быть первым (перехватывает GET /)
app.include_router(web.router)

# REST API
app.include_router(status.router)
app.include_router(events.router)
app.include_router(persons.router)
app.include_router(cameras.router)
app.include_router(ingest.router)
app.include_router(unclassified.router)
app.include_router(training.router)
