"""Сервис логики — точка входа FastAPI."""

from fastapi import FastAPI

from backend.routers import cameras, events, ingest, persons, status, training, unclassified, web

app = FastAPI(title="Video Surveillance Backend", version="0.1.0")

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
