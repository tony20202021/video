import httpx

from .config import settings

PAGE_SIZE = 5


class BackendClient:
    def __init__(self):
        self._base = settings.BACKEND_URL

    async def get_events(self, page: int = 0) -> dict:
        return await self._get("/events", params={"skip": page * PAGE_SIZE, "limit": PAGE_SIZE})

    async def get_event(self, event_id: str) -> dict:
        return await self._get(f"/events/{event_id}")

    async def get_event_image(self, event_id: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self._base}/events/{event_id}/image")
                r.raise_for_status()
                return r.content
        except Exception:
            return None

    async def get_persons(self, page: int = 0) -> dict:
        return await self._get("/persons", params={"skip": page * PAGE_SIZE, "limit": PAGE_SIZE})

    async def get_person(self, person_id: str) -> dict:
        return await self._get(f"/persons/{person_id}")

    async def get_cameras(self) -> dict:
        return await self._get("/cameras")

    async def get_unclassified(self, page: int = 0) -> dict:
        return await self._get("/unclassified", params={"skip": page * PAGE_SIZE, "limit": PAGE_SIZE})

    async def assign_person(self, record_id: str, person_id: str) -> dict:
        return await self._patch(f"/unclassified/{record_id}", json={"person_id": person_id})

    async def trigger_export(self, model: str = "classify") -> dict:
        return await self._post("/training/export", json={"model": model, "include_unclassified": True})

    async def _get(self, path: str, **kwargs) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self._base}{path}", **kwargs)
                r.raise_for_status()
                return {"ok": True, "data": r.json()}
        except httpx.ConnectError:
            return {"ok": False, "error": "Backend недоступен"}
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _patch(self, path: str, **kwargs) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.patch(f"{self._base}{path}", **kwargs)
                r.raise_for_status()
                return {"ok": True, "data": r.json()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _post(self, path: str, **kwargs) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(f"{self._base}{path}", **kwargs)
                r.raise_for_status()
                return {"ok": True, "data": r.json()}
        except Exception as e:
            return {"ok": False, "error": str(e)}


client = BackendClient()
