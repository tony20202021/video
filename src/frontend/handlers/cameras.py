from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ..api_client import client

router = Router()

STATUS_ICON = {True: "🟢", False: "🔴"}


@router.message(Command("cameras"))
async def cmd_cameras(message: Message) -> None:
    result = await client.get_cameras()
    if not result["ok"]:
        await message.answer(f"⚠️ {result['error']}")
        return

    data = result["data"]
    items = data if isinstance(data, list) else data.get("items", [])

    if not items:
        await message.answer("Камеры не настроены.")
        return

    lines = ["<b>Статус камер</b>\n"]
    for cam in items:
        icon = STATUS_ICON.get(cam.get("is_active"), "⚪")
        name = cam.get("name") or cam.get("camera_id", "—")
        location = cam.get("location", "")
        loc_str = f"  —  {location}" if location else ""
        lines.append(f"{icon} <b>{name}</b>{loc_str}")

    await message.answer("\n".join(lines), parse_mode="HTML")
