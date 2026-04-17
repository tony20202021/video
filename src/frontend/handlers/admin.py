from aiogram import Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message

from ..api_client import client
from ..config import settings

router = Router()


@router.message(Command("export"))
async def cmd_export(message: Message) -> None:
    if message.from_user.id not in settings.ADMIN_IDS:
        await message.answer("⛔ Нет доступа.")
        return

    await message.answer("⏳ Формирую обучающую выборку…")
    result = await client.trigger_export()

    if not result["ok"]:
        await message.answer(f"⚠️ {result['error']}")
        return

    data = result["data"]
    file_url = data.get("file_url")
    count = data.get("count", "?")

    if file_url:
        # Если backend вернул прямую ссылку на zip
        await message.answer(
            f"✅ Выборка готова: {count} изображений\n{file_url}"
        )
    else:
        await message.answer(f"✅ Экспорт запущен. Изображений: {count}")
