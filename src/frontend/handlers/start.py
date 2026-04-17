from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from ..config import settings

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    is_admin = settings.is_admin(message.from_user.id)
    admin_block = "\n\n<b>Команды администратора:</b>\n/unclassified — разметка нераспознанных\n/export — экспорт обучающей выборки" if is_admin else ""
    await message.answer(
        "👁 <b>Система видеонаблюдения</b>\n\n"
        "<b>Команды:</b>\n"
        "/events — последние события\n"
        "/persons — список жителей\n"
        "/cameras — статус камер"
        + admin_block,
        parse_mode="HTML",
    )
