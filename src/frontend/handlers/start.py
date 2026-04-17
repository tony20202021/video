from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

from ..config import settings

router = Router()


def main_menu(is_admin: bool) -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(text="📋 События"), KeyboardButton(text="👥 Жители")],
        [KeyboardButton(text="📷 Камеры")],
    ]
    if is_admin:
        rows.append([KeyboardButton(text="❓ Нераспознанные"), KeyboardButton(text="📦 Экспорт")])
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    is_admin = settings.is_admin(message.from_user.id)
    await message.answer(
        "👁 <b>Система видеонаблюдения</b>",
        parse_mode="HTML",
        reply_markup=main_menu(is_admin),
    )
