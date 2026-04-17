from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from ..api_client import client

router = Router()


def _format_person(p: dict) -> str:
    name = p.get("name") or "Без имени"
    pid = p.get("person_id", "—")
    apt = p.get("apartment_id")
    apt_str = f"  |  Кв. {apt}" if apt else ""
    photos = len(p.get("image_paths") or [])
    return f"👤 <b>{name}</b>  ({pid}){apt_str}\nФото в базе: {photos}"


@router.message(Command("persons"))
@router.message(F.text == "👥 Жители")
async def cmd_persons(message: Message) -> None:
    await _show_persons_page(message, page=0, edit=False)


@router.callback_query(F.data.startswith("persons:page:"))
async def cb_persons_page(call: CallbackQuery) -> None:
    page = int(call.data.split(":")[2])
    await _show_persons_page(call.message, page=page, edit=True)
    await call.answer()


@router.callback_query(F.data.startswith("persons:detail:"))
async def cb_person_detail(call: CallbackQuery) -> None:
    parts = call.data.split(":")
    person_id = parts[2]
    page = int(parts[3]) if len(parts) > 3 else 0

    result = await client.get_person(person_id)
    if not result["ok"]:
        await call.answer(result["error"], show_alert=True)
        return

    p = result["data"]
    text = _format_person(p)
    embeddings = len(p.get("embeddings") or [])
    text += f"\nEmbeddings: {embeddings}"

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="↩ К списку", callback_data=f"persons:page:{page}")]
    ])
    await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
    await call.answer()


async def _show_persons_page(message: Message, page: int, edit: bool) -> None:
    result = await client.get_persons(page=page)
    if not result["ok"]:
        text = f"⚠️ {result['error']}"
        if edit:
            await message.edit_text(text)
        else:
            await message.answer(text)
        return

    data = result["data"]
    items = data if isinstance(data, list) else data.get("items", [])
    has_next = len(items) == 5

    if not items:
        text = "Жители не добавлены." if page == 0 else "Больше жителей нет."
        if edit:
            await message.edit_text(text)
        else:
            await message.answer(text)
        return

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton(text="← Назад", callback_data=f"persons:page:{page - 1}"))
    if has_next:
        nav_row.append(InlineKeyboardButton(text="Вперёд →", callback_data=f"persons:page:{page + 1}"))

    detail_rows = [
        [InlineKeyboardButton(
            text=f"👤 {p.get('name') or p.get('person_id', '?')}",
            callback_data=f"persons:detail:{p.get('person_id')}:{page}"
        )]
        for p in items
    ]

    kb_rows = detail_rows + ([nav_row] if nav_row else [])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

    lines = [f"<b>Жители (стр. {page + 1})</b>\n"]
    for i, p in enumerate(items, 1):
        lines.append(f"{i}. {_format_person(p)}")

    text = "\n".join(lines)
    if edit:
        await message.edit_text(text, parse_mode="HTML", reply_markup=kb)
    else:
        await message.answer(text, parse_mode="HTML", reply_markup=kb)
