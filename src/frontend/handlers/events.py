from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from ..api_client import client
from ..keyboards.navigation import event_detail, pager

router = Router()

GROUP_CLASS_RU = {
    "resident": "Житель",
    "courier": "Курьер",
    "delivery": "Доставка",
    "utilities": "ЖКХ",
    "other": "Другой",
}


def _format_event(e: dict) -> str:
    cls = GROUP_CLASS_RU.get(e.get("group_class", ""), e.get("group_class", "—"))
    person = e.get("person_id") or "не определён"
    conf = e.get("person_confidence")
    conf_str = f" ({conf:.0%})" if conf else ""
    ts = e.get("timestamp", "")[:19].replace("T", " ")
    cam = e.get("camera_id", "—")
    return f"🕐 {ts}  📷 {cam}\nКласс: {cls}  |  Житель: {person}{conf_str}"


@router.message(Command("events"))
async def cmd_events(message: Message) -> None:
    await _show_events_page(message, page=0, edit=False)


@router.callback_query(F.data.startswith("events:page:"))
async def cb_events_page(call: CallbackQuery) -> None:
    page = int(call.data.split(":")[2])
    await _show_events_page(call.message, page=page, edit=True)
    await call.answer()


@router.callback_query(F.data.startswith("events:photo:"))
async def cb_event_photo(call: CallbackQuery) -> None:
    event_id = call.data.split(":")[2]

    result = await client.get_event(event_id)
    if not result["ok"]:
        await call.answer(result["error"], show_alert=True)
        return

    image = await client.get_event_image(event_id)
    if image:
        e = result["data"]
        caption = _format_event(e)
        await call.message.answer_photo(BufferedInputFile(image, filename="event.jpg"), caption=caption)
    else:
        await call.answer("Изображение недоступно", show_alert=True)

    await call.answer()


async def _show_events_page(message: Message, page: int, edit: bool) -> None:
    result = await client.get_events(page=page)
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
        text = "Событий нет." if page == 0 else "Больше событий нет."
        if edit:
            await message.edit_text(text)
        else:
            await message.answer(text)
        return

    lines = [f"<b>События (стр. {page + 1})</b>\n"]
    for e in items:
        eid = str(e.get("_id", e.get("id", "")))
        lines.append(_format_event(e))
        lines.append(f'<a href="tg://btn/{eid}">— подробнее (id: {eid[:8]}…)</a>\n')

    # Показываем список с кнопками навигации
    # Для просмотра деталей события используем отдельные inline-кнопки
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton(text="← Назад", callback_data=f"events:page:{page - 1}"))
    if has_next:
        nav_row.append(InlineKeyboardButton(text="Вперёд →", callback_data=f"events:page:{page + 1}"))

    detail_rows = [
        [InlineKeyboardButton(
            text=f"📷 {str(e.get('_id', e.get('id', '')))[:8]}…",
            callback_data=f"events:photo:{e.get('_id', e.get('id', ''))}"
        )]
        for e in items
    ]

    kb_rows = detail_rows + ([nav_row] if nav_row else [])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

    # Краткий список
    short_lines = [f"<b>События (стр. {page + 1})</b>\n"]
    for i, e in enumerate(items, 1):
        short_lines.append(f"{i}. {_format_event(e)}")

    text = "\n".join(short_lines)
    if edit:
        await message.edit_text(text, parse_mode="HTML", reply_markup=kb)
    else:
        await message.answer(text, parse_mode="HTML", reply_markup=kb)
