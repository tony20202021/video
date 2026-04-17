from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from ..api_client import client
from ..config import settings
from ..keyboards.navigation import unclassified_actions

router = Router()


class AssignState(StatesGroup):
    waiting_person_id = State()


@router.message(Command("unclassified"))
async def cmd_unclassified(message: Message) -> None:
    if message.from_user.id not in settings.ADMIN_IDS:
        await message.answer("⛔ Нет доступа.")
        return
    await _show_unclassified_page(message, page=0, edit=False)


@router.callback_query(F.data.startswith("unclassified:page:"))
async def cb_unclassified_page(call: CallbackQuery) -> None:
    if call.from_user.id not in settings.ADMIN_IDS:
        await call.answer("⛔ Нет доступа.", show_alert=True)
        return
    page = int(call.data.split(":")[2])
    await _show_unclassified_page(call.message, page=page, edit=True)
    await call.answer()


@router.callback_query(F.data.startswith("unclassified:assign:"))
async def cb_assign_start(call: CallbackQuery, state: FSMContext) -> None:
    if call.from_user.id not in settings.ADMIN_IDS:
        await call.answer("⛔ Нет доступа.", show_alert=True)
        return
    record_id = call.data.split(":")[2]
    await state.set_state(AssignState.waiting_person_id)
    await state.update_data(record_id=record_id)
    await call.message.answer(
        "Введите person_id жителя (например <code>p_0042</code>)\n"
        "или /cancel для отмены",
        parse_mode="HTML",
    )
    await call.answer()


@router.message(AssignState.waiting_person_id, F.text)
async def cb_assign_person_id(message: Message, state: FSMContext) -> None:
    if message.text.startswith("/"):
        await state.clear()
        await message.answer("Отменено.")
        return

    data = await state.get_data()
    record_id = data["record_id"]
    person_id = message.text.strip()

    result = await client.assign_person(record_id, person_id)
    if result["ok"]:
        await message.answer(f"✅ Запись <code>{record_id[:8]}…</code> назначена жителю <code>{person_id}</code>", parse_mode="HTML")
    else:
        await message.answer(f"⚠️ {result['error']}")

    await state.clear()


async def _show_unclassified_page(message: Message, page: int, edit: bool) -> None:
    result = await client.get_unclassified(page=page)
    if not result["ok"]:
        text = f"⚠️ {result['error']}"
        if edit:
            await message.edit_text(text)
        else:
            await message.answer(text)
        return

    data = result["data"]
    items = data if isinstance(data, list) else data.get("items", [])

    if not items:
        text = "Нераспознанных записей нет." if page == 0 else "Больше записей нет."
        if edit:
            await message.edit_text(text)
        else:
            await message.answer(text)
        return

    # Показываем первую запись страницы с фото
    rec = items[0]
    rec_id = str(rec.get("_id", rec.get("id", "")))
    ts = rec.get("timestamp", "")[:19].replace("T", " ")
    cam = rec.get("camera_id", "—")
    cls = rec.get("group_class") or "не определён"
    conf = rec.get("best_person_confidence")
    conf_str = f"  уверенность: {conf:.0%}" if conf else ""

    caption = (
        f"❓ <b>Нераспознан</b>  (стр. {page + 1})\n"
        f"🕐 {ts}  📷 {cam}\n"
        f"Класс: {cls}{conf_str}"
    )

    image = await client.get_event_image(rec_id)
    kb = unclassified_actions(rec_id, page)

    if image:
        if edit:
            await message.answer_photo(BufferedInputFile(image, "unclassified.jpg"), caption=caption, parse_mode="HTML", reply_markup=kb)
        else:
            await message.answer_photo(BufferedInputFile(image, "unclassified.jpg"), caption=caption, parse_mode="HTML", reply_markup=kb)
    else:
        text = caption + "\n\n(фото недоступно)"
        if edit:
            await message.edit_text(text, parse_mode="HTML", reply_markup=kb)
        else:
            await message.answer(text, parse_mode="HTML", reply_markup=kb)
