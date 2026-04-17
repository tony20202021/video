from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def pager(section: str, page: int, has_next: bool) -> InlineKeyboardMarkup:
    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton(text="← Назад", callback_data=f"{section}:page:{page - 1}"))
    if has_next:
        buttons.append(InlineKeyboardButton(text="Вперёд →", callback_data=f"{section}:page:{page + 1}"))
    if not buttons:
        return None
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def event_detail(event_id: str, page: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📷 Фото", callback_data=f"events:photo:{event_id}")],
        [InlineKeyboardButton(text="↩ К списку", callback_data=f"events:page:{page}")],
    ])


def person_detail(person_id: str, page: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="↩ К списку", callback_data=f"persons:page:{page}")],
    ])


def unclassified_actions(record_id: str, page: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✏️ Назначить жителя", callback_data=f"unclassified:assign:{record_id}")],
        [InlineKeyboardButton(text="⏭ Пропустить", callback_data=f"unclassified:page:{page + 1}")],
        [InlineKeyboardButton(text="↩ К списку", callback_data=f"unclassified:page:{page}")],
    ])
