"""Запуск Telegram-бота.

Использование:
    python scripts/bot/run_bot.py
    python scripts/bot/run_bot.py --log-level DEBUG

Переменные окружения (из .env):
    BOT_TOKEN      — токен Telegram-бота (обязательно)
    BACKEND_URL    — URL backend (default: http://localhost:8780)
    ADMIN_IDS      — ID администраторов через запятую
    CONDA_ENV      — имя conda-окружения (используется в sh-скрипте)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

from dotenv import load_dotenv
load_dotenv(_root / ".env")

from frontend.bot import main as bot_main


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Telegram-бот системы видеонаблюдения")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования (default: INFO)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)
    log.info("Запуск бота (root=%s)", _root)
    asyncio.run(bot_main())
