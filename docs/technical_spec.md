# Техническое описание

- [overview.md](overview.md) — архитектура, нефункциональные требования, структура репозитория
- [cameras.md](cameras.md) — камеры iCSee, RTSP, сетевой доступ, IP/пароли
- [ml.md](ml.md) — ML пайплайн, модели, frame diff, логика последовательности кадров
- [database.md](database.md) — схема MongoDB, коллекции, хранение изображений
- [services.md](services.md) — API backend, ML service, Telegram bot, конфиг, экспорт
- [transfer.md](transfer.md) — передача дифф-кадров клиент → сервер, IP whitelist, API key
- [remote.md](remote.md) — SSH/Tailscale, доступ к Transfer и Label UI на сервере
