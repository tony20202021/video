# Удалённое подключение и отладка

## Узлы

| Переменная | Роль | ОС |
|------------|------|----|
| `TS_SERVER` | Linux-сервер (VDS) | Linux |
| `TS_WIN_HOME` | Домашний Windows | Windows |
| `TS_WIN_ENTRY` | Подъездный Windows | Windows |

Все три узла — в одной Tailscale-сети под одним аккаунтом. IP (`100.x.x.x`) стабильны, не меняются при смене провайдера или роутера. Конкретные адреса — в `.env`.

---

## Tailscale

Mesh VPN — все три машины в одной сети, SSH работает через любой NAT без проброса портов.

### Установка

**Linux-сервер:**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh   # включить Tailscale SSH (не нужен OpenSSH)
```

**Windows (каждая машина):**
1. Скачать и установить: https://tailscale.com/download
2. Войти с тем же аккаунтом
3. Включить OpenSSH Server (от Администратора):
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic
Start-Service sshd
```

### Проверка сети

```bash
tailscale status          # список узлов и IP
tailscale ping $TS_WIN_HOME   # проверить связь
```

### SSH config (`~/.ssh/config`)

```
# Linux-сервер (Tailscale SSH — без пароля, авторизация через аккаунт)
Host video-server
    HostName $TS_SERVER      # из .env
    User <username>

# Домашний Windows (OpenSSH + ключ)
Host video-home
    HostName $TS_WIN_HOME    # из .env
    User <username>
    IdentityFile ~/.ssh/id_ed25519

# Подъездный Windows (OpenSSH + ключ)
Host video-entry
    HostName $TS_WIN_ENTRY   # из .env
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

### Авторизация ключа на Windows (один раз)

```powershell
# С Windows-машины — скопировать публичный ключ сервера
# (или с любой другой машины в сети)
type $env:USERPROFILE\.ssh\id_ed25519.pub |
    ssh <username>@<tailscale-ip> "cat >> C:\Users\<username>\.ssh\authorized_keys"
```

### ACL в Tailscale admin-консоли

`https://login.tailscale.com/admin/acls` → добавить блок `ssh`:

```json
"ssh": [
    {
        "action": "accept",
        "src":    ["autogroup:member"],
        "dst":    ["autogroup:member"],
        "users":  ["autogroup:nonroot", "root"]
    }
]
```

В Cursor: Remote Explorer → SSH → выбрать нужный хост.

---

## Веб-сервисы на сервере (Transfer, Label UI)

На Linux-сервере с публичным IP доступны HTTP-сервисы проекта. Оба слушают `0.0.0.0`, но
ограничены whitelist'ом IP из `.env` (если `ALLOWED_IPS` задан).

| Сервис | Порт | Переменная | Запуск |
|--------|------|------------|--------|
| Transfer (приём дифф-кадров) | `8765` | `TRANSFER_PORT` | `./sh/transfer/1_start_server.sh` |
| Label UI (разметка кропов) | `87**` | `LABEL_UI_PORT` | `./sh/train/2_label_ui.sh` |

Пример `.env` на сервере:

```dotenv
ALLOWED_IPS=<домашняя-подсеть>/24   # домашняя подсеть провайдера (/24 переживает смену последнего октета)
TRANSFER_PORT=8765
TRANSFER_API_KEY=<secret>
LABEL_UI_PORT=8789
```

**С домашней машины:** браузер → `http://<публичный-IP>:8789/` (разметчик) или клиент Transfer → `http://<публичный-IP>:8765`.

**Смена домашнего IP:** если Transfer отклоняет запрос с верным API-ключом, в терминале сервера
появится подсказка вида `ALLOWED_IPS=...,109.xxx.yyy.0/24`. Добавьте подсеть в `.env` и перезапустите сервис.

Подробнее: [transfer.md](transfer.md), раздел «Веб-разметчик» в [ml.md](ml.md).

---

## Альтернатива: обратный SSH-туннель

Если Tailscale по каким-то причинам не подходит — reverse tunnel через сервер:

На подъездном клиенте запустить постоянный туннель (через Task Scheduler или NSSM):
```powershell
# autossh нет на Windows — используем встроенный SSH с keepalive
ssh -N -R 2222:localhost:22 -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes <user>@<server-ip>
```

Подключение через ProxyJump:
```
Host video-entryway-tunnel
    HostName localhost
    Port 2222
    User <username>
    ProxyJump video-server
    IdentityFile ~/.ssh/id_ed25519
```

Минусы: туннель нужно держать живым (падает при перезагрузке, нестабильном интернете), сложнее в обслуживании. Tailscale надёжнее.
