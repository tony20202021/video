# Удалённое подключение и отладка

## Узлы

| Узел | ОС | Сеть | Подключение |
|------|----|------|-------------|
| Сервер | Linux | Публичный IP | SSH напрямую |
| Клиент дома | Windows | Тот же роутер что и дев-машина | SSH по локальному IP |
| Клиент подъезд | Windows | Другой роутер, нет внешнего IP | Tailscale |

---

## Сервер (Linux, публичный IP)

Стандартное SSH подключение в Cursor — работает из коробки.

`~/.ssh/config`:
```
Host video-server
    HostName <публичный-IP>
    User <user>
    IdentityFile ~/.ssh/id_ed25519
```

---

## Клиент дома (Windows, тот же роутер)

SSH по локальному IP — доступен напрямую с дев-машины.

```
Host video-home
    HostName 192.168.1.XXX
    User Anton
    IdentityFile ~/.ssh/id_ed25519
```

---

## Клиент подъезд (Windows, другой роутер)

Другой роутер без внешнего IP → входящий SSH невозможен.  
Решение: **Tailscale** — mesh VPN, работает через любой NAT без проброса портов.

### Как работает

Все три машины (дев, подъезд, сервер опционально) входят в одну Tailscale-сеть и получают постоянные IP вида `100.x.x.x`. Cursor подключается к этому IP как к обычному SSH — никаких прыжков и туннелей.

### Установка

**На каждой машине** (дев-машина Windows + клиент подъезд Windows):

1. Скачать: https://tailscale.com/download
2. Установить, войти с одним аккаунтом
3. На дев-машине: `tailscale status` — видны все узлы и их IP

На подъездном клиенте убедиться что SSH сервер включён:
```powershell
# Включить OpenSSH Server (один раз)
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic
Start-Service sshd
```

### SSH config для Cursor

```
Host video-entryway
    HostName 100.x.x.x        # Tailscale IP подъездного клиента
    User Anton
    IdentityFile ~/.ssh/id_ed25519
```

Tailscale IP стабилен — не меняется при смене роутера или IP-адреса.

### Авторизация ключа на подъездном клиенте

С дев-машины (один раз):
```powershell
# Скопировать публичный ключ на подъездный клиент
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh Anton@100.x.x.x "cat >> C:\Users\Anton\.ssh\authorized_keys"
```

---

## Итоговый ~/.ssh/config

```
Host video-server
    HostName <публичный-IP>
    User <user>
    IdentityFile ~/.ssh/id_ed25519

Host video-home
    HostName 192.168.1.XXX
    User Anton
    IdentityFile ~/.ssh/id_ed25519

Host video-entryway
    HostName 100.x.x.x
    User Anton
    IdentityFile ~/.ssh/id_ed25519
```

В Cursor: Remote Explorer → SSH → выбрать нужный хост.

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
    User Anton
    ProxyJump video-server
    IdentityFile ~/.ssh/id_ed25519
```

Минусы: туннель нужно держать живым (падает при перезагрузке, нестабильном интернете), сложнее в обслуживании. Tailscale надёжнее.
