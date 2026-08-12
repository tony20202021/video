# Удалённое подключение и отладка

## Узлы

| Переменная `.env` | Метка (`_LABEL`) | Роль | ОС |
|-------------------|------------------|------|----|
| `TS_SERVER` | — | Linux-сервер (VDS) | Linux |
| `TS_DEVELOP` | `TS_DEVELOP_LABEL` | DEVELOP — домашний Windows | Windows |
| `TS_CAMERAS_3` | `TS_CAMERAS_3_LABEL` | CAMERAS_3 — подъездный Windows | Windows |
| `TS_CAMERAS_1` | `TS_CAMERAS_1_LABEL` | CAMERAS_1 — Windows у Елены | Windows |

Все узлы — в одной Tailscale-сети под одним аккаунтом. IP (`100.x.x.x`) стабильны, не меняются при смене провайдера или роутера. Конкретные адреса и метки — в `.env`.

`_LABEL` — отображаемое имя в `sh/status/status.sh` (терминал + markdown-таблица). Менять в `.env`, не в скрипте.

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
tailscale ping $TS_DEVELOP   # проверить связь
```

### SSH config (`~/.ssh/config`)

```
# Linux-сервер (Tailscale SSH — без пароля, авторизация через аккаунт)
Host video-server
    HostName $TS_SERVER      # из .env
    User <username>

# DEVELOP (OpenSSH + ключ)
Host video-develop
    HostName $TS_DEVELOP     # из .env
    User <username>
    IdentityFile ~/.ssh/id_ed25519

# CAMERAS_3 (OpenSSH + ключ)
Host video-cameras-3
    HostName $TS_CAMERAS_3   # из .env
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

---

## Деплой и синхронизация скриптов на машины

Единственный источник правды — ветка `develop` на GitHub. Скрипты **никогда** не
копируются между машинами вручную (scp). Любое изменение проходит цикл:
правка на сервере → `commit` → `push` → `git pull` на машинах.

**Почему не scp:** ручное копирование разъезжается — на машинах остаются untracked-копии
и локальные правки, репозитории расходятся. git даёт единый источник правды и
воспроизводимость.

### Выкатить изменение на все машины

```bash
# 1. На сервере — закоммитить и запушить
cd /home/tony/repos/video
git add -A && git commit -m "..."
git push origin develop
```

**Linux-сервер** (обновление + перезапуск сервисов, если менялись unit-файлы/пути):
```bash
git pull origin develop
sudo ./sh/system/setup_systemd.sh
```

**Windows, дерево чистое** (нет локальных правок):
```powershell
cd <repo>            # путь машины — TS_*_REPO в .env
git pull origin develop
```

**Windows с локальными правками** (сделать байт-в-байт как репо):
```powershell
cd <repo>
git fetch origin
git reset --hard origin/develop
git clean -fdn       # сначала dry-run — посмотреть, что удалится
git clean -fd        # затем удалить untracked-файлы
```
> Забэкапить правки перед reset, если могут понадобиться: `git diff HEAD > backup.patch`.

### Машины: доступы и пути (в `.env`)

| Машина | SSH user | IP | Репозиторий (`TS_*_REPO`) | git |
|--------|----------|----|---------------------------|-----|
| Linux-сервер | — | `TS_SERVER` | `/home/tony/repos/video` | в PATH |
| DEVELOP | `Anton` | `TS_DEVELOP` | `E:\_Home\Tony\pet projects\video` | в PATH |
| CAMERAS_3 | `julia` | `TS_CAMERAS_3` | `C:\_Work\video` | в PATH |
| CAMERAS_1 | `evelina` | `TS_CAMERAS_1` | `C:\Work\video` | в PATH (портативный, `C:\Work\git\cmd`) |

`TS_*_REPO` в `.env` — оттуда `sh/status/status.sh` строит путь к `status_win.ps1`
на каждой машине (пути разные, поэтому хардкодить нельзя).

### Типовые проблемы

- **`git diff origin/develop` → «unknown revision»** — репозиторий склонирован single-branch
  (только `main`, как было на CAMERAS_1). Починка:
  ```powershell
  git remote set-branches origin main develop
  git fetch origin
  ```
- **SSH «Permission denied (publickey)»** — ключ сервера не авторизован или неверный
  SSH-user (на DEVELOP аккаунт — `Anton`, не `tony`). См. авторизацию ключа ниже.
- **`git pull` → exit 128, `fetch` при этом проходит** — у ветки не настроен upstream
  (`fatal: no tracking information`). Так было на CAMERAS_1. Починка (git по полному пути,
  если не в PATH):
  ```powershell
  & "C:\Work\git\cmd\git.exe" -C "C:\Work\video" branch --set-upstream-to=origin/develop develop
  & "C:\Work\git\cmd\git.exe" -C "C:\Work\video" pull --ff-only
  ```
- **git — портативный (`C:\Work\git`)** — это штатно (раздел 4a), «ставить заново» не нужно.
  Либо добавить `C:\Work\git\cmd` в User PATH (рецепт в 4a — так сделано на CAMERAS_1), либо
  звать по полному пути `C:\Work\git\cmd\git.exe`. Актуальное состояние — столбец «git» в таблице
  выше и `TS_*_GIT` в `.env`.

## Выполнение команд и управление процессами на Windows по SSH

Практические грабли (проверено в работе с CAMERAS_3/DEVELOP через Tailscale SSH из Linux).

### Кодировка вывода
Windows cmd/PowerShell отдают кириллицу в **cp866** → в Linux-терминале «крякозябры» (ASCII/числа читаются).
Приём: срезать не-печатное — `ssh ... "..." | sed 's/[^[:print:][:space:]]//g'`, а в команду вставлять
ASCII-маркеры (`'PYCOUNT=' + (Get-Process python).Count`), чтобы грепать по ним. Либо `chcp 65001` в начале.

### Кавычки и `$_`
Внешняя ssh-строка в двойных кавычках → локальный **zsh раскрывает `$_`, `$env:`** ДО отправки.
- Избегать `$_`: упрощённый синтаксис `Where-Object CommandLine -match 'X'` (без scriptblock и `$_`).
- Внутренние `"` PowerShell экранировать как `\"`; напр. `-Filter \"Name='python.exe'\"`.
- Экранировать `\$` где нужен буквальный `$` (`\$env:USERPROFILE`). Либо внешние **одинарные** кавычки для ssh-строки.

### Self-match в запросах процессов
`Get-CimInstance Win32_Process | Where CommandLine -match 'python'` матчит и СВОЙ запрос (его командная строка
содержит паттерн) → счётчик завышен на 1-2. Фильтровать по образу: `-Filter "Name='python.exe'"` (исключает powershell/cmd).

### Персистентность процессов ⚠️
`Start-Process` / `cmd start` / любой фоновый процесс, запущенный ЧЕРЕЗ SSH, **НЕ переживает закрытие SSH-сессии**
(Windows OpenSSH убивает job сессии — проверено, PYCOUNT=0 после выхода). Для постоянного процесса — **scheduled task**:
```
schtasks /Create /TN <Имя> /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <путь.ps1>" /SC ONCE /ST 23:59 /RU SYSTEM /F
schtasks /Run /TN <Имя>
```
`/RU SYSTEM` — без пароля (у `1_motion_diff.ps1` есть SYSTEM-фолбэк поиска conda). Задача живёт в своей сессии.
Управление: `schtasks /Query|/End|/Run /TN <Имя>`; `/Change /DISABLE` — чтобы не поднималась на boot/wake.
(Классификатор Claude может блокировать создание task как «эскалацию персистентности» — нужна явная авторизация.)

### Сначала проверить синхронно
Детач прячет ошибки старта. Убедиться, что `.ps1`→python стартует: запустить СИНХРОННО с длительностью —
`ssh ... "powershell -File ...\1_motion_diff.ps1 8"` (8 сек) — баннер/baseline/traceback видно прямо в выводе.

### Управление пайплайном камеры
- **watchdog** (`VideoWatchdog`, scheduled task) сторожит `1_motion_diff.ps1` + `2_send.ps1`, респавн каждые 30с.
  Убить процесс мало — watchdog поднимет; durable-стоп: `schtasks /End` **+** `/Change /DISABLE` (иначе wake-триггер вернёт).
- **Захват БЕЗ отправки** (напр. на время обучения на сервере): watchdog off + отдельная SYSTEM-task только на
  `1_motion_diff.ps1` (как выше). Файлы копятся в `.output/pipeline/1_motion_diff/images/` (transfer их не шлёт/не удаляет).
- **НЕ запускать два motion_diff одновременно** на одну камеру — A31 отдаёт один субпоток, второй клиент голодает.

### Ресурсы машин (для тяжёлых задач, ~2026-08)
| Машина | CPU | RAM | Заметка |
|--------|-----|-----|---------|
| Linux-сервер (VDS) | **1 ядро** | 3.8ГБ | обучение CNN медленно + тесно; стоп idle-сервисов освобождает мало (модели не загружены) |
| DEVELOP | 2 ядра | ~8ГБ | вдвое быстрее, без OOM; нужна перекачка датасета по Tailscale |
| CAMERAS_1/3 | ноуты | — | только захват/отправка |

### Sudo на сервере (systemctl)
`systemctl` требует пароль. Для управления сервисами из автоматизации — временное NOPASSWD-правило
(вернуть/удалить после): `echo -e 'Defaults:tony !use_pty\ntony ALL=(ALL) NOPASSWD: /usr/bin/systemctl, /bin/systemctl, /usr/bin/rm /etc/sudoers.d/tony-systemctl' | sudo tee /etc/sudoers.d/tony-systemctl`.

---

## Добавление новой Windows-машины

Чеклист для подключения нового Windows-клиента к инфраструктуре.

### 1. Tailscale + OpenSSH

```powershell
# 1. Установить Tailscale (https://tailscale.com/download), войти с тем же аккаунтом
# 2. Включить OpenSSH Server (от Администратора):
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic
Start-Service sshd
```

### 2. Авторизовать SSH-ключ сервера

**Важно:** если пользователь Windows входит в группу Administrators, sshd ищет ключи
в `C:\ProgramData\ssh\administrators_authorized_keys` (не в `.ssh\authorized_keys`).

Определить группу:
```bash
ssh <user>@<tailscale-ip> "whoami /groups | findstr S-1-5-32-544"
```
Если вывод не пуст — пользователь в Administrators.

#### Для пользователя из группы **Administrators**

```bash
PUB=$(cat ~/.ssh/id_ed25519.pub)

# Записать ключ в системный файл
ssh <user>@<tailscale-ip> "echo $PUB > C:\ProgramData\ssh\administrators_authorized_keys"

# Задать права через SID (имена групп на русской Windows отличаются — SID работают везде)
ssh <user>@<tailscale-ip> "icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant *S-1-5-18:(F) /grant *S-1-5-32-544:(F)"
```

#### Для обычного пользователя (не Administrators)

```bash
PUB=$(cat ~/.ssh/id_ed25519.pub)
WIN_USER=<username>

# Создать директорию и записать ключ
ssh <user>@<tailscale-ip> "mkdir C:\Users\$WIN_USER\.ssh 2>nul & echo $PUB > C:\Users\$WIN_USER\.ssh\authorized_keys"

# Убрать наследование; права через SID пользователя и SYSTEM
# SID пользователя можно узнать: wmic useraccount where name='<username>' get sid
ssh <user>@<tailscale-ip> "icacls C:\Users\$WIN_USER\.ssh /inheritance:r /grant *S-1-5-18:(OI)(CI)F /grant *<USER-SID>:(OI)(CI)F"
ssh <user>@<tailscale-ip> "icacls C:\Users\$WIN_USER\.ssh\authorized_keys /inheritance:r /grant *S-1-5-18:(F) /grant *<USER-SID>:(F)"
```

Проверка:
```bash
ssh -o BatchMode=yes <user>@<tailscale-ip> "echo ok"
```

### 3. Добавить в `.env`

```dotenv
TS_<NAME>=<tailscale-ip>
TS_<NAME>_USER=<windows-username>
TS_<NAME>_LABEL=<метка-в-статусе>
```

### 4. Клонировать репозиторий

#### 4a. PortableGit (если git не установлен)

> **Важно:** `Invoke-WebRequest` на новых Windows блокируется политикой безопасности.  
> Использовать `curl.exe` (встроен в Windows 10+).

```powershell
New-Item -ItemType Directory -Force C:\Work | Out-Null
curl.exe -L -o C:\Work\PortableGit.exe `
    "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/PortableGit-2.47.1-64-bit.7z.exe"
Start-Process -Wait "C:\Work\PortableGit.exe" -ArgumentList "-o `"C:\Work\git`" -y"
& "C:\Work\git\cmd\git.exe" --version   # проверка
```

**Добавить в User PATH** (чтобы `git` работал коротко, без полного пути; только для evelina, без прав админа):
```powershell
$p = [Environment]::GetEnvironmentVariable('Path','User')
$parts = @($p -split ';' | Where-Object { $_ -ne '' })          # @() — обязателен, иначе скаляр склеится
if ($parts -notcontains 'C:\Work\git\cmd') {
    [Environment]::SetEnvironmentVariable('Path', (@($parts) + 'C:\Work\git\cmd' -join ';'), 'User')
}
```
> Применяется в **новых** сессиях (sshd читает реестр при входе). Проверка из новой SSH-сессии: `ssh <user>@<ip> "git --version"`.

#### 4b. SSH-ключ для GitHub

С Linux-сервера скопировать приватный ключ (`~/repos/gh` — специальный ключ для GitHub):
```bash
scp ~/repos/gh <user>@<tailscale-ip>:'C:\Users\<username>\.ssh\gh'
```

Создать `C:\Users\<username>\.ssh\config` (через SSH с сервера):
```bash
ssh <user>@<tailscale-ip> "powershell -Command \
    \"Set-Content \$env:USERPROFILE\.ssh\config (\`\"Host github.com\`\`n    IdentityFile \$env:USERPROFILE\.ssh\gh\`\`n    StrictHostKeyChecking accept-new\`\")\""
```

Проверка доступа к GitHub:
```bash
ssh -o BatchMode=yes <user>@<tailscale-ip> \
    '"C:\Work\git\cmd\git.exe" ls-remote git@github.com:tony20202021/video.git HEAD 2>&1'
```

#### 4c. Клонирование ветки `develop`

> **Важно:** `main` — пустая ветка (только `.gitignore`). Весь код на `develop`.

```bash
ssh -o BatchMode=yes <user>@<tailscale-ip> \
    '"C:\Work\git\cmd\git.exe" -c core.compression=0 clone --depth=1 git@github.com:tony20202021/video.git "C:\Work\video" 2>&1'

ssh -o BatchMode=yes <user>@<tailscale-ip> 'powershell -NoProfile -Command "
    Set-Location C:\Work\video
    & \"C:\Work\git\cmd\git.exe\" fetch origin develop --depth=1
    & \"C:\Work\git\cmd\git.exe\" checkout -b develop FETCH_HEAD
    & \"C:\Work\git\cmd\git.exe\" log --oneline -1
"'
```

### 5. Скопировать `.env`

```bash
scp /home/tony/repos/video/.env <user>@<tailscale-ip>:'C:\Work\video\.env'
```

### 6. Проверить статус

```bash
bash sh/status/status.sh
```

Машина должна появиться в секции `WINDOWS (<метка> / <ip>)` со статусом скриптов.

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
