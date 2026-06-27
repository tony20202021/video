"""Общие утилиты детекции движения: фильтрация URL, метрика diff, проверка кадра."""

from __future__ import annotations

import os
import re
import threading
import time as _time_module

import cv2
import numpy as np

# Обе формы передачи учётных данных в URL: query (?...&password=...) и basic-auth (user:pass@host)
_RE_PASSWORD_QUERY = re.compile(r"(?i)(password=)[^&/\s'\"]*")
_RE_USERINFO = re.compile(r"(://)[^:@/\s]+:[^@/\s]+@")


def redact_url(text: str) -> str:
    """Маскирует пароль в строке (URL или сообщение об ошибке): и password=…, и user:pass@host."""
    if not text:
        return text
    text = _RE_PASSWORD_QUERY.sub(r"\1***", text)
    text = _RE_USERINFO.sub(r"\1***:***@", text)
    return text


def skip_url(url: str) -> bool:
    u = (url or "").strip()
    return not u or u.startswith("#") or ("<" in u and ">" in u) or not u.lower().startswith("rtsp://")


def ffmpeg_capture_options(
    *,
    use_tcp: bool,
    stimeout_us: int,
    rtbufsize: int = 1_048_576,  # 1 МБ — ограничивает compressed input ring-buffer FFMPEG
) -> str:
    parts: list[str] = [f"stimeout;{stimeout_us}", f"rtbufsize;{rtbufsize}"]
    if use_tcp:
        parts.insert(0, "rtsp_transport;tcp")
    return "|".join(parts)


def stem_from_var(var_name: str) -> str:
    return var_name.replace("_URL", "").lower()


def prepare_gray(frame: np.ndarray, width: int = 0) -> np.ndarray:
    # width ignored: cv2.norm(NORM_L1) быстрее resize+norm, ресайз контрпродуктивен
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def mean_abs_diff(prev: np.ndarray, cur: np.ndarray) -> float:
    return cv2.norm(prev, cur, cv2.NORM_L1) / prev.size


def frame_decode_plausible(
    bgr: np.ndarray,
    *,
    min_laplacian_var: float,
    min_gray_std: float,
) -> bool:
    if bgr is None or bgr.size == 0:
        return False
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if float(cv2.Laplacian(g, cv2.CV_64F).var()) < min_laplacian_var:
        return False
    if float(g.std()) < min_gray_std:
        return False

    h, w = g.shape[:2]

    # Горизонтальные полосы: ловим верхние/нижние серые бэнды
    if h >= 16:
        strip_h = max(1, h // 4)
        for i in range(4):
            strip = g[i * strip_h: min(h, (i + 1) * strip_h), :]
            if strip.size > 0 and float(strip.std()) < min_gray_std:
                return False

    # HEVC-специфичный артефакт: недекодированные блоки заполняются
    # YCbCr(128,128,128) → RGB(128,128,128) — точный нейтральный серый.
    # Используем строгий допуск ±6 и узкий диапазон [118,138] чтобы не
    # отбрасывать ночные сцены с тёплым освещением (R>G>B, но не нейтрально).
    if bgr.ndim == 3:
        b_ch = bgr[:, :, 0].astype(np.int16)
        g_ch = bgr[:, :, 1].astype(np.int16)
        r_ch = bgr[:, :, 2].astype(np.int16)
        neutral = (
            (np.abs(r_ch - g_ch) < 6) &
            (np.abs(g_ch - b_ch) < 6) &
            (g_ch > 118) &
            (g_ch < 138)
        )
        if float(neutral.mean()) > 0.35:
            return False

    return True


def read_first_plausible_frame(
    cap: cv2.VideoCapture,
    *,
    max_attempts: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    for _ in range(max_attempts):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0 and frame_decode_plausible(
            f, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std
        ):
            return f
    return None


def pick_frame_to_save(
    cap: cv2.VideoCapture,
    first_bgr: np.ndarray,
    *,
    max_extra_reads: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    if frame_decode_plausible(first_bgr, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
        return first_bgr
    for _ in range(max_extra_reads):
        ok, f = cap.read()
        if not ok or f is None or f.size == 0:
            continue
        if frame_decode_plausible(f, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
            return f
    return None


def drain_cap_buffer(cap: cv2.VideoCapture, max_drain: int = 16) -> None:
    """Дренирует накопившиеся кадры FFMPEG-буфера RTSP (FIFO).

    CAP_PROP_BUFFERSIZE=1 устанавливается при открытии потока (open_cap),
    поэтому буфер обычно мал. Дренируем не более max_drain кадров без
    таймаутов — тяжёлый timing-drain вешал процесс если поток умер
    (первый grab() мог блокироваться до stimeout = несколько секунд).
    """
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(max_drain):
        if not cap.grab():
            break


def read_hi_save_frame(
    cap: cv2.VideoCapture,
    *,
    max_extra_reads: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    """Дренирует буфер HI-потока и возвращает свежий годный кадр.

    После drain буфер может оказаться пустым (кадр ещё не пришёл от камеры
    на 12fps). Делаем до 3 попыток с паузой ~100мс между ними.
    """
    import time as _time
    drain_cap_buffer(cap)
    ok, f = None, None
    for _ in range(3):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            break
        _time.sleep(0.1)  # ждём следующий кадр (12fps ≈ 83мс)
    if not ok or f is None or f.size == 0:
        return None
    return pick_frame_to_save(
        cap, f,
        max_extra_reads=max_extra_reads,
        min_laplacian_var=min_laplacian_var,
        min_gray_std=min_gray_std,
    )


def reopen_cap(
    url: str,
    caps_by_url: dict,
    *,
    open_timeout_ms: int,
    read_timeout_ms: int,
    pause_sec: float = 2.0,
) -> bool:
    """Закрывает старый и открывает новый VideoCapture для url.

    Возвращает True если переподключение успешно. При неудаче url
    удаляется из caps_by_url — основной цикл пропустит его.
    """
    import time as _t
    old = caps_by_url.pop(url, None)
    if old is not None:
        try:
            old.release()
        except Exception:
            pass
    _t.sleep(pause_sec)
    cap = open_cap(url, open_timeout_ms=open_timeout_ms, read_timeout_ms=read_timeout_ms)
    if cap is None:
        return False
    caps_by_url[url] = cap
    return True


def open_cap(
    url: str,
    *,
    open_timeout_ms: int,
    read_timeout_ms: int,
) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(open_timeout_ms))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(read_timeout_ms))
        # Ограничиваем буфер одним кадром — меньше устаревших данных
        # и быстрее drain при необходимости
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        cap.release()
        return None
    return cap


class StreamReader:
    """Фоновый поток, непрерывно читающий один RTSP URL.

    Главный цикл вызывает get_latest() — **никогда не блокируется**.
    Поток сам переподключается при обрыве.

    Использование:
        reader = StreamReader(url, open_timeout_ms=10000, read_timeout_ms=5000)
        reader.start()
        ...
        frame, ts = reader.get_latest()   # мгновенно
        if frame is not None and reader.age_sec() < 3.0:
            ...  # используем кадр
        ...
        reader.stop()
    """

    def __init__(
        self,
        url: str,
        *,
        open_timeout_ms: int,
        read_timeout_ms: int,
        min_laplacian_var: float = 12.0,
        min_gray_std: float = 2.5,
        reconnect_pause_sec: float = 2.0,
        scale: float = 1.0,
        extract_cam_ts: bool = False,
        decode_max_fps: float = 0.0,
    ) -> None:
        self._url = url
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._min_laplacian_var = min_laplacian_var
        self._min_gray_std = min_gray_std
        self._reconnect_pause = reconnect_pause_sec
        self._scale = scale  # <1.0 → уменьшить кадр перед хранением
        self._extract_cam_ts = extract_cam_ts
        # decode_max_fps > 0: после каждого годного кадра спим, ограничивая декодирование
        self._decode_min_interval = 1.0 / decode_max_fps if decode_max_fps > 0 else 0.0

        self._lock = threading.Lock()
        self._good_frame: np.ndarray | None = None
        self._good_ts: float = 0.0
        self._good_cam_ts: "datetime | None" = None  # время из OSD камеры
        self._rtcp_calib: "object | None" = None      # RtcpCalibration | None

        self._running = False
        self._thread: threading.Thread | None = None
        self._rtcp_thread: "threading.Thread | None" = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"sr-{self._url[-25:]}",
        )
        self._thread.start()
        # Фоновый поток для RTCP NTP калибровки
        self._rtcp_thread = threading.Thread(
            target=self._run_rtcp, daemon=True,
            name=f"rtcp-{self._url[-20:]}",
        )
        self._rtcp_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=6)

    def get_latest(self) -> tuple[np.ndarray | None, float]:
        """Возвращает (frame, monotonic_timestamp). Никогда не блокируется."""
        with self._lock:
            return self._good_frame, self._good_ts

    def get_cam_ts(self) -> "datetime | None":
        """Время камеры из OSD (None если extract_cam_ts=False или не распознано)."""
        with self._lock:
            return self._good_cam_ts

    def get_rtcp_calib(self) -> "object | None":
        """RtcpCalibration если получена, иначе None."""
        with self._lock:
            return self._rtcp_calib

    def age_sec(self) -> float:
        """Секунд с момента последнего годного кадра."""
        with self._lock:
            ts = self._good_ts
        return _time_module.monotonic() - ts if ts > 0 else float("inf")

    def _run(self) -> None:
        while self._running:
            cap = open_cap(
                self._url,
                open_timeout_ms=self._open_timeout_ms,
                read_timeout_ms=self._read_timeout_ms,
            )
            if cap is None:
                _time_module.sleep(self._reconnect_pause)
                continue
            try:
                fail_streak = 0
                implausible_streak = 0
                while self._running:
                    _t0 = _time_module.monotonic()
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        fail_streak = 0
                        if frame_decode_plausible(
                            frame,
                            min_laplacian_var=self._min_laplacian_var,
                            min_gray_std=self._min_gray_std,
                        ):
                            implausible_streak = 0
                            if self._scale != 1.0:
                                h, w = frame.shape[:2]
                                frame = cv2.resize(
                                    frame,
                                    (max(1, int(w * self._scale)), max(1, int(h * self._scale))),
                                    interpolation=cv2.INTER_AREA,
                                )
                            # Извлекаем время камеры из OSD
                            if self._extract_cam_ts:
                                try:
                                    from common.utils.osd_time import extract_osd_time
                                    cam_ts = extract_osd_time(frame)
                                except Exception:
                                    cam_ts = None
                            else:
                                cam_ts = None
                            with self._lock:
                                self._good_frame = frame
                                self._good_ts = _time_module.monotonic()
                                if cam_ts is not None:
                                    self._good_cam_ts = cam_ts
                            # Ограничение частоты декодирования: спим остаток интервала.
                            # Во время сна буфер VideoCapture (размер=1) накапливает
                            # последний кадр — при следующем cap.read() получим свежий кадр.
                            if self._decode_min_interval > 0:
                                _elapsed = _time_module.monotonic() - _t0
                                _sleep = self._decode_min_interval - _elapsed
                                if _sleep > 0:
                                    _time_module.sleep(_sleep)
                        else:
                            # ok=True но кадр битый (HEVC после обрыва/сна).
                            # fail_streak не поможет — нужен отдельный счётчик.
                            implausible_streak += 1
                            if implausible_streak >= 40:
                                break  # принудительное переподключение
                    else:
                        fail_streak += 1
                        if fail_streak >= 5:
                            break  # поток умер — переподключаемся
                        _time_module.sleep(0.05)
            finally:
                cap.release()
            if self._running:
                _time_module.sleep(self._reconnect_pause)

    def _run_rtcp(self) -> None:
        """Фоновый поток: получает RTCP SR и обновляет калибровку периодически."""
        try:
            from common.utils.rtcp_time import RtcpTimingReader
        except ImportError:
            return

        reader = RtcpTimingReader(self._url)
        while self._running:
            calib = reader.get_calibration(timeout=10.0)
            if calib is not None:
                with self._lock:
                    self._rtcp_calib = calib
            # Обновляем калибровку раз в 5 минут (RTCP SR приходят ~каждые 5 сек)
            _time_module.sleep(300 if calib else 30)
