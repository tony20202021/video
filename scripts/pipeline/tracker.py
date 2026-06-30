"""IoU-трекер для связывания детекций людей между кадрами.

Не требует внешних ML-моделей — только numpy.
Используется скриптом 5_track_direction.py.

Алгоритм:
  - Каждый кадр: сопоставить новые боксы с активными треками по IoU
  - Треки без совпадений «стареют»; если age > max_age — трек завершён
  - Новые боксы без совпадений → новые треки
  - Только треки с hits >= min_hits считаются подтверждёнными
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    age: int = 0                    # кадров без совпадения подряд
    hits: int = 1                   # всего совпадений
    positions: list[tuple] = field(default_factory=list)   # [(cx, cy), ...]
    frame_ids: list[str]  = field(default_factory=list)    # имена кадров

    @property
    def start_center(self) -> tuple[float, float] | None:
        return self.positions[0] if self.positions else None

    @property
    def end_center(self) -> tuple[float, float] | None:
        return self.positions[-1] if self.positions else None


class IoUTracker:
    """Простой IoU-трекер без Kalman-фильтра.

    Args:
        iou_threshold: минимальный IoU для совпадения бокса с треком
        max_age:       сколько кадров трек живёт без совпадений
        min_hits:      минимум совпадений для «подтверждённого» трека
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_age: int = 5,
        min_hits: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id: int = 0
        self.finished_tracks: list[Track] = []

    def update(
        self,
        frame_id: str,
        bboxes: list[list[float]],
    ) -> list[Track]:
        """Обновить треки новыми боксами из кадра frame_id.

        Returns:
            Список активных подтверждённых треков после обновления.
        """
        # Сопоставление: greedy по убыванию IoU
        matched_track_ids: set[int] = set()
        matched_bbox_ids:  set[int] = set()

        if self._tracks and bboxes:
            for bi, bbox in enumerate(bboxes):
                best_iou, best_ti = 0.0, -1
                for ti, track in enumerate(self._tracks):
                    if ti in matched_track_ids:
                        continue
                    score = _iou(track.bbox, bbox)
                    if score > best_iou:
                        best_iou, best_ti = score, ti
                if best_iou >= self.iou_threshold:
                    matched_track_ids.add(best_ti)
                    matched_bbox_ids.add(bi)
                    t = self._tracks[best_ti]
                    t.bbox = bbox
                    t.age = 0
                    t.hits += 1
                    t.positions.append(_center(bbox))
                    t.frame_ids.append(frame_id)

        # Новые треки из несопоставленных боксов
        for bi, bbox in enumerate(bboxes):
            if bi not in matched_bbox_ids:
                t = Track(
                    track_id=self._next_id,
                    bbox=bbox,
                    positions=[_center(bbox)],
                    frame_ids=[frame_id],
                )
                self._next_id += 1
                self._tracks.append(t)

        # Старение несопоставленных треков
        alive: list[Track] = []
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids and track.hits > 0:
                track.age += 1
            if track.age > self.max_age:
                self.finished_tracks.append(track)
            else:
                alive.append(track)
        self._tracks = alive

        return [t for t in self._tracks if t.hits >= self.min_hits]

    def flush(self) -> list[Track]:
        """Завершить все оставшиеся активные треки (конец последовательности)."""
        self.finished_tracks.extend(self._tracks)
        done = self._tracks[:]
        self._tracks = []
        return [t for t in done if t.hits >= self.min_hits]

    def all_confirmed_tracks(self) -> list[Track]:
        """Все подтверждённые треки: завершённые + ещё активные."""
        return [
            t for t in self.finished_tracks + self._tracks
            if t.hits >= self.min_hits
        ]
