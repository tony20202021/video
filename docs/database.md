# База данных (MongoDB)

## Хранение изображений

Изображения хранятся **на диске**, в MongoDB — только пути.

```
/data/images/
  events/         YYYY/MM/DD/<camera_id>/<timestamp>_<bbox_id>.jpg
  persons/        <person_id>/img_<NNN>.jpg
  classes/        <class_label>/img_<NNN>.jpg
  unclassified/   YYYY/MM/DD/<timestamp>.jpg
```

---

## Коллекция `events`

| Поле                | Тип        | Описание                                          |
|---------------------|------------|---------------------------------------------------|
| `_id`               | ObjectId   | ID события                                        |
| `timestamp`         | DateTime   | Дата и время                                      |
| `camera_id`         | String     | ID камеры                                         |
| `frame_group_id`    | String     | Группирует людей из одного кадра                  |
| `image_path`        | String     | Путь к кадру (основной поток)                     |
| `bbox`              | [Int]      | Bounding box человека [x1, y1, x2, y2]            |
| `group_class`       | String     | `1_resident` / `2_delivery` / `3_utilities` / `4_guest` |
| `group_confidence`  | Float      | Уверенность классификации группы                  |
| `person_id`         | String?    | ID жителя (null если не определён)                |
| `person_confidence` | Float?     | Уверенность идентификации                         |
| `identify_method`   | String?    | face / body                                       |
| `event_type`        | String     | single_pass / group_pass                          |
| `model_version`     | String     | Версия набора моделей                             |

---

## Коллекция `persons`

| Поле            | Тип       | Описание                                          |
|-----------------|-----------|---------------------------------------------------|
| `_id`           | ObjectId  |                                                   |
| `person_id`     | String    | Уникальный ID (например, `p_0042`)                |
| `name`          | String?   | Имя (опционально)                                 |
| `apartment_id`  | String?   | Номер квартиры (для группировки семей)            |
| `image_paths`   | [String]  | Эталонные фото на диске                           |
| `embeddings`    | [[Float]] | Предвычисленные face/body embeddings              |
| `created_at`    | DateTime  |                                                   |
| `updated_at`    | DateTime  |                                                   |

Предвычисленные embeddings — при идентификации только cosine similarity, модель не запускается повторно.

---

## Коллекция `class_samples`

Размеченные образцы для обучения классификатора групп.

| Поле          | Тип      | Описание                              |
|---------------|----------|---------------------------------------|
| `_id`         | ObjectId |                                       |
| `image_path`  | String   |                                       |
| `class_label` | String   | `1_resident` / `2_delivery` / `3_utilities` / `4_guest` |
| `person_id`   | String?  | Привязка к жителю                     |
| `added_at`    | DateTime |                                       |
| `source`      | String   | manual / from_unclassified            |

---

## Коллекция `unclassified_persons`

Люди, которых модель не смогла идентифицировать — для дообучения.

| Поле                     | Тип      | Описание                                   |
|--------------------------|----------|--------------------------------------------|
| `_id`                    | ObjectId |                                            |
| `timestamp`              | DateTime |                                            |
| `camera_id`              | String   |                                            |
| `image_path`             | String   |                                            |
| `group_class`            | String?  | Класс группы (если определился)            |
| `group_confidence`       | Float?   |                                            |
| `best_person_confidence` | Float    | Макс. уверенность при попытке идентификации|
| `reviewed`               | Bool     | Проверено вручную                          |
| `assigned_person_id`     | String?  | Назначен вручную после просмотра           |

---

## Коллекция `model_versions`

| Поле                     | Тип      | Описание                             |
|--------------------------|----------|--------------------------------------|
| `_id`                    | ObjectId |                                      |
| `model_type`             | String   | detect / classify / identify         |
| `version`                | String   | Например, `1.3.0`                    |
| `file_path`              | String   | Путь к `.onnx` файлу                 |
| `is_active`              | Bool     |                                      |
| `trained_at`             | DateTime |                                      |
| `metrics`                | Object   | accuracy, precision, recall, f1      |
| `training_samples_count` | Int      |                                      |

---

## Коллекция `cameras`

| Поле         | Тип      | Описание                             |
|--------------|----------|--------------------------------------|
| `_id`        | ObjectId |                                      |
| `camera_id`  | String   | Уникальный ID (напр. `cam_01`)        |
| `name`       | String   | Человекочитаемое название            |
| `rtsp_url`   | String   | Полный RTSP URL (субпоток)           |
| `location`   | String   | Расположение (этаж, подъезд)         |
| `is_active`  | Bool     |                                      |
