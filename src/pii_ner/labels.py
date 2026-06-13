"""BIO label schema: single source of truth for label <-> id mapping."""

from __future__ import annotations

ENTITY_LABELS: list[str] = [
    "API ключи",
    "CVV/CVC",
    "Email",
    "Водительское удостоверение",
    "Временное удостоверение личности",
    "Гражданство и названия стран",
    "Данные об автомобиле клиента",
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)",
    "Дата окончания срока действия карты",
    "Дата регистрации по месту жительства или пребывания",
    "Дата рождения",
    "Имя держателя карты",
    "Кодовые слова",
    "Место рождения",
    "Наименование банка",
    "Номер банковского счета",
    "Номер карты",
    "Номер телефона",
    "Одноразовые коды",
    "ПИН код",
    "Пароли",
    "Паспортные данные",
    "Полный адрес",
    "Разрешение на работу / визу",
    "СНИЛС клиента",
    "Сведения об ИНН",
    "Свидетельство о рождении",
    "Серия и номер вида на жительство",
    "Содержимое магнитной полосы",
    "ФИО",
]

BIO_LABELS: list[str] = ["O"]
for _label in ENTITY_LABELS:
    BIO_LABELS.append(f"B-{_label}")
    BIO_LABELS.append(f"I-{_label}")

label2id: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
id2label: dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}

NUM_LABELS: int = len(BIO_LABELS)
