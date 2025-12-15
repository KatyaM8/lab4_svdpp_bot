from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

Rating = Tuple[int, int, float]  # (user_id, item_id, rating)


@dataclass
class MovieLensData:
    ratings: List[Rating]
    titles: Dict[int, str]


def load_movielens_100k(folder: Path) -> MovieLensData:
    """
    ВХОД:
      folder: Path — путь до папки, где лежат u.data и u.item
    ЧТО ДЕЛАЕТ:
      Загружает датасет MovieLens 100k: список рейтингов и названия фильмов
    ВОЗВРАЩАЕТ:
      MovieLensData — ratings и titles (словарь item_id -> title)
    """
    data_file: Path = folder / "u.data"
    item_file: Path = folder / "u.item"

    if not data_file.exists():
        raise FileNotFoundError(f"Не найден файл {data_file}")

    ratings: List[Rating] = []
    with data_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts: List[str] = line.strip().split("\t")
            if len(parts) < 3:
                continue
            user_id: int = int(parts[0])
            item_id: int = int(parts[1])
            rating: float = float(parts[2])
            ratings.append((user_id, item_id, rating))

    titles: Dict[int, str] = {}
    if item_file.exists():
        with item_file.open("r", encoding="latin-1", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                item_id = int(parts[0])
                title = parts[1]
                titles[item_id] = title

    return MovieLensData(ratings=ratings, titles=titles)


def build_proxy_implicit(ratings: List[Rating]) -> Dict[int, List[int]]:
    """
    ВХОД:
      ratings: List[Rating] — список (user_id, item_id, rating)
    ЧТО ДЕЛАЕТ:
      Строит proxy-неявный фидбэк N(u), если нет данных просмотров/кликов.
      Тут N(u) = список item_id, которые пользователь оценивал (факт взаимодействия).
      Значение рейтинга НЕ используется, только факт, что рейтинг существует.
    ВОЗВРАЩАЕТ:
      Dict[int, List[int]] — user_id -> отсортированный список item_id без повторов
    """

    tmp: Dict[int, set[int]] = {}
    for u, i, _r in ratings:
        if u not in tmp:
            tmp[u] = set()
        tmp[u].add(i)

    result: Dict[int, List[int]] = {}
    for u, items in tmp.items():
        result[u] = sorted(list(items))

    return result
