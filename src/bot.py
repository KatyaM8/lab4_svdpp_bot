import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message

from config import TELEGRAM_TOKEN


from load_data import load_movielens_100k
from svdpp_model import SVDPPModel, SVDPPConfig

# ===== Настройки =====
MIN_RATINGS = 5 #мин колво оценок, после которых бот может начать что-то рекомендовать
PICK_N = 5 #сколько фильмов предложим пользователю для оценки
RECS_N = 5

# ===== Глобальные переменные =====
MODEL: SVDPPModel | None = None
TITLES: Dict[int, str] = {}  #словарь movie_id → название

# telegram_user_id -> list[(telegram_user_id, item_id, rating)]
USER_RATINGS: Dict[int, List[Tuple[int, int, float]]] = {}

# telegram_user_id -> set[item_id] (чтобы не предлагать одно и то же)
USER_SUGGESTED: Dict[int, set[int]] = {}

router = Router()


def title_of(item_id: int) -> str:
    """
    ВХОД:
        item_id (int) — идентификатор фильма
    ЧТО ДЕЛАЕТ:
        Пытается получить название фильма из словаря TITLES по item_id.
        Если названия нет, формирует строку-заглушку.
    ВЫХОД:
        str — название фильма или строка "movie_id=<id>"
    """
    return TITLES.get(item_id, f"movie_id={item_id}")


def ensure_user(tg_user_id: int) -> None:
    """
    ВХОД:
        tg_user_id (int) — Telegram ID пользователя
    ЧТО ДЕЛАЕТ:
        Проверяет, есть ли пользователь в словарях USER_RATINGS и USER_SUGGESTED.
        Если пользователя нет, создаёт:
            - пустой список для хранения его оценок
            - пустое множество для хранения уже предложенных фильмов
    ВЫХОД:
        None — функция ничего не возвращает
    """
    USER_RATINGS.setdefault(tg_user_id, [])
    USER_SUGGESTED.setdefault(tg_user_id, set())


def pick_movies_for_user(tg_user_id: int, n: int) -> List[int]:
    """
    ВХОД:
        tg_user_id (int) — Telegram ID пользователя
        n (int) — количество фильмов, которые нужно предложить пользователю

    ЧТО ДЕЛАЕТ:
        Гарантирует, что пользователь инициализирован в памяти бота.
        Случайным образом выбирает фильмы из общего списка фильмов,
        исключая те, которые уже были предложены данному пользователю ранее.
        Выбранные фильмы помечаются как уже предложенные.

    ВЫХОД:
        List[int] — список идентификаторов (movie_id) выбранных фильмов
    """
    ensure_user(tg_user_id)
    all_ids = list(TITLES.keys())
    random.shuffle(all_ids)

    picked: List[int] = []
    for mid in all_ids:
        if mid in USER_SUGGESTED[tg_user_id]:
            continue
        picked.append(mid)  #добавляем фильм в список выбранных
        USER_SUGGESTED[tg_user_id].add(mid)  #запоминаем, что этот фильм уже показан пользователю
        if len(picked) >= n:
            break
    return picked


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """
    ВХОД:
        message (Message) — сообщение от пользователя с командой /start

    ЧТО ДЕЛАЕТ:
        Обрабатывает команду /start.
        Получает Telegram ID пользователя, инициализирует его в системе
        и отправляет приветственное сообщение с описанием возможностей бота
        и списком доступных команд.

    ВЫХОД:
        None — функция ничего не возвращает, но отправляет сообщение пользователю
    """
    tg_user_id = message.from_user.id
    ensure_user(tg_user_id)

    await message.answer(
        "Привет! Я бот с моделью SVD++.\n\n"
        f"Я НЕ рекомендую, пока ты не оценишь минимум {MIN_RATINGS} фильмов.\n"
        "Команды:\n"
        "  /pick — предложу фильмы для оценки\n"
        "  /rate <movie_id> <rating 1..5> — поставить оценку\n"
        "  /status — сколько оценок уже есть\n"
        "  /recommend — рекомендации (после 5 оценок)\n"
    )


@router.message(Command("pick"))
async def cmd_pick(message: Message) -> None:
    """
    ВХОД:
        message (Message) — сообщение от пользователя с командой /pick

    ЧТО ДЕЛАЕТ:
        Обрабатывает команду /pick.
        Выбирает случайный набор фильмов, которые пользователь ещё не оценивал,
        и отправляет их пользователю для выставления оценок.

    ВЫХОД:
        None — функция ничего не возвращает, но отправляет сообщение пользователю
    """
    tg_user_id = message.from_user.id
    movies = pick_movies_for_user(tg_user_id, PICK_N)

    if not movies:
        await message.answer("Не могу подобрать фильмы (проверь, что titles загрузились).")
        return

    lines = ["Оцени любые фильмы из списка (1..5). Пример: /rate 50 4\n"] #список строк будущего сообщения
    for mid in movies:
        lines.append(f"{mid} — {title_of(mid)}")
    await message.answer("\n".join(lines))


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    tg_user_id = message.from_user.id
    ensure_user(tg_user_id) #если пользователь новый, создаём для него пустые структуры
    cnt = len(USER_RATINGS[tg_user_id]) #считаем количество оценок, которые пользователь уже поставил
    await message.answer(f"У тебя {cnt} оценок. Нужно минимум {MIN_RATINGS}.")


@router.message(Command("rate"))
async def cmd_rate(message: Message) -> None:
    global MODEL
    tg_user_id = message.from_user.id
    ensure_user(tg_user_id)

    if MODEL is None or not MODEL.fitted:
        await message.answer("Модель ещё не готова. Попробуй позже.")
        return

    parts = message.text.strip().split()
    if len(parts) != 3:
        await message.answer("Формат: /rate <movie_id> <rating 1..5>")
        return

    try:
        movie_id = int(parts[1])
        rating = float(parts[2])
    except ValueError:
        await message.answer("movie_id должен быть int, rating — число 1..5.")
        return

    if not (1.0 <= rating <= 5.0):
        await message.answer("rating должен быть от 1 до 5.")
        return

    if movie_id not in MODEL.item2idx:
        await message.answer("Я не знаю такой movie_id (нет в датасете). Возьми фильм из /pick.")
        return

    # сохраняем оценку
    USER_RATINGS[tg_user_id].append((tg_user_id, movie_id, rating))

    # обновляем историю N(u) в модели
    MODEL.add_user_rating(tg_user_id, movie_id, rating)

    cnt = len(USER_RATINGS[tg_user_id])
    await message.answer(f"Ок! {title_of(movie_id)} = {rating}. Всего оценок: {cnt}.")

    if cnt == MIN_RATINGS:
        MODEL.finetune_user(tg_user_id, USER_RATINGS[tg_user_id], n_epochs=10)
        await message.answer("Супер! 5 оценок есть — профиль донастроен. Теперь /recommend.")
    elif cnt < MIN_RATINGS:
        await message.answer(f"Ещё {MIN_RATINGS - cnt} оценок. /pick")


@router.message(Command("recommend"))
async def cmd_recommend(message: Message) -> None:
    global MODEL
    tg_user_id = message.from_user.id
    ensure_user(tg_user_id)

    if MODEL is None or not MODEL.fitted:
        await message.answer("Модель ещё не готова. Попробуй позже.")
        return

    if len(USER_RATINGS[tg_user_id]) < MIN_RATINGS:
        await message.answer(
            f"Нельзя рекомендовать без {MIN_RATINGS} оценок. Сейчас: {len(USER_RATINGS[tg_user_id])}.\n"
            "Используй /pick и /rate."
        )
        return

    recs = MODEL.recommend_for_user(tg_user_id, n=RECS_N)

    lines = ["Твои рекомендации:"]
    for mid, score in recs:
        lines.append(f"{mid} — {title_of(int(mid))} (pred={score:.2f})")
    await message.answer("\n".join(lines))


async def main() -> None:
    global MODEL, TITLES
    print("Bot is running...")

    token = TELEGRAM_TOKEN

    # Загружаем MovieLens
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"  
    data = load_movielens_100k(data_dir)
    TITLES = data.titles


    cfg = SVDPPConfig(
        n_factors=50,
        n_epochs=20,
        lr=0.01,
        reg=0.02,
        verbose=True,
        clip_min=1.0,
        clip_max=5.0,
        seed=42
    )
    MODEL = SVDPPModel(cfg).fit(data.ratings)

    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)

    print("Bot is running...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
