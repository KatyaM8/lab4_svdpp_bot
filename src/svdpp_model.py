
# Идея: предсказываем рейтинг r(u,i) через bias + матричную факторизацию + implicit часть

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional
import numpy as np

class UserNotFoundError(ValueError):
    pass

class NotEnoughRatingsError(ValueError):
    pass



@dataclass
class SVDPPConfig:
    # k — размер латентных векторов
    n_factors: int = 50
    # сколько раз прогоняем весь train
    n_epochs: int = 20

    # gamma — learning rate
    lr: float = 0.01
    # lambda — регуляризация (L2) (для простоты одна на всё)
    reg: float = 0.02

    # std для инициализации векторов (маленькие рандомные числа)
    init_std: float = 0.1

    seed: int = 42
    shuffle: bool = True
    verbose: bool = True

    clip_min: Optional[float] = 1.0
    clip_max: Optional[float] = 5.0


class SVDPPModel:
    """
    Модель рекомендаций SVD++
    """

    """
    SVD++ 

    Формула:
        r_hat(u,i) = mu + b_u + b_i + q_i^T ( p_u + |N(u)|^{-1/2} * sum_{j in N(u)} y_j )

    Где:
        mu  — средняя оценка по всему train
        b_u — bias пользователя
        b_i — bias item
        p_u — вектор пользователя (k)
        q_i — вектор item (k)
        y_j — implicit-вектор item (k), участвует через историю пользователя N(u)
        N(u) — множество item-ов, с которыми пользователь взаимодействовал в train

    Обновляем параметры на каждом примере SGD:
        e = r - r_hat

        b_u += lr * (e - reg*b_u)
        b_i += lr * (e - reg*b_i)

        q_i += lr * (e * x_u - reg*q_i)
        p_u += lr * (e * q_i_old - reg*p_u)

        y_j += lr * (e * |N(u)|^{-1/2} * q_i_old - reg*y_j) для всех j in N(u)
    """

    def __init__(self, cfg: Optional[SVDPPConfig] = None):
        self.cfg = cfg or SVDPPConfig() #либо исп конкретный конфиг либо по умолчанию
        self.rng = np.random.default_rng(self.cfg.seed) #Генератор случайных чисел

        # "маппинги": реальные айдишники -> индексы в массивах
        self.user2idx: Dict[Any, int] = {}
        self.item2idx: Dict[Any, int] = {}
        self.idx2user: List[Any] = []
        self.idx2item: List[Any] = []

        # параметры 
        self.mu: float = 0.0
        self.bu: Optional[np.ndarray] = None   # (n_users,)
        self.bi: Optional[np.ndarray] = None   # (n_items,)
        self.pu: Optional[np.ndarray] = None   # (n_users, k)
        self.qi: Optional[np.ndarray] = None   # (n_items, k)
        self.yj: Optional[np.ndarray] = None   # (n_items, k)

        # Nu[u] = массив item-индексов, которые встречались у пользователя u в train
        self.Nu: List[np.ndarray] = []

        self.fitted: bool = False

    # --------------------------
    # Основные методы
    # --------------------------

    def fit(self, ratings: Sequence[Tuple[Any, Any, float]]) -> "SVDPPModel":
        # ВХОД:
        #   ratings — список (user_id, item_id, rating)
        #
        # ЧТО ДЕЛАЕТ:
        #   строит маппинги пользователей и фильмов
        #   считает среднюю оценку mu
        #   инициализирует параметры модели
        #   строит историю взаимодействий N(u)
        #   обучает модель с помощью SGD
        #
        # ВЫХОД:
        #   возвращает self (обученную модель)
        if len(ratings) == 0:
            raise ValueError("ratings пустой — обучать нечего")

        # 1) строим маппинги user/item -> индекс
        self._build_mappings(ratings)

        n_users = len(self.idx2user) #ск пользователей
        n_items = len(self.idx2item) #ск фильмов
        k = self.cfg.n_factors #размер латентного вектора

        # 2) mu — глобальная средняя оценка 
        self.mu = float(np.mean([r for _, _, r in ratings]))

        # 3) инициализируем параметры
        self.bu = np.zeros(n_users, dtype=np.float64) 
        self.bi = np.zeros(n_items, dtype=np.float64)

        self.pu = self.rng.normal(0.0, self.cfg.init_std, size=(n_users, k)).astype(np.float64)
        self.qi = self.rng.normal(0.0, self.cfg.init_std, size=(n_items, k)).astype(np.float64)
        self.yj = self.rng.normal(0.0, self.cfg.init_std, size=(n_items, k)).astype(np.float64)

        # 4) строим историю N(u) по train
        self.Nu = self._build_Nu(ratings, n_users)

        # 5) переводим ratings в индексный вид 
        data = [(self.user2idx[u], self.item2idx[i], float(r)) for (u, i, r) in ratings]

        # 6) SGD по эпохам
        for epoch in range(1, self.cfg.n_epochs + 1):
            if self.cfg.shuffle:
                self.rng.shuffle(data)

            se = 0.0  # sum squared error (для rmse)

            for (u, i, r) in data:
                pred = self._predict_idx(u, i) #считаем предсказание 
                e_for_rmse = (r - pred) #считаем ошибку
                se += e_for_rmse * e_for_rmse

                # основной шаг обучения
                self._sgd_step(u, i, r) 

            rmse = float(np.sqrt(se / len(data)))
            if self.cfg.verbose:
                print(f"epoch {epoch}/{self.cfg.n_epochs} rmse={rmse:.5f}")

        self.fitted = True
        return self

    def predict_single(self, user_id: Any, item_id: Any) -> float:
        """
        Предсказание для одного (user_id, item_id).
        Если user/item неизвестны — возвращаем baseline (mu + известные bias).
        """
        if not self.fitted:
            raise RuntimeError("Сначала вызови fit()")

        base = self.mu

        # если пользователя нет — мы вообще не знаем его параметры
        if user_id not in self.user2idx:
            return self._clip(base)

        u = self.user2idx[user_id]
        base += float(self.bu[u])

        # если item нет — аналогично
        if item_id not in self.item2idx:
            return self._clip(base)

        i = self.item2idx[item_id]
        base += float(self.bi[i])

        # полный прогноз
        pred = self._predict_idx(u, i)  # там уже mu/bias тоже учитываются, но ок
        return self._clip(pred)

    def predict(self, pairs: Sequence[Tuple[Any, Any]]) -> np.ndarray:
        """
        pairs: список [(user_id, item_id), ...]
        """
        return np.array([self.predict_single(u, i) for (u, i) in pairs], dtype=np.float64)


    def _build_mappings(self, ratings: Sequence[Tuple[Any, Any, float]]) -> None:
        # Заполняем user2idx / item2idx в порядке первого появления (без сортировок).
        for (u, i, _) in ratings:
            if u not in self.user2idx:
                self.user2idx[u] = len(self.idx2user)
                self.idx2user.append(u)
            if i not in self.item2idx:
                self.item2idx[i] = len(self.idx2item)
                self.idx2item.append(i)

    def _build_Nu(self, ratings: Sequence[Tuple[Any, Any, float]], n_users: int) -> List[np.ndarray]:
        # Nu[u] = какие item индексы встречались у пользователя u
        tmp = [set() for _ in range(n_users)]
        for (u_id, i_id, _) in ratings:
            u = self.user2idx[u_id]
            i = self.item2idx[i_id]
            tmp[u].add(i)

        Nu: List[np.ndarray] = []
        for u in range(n_users):
            Nu.append(np.array(list(tmp[u]), dtype=np.int32))
        return Nu

    def _implicit_sum(self, u: int) -> Tuple[np.ndarray, float]:
        """
        Возвращает:
            s_u = |N(u)|^{-1/2} * sum_{j in N(u)} y_j
            norm = |N(u)|^{-1/2}
        """
        items = self.Nu[u]
        if items.size == 0:
            return np.zeros(self.cfg.n_factors, dtype=np.float64), 0.0

        norm = 1.0 / np.sqrt(items.size)
        su = norm * self.yj[items].sum(axis=0)
        return su, norm

    def _predict_idx(self, u: int, i: int) -> float:
        """
        Предсказание по индексам u,i
        """
        su, _ = self._implicit_sum(u)    # s_u
        xu = self.pu[u] + su             # x_u = p_u + s_u

        r_hat = self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.qi[i], xu))
        return self._clip(r_hat)

    def _sgd_step(self, u: int, i: int, r: float) -> None:
        # ВХОД:
        #   u — индекс пользователя (не user_id, а уже индекс из маппинга)
        #   i — индекс фильма (не item_id, а индекс)
        #   r — реальная оценка пользователя u фильму i
        #
        # ЧТО ДЕЛАЕТ:
        #   - считает предсказание r_hat для (u, i)
        #   - считает ошибку e = r - r_hat
        #   - делает один шаг обучения SGD:
        #       обновляет bu[u], bi[i], pu[u], qi[i]
        #       и обновляет yj[j] для всех фильмов j из истории пользователя Nu[u]
        #     (регуляризация reg уменьшает параметры, чтобы они не раздувались)
        #

        #берём настройки обучения
        lr = self.cfg.lr
        reg = self.cfg.reg

        # 1) считаем implicit часть
        Nu_items = self.Nu[u]
        if Nu_items.size > 0:
            norm = 1.0 / np.sqrt(Nu_items.size)
            su = norm * self.yj[Nu_items].sum(axis=0)
        else:
            norm = 0.0
            su = np.zeros(self.cfg.n_factors, dtype=np.float64)

        xu = self.pu[u] + su
        #pu[u] - личный вектор пользователя (его вкусы)
        #su -что говорит история

        # 2) прогноз и ошибка
        r_hat = self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.qi[i], xu))
        r_hat = self._clip(r_hat)
        e = r - r_hat  # e_ui

        # 3) сохраняем старый q_i (чтобы имплисит и p_u обновлялись "в одном времени")
        qi_old = self.qi[i].copy()

        # 4) обновляем bias'ы
        self.bu[u] += lr * (e - reg * self.bu[u])
        self.bi[i] += lr * (e - reg * self.bi[i])

        # 5) обновляем q_i и p_u
        # q_i тянем в сторону x_u (если e положит., значит надо увеличить dot)
        self.qi[i] += lr * (e * xu - reg * self.qi[i])

        # p_u тянем в сторону старого q_i
        self.pu[u] += lr * (e * qi_old - reg * self.pu[u])

        # 6) обновляем y_j для всех j из истории пользователя
        if Nu_items.size > 0:
            # общая часть градиента для каждого y_j:
            # e * |N(u)|^{-1/2} * q_i
            common = e * norm * qi_old

            # vectorized update: сразу всем y_j из Nu_items
            self.yj[Nu_items] += lr * (common - reg * self.yj[Nu_items])

    def _clip(self, x: float) -> float:
        if self.cfg.clip_min is not None:
            x = max(self.cfg.clip_min, x)
        if self.cfg.clip_max is not None:
            x = min(self.cfg.clip_max, x)
        return float(x)
    
    def recommend_for_user(self, user_id, n=5):
        # ВХОД:
        #   user_id — идентификатор пользователя, для которого мы хотим сделать рекомендации
        #   n — количество фильмов, которые нужно порекомендовать (по умолчанию 5)
        #
        # ЧТО ДЕЛАЕТ:
        #   - проверяет, обучена ли модель (если нет, вызывает ошибку)
        #   - проверяет, есть ли пользователь в модели
        #   - собирает список фильмов, которые пользователь ещё не оценивал
        #   - для каждого фильма, который не был оценён, предсказывает рейтинг
        #   - сортирует фильмы по предсказанным рейтингам
        #   - выбирает топ-N фильмов
        #   - возвращает реальные идентификаторы этих фильмов с их предсказанным рейтингом
        #
        # ВЫХОД:
        #   Список из N кортежей (item_id, score), где:
        #     - item_id - реальный идентификатор фильма
        #     - score - предсказанный рейтинг этого фильма для пользователя
        if not self.fitted:
            raise RuntimeError("fit() first")

        if user_id not in self.user2idx:
            raise ValueError("unknown user")

        u = self.user2idx[user_id]
        seen_items = set(self.Nu[u].tolist())

        scores = []
        for i in range(len(self.idx2item)):
            if i in seen_items:
                continue
            score = self._predict_idx(u, i)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:n]

        # возвращаем реальные item_id
        return [(self.idx2item[i], score) for i, score in top]
    
    def add_user(self, user_id: Any) -> int:
        # ВХОД:
        #   user_id - реальный id пользователя (например telegram id)
        #
        # ЧТО ДЕЛАЕТ:
        #   - если пользователь уже есть в маппинге user2idx, возвращает его индекс
        #   - если пользователя нет:
        #       добавляет его в user2idx и idx2user
        #       расширяет параметры пользователя:
        #         bu (добавляет 0.0)
        #         pu (добавляет новый случайный вектор)
        #       добавляет пустую историю Nu[u]
        #
        # ВЫХОД:
        #   int - индекс пользователя u (который используется в массивах numpy)
        if user_id in self.user2idx:
            return self.user2idx[user_id]

        u = len(self.idx2user)
        self.user2idx[user_id] = u
        self.idx2user.append(user_id)

        # расширяем параметры пользователя
        self.bu = np.append(self.bu, 0.0)
        new_pu = self.rng.normal(0.0, self.cfg.init_std, size=(1, self.cfg.n_factors)).astype(np.float64)
        self.pu = np.vstack([self.pu, new_pu])

        # история оценок (N(u)) пустая
        self.Nu.append(np.array([], dtype=np.int32))

        return u
    
    def add_user_rating(self, user_id: Any, item_id: Any, rating: float) -> None:
        # ВХОД:
        #   user_id — id пользователя
        #   item_id — id фильма
        #   rating — оценка (в этой версии внутри модели почти не сохраняется)
        #
        # ЧТО ДЕЛАЕТ:
        #   - проверяет, что модель уже обучена (fit() был вызван)
        #   - добавляет пользователя, если его ещё нет (через add_user)
        #   - проверяет, что фильм известен модели
        #   - добавляет фильм в историю Nu[u] (без дубликатов)
        #   - НЕ сохраняет сам рейтинг внутри модели (его хранит бот)

        if not self.fitted:
            raise RuntimeError("Сначала fit(), потом можно добавлять пользователей/оценки.")

        u = self.add_user(user_id)

        if item_id not in self.item2idx:
            raise ValueError(f"Unknown item_id={item_id}")

        i = self.item2idx[item_id]

        items = self.Nu[u]
        if items.size == 0:
            self.Nu[u] = np.array([i], dtype=np.int32)
        else:
            # не добавляем дубликаты
            if i not in set(items.tolist()):
                self.Nu[u] = np.append(items, i).astype(np.int32)

        # сами оценки мы в этой простой версии не сохраняем внутри модели
        # их будет хранить бот (в памяти/БД) и потом передаст в finetune_user()

    def can_recommend(self, user_id: Any, min_ratings: int = 5) -> bool:
        # ВХОД:
        #   user_id - id пользователя
        #   min_ratings -сколько фильмов минимум должно быть в истории пользователя
        #
        # ЧТО ДЕЛАЕТ:
        #   - проверяет, что пользователь есть в модели
        #   - проверяет, что длина истории Nu[u] >= min_ratings
        #
        # ВЫХОД:
        #   bool - True если можно рекомендовать, иначе False
        if user_id not in self.user2idx:
            return False
        u = self.user2idx[user_id]
        return len(self.Nu[u]) >= min_ratings

    def recommend_for_user(self, user_id: Any, n: int = 5, min_ratings: int = 5):
        # ВХОД:
        #   user_id - id пользователя
        #   n — сколько рекомендаций вернуть
        #   min_ratings - минимум оценённых/просмотренных фильмов перед рекомендациями
        #
        # ЧТО ДЕЛАЕТ:
        #   - проверяет, что модель обучена
        #   - проверяет, что пользователь существует
        #   - проверяет, что у пользователя достаточно истории (Nu[u])
        #   - для всех фильмов, которые пользователь ещё не видел:
        #       считает предсказанный рейтинг
        #   - сортирует по рейтингу и возвращает top-n
        #
        # ВЫХОД:
        #   list[(item_id, score)] — список рекомендаций:
        #     item_id - реальный id фильма
        #     score - предсказанная оценка
        if not self.fitted:
            raise RuntimeError("fit() first")

        if user_id not in self.user2idx:
            raise UserNotFoundError("Пользователь не найден. Сначала соберите оценки.")

        u = self.user2idx[user_id]
        if len(self.Nu[u]) < min_ratings:
            raise NotEnoughRatingsError(f"Недостаточно оценок: нужно минимум {min_ratings}")

        seen = set(self.Nu[u].tolist())
        scores = []
        for i in range(len(self.idx2item)):
            if i in seen:
                continue
            score = self._predict_idx(u, i)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:n]
        return [(self.idx2item[i], score) for i, score in top]

    def finetune_user(
        self,
        user_id: Any,
        user_ratings: List[Tuple[Any, Any, float]],
        n_epochs: int = 10,
        lr: Optional[float] = None,
        reg: Optional[float] = None,
    ) -> None:
    # ВХОД:
    #   user_id - id пользователя
    #   user_ratings - список оценок пользователя: [(user_id, item_id, rating), ...]
    #   n_epochs - сколько эпох дообучать пользователя
    #   lr - learning rate (если None, берётся из cfg)
    #   reg - регуляризация (если None, берётся из cfg)
    #
    # ЧТО ДЕЛАЕТ:
    #   - проверяет, что модель обучена
    #   - добавляет пользователя, если его ещё нет
    #   - переводит item_id в индексы и формирует prepared = [(i_idx, rating), ...]
    #   - обновляет историю Nu[u], чтобы implicit часть работала
    #   - делает несколько эпох обучения, обновляя ТОЛЬКО параметры пользователя:
    #       bu[u] и pu[u]
    #     (фильмы и yj не трогает)
    #
    # ВЫХОД:
    #   ничего (None), но изменяет bu[u] и pu[u] для этого пользователя
        if not self.fitted:
            raise RuntimeError("Сначала fit().")

        lr = lr if lr is not None else self.cfg.lr
        reg = reg if reg is not None else self.cfg.reg

        u = self.add_user(user_id)

        prepared = []
        for _u, item_id, r in user_ratings:
            if item_id not in self.item2idx:
                continue
            i = self.item2idx[item_id]
            prepared.append((i, float(r)))

            # обновим N(u), чтобы implicit часть работала
            items = self.Nu[u]
            if items.size == 0:
                self.Nu[u] = np.array([i], dtype=np.int32)
            else:
                if i not in set(items.tolist()):
                    self.Nu[u] = np.append(items, i).astype(np.int32)

        if not prepared:
            raise ValueError("Нет валидных оценок для дообучения")

        for _ in range(n_epochs):
            for i, r in prepared:
                su, _ = self._implicit_sum(u)
                xu = self.pu[u] + su
                r_hat = self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.qi[i], xu))
                r_hat = self._clip(r_hat)
                e = r - r_hat

                # обновляем ТОЛЬКО пользователя
                self.bu[u] += lr * (e - reg * self.bu[u])
                self.pu[u] += lr * (e * self.qi[i] - reg * self.pu[u])




