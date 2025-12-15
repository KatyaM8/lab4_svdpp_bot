# train.py
from pathlib import Path
from svdpp_model import SVDPPModel, SVDPPConfig
from load_data import load_movielens_100k


def main():
    BASE_DIR = Path(__file__).resolve().parent      # src/
    DATA_DIR = BASE_DIR.parent / "data"             # ../data

    data = load_movielens_100k(DATA_DIR)
    ratings = data.ratings

    print("num ratings:", len(ratings))

    cfg = SVDPPConfig(
        n_factors=50,
        n_epochs=20,
        lr=0.01,
        reg=0.02,
        verbose=True,
        clip_min=1.0,
        clip_max=5.0
    )

    model = SVDPPModel(cfg)
    model.fit(ratings)

    u0, i0, _ = ratings[0]
    print("example pred:", model.predict_single(u0, i0))
    u0 = model.idx2user[0]
    recs = model.recommend_for_user(u0, n=5)
    print("Recommendations:")
    for item_id, score in recs:
        print(item_id, score)



if __name__ == "__main__":
    main()
