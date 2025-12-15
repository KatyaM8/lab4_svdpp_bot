from dotenv import load_dotenv
import os

# загрузка .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if TELEGRAM_TOKEN is None:
    raise RuntimeError("TELEGRAM_TOKEN not found in .пппп")
