"""Запускает весь пайплайн подготовки данных."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "scripts/01_download.py",
    "scripts/02_build_catalog.py",
    "scripts/03_embed.py",
    "scripts/04_train_als.py",
    "scripts/05_train_itemknn.py",
    "scripts/06_train_bpr.py",
    "scripts/07_train_twotower.py",
    "scripts/08_train_lightfm.py",   # опционально, пропускается без lightfm
    "scripts/09_evaluate.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        try:
            subprocess.run([sys.executable, str(ROOT / step)], cwd=str(ROOT), check=True)
        except subprocess.CalledProcessError as e:
            if "lightfm" in step:
                print("  пропускаю опциональный шаг (lightfm недоступен)")
                continue
            raise e
    print("\nПайплайн завершён. Запуск сервера:  python -m app.server --open")


if __name__ == "__main__":
    main()
