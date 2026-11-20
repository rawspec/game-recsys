"""Вставляет таблицу метрик из data/results/metrics.md в README.md и README.ru.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = (ROOT / "data" / "results" / "metrics.md").read_text(encoding="utf-8").splitlines()[2:]

for readme, marker in (("README.md", "<!-- METRICS_TABLE -->"),
                       ("README.ru.md", "<!-- METRICS_TABLE_RU -->")):
    path = ROOT / readme
    text = path.read_text(encoding="utf-8")
    assert marker in text, f"нет маркера в {readme}"
    text = text.replace(marker, "\n".join(rows))
    path.write_text(text, encoding="utf-8")
    print(f"{readme}: таблица вставлена ({len(rows)} строк)")
