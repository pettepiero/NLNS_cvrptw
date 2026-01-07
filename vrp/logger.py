import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
    max_bytes: int = 5_000_000,  # ~5MB
    backup_count: int = 3,
    to_console: bool = False,
) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))

    root = logging.getLogger()
    root.setLevel(level)

    for h in root.handlers[:]:
        root.removeHandler(h)

    root.addHandler(file_handler)

    if to_console:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(console)
