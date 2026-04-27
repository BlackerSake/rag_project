import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = LOG_DIR / "app.log"


def configure_logging(level: str | int | None = None, force: bool = False) -> Path:
    """配置项目日志，将日志统一写入 logs/app.log。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level or os.getenv("APP_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                DEFAULT_LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
        ],
        force=force,
    )
    return DEFAULT_LOG_FILE


configure_logging()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
