"""Production process entry point."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from .application import ResearchApplication
from .settings import Settings
from .ui import create_ui


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    load_dotenv()
    configure_logging()
    settings = Settings.from_environment()
    application = ResearchApplication(settings)
    create_ui(application).launch(
        server_name=settings.host,
        server_port=settings.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
