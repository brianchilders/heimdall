"""Entry point for the pipeline worker service.

Run with:
    python -m pipeline_worker.main
or:
    cd pipeline_worker && python main.py
"""

import uvicorn

from pipeline_worker.server import create_app
from pipeline_worker.settings import Settings


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.pipeline_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
