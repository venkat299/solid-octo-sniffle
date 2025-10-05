"""Launcher that waits for the configured LLM service before starting the app."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Callable

import httpx
import uvicorn

from job_role_analyzer import LLMEndpointConfig, load_config


logger = logging.getLogger(__name__)


def _wait_for_llm(
    target: LLMEndpointConfig,
    *,
    retry_interval: float,
    max_attempts: int,
    request_factory: Callable[[str, float], httpx.Response] | None = None,
) -> None:
    if not target.base_url:
        raise ValueError("LLM endpoint must define a base_url")

    request = request_factory or (lambda url, timeout: httpx.get(url, timeout=timeout))
    health_url = target.base_url.rstrip("/")
    attempt = 0
    while max_attempts <= 0 or attempt < max_attempts:
        attempt += 1
        try:
            response = request(health_url, target.timeout)
            if response.status_code < 500:
                logger.info("LLM service reachable at %s (status %s)", health_url, response.status_code)
                return
            logger.warning(
                "Attempt %s: LLM service at %s responded with status %s",
                attempt,
                health_url,
                response.status_code,
            )
        except httpx.HTTPError as exc:
            logger.warning("Attempt %s: unable to reach LLM service %s: %s", attempt, health_url, exc)
        time.sleep(retry_interval)

    raise RuntimeError(f"LLM service at {health_url} did not become available")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start the Job Role Analyzer UI after LLM warmup")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the FastAPI app")
    parser.add_argument("--port", type=int, default=8000, help="Port for the FastAPI app")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode")
    parser.add_argument("--llm-attempts", type=int, default=10, help="Maximum LLM connectivity attempts (0 for infinite)")
    parser.add_argument(
        "--llm-interval",
        type=float,
        default=3.0,
        help="Seconds between LLM connectivity checks",
    )
    parser.add_argument(
        "--app-target",
        default="job_role_analyzer",
        help="LLM target key to validate before startup",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config()
    llm_config = config.get_llm_config(args.app_target)

    try:
        _wait_for_llm(
            llm_config,
            retry_interval=args.llm_interval,
            max_attempts=args.llm_attempts,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error("LLM preflight failed: %s", exc)
        sys.exit(1)

    uvicorn.run(
        "webapp.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
