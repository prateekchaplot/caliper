"""Caliper command surface (§7.9, §9.1).

Subcommands land with their owning tasks per ARCHITECTURE §14.2: ingest (T07),
ask (T12), eval/report/spotcheck (T14/T15/T19), setup/serve (T22). The only
argument parsing in the project lives here (§7.9). Every subcommand maps typed
component errors to the §9 envelope exactly once at this edge (§16), prints
machine-readable JSON to stdout, and logs human prose to stderr (§11.3).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from caliper.config import ConfigError, load_config
from caliper.db import DbError
from caliper.ingest import run_ingest
from caliper.ingest.source import SourceError
from caliper.providers import ProviderError

app = typer.Typer(
    name="caliper",
    help="Local Q&A assistant over one frozen documentation set, with measured quality.",
    no_args_is_help=True,
)

# §11.2: paths and log level live in the environment, never in the config model.
_USER_CONFIG_ENV_VAR = "CALIPER_CONFIG"
_DEFAULT_USER_CONFIG = "configs/caliper.sample.toml"
_PRESETS_DIR = Path("configs")
_LOG_LEVEL_ENV_VAR = "CALIPER_LOG_LEVEL"

# §9 error-code table: the CLI exit codes this surface maps to.
_EXIT_CONFIG_INVALID = 2
_EXIT_PROVIDER_UNAVAILABLE = 4
_EXIT_QUOTA_EXHAUSTED = 5


class Version(StrEnum):
    """The three measured versions (§9.7); selects the committed preset file."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


def _configure_logging() -> None:
    """§11.3: Python logging to stderr in the standard format, level from
    CALIPER_LOG_LEVEL (default INFO). Machine-readable JSON goes to stdout."""
    level = os.environ.get(_LOG_LEVEL_ENV_VAR, "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def _print_error(code: str, message: str) -> None:
    """Print the §9 error envelope as JSON to stderr (§9)."""
    typer.echo(json.dumps({"error": {"code": code, "message": message}}), err=True)


def _map_error(exc: Exception) -> tuple[str, int]:
    """Map one typed component exception to its §9 envelope code and CLI exit
    code (§9 error-code table). `SourceError`/`DbError` have no dedicated §9
    code (recorded in docs/TASKS.md Notes): they map to `provider_unavailable`
    — an ingest whose corpus source or local store is unavailable."""
    if isinstance(exc, ConfigError):
        return "config_invalid", _EXIT_CONFIG_INVALID
    if isinstance(exc, ProviderError):
        if exc.code == "quota_exhausted":
            return "quota_exhausted", _EXIT_QUOTA_EXHAUSTED
        return "provider_unavailable", _EXIT_PROVIDER_UNAVAILABLE
    if isinstance(exc, SourceError | DbError):
        return "provider_unavailable", _EXIT_PROVIDER_UNAVAILABLE
    raise AssertionError(f"unmapped error type: {type(exc).__name__}")


@app.callback()
def main() -> None:
    """Caliper: ask questions, get cited answers, measure quality."""
    _configure_logging()


@app.command()
def ingest(
    version: Annotated[
        Version, typer.Option("--version", help="Version preset to ingest (§9.7).")
    ] = Version.V1,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-resolve and refetch the corpus even when cached.")
    ] = False,
) -> None:
    """Fetch, parse, chunk, embed, and index the frozen corpus (§10.3)."""
    try:
        cfg = load_config(
            Path(os.environ.get(_USER_CONFIG_ENV_VAR, _DEFAULT_USER_CONFIG)),
            _PRESETS_DIR / f"{version.value}.toml",
        )
        summary = run_ingest(cfg, refresh=refresh)
    except (ConfigError, ProviderError, SourceError, DbError) as exc:
        code, exit_code = _map_error(exc)
        _print_error(code, str(exc))
        raise typer.Exit(code=exit_code) from exc
    typer.echo(json.dumps(summary.as_dict()))


if __name__ == "__main__":
    app()
