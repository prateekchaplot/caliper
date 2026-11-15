"""Caliper command surface (skeleton).

Subcommands land with their owning tasks per ARCHITECTURE §14.2:
ingest (T07), ask (T12), eval/report/spotcheck (T14/T15/T19), setup/serve (T22).
The only argument parsing in the project lives here (§7.9).
"""

import typer

app = typer.Typer(
    name="caliper",
    help="Local Q&A assistant over one frozen documentation set, with measured quality.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Caliper: ask questions, get cited answers, measure quality."""


if __name__ == "__main__":
    app()
