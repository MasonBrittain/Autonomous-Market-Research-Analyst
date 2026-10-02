"""Command line interface.

`research --stub` runs the entire pipeline with no API key and no network beyond
the source adapters, which is the fastest way to see the shape of the output and
to check that a change did not break the renderer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .cache import FetchCache
from .config import settings
from .models import RunConfig, RunStatus
from .orchestrator.pipeline import AmbiguousTarget, research
from .orchestrator.store import RunStore
from .tools import entity as entity_tools
from .tools.fetch import Fetcher

app = typer.Typer(
    add_completion=False,
    help="Autonomous market research analyst: gather evidence, build a cited brief, red-team it.",
    no_args_is_help=True,
)
console = Console()

NODE_LABELS = {
    "resolve": "Resolve",
    "scout": "Scout",
    "curate": "Librarian",
    "analyze": "Analyst",
    "challenge": "Adversary",
    "compose": "Scribe",
}


def _progress(node: str, status: str, detail: str) -> None:
    label = NODE_LABELS.get(node, node)
    if status == "start":
        console.print(f"[dim]->[/dim] [bold]{label}[/bold] [dim]working...[/dim]")
    elif status == "done":
        console.print(f"   [green]ok[/green] {label}: {detail}")
    elif status == "skipped":
        console.print(f"   [yellow]skip[/yellow] {label}: {detail}")
    elif status == "ambiguous":
        console.print(f"   [yellow]stop[/yellow] {label}: {detail}")
    else:
        console.print(f"   [red]fail[/red] {label}: {detail}")


def research_cmd(
    query: Annotated[str, typer.Argument(help="Company name, ticker, or industry")],
    stub: Annotated[
        bool, typer.Option("--stub", help="Run without an API key (offline model)")
    ] = False,
    offline: Annotated[
        bool, typer.Option("--offline", help="Cache only; fail on cache miss")
    ] = False,
    lookback: Annotated[int, typer.Option("--lookback", help="Evidence window in days")] = 120,
    model: Annotated[str, typer.Option("--model", help="Model id")] = "",
    rounds: Annotated[int, typer.Option("--rounds", help="Max scout rounds")] = 4,
    max_evidence: Annotated[int, typer.Option("--max-evidence", help="Document ceiling")] = 80,
    resume: Annotated[str, typer.Option("--resume", help="Resume an existing run id")] = "",
    out: Annotated[str, typer.Option("--out", help="Output directory")] = "",
    show: Annotated[bool, typer.Option("--show", help="Print the brief to stdout")] = False,
) -> None:
    """Produce a research brief."""
    cfg = settings()
    problem = cfg.sec_user_agent_problem()
    if problem:
        console.print(f"[red]{problem}[/red]")
        raise typer.Exit(code=1)

    config = RunConfig(
        model=model or cfg.model,
        lookback_days=lookback,
        max_scout_rounds=rounds,
        max_evidence=max_evidence,
        stub=stub or not cfg.has_api_key,
    )
    if config.stub and not stub:
        console.print(
            "[yellow]No ANTHROPIC_API_KEY found -- running in stub mode. "
            "Set it in .env for real analysis.[/yellow]"
        )

    console.print(
        Panel(
            f"[bold]{query}[/bold]\n"
            f"model={config.model}  window={lookback}d  "
            f"stub={config.stub}  offline={offline}",
            title="research",
            border_style="dim",
        )
    )

    try:
        run, stats = asyncio.run(
            research(
                query,
                config=config,
                progress=_progress,
                offline=offline,
                run_id=resume or None,
            )
        )
    except AmbiguousTarget as exc:
        console.print(f"\n[yellow]Ambiguous target.[/yellow] {exc}")
        console.print("[dim]Re-run with the ticker, or the full legal name.[/dim]")
        raise typer.Exit(code=2) from exc

    if run.status is RunStatus.AMBIGUOUS:
        console.print(f"\n[yellow]{run.error}[/yellow]")
        raise typer.Exit(code=2)

    out_dir = Path(out) if out else cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = (run.entity.ticker or run.entity.name if run.entity else run.query).replace(" ", "_")
    stem = f"{slug}_{run.id.split('_')[-1]}"
    if run.report:
        (out_dir / f"{stem}.md").write_text(run.report.markdown, encoding="utf-8")
        (out_dir / f"{stem}.html").write_text(run.report.html, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")

    _print_summary(run, stats)
    console.print(f"\n[dim]written to[/dim] {out_dir / stem}.{{md,html,json}}")
    if show and run.report:
        console.print("\n" + run.report.markdown)


def _print_summary(run, stats: dict) -> None:  # noqa: ANN001
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    published = sum(1 for c in run.claims if c.survived)
    coverage = run.latest_coverage
    rows = [
        ("run", run.id),
        ("entity", f"{run.entity.name} ({run.entity.ticker or 'unlisted'})" if run.entity else "-"),
        ("documents", f"{len(run.evidence)} gathered / {len(run.usable_evidence())} usable"),
        ("duplicates", f"{run.dedup_rate:.0%}"),
        ("facts", str(len(run.facts))),
        ("claims", f"{published} published / {len(run.claims) - published} rejected"),
        ("rejection rate", f"{run.rejection_rate:.0%}"),
        (
            "coverage",
            f"{(coverage.score if coverage else 0):.0%} over {len(run.coverage)} round(s)",
        ),
        ("llm calls", str(run.ledger.total_calls)),
        ("cache reads", f"{run.ledger.cache_read_tokens:,} tokens"),
        ("cost", f"${run.ledger.total_usd:.4f}"),
    ]
    for label, value in rows:
        table.add_row(f"[dim]{label}[/dim]", str(value))
    console.print(Panel(table, title="result", border_style="dim"))

    if stats.get("librarian"):
        lib = stats["librarian"]
        console.print(
            f"[dim]quote integrity {lib['quote_integrity']:.0%} "
            f"({lib['facts_dropped_bad_quote']} facts dropped for ungrounded quotes)[/dim]"
        )
    if stats.get("analyst"):
        an = stats["analyst"]
        console.print(
            f"[dim]citation validity {an['citation_validity']:.0%} "
            f"({an['fact_ids_hallucinated']} invalid fact ids)[/dim]"
        )


@app.command()
def runs(limit: Annotated[int, typer.Option("--limit")] = 20) -> None:
    """List stored runs."""
    store = RunStore()
    summaries = store.list_runs(limit)
    if not summaries:
        console.print("[dim]no runs yet[/dim]")
        return
    table = Table(title="runs")
    for col in ("id", "query", "entity", "status", "updated", "cost"):
        table.add_column(col)
    for s in summaries:
        table.add_row(
            s.id,
            s.query[:28],
            f"{s.entity_name or '-'}{f' ({s.ticker})' if s.ticker else ''}"[:30],
            s.status,
            s.updated_at[:19].replace("T", " "),
            f"${s.cost_usd:.4f}",
        )
    console.print(table)


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run id")],
    fmt: Annotated[str, typer.Option("--format", help="md | json | stats")] = "md",
) -> None:
    """Print a stored run."""
    run = RunStore().load(run_id)
    if run is None:
        console.print(f"[red]no run {run_id}[/red]")
        raise typer.Exit(code=1)
    if fmt == "json":
        console.print_json(run.model_dump_json())
    elif fmt == "stats":
        console.print_json(
            json.dumps(
                {
                    "status": run.status.value,
                    "documents": len(run.evidence),
                    "facts": len(run.facts),
                    "claims": len(run.claims),
                    "rejection_rate": run.rejection_rate,
                    "dedup_rate": run.dedup_rate,
                    "cost_usd": run.ledger.total_usd,
                    "cost_by_node": run.ledger.by_node(),
                    "nodes": {k: v.status.value for k, v in run.nodes.items()},
                },
                indent=2,
            )
        )
    elif run.report:
        console.print(run.report.markdown)
    else:
        console.print("[yellow]run has no report[/yellow]")


@app.command()
def resolve(query: Annotated[str, typer.Argument(help="Company name or ticker")]) -> None:
    """Resolve a query to an entity without running the pipeline."""

    async def _go() -> None:
        fetcher = Fetcher(FetchCache(), settings())
        index = await entity_tools.load_index(fetcher)
        entity = entity_tools.resolve_from_index(query, index)
        console.print(f"[bold]{entity.name}[/bold]")
        console.print(f"  ticker: {entity.ticker or '-'}   cik: {entity.cik or '-'}")
        console.print(f"  confidence: {entity.confidence:.3f}   industry: {entity.is_industry}")
        console.print(f"  needs clarification: {entity.needs_clarification}")
        if entity.candidates:
            table = Table("candidate", "ticker", "score", box=None)
            for c in entity.candidates:
                table.add_row(c.name[:46], c.ticker or "-", f"{c.score:.3f}")
            console.print(table)
        console.print(f"[dim]index size: {len(index):,} companies[/dim]")

    asyncio.run(_go())


@app.command()
def cache_stats() -> None:
    """Show fetch cache size."""
    cache = FetchCache()
    files = list(cache.root.rglob("*.json"))
    size = sum(f.stat().st_size for f in files)
    console.print(f"cached documents: {len(files):,}")
    console.print(f"on disk: {size / 1_048_576:.1f} MB")
    console.print(f"location: {cache.root}")


# `research` is the primary verb; keep it as the command name.
app.command(name="research")(research_cmd)


if __name__ == "__main__":
    app()
