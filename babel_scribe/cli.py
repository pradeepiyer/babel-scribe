import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from babel_scribe.pipeline import scribe
from babel_scribe.providers import normalize_language_code
from babel_scribe.transcriber import Transcriber, create_transcriber
from babel_scribe.translator import Translator, create_translator
from babel_scribe.types import ScribeError, ScribeResult

console = Console()


def _format_text_output(result: ScribeResult, timestamps: bool) -> str:
    text = result.translation.text if result.translation else result.transcription.text

    if timestamps and result.transcription.segments:
        lines = []
        for seg in result.transcription.segments:
            start_min, start_sec = divmod(seg.start, 60)
            end_min, end_sec = divmod(seg.end, 60)
            start = f"{int(start_min):02d}:{start_sec:05.2f}"
            end = f"{int(end_min):02d}:{end_sec:05.2f}"
            prefix = f"[Speaker {seg.speaker}] " if seg.speaker else ""
            lines.append(f"{prefix}[{start} - {end}] {seg.text}")
        return "\n".join(lines)

    return text


def _format_json_output(result: ScribeResult) -> str:
    data: dict[str, object] = {
        "transcription": {
            "text": result.transcription.text,
            "source_language": result.transcription.source_language,
        },
    }

    if result.transcription.segments:
        segments_data = []
        for s in result.transcription.segments:
            seg: dict[str, object] = {"text": s.text, "start": s.start, "end": s.end}
            if s.speaker is not None:
                seg["speaker"] = s.speaker
            segments_data.append(seg)
        data["segments"] = segments_data

    if result.translation:
        data["translation"] = {
            "text": result.translation.text,
            "source_language": result.translation.source_language,
            "target_language": result.translation.target_language,
        }

    return json.dumps(data, indent=2, ensure_ascii=False)


def _output_path_for(audio_path: Path, output_folder: Path | None) -> Path:
    folder = output_folder or audio_path.parent
    return folder / f"{audio_path.stem}.txt"


async def _process_local_files(
    paths: list[Path],
    transcriber: Transcriber,
    translator: Translator | None,
    source_language: str | None,
    target_language: str,
    timestamps: bool,
    output_format: str,
    output_folder: Path | None,
    concurrency: int,
) -> None:
    remaining = [p for p in paths if not _output_path_for(p, output_folder).exists()]
    if len(remaining) < len(paths):
        console.print(f"Skipping {len(paths) - len(remaining)} already transcribed file(s)")
    if not remaining:
        return
    paths = remaining

    if len(paths) == 1:
        console.print(f"Transcribing [bold]{paths[0].name}[/bold]...")
        result = await scribe(paths[0], transcriber, translator, source_language, target_language, timestamps)
        output = _format_json_output(result) if output_format == "json" else _format_text_output(result, timestamps)
        out_path = _output_path_for(paths[0], output_folder)
        out_path.write_text(output, encoding="utf-8")
        console.print(f"Output written to [bold]{out_path}[/bold]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files", total=len(paths))

        semaphore = asyncio.Semaphore(concurrency)

        failed: list[tuple[Path, Exception]] = []

        async def process_one(path: Path) -> None:
            async with semaphore:
                try:
                    result = await scribe(path, transcriber, translator, source_language, target_language, timestamps)
                except Exception as e:
                    failed.append((path, e))
                    progress.console.print(f"  [red]FAILED[/red] [bold]{path.name}[/bold]: {e}")
                    progress.advance(task)
                    return
                output = (
                    _format_json_output(result) if output_format == "json" else _format_text_output(result, timestamps)
                )
                out_path = _output_path_for(path, output_folder)
                out_path.write_text(output, encoding="utf-8")
                progress.console.print(f"  [bold]{path.name}[/bold] → {out_path}")
                progress.advance(task)

        async with asyncio.TaskGroup() as tg:
            for p in paths:
                tg.create_task(process_one(p))

    if failed:
        console.print(f"\n[red]{len(failed)} file(s) failed:[/red]")
        for path, err in failed:
            console.print(f"  [bold]{path.name}[/bold]: {err}")
        raise SystemExit(1)


@click.group()
def main() -> None:
    """Audio transcription and translation tool."""


@main.command()
@click.argument("sources", nargs=-1, required=True)
@click.option("--from", "from_lang", required=True, help="Source language code")
@click.option("--to", "to_lang", default="en", help="Target language code")
@click.option("-o", "--output-format", type=click.Choice(["text", "json"]), default="text")
@click.option("--output-folder", default=None, help="Output folder")
@click.option("--concurrency", type=int, default=5, help="Max parallel tasks")
@click.option("--job-timeout", type=int, default=1800, help="Sarvam batch job timeout in seconds")
@click.option("--timestamps", is_flag=True, help="Include segment timestamps")
def transcribe(
    sources: tuple[str, ...],
    from_lang: str,
    to_lang: str,
    output_format: str,
    output_folder: str | None,
    concurrency: int,
    job_timeout: int,
    timestamps: bool,
) -> None:
    """Transcribe and translate audio files."""
    try:
        transcriber = create_transcriber(from_lang, to_lang, job_timeout)
        translator = create_translator() if normalize_language_code(to_lang) != "en" else None

        asyncio.run(
            _run_transcribe(
                sources,
                transcriber,
                translator,
                from_lang,
                to_lang,
                timestamps,
                output_format,
                output_folder,
                concurrency,
            )
        )
    except ScribeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None


async def _run_transcribe(
    sources: tuple[str, ...],
    transcriber: Transcriber,
    translator: Translator | None,
    source_language: str | None,
    target_language: str,
    timestamps: bool,
    output_format: str,
    output_folder: str | None,
    concurrency: int,
) -> None:
    paths: list[Path] = []
    for source in sources:
        path = Path(source)
        if not path.exists():
            raise ScribeError(f"File not found: {source}")
        paths.append(path)

    local_output_folder: Path | None = None
    if output_folder:
        local_output_folder = Path(output_folder)
        local_output_folder.mkdir(parents=True, exist_ok=True)

    await _process_local_files(
        paths,
        transcriber,
        translator,
        source_language,
        target_language,
        timestamps,
        output_format,
        local_output_folder,
        concurrency,
    )
