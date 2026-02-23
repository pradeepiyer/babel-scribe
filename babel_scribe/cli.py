import asyncio
import json
import mimetypes
from collections.abc import Awaitable, Callable
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from babel_scribe.pipeline import scribe, translate
from babel_scribe.providers import normalize_language_code
from babel_scribe.transcriber import Transcriber, create_transcriber
from babel_scribe.translator import Translator, create_translator
from babel_scribe.types import ScribeError, ScribeResult, TranslationResult

console = Console()


def _format_scribe_text(result: ScribeResult, timestamps: bool) -> str:
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


def _format_scribe_json(result: ScribeResult) -> str:
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


def _format_translation_text(result: TranslationResult) -> str:
    return result.text


def _format_translation_json(result: TranslationResult) -> str:
    data = {
        "translation": {
            "text": result.text,
            "source_language": result.source_language,
            "target_language": result.target_language,
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def _audio_output_path(source: Path, output_folder: Path | None) -> Path:
    folder = output_folder or source.parent
    return folder / f"{source.stem}.txt"


def _text_output_path(source: Path, output_folder: Path | None) -> Path:
    folder = output_folder or source.parent
    return folder / f"{source.stem}.translated.txt"


def _detect_mode(paths: list[Path]) -> str:
    """Detect whether input files are text or audio based on MIME type.

    Returns 'text' or 'audio'. Raises ScribeError for unrecognized or mixed types.
    """
    modes: set[str] = set()
    for path in paths:
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            raise ScribeError(f"Cannot determine file type for {path.name} — use a recognized file extension")
        major = mime_type.split("/")[0]
        if major == "text":
            modes.add("text")
        elif major in ("audio", "video"):
            modes.add("audio")
        else:
            raise ScribeError(f"Unsupported file type '{mime_type}' for {path.name}")
    if len(modes) > 1:
        raise ScribeError("Cannot mix text and audio files in a single invocation")
    return modes.pop()


async def _process_files(
    paths: list[Path],
    process_one: Callable[[Path], Awaitable[str]],
    output_path_for: Callable[[Path], Path],
    concurrency: int,
    verb: str,
) -> None:
    remaining = [p for p in paths if not output_path_for(p).exists()]
    if len(remaining) < len(paths):
        console.print(f"Skipping {len(paths) - len(remaining)} already processed file(s)")
    if not remaining:
        return
    paths = remaining

    if len(paths) == 1:
        console.print(f"{verb} [bold]{paths[0].name}[/bold]...")
        output = await process_one(paths[0])
        out_path = output_path_for(paths[0])
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

        async def _run_one(path: Path) -> None:
            async with semaphore:
                try:
                    output = await process_one(path)
                except Exception as e:
                    failed.append((path, e))
                    progress.console.print(f"  [red]FAILED[/red] [bold]{path.name}[/bold]: {e}")
                    progress.advance(task)
                    return
                out_path = output_path_for(path)
                out_path.write_text(output, encoding="utf-8")
                progress.console.print(f"  [bold]{path.name}[/bold] → {out_path}")
                progress.advance(task)

        async with asyncio.TaskGroup() as tg:
            for p in paths:
                tg.create_task(_run_one(p))

    if failed:
        console.print(f"\n[red]{len(failed)} file(s) failed:[/red]")
        for path, err in failed:
            console.print(f"  [bold]{path.name}[/bold]: {err}")
        raise SystemExit(1)


async def _run_transcribe(
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
    async def process_one(path: Path) -> str:
        result = await scribe(path, transcriber, translator, source_language, target_language, timestamps)
        if output_format == "json":
            return _format_scribe_json(result)
        return _format_scribe_text(result, timestamps)

    await _process_files(
        paths,
        process_one,
        lambda p: _audio_output_path(p, output_folder),
        concurrency,
        verb="Transcribing",
    )


async def _run_translate(
    paths: list[Path],
    translator: Translator,
    source_language: str,
    target_language: str,
    output_format: str,
    output_folder: Path | None,
    concurrency: int,
) -> None:
    async def process_one(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        result = await translate(text, translator, source_language, target_language)
        if output_format == "json":
            return _format_translation_json(result)
        return _format_translation_text(result)

    await _process_files(
        paths,
        process_one,
        lambda p: _text_output_path(p, output_folder),
        concurrency,
        verb="Translating",
    )


@click.command(epilog="""\b
Examples:
  babel-scribe recording.mp3 --from hi
  babel-scribe recording.mp3 --from es --to fr
  babel-scribe recording.mp3 --from hi --timestamps
  babel-scribe essay.txt --from hi --to en
  babel-scribe '*.mp3' --from ta --output-format json
""")
@click.argument("sources", nargs=-1, required=True)
@click.option("--from", "from_lang", required=True, help="Source language code (e.g. hi, es, en-US)")
@click.option("--to", "to_lang", default="en", show_default=True, help="Target language code")
@click.option(
    "--output-format", type=click.Choice(["text", "json"]),
    default="text", show_default=True, help="Output format",
)
@click.option("--output-folder", default=None, help="Write output files to this directory")
@click.option("--concurrency", type=int, default=5, show_default=True, help="Max parallel file processing tasks")
@click.option("--job-timeout", type=int, default=1800, show_default=True, help="Sarvam batch job timeout in seconds")
@click.option("--timestamps", is_flag=True, help="Include word-level timestamps in output")
def main(
    sources: tuple[str, ...],
    from_lang: str,
    to_lang: str,
    output_format: str,
    output_folder: str | None,
    concurrency: int,
    job_timeout: int,
    timestamps: bool,
) -> None:
    """Transcribe and translate audio files, or translate text files."""
    try:
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

        mode = _detect_mode(paths)

        if mode == "text":
            if normalize_language_code(from_lang) == normalize_language_code(to_lang):
                raise ScribeError("Source and target languages are the same — nothing to translate")
            translator = create_translator(from_lang, to_lang)
            asyncio.run(
                _run_translate(paths, translator, from_lang, to_lang, output_format, local_output_folder, concurrency)
            )
        else:
            transcriber = create_transcriber(from_lang, to_lang, job_timeout)
            translator = create_translator(from_lang, to_lang) if normalize_language_code(to_lang) != "en" else None
            asyncio.run(
                _run_transcribe(
                    paths, transcriber, translator, from_lang, to_lang, timestamps,
                    output_format, local_output_folder, concurrency,
                )
            )
    except ScribeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None
