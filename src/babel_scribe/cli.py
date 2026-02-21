import asyncio
import json
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from babel_scribe.config import load_config
from babel_scribe.drive import (
    DriveError,
    authenticate,
    download_file,
    list_audio_files,
    parse_drive_url,
    upload_file,
)
from babel_scribe.errors import ScribeError
from babel_scribe.pipeline import scribe, scribe_batch
from babel_scribe.transcriber import LitellmTranscriber
from babel_scribe.translator import LitellmTranslator
from babel_scribe.types import ScribeResult

console = Console()


def _is_drive_url(source: str) -> bool:
    return "drive.google.com" in source


def _format_text_output(result: ScribeResult, timestamps: bool) -> str:
    if result.translation:
        text = result.translation.text
    else:
        text = result.transcription.text

    if timestamps and result.transcription.segments:
        lines = []
        for seg in result.transcription.segments:
            start_min, start_sec = divmod(seg.start, 60)
            end_min, end_sec = divmod(seg.end, 60)
            lines.append(f"[{int(start_min):02d}:{start_sec:05.2f} - {int(end_min):02d}:{end_sec:05.2f}] {seg.text}")
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
        data["segments"] = [
            {"text": s.text, "start": s.start, "end": s.end}
            for s in result.transcription.segments
        ]

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
    transcriber: LitellmTranscriber,
    translator: LitellmTranslator,
    source_language: str | None,
    target_language: str,
    timestamps: bool,
    output_format: str,
    output_folder: Path | None,
    concurrency: int,
) -> None:
    if len(paths) == 1:
        console.print(f"Transcribing [bold]{paths[0].name}[/bold]...")
        result = await scribe(
            paths[0], transcriber, translator, source_language, target_language, timestamps
        )
        output = (
            _format_json_output(result)
            if output_format == "json"
            else _format_text_output(result, timestamps)
        )
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

        async def process_one(path: Path) -> ScribeResult:
            async with semaphore:
                result = await scribe(
                    path, transcriber, translator, source_language, target_language, timestamps
                )
                progress.advance(task)
                return result

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_one(p)) for p in paths]

        results = [t.result() for t in tasks]

    for path, result in zip(paths, results):
        output = (
            _format_json_output(result)
            if output_format == "json"
            else _format_text_output(result, timestamps)
        )
        out_path = _output_path_for(path, output_folder)
        out_path.write_text(output, encoding="utf-8")
        console.print(f"  [bold]{path.name}[/bold] → {out_path}")


async def _process_drive_source(
    source: str,
    transcriber: LitellmTranscriber,
    translator: LitellmTranslator,
    source_language: str | None,
    target_language: str,
    timestamps: bool,
    output_format: str,
    output_folder_str: str | None,
    concurrency: int,
) -> None:
    drive_id, drive_type = parse_drive_url(source)

    # Determine output destination
    drive_output_folder_id: str | None = None
    local_output_folder: Path | None = None

    if output_folder_str:
        if _is_drive_url(output_folder_str):
            drive_output_folder_id, _ = parse_drive_url(output_folder_str)
        else:
            local_output_folder = Path(output_folder_str)
            local_output_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        if drive_type == "file":
            console.print("Downloading file from Drive...")
            local_file = await download_file(drive_id, tmp_path)
            console.print(f"Transcribing [bold]{local_file.name}[/bold]...")
            result = await scribe(
                local_file, transcriber, translator, source_language, target_language, timestamps
            )
            output = (
                _format_json_output(result)
                if output_format == "json"
                else _format_text_output(result, timestamps)
            )
            out_name = f"{local_file.stem}.txt"

            if local_output_folder:
                out_path = local_output_folder / out_name
                out_path.write_text(output, encoding="utf-8")
                console.print(f"Output written to [bold]{out_path}[/bold]")
            else:
                out_path = tmp_path / out_name
                out_path.write_text(output, encoding="utf-8")
                folder_id = drive_output_folder_id or drive_id
                uploaded_id = await upload_file(out_path, folder_id, out_name)
                console.print(f"Uploaded [bold]{out_name}[/bold] to Drive (ID: {uploaded_id})")

        else:  # folder
            console.print("Listing audio files in Drive folder...")
            audio_files = await list_audio_files(drive_id)
            if not audio_files:
                console.print("[yellow]No audio files found in folder.[/yellow]")
                return

            console.print(f"Found {len(audio_files)} audio file(s). Downloading...")
            local_files = []
            for af in audio_files:
                local_file = await download_file(af.id, tmp_path)
                local_files.append((af, local_file))

            console.print("Processing files...")
            paths = [lf for _, lf in local_files]
            results = await scribe_batch(
                paths, transcriber, translator, source_language, target_language, timestamps,
                concurrency,
            )

            for (af, local_file), result in zip(local_files, results):
                output = (
                    _format_json_output(result)
                    if output_format == "json"
                    else _format_text_output(result, timestamps)
                )
                out_name = f"{local_file.stem}.txt"

                if local_output_folder:
                    out_path = local_output_folder / out_name
                    out_path.write_text(output, encoding="utf-8")
                    console.print(f"  [bold]{af.name}[/bold] → {out_path}")
                else:
                    out_path = tmp_path / out_name
                    out_path.write_text(output, encoding="utf-8")
                    folder_id = drive_output_folder_id or drive_id
                    uploaded_id = await upload_file(out_path, folder_id, out_name)
                    console.print(f"  [bold]{af.name}[/bold] → Drive (ID: {uploaded_id})")


@click.group()
def main() -> None:
    """Audio transcription and translation tool."""


@main.command()
def auth() -> None:
    """Authenticate with Google Drive."""
    try:
        asyncio.run(authenticate())
        console.print("[green]Authentication successful![/green]")
    except ScribeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("sources", nargs=-1, required=True)
@click.option("--from", "from_lang", default=None, help="Source language code")
@click.option("--to", "to_lang", default=None, help="Target language code")
@click.option("--transcription-model", default=None, help="Transcription model")
@click.option("--translation-model", default=None, help="Translation model")
@click.option("-o", "--output-format", type=click.Choice(["text", "json"]), default="text")
@click.option("--output-folder", default=None, help="Output folder (local path or Drive URL)")
@click.option("--concurrency", type=int, default=None, help="Max parallel tasks")
@click.option("--timestamps", is_flag=True, help="Include segment timestamps")
def transcribe(
    sources: tuple[str, ...],
    from_lang: str | None,
    to_lang: str | None,
    transcription_model: str | None,
    translation_model: str | None,
    output_format: str,
    output_folder: str | None,
    concurrency: int | None,
    timestamps: bool,
) -> None:
    """Transcribe and translate audio files.

    SOURCES can be local file paths or Google Drive URLs.
    """
    config = load_config()

    target_language = to_lang or config.target_language
    effective_concurrency = concurrency or config.concurrency
    t_model = transcription_model or config.transcription_model
    l_model = translation_model or config.translation_model

    transcriber = LitellmTranscriber(model=t_model)
    translator = LitellmTranslator(model=l_model)

    try:
        asyncio.run(
            _run_transcribe(
                sources, transcriber, translator, from_lang, target_language,
                timestamps, output_format, output_folder, effective_concurrency,
            )
        )
    except ScribeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


async def _run_transcribe(
    sources: tuple[str, ...],
    transcriber: LitellmTranscriber,
    translator: LitellmTranslator,
    source_language: str | None,
    target_language: str,
    timestamps: bool,
    output_format: str,
    output_folder: str | None,
    concurrency: int,
) -> None:
    local_paths: list[Path] = []
    drive_sources: list[str] = []

    for source in sources:
        if _is_drive_url(source):
            drive_sources.append(source)
        else:
            path = Path(source)
            if not path.exists():
                raise ScribeError(f"File not found: {source}")
            local_paths.append(path)

    # Determine local output folder (only for local files)
    local_output_folder: Path | None = None
    if output_folder and not _is_drive_url(output_folder):
        local_output_folder = Path(output_folder)
        local_output_folder.mkdir(parents=True, exist_ok=True)

    if local_paths:
        await _process_local_files(
            local_paths, transcriber, translator, source_language, target_language,
            timestamps, output_format, local_output_folder, concurrency,
        )

    for drive_source in drive_sources:
        await _process_drive_source(
            drive_source, transcriber, translator, source_language, target_language,
            timestamps, output_format, output_folder, concurrency,
        )
