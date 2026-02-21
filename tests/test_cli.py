import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from babel_scribe.cli import main
from babel_scribe.types import ScribeResult, Segment, TranscriptionResult, TranslationResult


def test_transcribe_requires_source() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_transcribe_nonexistent_file() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "/nonexistent/file.mp3"])
    assert result.exit_code == 1
    assert "File not found" in result.output


def test_transcribe_local_file_text_output(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake audio")

    scribe_result = ScribeResult(
        transcription=TranscriptionResult(text="hola mundo", source_language="es"),
        translation=TranslationResult(text="hello world", source_language="es", target_language="en"),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio), "--to", "en"])

    assert result.exit_code == 0
    out_file = tmp_path / "test.txt"
    assert out_file.exists()
    assert out_file.read_text() == "hello world"


def test_transcribe_local_file_json_output(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake audio")

    scribe_result = ScribeResult(
        transcription=TranscriptionResult(text="hola", source_language="es"),
        translation=TranslationResult(text="hello", source_language="es", target_language="en"),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio), "-o", "json"])

    assert result.exit_code == 0
    out_file = tmp_path / "test.txt"
    data = json.loads(out_file.read_text())
    assert data["transcription"]["text"] == "hola"
    assert data["translation"]["text"] == "hello"


def test_transcribe_with_timestamps_text(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake audio")

    segments = [
        Segment(text="hello", start=0.0, end=1.5),
        Segment(text="world", start=1.5, end=3.0),
    ]
    scribe_result = ScribeResult(
        transcription=TranscriptionResult(
            text="hello world", source_language="en", segments=segments
        ),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio), "--timestamps", "--to", "en"])

    assert result.exit_code == 0
    out_file = tmp_path / "test.txt"
    content = out_file.read_text()
    assert "[00:00.00 - 00:01.50] hello" in content
    assert "[00:01.50 - 00:03.00] world" in content


def test_transcribe_with_timestamps_json(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake audio")

    segments = [
        Segment(text="hello", start=0.0, end=1.5),
    ]
    scribe_result = ScribeResult(
        transcription=TranscriptionResult(
            text="hello", source_language="en", segments=segments
        ),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio), "--timestamps", "-o", "json"])

    assert result.exit_code == 0
    out_file = tmp_path / "test.txt"
    data = json.loads(out_file.read_text())
    assert data["segments"][0]["text"] == "hello"
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["end"] == 1.5


def test_transcribe_output_folder(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake audio")
    out_dir = tmp_path / "output"

    scribe_result = ScribeResult(
        transcription=TranscriptionResult(text="hello", source_language="en"),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(
            main, ["transcribe", str(audio), "--output-folder", str(out_dir), "--to", "en"]
        )

    assert result.exit_code == 0
    out_file = out_dir / "test.txt"
    assert out_file.exists()
    assert out_file.read_text() == "hello"


def test_transcribe_multiple_local_files(tmp_path: Path) -> None:
    files = []
    for i in range(3):
        f = tmp_path / f"test{i}.mp3"
        f.write_bytes(b"fake audio")
        files.append(str(f))

    scribe_result = ScribeResult(
        transcription=TranscriptionResult(text="hello", source_language="en"),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", *files, "--to", "en"])

    assert result.exit_code == 0
    for i in range(3):
        assert (tmp_path / f"test{i}.txt").exists()


def test_auth_command() -> None:
    with patch("babel_scribe.cli.authenticate", new_callable=AsyncMock) as mock_auth:
        runner = CliRunner()
        result = runner.invoke(main, ["auth"])

    assert result.exit_code == 0
    assert "successful" in result.output
    mock_auth.assert_called_once()


def test_transcribe_no_translation_when_same_language(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake audio")

    scribe_result = ScribeResult(
        transcription=TranscriptionResult(text="hello world", source_language="en"),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio), "--to", "en"])

    assert result.exit_code == 0
    out_file = tmp_path / "test.txt"
    assert out_file.read_text() == "hello world"
