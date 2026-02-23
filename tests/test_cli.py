import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from babel_scribe.cli import main
from babel_scribe.types import ScribeResult, Segment, TranscriptionResult, TranslationResult


@pytest.fixture(autouse=True)
def _set_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")


def test_transcribe_requires_source() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "es"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_transcribe_requires_from_lang() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["file.mp3"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_transcribe_nonexistent_file() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "es", "/nonexistent/file.mp3"])
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
        result = runner.invoke(main, ["--from", "es", "--to", "en", str(audio)])

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
        result = runner.invoke(main, ["--from", "es", str(audio), "--output-format", "json"])

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
        transcription=TranscriptionResult(text="hello world", source_language="en", segments=segments),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["--from", "en", "--to", "en", str(audio), "--timestamps"])

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
        transcription=TranscriptionResult(text="hello", source_language="en", segments=segments),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["--from", "en", str(audio), "--timestamps", "--output-format", "json"])

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
            main, ["--from", "en", "--to", "en", str(audio), "--output-folder", str(out_dir)]
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
        result = runner.invoke(main, ["--from", "en", "--to", "en", *files])

    assert result.exit_code == 0
    for i in range(3):
        assert (tmp_path / f"test{i}.txt").exists()


def test_transcribe_no_translation_when_same_language(tmp_path: Path) -> None:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake audio")

    scribe_result = ScribeResult(
        transcription=TranscriptionResult(text="hello world", source_language="en"),
    )

    with patch("babel_scribe.cli.scribe", new_callable=AsyncMock, return_value=scribe_result):
        runner = CliRunner()
        result = runner.invoke(main, ["--from", "en", "--to", "en", str(audio)])

    assert result.exit_code == 0
    out_file = tmp_path / "test.txt"
    assert out_file.read_text() == "hello world"


# --- Text-mode CLI tests ---


def test_translate_text_file_text_output(tmp_path: Path) -> None:
    text_file = tmp_path / "essay.txt"
    text_file.write_text("नमस्ते दुनिया", encoding="utf-8")

    translation = TranslationResult(text="hello world", source_language="hi", target_language="en")

    with patch("babel_scribe.cli.translate", new_callable=AsyncMock, return_value=translation):
        runner = CliRunner()
        result = runner.invoke(main, ["--from", "hi", "--to", "en", str(text_file)])

    assert result.exit_code == 0
    out_file = tmp_path / "essay.translated.txt"
    assert out_file.exists()
    assert out_file.read_text() == "hello world"


def test_translate_text_file_json_output(tmp_path: Path) -> None:
    text_file = tmp_path / "essay.txt"
    text_file.write_text("नमस्ते दुनिया", encoding="utf-8")

    translation = TranslationResult(text="hello world", source_language="hi", target_language="en")

    with patch("babel_scribe.cli.translate", new_callable=AsyncMock, return_value=translation):
        runner = CliRunner()
        result = runner.invoke(main, ["--from", "hi", "--to", "en", str(text_file), "--output-format", "json"])

    assert result.exit_code == 0
    out_file = tmp_path / "essay.translated.txt"
    data = json.loads(out_file.read_text())
    assert data["translation"]["text"] == "hello world"
    assert data["translation"]["source_language"] == "hi"
    assert data["translation"]["target_language"] == "en"


def test_translate_multiple_text_files(tmp_path: Path) -> None:
    files = []
    for i in range(3):
        f = tmp_path / f"doc{i}.txt"
        f.write_text(f"text {i}", encoding="utf-8")
        files.append(str(f))

    translation = TranslationResult(text="translated", source_language="hi", target_language="en")

    with patch("babel_scribe.cli.translate", new_callable=AsyncMock, return_value=translation):
        runner = CliRunner()
        result = runner.invoke(main, ["--from", "hi", "--to", "en", *files])

    assert result.exit_code == 0
    for i in range(3):
        assert (tmp_path / f"doc{i}.translated.txt").exists()


def test_translate_text_file_output_folder(tmp_path: Path) -> None:
    text_file = tmp_path / "essay.txt"
    text_file.write_text("texto", encoding="utf-8")
    out_dir = tmp_path / "output"

    translation = TranslationResult(text="text", source_language="es", target_language="en")

    with patch("babel_scribe.cli.translate", new_callable=AsyncMock, return_value=translation):
        runner = CliRunner()
        result = runner.invoke(
            main, ["--from", "es", "--to", "en", str(text_file), "--output-folder", str(out_dir)]
        )

    assert result.exit_code == 0
    out_file = out_dir / "essay.translated.txt"
    assert out_file.exists()
    assert out_file.read_text() == "text"


def test_translate_text_same_language_error(tmp_path: Path) -> None:
    text_file = tmp_path / "essay.txt"
    text_file.write_text("hello", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["--from", "en", "--to", "en", str(text_file)])

    assert result.exit_code == 1
    assert "same" in result.output.lower()


def test_mixed_text_and_audio_error(tmp_path: Path) -> None:
    text_file = tmp_path / "essay.txt"
    text_file.write_text("text", encoding="utf-8")
    audio_file = tmp_path / "recording.mp3"
    audio_file.write_bytes(b"fake audio")

    runner = CliRunner()
    result = runner.invoke(main, ["--from", "hi", "--to", "en", str(text_file), str(audio_file)])

    assert result.exit_code == 1
    assert "mix" in result.output.lower()


def test_translate_text_does_not_overwrite_source(tmp_path: Path) -> None:
    text_file = tmp_path / "essay.txt"
    original_content = "original hindi text"
    text_file.write_text(original_content, encoding="utf-8")

    translation = TranslationResult(text="translated english", source_language="hi", target_language="en")

    with patch("babel_scribe.cli.translate", new_callable=AsyncMock, return_value=translation):
        runner = CliRunner()
        result = runner.invoke(main, ["--from", "hi", "--to", "en", str(text_file)])

    assert result.exit_code == 0
    # Source file should be untouched
    assert text_file.read_text() == original_content
    # Output goes to .translated.txt
    out_file = tmp_path / "essay.translated.txt"
    assert out_file.read_text() == "translated english"
