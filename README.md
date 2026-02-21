# babel-scribe

Audio transcription and translation CLI. Transcribes audio files and optionally translates them to a target language, with support for multiple providers including Groq, OpenAI, and Sarvam AI.

## Installation

```bash
uv pip install -e .
```

## Configuration

babel-scribe reads configuration from `~/.babel-scribe/config.toml`. All settings are optional — sensible defaults are used when omitted.

```toml
[defaults]
target_language = "en"
concurrency = 5

[models]
transcription = "groq/whisper-large-v3-turbo"
translation = "groq/llama-3.3-70b-versatile"
```

## Environment Variables

Set the API key for whichever provider you're using:

| Provider | Variable |
|----------|----------|
| Groq | `GROQ_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Sarvam | `SARVAM_API_KEY` |

## Usage

### Basic transcription

```bash
babel-scribe transcribe recording.mp3 --to en
```

### With timestamps

```bash
babel-scribe transcribe recording.mp3 --to en --timestamps
```

### Using a different provider

```bash
babel-scribe transcribe recording.mp3 --to en --transcription-model sarvam/saaras:v3
```

### Multiple files

```bash
babel-scribe transcribe file1.mp3 file2.mp3 --to en
```

### JSON output

```bash
babel-scribe transcribe recording.mp3 --to en -o json
```

## Supported Providers

| Provider | Transcription | Translation | Model prefix |
|----------|--------------|-------------|-------------|
| Groq | whisper-large-v3-turbo | llama-3.3-70b-versatile | `groq/` |
| OpenAI | whisper-1 | gpt-4o-mini | `openai/` |
| Sarvam | saaras:v3 | — | `sarvam/` |

Sarvam's `saaras:v3` model can transcribe and translate Indian languages to English in a single step.
