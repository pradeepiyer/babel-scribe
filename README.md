# babel-scribe

Audio transcription and translation CLI. Transcribes audio files and optionally translates them to a target language, automatically selecting the right provider based on the source language.

## Installation

```bash
uv pip install -e .
```

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `SARVAM_API_KEY` | Indian language transcription |
| `OPENAI_API_KEY` | Non-Indian language transcription, all translation |

## Usage

### Basic transcription (Hindi → English)

```bash
babel-scribe transcribe recording.mp3 --from hi
```

### Transcribe and translate (Spanish → French)

```bash
babel-scribe transcribe recording.mp3 --from es --to fr
```

### With timestamps

```bash
babel-scribe transcribe recording.mp3 --from hi --timestamps
```

### Multiple files

```bash
babel-scribe transcribe file1.mp3 file2.mp3 --from ta --to en
```

### JSON output

```bash
babel-scribe transcribe recording.mp3 --from hi -o json
```

## CLI Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--from` | Yes | — | Source language code (ISO 639-1 or BCP-47) |
| `--to` | No | `en` | Target language code (locale specificity preserved for translation) |
| `--concurrency` | No | `5` | Max parallel tasks |
| `--job-timeout` | No | `1800` | Sarvam batch job timeout in seconds |
| `--timestamps` | No | off | Include segment timestamps |
| `-o` | No | `text` | Output format (`text` or `json`) |
| `--output-folder` | No | same as input | Output directory |

## Provider Routing

The source language determines which transcription provider is used:

- **Indian languages** → Sarvam AI (`saaras:v3`). When the target is English, Sarvam translates in a single step.
- **All other languages** → OpenAI Whisper (`whisper-1`). When the target is English, the Whisper translations endpoint is used.

Translation (when target is not English) always uses OpenAI (`gpt-5-mini`).

## Supported Indian Languages

`as` (Assamese), `bn` (Bengali), `brx` (Bodo), `doi` (Dogri), `gu` (Gujarati), `hi` (Hindi), `kn` (Kannada), `kok` (Konkani), `ks` (Kashmiri), `mai` (Maithili), `ml` (Malayalam), `mni` (Manipuri), `mr` (Marathi), `ne` (Nepali), `or` (Odia), `pa` (Punjabi), `sa` (Sanskrit), `sat` (Santali), `sd` (Sindhi), `ta` (Tamil), `te` (Telugu), `ur` (Urdu)

Region subtags are stripped for routing (e.g., `hi-IN` routes to Sarvam), but the full `--to` value is preserved in translation prompts (e.g., `pt-BR` specificity is kept).
