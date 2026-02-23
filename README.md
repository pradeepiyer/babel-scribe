# babel-scribe

Audio transcription and text translation CLI. Transcribes audio files and translates text files, automatically selecting the right provider based on the source language.

## Installation

```bash
uv pip install -e .
```

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `SARVAM_API_KEY` | Indian language transcription and translation |
| `OPENAI_API_KEY` | Non-Indian language transcription and translation |

## Usage

### Basic transcription (Hindi → English)

```bash
babel-scribe recording.mp3 --from hi
```

### Transcribe and translate (Spanish → French)

```bash
babel-scribe recording.mp3 --from es --to fr
```

### With timestamps

```bash
babel-scribe recording.mp3 --from hi --timestamps
```

### Translate a text file (Hindi → English)

```bash
babel-scribe essay.txt --from hi --to en
```

### Multiple files

```bash
babel-scribe file1.mp3 file2.mp3 --from ta --to en
```

### JSON output

```bash
babel-scribe recording.mp3 --from hi --output-format json
```

For all options and examples, run `babel-scribe --help`.

## Provider Routing

### Transcription (audio files)

- **Indian languages** → [Sarvam AI](https://docs.sarvam.ai/api-reference-docs/speech-to-text/saaras) (`saaras:v3`). When the target is English, Sarvam translates in a single step.
- **All other languages** → [OpenAI Whisper](https://platform.openai.com/docs/guides/speech-to-text#supported-languages) (`whisper-1`). When the target is English, the Whisper translations endpoint is used.

### Translation (text files and post-transcription)

- **Indian↔English** → [Sarvam AI](https://docs.sarvam.ai/api-reference-docs/translate) (`sarvam-translate:v1`)
- **Indian↔non-English** → chained via English (Sarvam + OpenAI)
- **All other pairs** → OpenAI (`gpt-5-mini`)

