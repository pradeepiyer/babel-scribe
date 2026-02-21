import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from babel_scribe.config import CONFIG_DIR
from babel_scribe.errors import DriveError

CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

AUDIO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
    "audio/mp4",
    "audio/x-m4a",
    "audio/webm",
}

# Patterns for Drive URLs
_FILE_PATTERN = re.compile(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)")
_FOLDER_PATTERN = re.compile(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)")


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    mime_type: str


def parse_drive_url(url: str) -> tuple[str, str]:
    """Extract file/folder ID and type from a Google Drive URL.

    Returns (id, type) where type is "file" or "folder".
    Raises DriveError if the URL doesn't match known patterns.
    """
    file_match = _FILE_PATTERN.search(url)
    if file_match:
        return file_match.group(1), "file"

    folder_match = _FOLDER_PATTERN.search(url)
    if folder_match:
        return folder_match.group(1), "folder"

    raise DriveError(f"Could not parse Drive URL: {url}")


def get_credentials() -> "google.oauth2.credentials.Credentials":  # type: ignore[name-defined]  # noqa: F821
    """Load stored OAuth2 credentials from disk."""
    from google.oauth2.credentials import Credentials

    if not CREDENTIALS_FILE.exists():
        raise DriveError(
            "No credentials found. Run 'babel-scribe auth' to authenticate with Google Drive."
        )

    creds = Credentials.from_authorized_user_file(str(CREDENTIALS_FILE), SCOPES)

    if creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request

        creds.refresh(Request())
        CREDENTIALS_FILE.write_text(creds.to_json())

    if not creds.valid:
        raise DriveError("Credentials are invalid. Run 'babel-scribe auth' to re-authenticate.")

    return creds


def _build_service() -> "googleapiclient.discovery.Resource":  # type: ignore[name-defined]  # noqa: F821
    """Build a Google Drive API service instance."""
    from googleapiclient.discovery import build

    creds = get_credentials()
    return build("drive", "v3", credentials=creds)


async def download_file(file_id: str, dest: Path) -> Path:
    """Download a Drive file to a local path."""
    from googleapiclient.http import MediaIoBaseDownload

    def _download() -> Path:
        service = _build_service()
        # Get file metadata for the name
        meta = service.files().get(fileId=file_id, fields="name").execute()
        file_name = meta.get("name", file_id)
        target = dest / file_name

        request = service.files().get_media(fileId=file_id)
        with open(target, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        return target

    try:
        return await asyncio.to_thread(_download)
    except DriveError:
        raise
    except Exception as e:
        raise DriveError(f"Failed to download file {file_id}: {e}") from e


async def upload_file(local_path: Path, folder_id: str, name: str) -> str:
    """Upload a file to a Drive folder. Returns the new file's ID."""
    from googleapiclient.http import MediaFileUpload

    def _upload() -> str:
        service = _build_service()
        file_metadata = {"name": name, "parents": [folder_id]}
        media = MediaFileUpload(str(local_path), mimetype="text/plain")
        result = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        return result["id"]

    try:
        return await asyncio.to_thread(_upload)
    except DriveError:
        raise
    except Exception as e:
        raise DriveError(f"Failed to upload {name}: {e}") from e


async def list_audio_files(folder_id: str) -> list[DriveFile]:
    """List audio files in a Drive folder."""

    def _list() -> list[DriveFile]:
        service = _build_service()
        mime_query = " or ".join(f"mimeType='{m}'" for m in sorted(AUDIO_MIME_TYPES))
        query = f"'{folder_id}' in parents and ({mime_query}) and trashed=false"

        files: list[DriveFile] = []
        page_token = None

        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                )
                .execute()
            )

            for item in response.get("files", []):
                files.append(
                    DriveFile(id=item["id"], name=item["name"], mime_type=item["mimeType"])
                )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return files

    try:
        return await asyncio.to_thread(_list)
    except DriveError:
        raise
    except Exception as e:
        raise DriveError(f"Failed to list files in folder {folder_id}: {e}") from e


async def authenticate() -> None:
    """Run the OAuth2 flow and store credentials."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    def _auth() -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Look for client secrets file
        client_secrets = CONFIG_DIR / "client_secrets.json"
        if not client_secrets.exists():
            raise DriveError(
                f"Place your OAuth2 client secrets file at {client_secrets}\n"
                "Download it from Google Cloud Console > APIs & Services > Credentials."
            )

        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
        creds = flow.run_local_server(port=0)
        CREDENTIALS_FILE.write_text(creds.to_json())

    try:
        await asyncio.to_thread(_auth)
    except DriveError:
        raise
    except Exception as e:
        raise DriveError(f"Authentication failed: {e}") from e
