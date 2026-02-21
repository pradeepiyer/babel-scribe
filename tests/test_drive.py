import os

import pytest

from babel_scribe.drive import DriveError, parse_drive_url

pytestmark_integration = pytest.mark.integration


def test_parse_file_url() -> None:
    url = "https://drive.google.com/file/d/1abc123_def/view?usp=sharing"
    file_id, file_type = parse_drive_url(url)
    assert file_id == "1abc123_def"
    assert file_type == "file"


def test_parse_folder_url() -> None:
    url = "https://drive.google.com/drive/folders/1xyz789_abc?usp=sharing"
    file_id, file_type = parse_drive_url(url)
    assert file_id == "1xyz789_abc"
    assert file_type == "folder"


def test_parse_invalid_url() -> None:
    with pytest.raises(DriveError, match="Could not parse"):
        parse_drive_url("https://example.com/not-a-drive-url")


def test_parse_file_url_without_view() -> None:
    url = "https://drive.google.com/file/d/1abc123"
    file_id, file_type = parse_drive_url(url)
    assert file_id == "1abc123"
    assert file_type == "file"


@pytest.mark.integration
async def test_download_requires_credentials() -> None:
    if not os.environ.get("GOOGLE_DRIVE_TEST"):
        pytest.skip("GOOGLE_DRIVE_TEST not set")
    # Integration test placeholder — requires real Drive credentials
