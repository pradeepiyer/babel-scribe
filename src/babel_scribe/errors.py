class ScribeError(Exception):
    pass


class TranscriptionError(ScribeError):
    pass


class TranslationError(ScribeError):
    pass


class DriveError(ScribeError):
    pass
