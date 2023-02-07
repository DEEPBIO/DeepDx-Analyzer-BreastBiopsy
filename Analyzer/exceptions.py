from typing import Literal

# Consider using Literal if python version changes to 3.8
ErrorCodes = Literal["UNSUPPORTED_MAGNIFICATION", "TISSUE_NOT_FOUND"]

class AnalyzerError(Exception):
    """
    Error produced by the DeepDx Analyzer Library.
    """


class AnalyzerNotLoadedError(AnalyzerError):
    """
    Raised when analyze() is called when load is not finished.
    """
    pass


class AnalyzerUnsupportedError(AnalyzerError):
    """
    Raised when requested file or cropped region cannot be analyzed.
    """
    def __init__(self, code: ErrorCodes, message: str):
        super().__init__()
        self.code = code
        self.message = message

class MagnificationError(Exception):
    """
    This error occurs when the file does not support the proper magnification.
    """