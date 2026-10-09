class RagError(Exception):
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(RagError):
    status_code = 415


class FileTooLargeError(RagError):
    status_code = 413


class EmptyDocumentError(RagError):
    status_code = 422


class DocumentExtractionError(RagError):
    """PDF/DOCX/TXT could not be parsed (corrupt, encrypted, wrong format...)."""
    status_code = 422


class InvalidRequestError(RagError):
    status_code = 400


class DocumentNotFoundError(RagError):
    status_code = 404


class MissingApiKeyError(RagError):
    status_code = 503


class EmbeddingError(RagError):
    status_code = 502


class VectorStoreError(RagError):
    status_code = 500


class GenerationError(RagError):
    status_code = 502


class UpstreamTimeoutError(RagError):
    status_code = 504
