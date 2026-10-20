"""Optional fit predictor placeholder intentionally not part of the minimum scope.

A meaningful supervised fit predictor requires labelled resume-job pairs. Do not train one
on fabricated labels; collect and document suitable labelled data before implementing it.
"""

class FitPredictorNotImplemented(NotImplementedError):
    """Raised when a fit predictor is requested without labelled training data."""
