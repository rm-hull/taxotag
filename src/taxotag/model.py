"""Public model API.

The inference pipeline is added in the later implementation phases.
"""

from dataclasses import dataclass


class GistError(Exception):
    """Base exception for taxotag errors."""


@dataclass(frozen=True)
class Topic:
    """A classified topic and its score."""

    slug: str
    name: str
    score: float


class Gist:
    """Placeholder for the Gist model assembled in the later phases."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise GistError("The inference pipeline is not implemented yet")
