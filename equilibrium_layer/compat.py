"""Small compatibility helpers for attribute-or-mapping result access."""

from __future__ import annotations

from typing import Any


class DictAccessMixin:
    """Allow ``result["field"]`` in addition to ``result.field``."""

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        public = getattr(self, "public_dict", None)
        if callable(public):
            data = public()
            if key in data:
                return data[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
