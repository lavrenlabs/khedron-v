from __future__ import annotations

import importlib


def test_package_can_be_imported() -> None:
    module = importlib.import_module("khedron")
    if module.__name__ != "khedron":
        raise AssertionError(module.__name__)
