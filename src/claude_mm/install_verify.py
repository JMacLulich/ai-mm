"""Fail-closed verification for the managed global ``ai-mm`` installation."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


class InstallVerificationError(RuntimeError):
    """The installed package is not self-contained inside its managed venv."""


def validate_managed_install(
    *, module_path: Path, venv_path: Path, direct_url: str | None
) -> None:
    """Validate resolved package location and PEP 610 editable metadata."""
    resolved_module = module_path.resolve()
    resolved_venv = venv_path.resolve()
    if not resolved_module.is_relative_to(resolved_venv):
        raise InstallVerificationError(
            "ai-mm resolved outside its managed venv: "
            f"{resolved_module} (venv: {resolved_venv})"
        )

    if direct_url is None:
        # Registry/wheel installs may omit PEP 610 metadata. The resolved module
        # location above remains the primary self-containment postcondition.
        return
    try:
        metadata = json.loads(direct_url)
    except json.JSONDecodeError as exc:
        raise InstallVerificationError("ai-mm direct_url.json is malformed") from exc
    if not isinstance(metadata, dict):
        raise InstallVerificationError("ai-mm direct_url.json is not an object")
    dir_info = metadata.get("dir_info", {})
    if not isinstance(dir_info, dict):
        raise InstallVerificationError("ai-mm direct_url.json dir_info is not an object")
    if dir_info.get("editable") is True:
        raise InstallVerificationError("ai-mm is still installed in editable mode")


def verify_current_install() -> None:
    """Verify the import currently selected by the active Python interpreter."""
    import claude_mm

    if claude_mm.__file__ is None:
        raise InstallVerificationError("ai-mm has no resolved module file")
    try:
        direct_url = distribution("ai-mm").read_text("direct_url.json")
    except PackageNotFoundError as exc:
        raise InstallVerificationError("ai-mm distribution metadata was not found") from exc
    validate_managed_install(
        module_path=Path(claude_mm.__file__),
        venv_path=Path(sys.prefix),
        direct_url=direct_url,
    )


def main() -> None:
    try:
        verify_current_install()
    except InstallVerificationError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
