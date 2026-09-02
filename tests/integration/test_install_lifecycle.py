"""Real pip lifecycle coverage for the managed non-editable installation."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _copy_installer(target: Path) -> None:
    (target / "commands" / "install").mkdir(parents=True)
    (target / "bin").mkdir()
    shutil.copy2(
        ROOT / "commands" / "install" / "run", target / "commands" / "install" / "run"
    )
    shutil.copy2(ROOT / "bin" / "ai", target / "bin" / "ai")


def _build_test_wheel(source: Path, wheel_dir: Path) -> Path:
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = project["version"]
    wheel = wheel_dir / f"ai_mm-{version}-py3-none-any.whl"
    records: list[str] = []
    dist_info = f"ai_mm-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((source / "src" / "claude_mm").rglob("*.py")):
            member = path.relative_to(source / "src").as_posix()
            archive.write(path, member)
            records.append(f"{member},,")
        metadata = f"{dist_info}/METADATA"
        wheel_metadata = f"{dist_info}/WHEEL"
        record = f"{dist_info}/RECORD"
        archive.writestr(
            metadata,
            f"Metadata-Version: 2.1\nName: ai-mm\nVersion: {version}\n",
        )
        archive.writestr(
            wheel_metadata,
            "Wheel-Version: 1.0\nGenerator: ai-mm-test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
        )
        records.extend((f"{metadata},,", f"{wheel_metadata},,", f"{record},,"))
        archive.writestr(record, "\n".join(records) + "\n")
    return wheel


def test_noneditable_install_survives_source_checkout_removal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _copy_installer(source)
    wheel = _build_test_wheel(ROOT, tmp_path)

    home = tmp_path / "home"
    home.mkdir()
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.update(
        {
            "AI_MM_INSTALL_ARTIFACT": str(wheel),
            "HOME": str(home),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    subprocess.run(
        ["bash", str(source / "commands" / "install" / "run")],
        check=True,
        capture_output=True,
        env=clean_env,
        text=True,
    )

    shutil.rmtree(source)
    wheel.unlink()

    venv = home / ".local" / "venvs" / "ai"
    python = venv / "bin" / "python3"
    subprocess.run(
        [str(python), "-m", "claude_mm.install_verify"],
        check=True,
        capture_output=True,
        env=clean_env,
        text=True,
    )
    result = subprocess.run(
        [str(python), "-c", "import claude_mm; print(claude_mm.__file__)"],
        check=True,
        capture_output=True,
        env=clean_env,
        text=True,
    )
    assert Path(result.stdout.strip()).resolve().is_relative_to(venv.resolve())
    ai_bin = home / ".local" / "bin" / "ai"
    assert ai_bin.read_text(encoding="utf-8").splitlines()[0] == f"#!{python}"


def test_installer_rejects_missing_artifact_before_creating_venv(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _copy_installer(source)
    home = tmp_path / "home"
    home.mkdir()
    clean_env = os.environ.copy()
    clean_env.update(
        {
            "AI_MM_INSTALL_ARTIFACT": str(tmp_path / "missing.whl"),
            "HOME": str(home),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(source / "commands" / "install" / "run")],
        check=False,
        capture_output=True,
        env=clean_env,
        text=True,
    )

    assert result.returncode != 0
    assert "AI_MM_INSTALL_ARTIFACT does not exist" in result.stderr
    assert not (home / ".local" / "venvs" / "ai").exists()
