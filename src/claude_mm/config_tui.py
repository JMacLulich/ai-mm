"""
Interactive TUI for managing API keys.

Provides a nice interface similar to occtl's session manager for configuring
API keys for different AI providers.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import termios
import tty

from claude_mm.env import CONFIG_DIR, ENV_FILE, load_env_file, save_env_file

PROVIDERS = [
    ("openai", "OPENAI_API_KEY", "GPT-5.2 (reviews)", "sk-"),
    ("google", "GOOGLE_AI_API_KEY", "Gemini 3 Pro (reviews)", None),
    ("anthropic", "ANTHROPIC_API_KEY", "Claude Opus 4.5 (reviews)", "sk-ant-"),
    ("ollama", None, "Local LLM (no key needed)", None),
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _test_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """Test an API key by making a minimal API call.

    Returns:
        Tuple of (success, message)
    """
    try:
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True, "Valid"

        elif provider == "google":
            from google import genai

            client = genai.Client(api_key=api_key)
            client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hi",
            )
            return True, "Valid"

        elif provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, "Valid"

        elif provider == "ollama":
            import urllib.request

            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True, "Running"
            return False, "Ollama not running"

        return False, "Unknown provider"
    except Exception as e:
        return False, str(e)


def _fit_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _menu_border(inner_width: int) -> str:
    return "+" + ("-" * inner_width) + "+"


def _menu_row(text: str, inner_width: int) -> str:
    visible = len(ANSI_RE.sub("", text))
    if visible > inner_width:
        text = _fit_text(ANSI_RE.sub("", text), inner_width)
        visible = len(text)
    return "|" + text + (" " * max(0, inner_width - visible)) + "|"


def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    term = os.environ.get("TERM", "")
    return term != "dumb"


def _colorize(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def load_existing_keys() -> dict[str, str]:
    return load_env_file()


def mask_key(key: str) -> str:
    if not key or len(key) < 12:
        return "(not set)"
    return f"{key[:8]}...{key[-4:]}"


def save_keys(keys: dict[str, str]) -> None:
    save_env_file(keys)


def _read_menu_key() -> str:
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        nxt = sys.stdin.read(1)
        if nxt == "[":
            third = sys.stdin.read(1)
            if third == "A":
                return "up"
            if third == "B":
                return "down"
        return "esc"
    if ch in {"k", "K"}:
        return "up"
    if ch in {"j", "J"}:
        return "down"
    if ch in {"\r", "\n"}:
        return "enter"
    if ch in {"q", "Q"}:
        return "quit"
    if ch in {"d", "D"}:
        return "delete"
    if ch in {"t", "T"}:
        return "test"
    return "other"


def _prompt_for_key(
    provider: str, env_key: str, description: str, prefix: str | None
) -> str | None:
    print("\033[2J\033[H", end="")

    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    inner = cols - 4

    print(_menu_border(inner))
    print(_menu_row(f" ADD API KEY: {provider.upper()} ", inner))
    print(_menu_border(inner))
    print(_menu_row(f" Service: {description} ", inner))
    if prefix:
        print(_menu_row(f" Expected prefix: {prefix} ", inner))
    print(_menu_border(inner))
    print(_menu_row("", inner))
    print(_menu_row(" Enter API key: ", inner))
    input_row = 8
    print(_menu_row("", inner))
    print(_menu_border(inner))
    print(_menu_row(" Press Enter to save, Esc to cancel ", inner))
    print(_menu_border(inner))

    key_input: list[str] = []

    while True:
        if key_input:
            masked = "*" * min(len(key_input), 20) + ("..." if len(key_input) > 20 else "")
        else:
            masked = "(typing...)"
        sys.stdout.write(f"\033[{input_row};3H")
        sys.stdout.write(f"  {masked}" + " " * (inner - len(masked) - 4) + "  ")
        sys.stdout.flush()

        ch = sys.stdin.read(1)

        if ch == "\x1b":
            return None
        elif ch in {"\r", "\n"}:
            break
        elif ch in {"\x7f", "\x08"}:
            if key_input:
                key_input.pop()
        elif ch.isprintable():
            key_input.append(ch)

    return "".join(key_input) if key_input else None


def _build_menu_rows(
    keys: dict[str, str], test_results: dict[str, tuple[bool, str]] | None = None
) -> list[dict]:
    test_results = test_results or {}
    rows = []
    for provider, env_key, description, prefix in PROVIDERS:
        is_local = provider == "ollama"
        has_key = bool(env_key and env_key in keys and keys[env_key])
        test_status = test_results.get(provider)
        rows.append(
            {
                "provider": provider,
                "env_key": env_key,
                "description": description,
                "prefix": prefix,
                "has_key": has_key,
                "is_local": is_local,
                "masked": "(local)"
                if is_local
                else (mask_key(keys.get(env_key, "")) if has_key else "(not set)"),
                "is_exit": False,
                "is_cancel": False,
                "test_status": test_status,
            }
        )
    rows.append(
        {
            "provider": "Save & Exit",
            "env_key": "",
            "description": "",
            "prefix": None,
            "has_key": False,
            "is_local": False,
            "masked": "",
            "is_exit": True,
            "is_cancel": False,
            "test_status": None,
        }
    )
    rows.append(
        {
            "provider": "Exit without saving",
            "env_key": "",
            "description": "",
            "prefix": None,
            "has_key": False,
            "is_local": False,
            "masked": "",
            "is_exit": False,
            "is_cancel": True,
            "test_status": None,
        }
    )
    return rows


def _render_menu(
    rows: list[dict],
    idx: int,
    keys: dict[str, str],
    last_error: str = "",
    has_changes: bool = False,
) -> None:
    print("\033[2J\033[H", end="")

    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    inner = cols - 4
    name_w = max(14, min(20, inner - 40))

    print(_menu_border(inner))
    title = " AI CONFIG MANAGER "
    if has_changes:
        title = " AI CONFIG MANAGER (unsaved changes) "
    print(_menu_row(title, inner))
    print(_menu_row(f" Config: {CONFIG_DIR} ", inner))
    print(_menu_border(inner))
    print(_menu_row(" Up/Down: nav   Enter: edit   t: test   d: delete   q: quit ", inner))
    print(_menu_border(inner))
    print(_menu_row(f"   {'PROVIDER'.ljust(name_w)}  {'KEY STATUS'.ljust(24)}", inner))
    print(_menu_border(inner))

    for i, row in enumerate(rows):
        selected = i == idx
        cursor = ">" if selected else " "

        if row["is_exit"] or row.get("is_cancel"):
            line = _menu_row(f" {cursor} {row['provider']}", inner)
            if selected:
                print(f"\033[7m{line}\033[0m")
            else:
                print(_colorize(line, "2"))
            continue

        provider_name = _fit_text(row["provider"].upper(), name_w)

        if row.get("is_local"):
            base_status = "local (no key needed)"
            if row["test_status"]:
                success, _msg = row["test_status"]
                if success:
                    test_str = _colorize("[OK]", "32")
                else:
                    test_str = _colorize("[OFF]", "31")
                status = f"{base_status} {test_str}"
            else:
                status = _colorize(base_status, "36")
        elif row["has_key"]:
            base_status = f"✓ {row['masked']}"
            if row["test_status"]:
                success, _msg = row["test_status"]
                if success:
                    test_str = _colorize("[OK]", "32")
                else:
                    test_str = _colorize("[FAIL]", "31")
                status = f"{base_status} {test_str}"
            else:
                status = _colorize(base_status, "32")
        else:
            status = _colorize("○ not configured", "33")

        line = _menu_row(f" {cursor} {provider_name.ljust(name_w)}  {status}", inner)

        if selected:
            print(f"\033[7m{line}\033[0m")
        else:
            print(line)

    print(_menu_border(inner))

    selected = rows[idx]
    if selected["is_exit"]:
        footer = " Save changes and exit "
    elif selected.get("is_cancel"):
        footer = " Discard changes and exit "
    else:
        footer = f" {selected['description']} | t: test key "
    print(_menu_row(footer, inner))

    if last_error:
        for i in range(0, len(last_error), inner - 4):
            chunk = last_error[i : i + inner - 4]
            print(_colorize(_menu_row(chunk, inner), "31"))

    print(_menu_border(inner))


def _show_testing_screen(provider: str) -> None:
    """Show a 'testing...' screen while validating API key."""
    print("\033[2J\033[H", end="")
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    inner = cols - 4

    print(_menu_border(inner))
    print(_menu_row(f" TESTING {provider.upper()} KEY ", inner))
    print(_menu_border(inner))
    print(_menu_row("", inner))
    print(_menu_row("  Making test API call... ", inner))
    print(_menu_row("", inner))
    print(_menu_border(inner))
    sys.stdout.flush()


def run_config_tui() -> bool:
    original_keys = load_existing_keys()
    keys = dict(original_keys)
    test_results: dict[str, tuple[bool, str]] = {}
    rows = _build_menu_rows(keys, test_results)
    idx = 0
    last_error = ""

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("config requires an interactive terminal")
        return False

    def has_changes():
        return keys != original_keys

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        while True:
            _render_menu(rows, idx, keys, last_error, has_changes())
            key = _read_menu_key()

            if key == "up":
                idx = (idx - 1) % len(rows)
                last_error = ""
            elif key == "down":
                idx = (idx + 1) % len(rows)
                last_error = ""
            elif key == "enter":
                row = rows[idx]
                if row["is_exit"]:
                    save_keys(keys)
                    print("\033[2J\033[H", end="")
                    return True
                elif row.get("is_cancel"):
                    if has_changes():
                        print("\033[2J\033[H", end="")
                        print(_colorize("Changes discarded.", "33"))
                        print()
                    return False
                else:
                    if row.get("is_local"):
                        last_error = "Local providers don't need a key. Press 't' to test."
                    elif row["env_key"]:
                        new_key = _prompt_for_key(
                            row["provider"],
                            row["env_key"],
                            row["description"],
                            row["prefix"],
                        )
                        if new_key is not None:
                            keys[row["env_key"]] = new_key
                            if row["provider"] in test_results:
                                del test_results[row["provider"]]
                            rows = _build_menu_rows(keys, test_results)
                        last_error = ""
            elif key == "test":
                if not rows[idx]["is_exit"] and not rows[idx].get("is_cancel"):
                    row = rows[idx]
                    _show_testing_screen(row["provider"])
                    if row.get("is_local"):
                        success, msg = _test_api_key(row["provider"], "")
                    else:
                        success, msg = _test_api_key(row["provider"], keys.get(row["env_key"], ""))
                    test_results[row["provider"]] = (success, msg)
                    rows = _build_menu_rows(keys, test_results)
                    if not success:
                        last_error = msg
                    else:
                        last_error = ""
            elif key == "delete":
                if (
                    not rows[idx]["is_exit"]
                    and not rows[idx].get("is_cancel")
                    and rows[idx]["has_key"]
                    and not rows[idx].get("is_local")
                ):
                    del keys[rows[idx]["env_key"]]
                    if rows[idx]["provider"] in test_results:
                        del test_results[rows[idx]["provider"]]
                    rows = _build_menu_rows(keys, test_results)
                    last_error = ""
            elif key in {"quit", "esc"}:
                if has_changes():
                    last_error = "Unsaved changes! Use 'Save & Exit' or 'Exit without saving'"
                else:
                    print("\033[2J\033[H", end="")
                    return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> int:
    saved = run_config_tui()

    if saved:
        print()
        print(_colorize(f"✓ Configuration saved to {ENV_FILE}", "32"))
        print()
        print("To apply changes in your current shell, run:")
        print(_colorize(f"  source {ENV_FILE}", "36"))
        print()
        return 0
    else:
        print("Cancelled.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
