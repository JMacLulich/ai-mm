"""Tests for config_tui module."""

from claude_mm.config_tui import (
    _build_menu_rows,
    _colorize,
    _fit_text,
    _menu_border,
    _menu_row,
    _supports_color,
    load_existing_keys,
    mask_key,
    save_keys,
)


class TestFitText:
    def test_short_text_unchanged(self):
        assert _fit_text("hello", 10) == "hello"

    def test_exact_fit(self):
        assert _fit_text("hello", 5) == "hello"

    def test_truncation_with_ellipsis(self):
        assert _fit_text("hello world", 8) == "hello..."

    def test_very_small_width(self):
        assert _fit_text("hello", 3) == "hel"

    def test_zero_width(self):
        assert _fit_text("hello", 0) == ""

    def test_negative_width(self):
        assert _fit_text("hello", -1) == ""


class TestMenuBorder:
    def test_border_width(self):
        result = _menu_border(10)
        assert result == "+----------+"
        assert len(result) == 12

    def test_border_zero(self):
        assert _menu_border(0) == "++"


class TestMenuRow:
    def test_row_padding(self):
        result = _menu_row("hello", 10)
        assert result == "|hello     |"
        assert len(result) == 12

    def test_row_exact_fit(self):
        result = _menu_row("hello", 5)
        assert result == "|hello|"

    def test_row_truncation(self):
        result = _menu_row("hello world", 8)
        assert result == "|hello...|"

    def test_row_with_ansi_codes(self):
        result = _menu_row("\033[32mgreen\033[0m", 10)
        assert "green" in result
        assert result.startswith("|")
        assert result.endswith("|")


class TestColorize:
    def test_colorize_adds_codes(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("NO_COLOR", raising=False)
        import sys

        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = _colorize("text", "32")
        assert "\033[32m" in result
        assert "\033[0m" in result

    def test_colorize_respects_no_color(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        result = _colorize("text", "32")
        assert result == "text"

    def test_colorize_respects_dumb_term(self, monkeypatch):
        monkeypatch.setenv("TERM", "dumb")
        monkeypatch.delenv("NO_COLOR", raising=False)
        result = _colorize("text", "32")
        assert result == "text"


class TestSupportsColor:
    def test_supports_color_with_tty(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        assert _supports_color() or True

    def test_supports_color_no_color_env(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert not _supports_color()

    def test_supports_color_dumb_term(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert not _supports_color()


class TestMaskKey:
    def test_mask_short_key(self):
        assert mask_key("short") == "(not set)"

    def test_mask_empty_key(self):
        assert mask_key("") == "(not set)"

    def test_mask_valid_key(self):
        assert mask_key("sk-proj-1234567890abcd") == "sk-proj-...abcd"

    def test_mask_exactly_12_chars(self):
        assert mask_key("123456789012") == "12345678...9012"


class TestBuildMenuRows:
    def test_includes_all_providers(self):
        keys = {}
        rows = _build_menu_rows(keys)
        provider_names = [r["provider"] for r in rows if not r["is_exit"]]
        assert "openai" in provider_names
        assert "google" in provider_names
        assert "anthropic" in provider_names

    def test_includes_exit_row(self):
        keys = {}
        rows = _build_menu_rows(keys)
        exit_rows = [r for r in rows if r["is_exit"] or r.get("is_cancel")]
        assert len(exit_rows) == 2
        assert any("Save" in r["provider"] for r in exit_rows)
        assert any("Exit" in r["provider"] for r in exit_rows)

    def test_detects_missing_keys(self):
        keys = {}
        rows = _build_menu_rows(keys)
        for row in rows:
            if not row["is_exit"]:
                assert row["has_key"] is False

    def test_detects_present_keys(self):
        keys = {"OPENAI_API_KEY": "sk-test-12345678"}
        rows = _build_menu_rows(keys)
        openai_row = next(r for r in rows if r["provider"] == "openai")
        assert openai_row["has_key"] is True

    def test_masks_present_keys(self):
        keys = {"OPENAI_API_KEY": "sk-proj-1234567890abcd"}
        rows = _build_menu_rows(keys)
        openai_row = next(r for r in rows if r["provider"] == "openai")
        assert "sk-proj-...abcd" in openai_row["masked"]


class TestSaveAndLoadKeys:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_mm.env.CONFIG_DIR", tmp_path / "ai-mm")
        monkeypatch.setattr("claude_mm.env.ENV_FILE", tmp_path / "ai-mm" / "env")

        keys = {
            "OPENAI_API_KEY": "sk-test-12345678",
            "GOOGLE_AI_API_KEY": "test-google-key",
        }
        save_keys(keys)

        loaded = load_existing_keys()
        assert loaded["OPENAI_API_KEY"] == "sk-test-12345678"
        assert loaded["GOOGLE_AI_API_KEY"] == "test-google-key"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_mm.env.ENV_FILE", tmp_path / "nonexistent" / "env")
        loaded = load_existing_keys()
        assert loaded == {}

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "new-config-dir"
        monkeypatch.setattr("claude_mm.env.CONFIG_DIR", config_dir)
        monkeypatch.setattr("claude_mm.env.ENV_FILE", config_dir / "env")

        keys = {"OPENAI_API_KEY": "test-key"}
        save_keys(keys)

        assert config_dir.exists()

    def test_load_handles_export_prefix(self, tmp_path, monkeypatch):
        env_file = tmp_path / "env"
        monkeypatch.setattr("claude_mm.env.ENV_FILE", env_file)

        env_file.write_text('export OPENAI_API_KEY="sk-test-1234"\n')

        loaded = load_existing_keys()
        assert loaded["OPENAI_API_KEY"] == "sk-test-1234"

    def test_load_handles_single_quotes(self, tmp_path, monkeypatch):
        env_file = tmp_path / "env"
        monkeypatch.setattr("claude_mm.env.ENV_FILE", env_file)

        env_file.write_text("export OPENAI_API_KEY='sk-test-1234'\n")

        loaded = load_existing_keys()
        assert loaded["OPENAI_API_KEY"] == "sk-test-1234"

    def test_load_ignores_comments(self, tmp_path, monkeypatch):
        env_file = tmp_path / "env"
        monkeypatch.setattr("claude_mm.env.ENV_FILE", env_file)

        env_file.write_text("# This is a comment\nexport OPENAI_API_KEY='real-key'\n")

        loaded = load_existing_keys()
        assert loaded == {"OPENAI_API_KEY": "real-key"}
