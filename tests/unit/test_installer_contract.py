"""Installation contracts for shared review-skill discovery."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_installer_refreshes_claude_and_shared_agent_skills() -> None:
    installer = (ROOT / "commands" / "install" / "run").read_text(encoding="utf-8")

    assert '"$HOME/.claude/skills" "$HOME/.agents/skills"' in installer
    assert "for skill_name in mm-review mm-review-loop" in installer
    assert 'mktemp "$skill_target/.SKILL.md.XXXXXX"' in installer
    assert 'mv -f "$skill_tmp" "$skill_target/SKILL.md"' in installer
    assert 'mv -f "$marker_tmp" "$skill_target/.ai-mm-managed"' in installer
    assert '.ai-mm-managed' in installer
    assert "Skipping unmanaged customized skill" in installer
    assert 'unlink "$skill_target"' in installer
    assert "Skipping skill with nested symlink" in installer
    assert "Skipping skill with invalid ownership marker" in installer
    assert '[[ "$(<"$managed_marker")" != "managed-by=ai-mm" ]]' in installer
    assert "shopt -s nullglob dotglob" in installer
    assert '[[ "${entries[0]}" == "$skill_target/SKILL.md" ]]' in installer
    assert '[[ -f "${entries[0]}" && ! -L "${entries[0]}" ]]' in installer


def _review_skills() -> list[str]:
    return [
        (ROOT / "claude" / "skills" / name / "SKILL.md").read_text(
            encoding="utf-8"
        )
        for name in ("mm-review", "mm-review-loop")
    ]


def test_review_docs_use_semantic_selectors_not_provider_endpoints() -> None:
    skills = _review_skills()

    for skill in skills:
        assert "ai review --model local" in skill
        assert "ai review --model deepseek" in skill
        assert "--model lmstudio" not in skill
        assert "OLLAMA_BASE_URL" not in skill
        assert "LM_STUDIO" not in skill

    models = (ROOT / "src" / "claude_mm" / "models.py").read_text(encoding="utf-8")
    assert '"local": ["profile:local_only"]' in models


def test_normal_review_docs_use_one_bounded_deepseek_seat() -> None:
    normal_skill = _review_skills()[0]
    assert (
        "ai review --model deepseek --focus {chosen_focus} "
        "--per-model-timeout 180"
    ) in normal_skill
    assert "never an automatic fallback" in normal_skill
