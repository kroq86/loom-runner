from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_name_avoids_occupied_loom_agent_package() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "loom-runner"
    assert pyproject["project"]["name"] != "loom-agent"


def test_readme_and_console_script_match_distribution_name() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert pyproject["project"]["scripts"] == {"loom-runner": "loom_agent.cli:main"}
    assert readme.startswith("# loom-runner")
    assert "loom-runner run examples/counter_agent.py" in readme
    assert "loom-agent run examples/counter_agent.py" not in readme
