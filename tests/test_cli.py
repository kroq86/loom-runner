from __future__ import annotations

import json

from loom_agent.cli import main


def test_cli_explain_outputs_json(tmp_path, capsys) -> None:
    db = tmp_path / "runs.sqlite"
    module = "examples/counter_agent.py"

    assert main(["run", module, "--run-id", "demo", "--db", str(db), "--max-steps", "5"]) == 0
    capsys.readouterr()

    assert main(["explain", module, "--run-id", "demo", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["run_id"] == "demo"
    assert payload["status"] == "paused"
    assert payload["step_index"] == 5
    assert payload["checkpoint_count"] == 6
    assert payload["state"] == {"current": 5, "target": 10}


def test_cli_history_supports_limit_and_offset(tmp_path, capsys) -> None:
    db = tmp_path / "runs.sqlite"
    module = "examples/counter_agent.py"

    assert main(["run", module, "--run-id", "demo", "--db", str(db), "--max-steps", "5"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "history",
                module,
                "--run-id",
                "demo",
                "--db",
                str(db),
                "--limit",
                "2",
                "--offset",
                "2",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert [step["step_index"] for step in payload] == [2, 3]
