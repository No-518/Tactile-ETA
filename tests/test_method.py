"""Check task entrypoints and tactile selection without a simulator."""
from pathlib import Path

import pytest

from run import TASKS, check_demonstrations, config_for
from scripts.univtac.run_autonomous_insert_hole import (
    codex_command, demonstration_projection, operator_prompt,
)
from sim.envs.univtac.tactile_history import select_times


@pytest.mark.parametrize("task", TASKS)
def test_task_context_and_tools(task: str, tmp_path: Path) -> None:
    config = config_for(task, 1000000)
    prompt = operator_prompt(config)
    command = codex_command(tmp_path, "http://127.0.0.1:12345", config)
    projection = demonstration_projection(config)
    assert config["task_instruction"] in prompt
    assert "review_demonstrations" in prompt
    assert "gpt-6-astra" in command
    assert config["current_tactile"] is True
    assert len(projection["text"]["examples"]) == 2
    assert {image["label"].split()[-2] for image in projection["images"] if "history" in image["label"]} == {
        "left_tactile", "right_tactile",
    }
    assert check_demonstrations(task)["images"] == len(projection["images"])


@pytest.mark.parametrize("strengths, expected", [
    ([0, 0.2, 0.9, 0.1, 0], [0, 4]),
    ([0, 0.2, 1.0, 0.1, 0], [0, 1, 2, 4]),
    ([0, 0.2, 1.2, 1.2, 0], [0, 1, 2, 4]),
    ([0], [0]),
])
def test_bilateral_threshold_and_first_peak(strengths, expected) -> None:
    rows = [
        {"sample_id": i, "features": {
            "left_tactile": {"change_strength": 0},
            "right_tactile": {"change_strength": value},
        }}
        for i, value in enumerate(strengths)
    ]
    assert [row["sample_id"] for row in select_times(rows)] == expected
