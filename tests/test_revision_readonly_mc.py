from __future__ import annotations

import pandas as pd

from scripts.audit_revision_mc1_mc3_mc5 import action_effective_updates


def test_action_effective_updates_excludes_terminal_only_update() -> None:
    row = pd.Series(
        {
            "variant": "AdaptiveSAOS",
            "Budget": 50,
            "Transition_gen": 35,
            "PPO_rollout_sizes": "[15]",
        }
    )
    assert action_effective_updates(row) == 0


def test_generation_zero_ppo_only_counts_only_action_affecting_updates() -> None:
    row = pd.Series(
        {
            "variant": "PPOOnly",
            "Budget": 50,
            "Transition_gen": -1,
            "PPO_rollout_sizes": "[16, 16, 16, 2]",
        }
    )
    assert action_effective_updates(row) == 3


def test_preterminal_update_is_action_effective() -> None:
    row = pd.Series(
        {
            "variant": "AdaptiveSAOS",
            "Budget": 50,
            "Transition_gen": 33,
            "PPO_rollout_sizes": "[16]",
        }
    )
    assert action_effective_updates(row) == 1
