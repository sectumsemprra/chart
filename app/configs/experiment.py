"""Predefined experiment configurations."""
 
from .base import TrainingConfig, RewardConfig, CheckpointConfig
from typing import Dict, List
 
 
def _make_reward_config(
    use_hcpc: bool = False,
    use_hcpc_v2: bool = False,
) -> RewardConfig:
    """Create reward config."""
    return RewardConfig(
        use_hcpc=use_hcpc,
        use_hcpc_v2=use_hcpc_v2,
        use_clc=False,
        use_process_reward=not (use_hcpc or use_hcpc_v2),
        w_type=1.0,
        w_table=2.0,
        w_reason=1.5,
        w_clc=1.0,
        table_sim_threshold=0.6,
        hcpc_v2_semantic_weight=0.5,
    )
 
 
# Shared base config for all experiments
_SHARED = dict(
    seed=2026,
    learning_rate=1e-5,
    gradient_accumulation_steps=2,
    warmup_ratio=None,
    weight_decay=None,
    kl_coef=None,
    temperature=1.0,
    top_p=1.0,
    remove_unused_columns=False,
    apply_advantages_in_reward_fn=False,
    image_min_pixels=320 * 28 * 28,
    image_max_pixels=320 * 28 * 28,
    image_resample="bicubic",
    lora_target_modules=["q_proj", "v_proj"],
    torch_dtype_auto=True,
    attn_implementation=None,
    use_flash_attention=False,
    use_python_list_dataset=True,
    wandb_project="chartrl-nsr",
    checkpoint=CheckpointConfig(
        save_every_n_steps=10,
        keep_last_n=3,
        keep_best=False,
    ),
)
 
# DAPO uses apply_advantages_in_reward_fn=True so dynamic-sampling
# advantage computation in DAPOTrainer.compute_advantages() fires.
_DAPO_SHARED = {**_SHARED, "apply_advantages_in_reward_fn": True, "num_epochs": 2}
 
 
EXPERIMENTS: Dict[str, TrainingConfig] = {
    # ── Original 6 experiments ────────────────────────────────────────────
    "grpo_baseline": TrainingConfig(
        experiment_name="grpo_baseline",
        policy_method="grpo",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),
    "grpo_hcpc": TrainingConfig(
        experiment_name="grpo_hcpc",
        policy_method="grpo",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),
    "nsr_baseline": TrainingConfig(
        experiment_name="nsr_baseline",
        policy_method="nsr",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),
    "nsr_hcpc": TrainingConfig(
        experiment_name="nsr_hcpc",
        policy_method="nsr",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),
    "w_reinforce_baseline": TrainingConfig(
        experiment_name="w_reinforce_baseline",
        policy_method="w_reinforce",
        lambda_psr=0.1,
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),
    "w_reinforce_hcpc": TrainingConfig(
        experiment_name="w_reinforce_hcpc",
        policy_method="w_reinforce",
        lambda_psr=0.1,
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),
 
    # ── DAPO experiments ──────────────────────────────────────────────────
    # Experiment 7: DAPO baseline (no HCPC)
    "dapo_baseline": TrainingConfig(
        experiment_name="dapo_baseline",
        policy_method="dapo",
        **_DAPO_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),
 
    # Experiment 8: DAPO + original HCPC (ablation)
    "dapo_hcpc": TrainingConfig(
        experiment_name="dapo_hcpc",
        policy_method="dapo",
        **_DAPO_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),
 
    # Experiment 9: DAPO + HCPC-v2 (flagship — full HCPC-RLVR + DAPO)
    "dapo_hcpc_v2": TrainingConfig(
        experiment_name="dapo_hcpc_v2",
        policy_method="dapo",
        **_DAPO_SHARED,
        rewards=_make_reward_config(use_hcpc_v2=True),
    ),
}
 
 
def get_experiment_config(name: str, **overrides) -> TrainingConfig:
    """
    Get experiment config by name with optional overrides.
 
    Example:
        config = get_experiment_config("dapo_hcpc_v2", subset_size=2000)
    """
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {name!r}. Available: {list(EXPERIMENTS.keys())}"
        )
 
    config_dict = EXPERIMENTS[name].to_dict()
 
    for key, value in overrides.items():
        if key == "rewards" and isinstance(value, dict):
            for rk, rv in value.items():
                config_dict["rewards"][rk] = rv
        elif key == "checkpoint" and isinstance(value, dict):
            for ck, cv in value.items():
                config_dict["checkpoint"][ck] = cv
        else:
            config_dict[key] = value
 
    return TrainingConfig.from_dict(config_dict)
 
 
def list_experiments() -> List[str]:
    """Print and return available experiment names."""
    print("Available experiments:")
    print("-" * 60)
    for name, config in EXPERIMENTS.items():
        if config.rewards.use_hcpc_v2:
            reward_tag = "HCPC-v2"
        elif config.rewards.use_hcpc:
            reward_tag = "HCPC   "
        else:
            reward_tag = "       "
        print(f"  {name:30} | {config.policy_method:12} | {reward_tag}")
    print("-" * 60)
    return list(EXPERIMENTS.keys())