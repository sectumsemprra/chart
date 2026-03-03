"""
DAPO Trainer: Decoupled Clip and Dynamic Sampling Policy Optimization.
 
Key differences from GRPO:
  1. Clip-Higher: asymmetric clip bounds (epsilon_low < epsilon_high).
     Passes dapo_epsilon_low / dapo_epsilon_high to GRPOConfig if the
     installed TRL version exposes them; otherwise falls back gracefully.
  2. Dynamic sampling: groups where ALL rollouts are identical-reward
     (all correct or all wrong) contribute zero advantage — skipping
     uninformative gradient updates.
  3. No KL penalty: beta=0.0, kl_coef=None.
  4. Higher entropy: temperature bumped to 1.0 by default.
"""
 
import inspect
from typing import List
 
from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
 
from .base_trainer import BaseTrainer, PolicyMethodMixin
 
 
class DAPOTrainer(BaseTrainer, PolicyMethodMixin):
    """
    DAPO trainer backed by TRL's GRPOTrainer.
 
    compute_advantages implements dynamic sampling (skip homogeneous groups).
    Clip-Higher parameters are forwarded to GRPOConfig when supported.
    """
 
    def _create_trl_trainer(self):
        """Override to inject DAPO-specific GRPOConfig params."""
        allowed = set(inspect.signature(GRPOConfig).parameters.keys())
 
        config_kwargs = {
            "output_dir": str(
                self._checkpoint_manager_run_dir() / "trl_output"
            ),
            "per_device_train_batch_size": self.config.batch_size,
            "num_train_epochs": self.config.num_epochs,
            "num_generations": self.config.num_generations,
            "max_prompt_length": self.config.max_prompt_length,
            "max_completion_length": self.config.max_completion_length,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": self.config.warmup_ratio,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "logging_steps": self.config.logging_steps,
            "logging_first_step": self.config.logging_first_step,
            "save_steps": self.config.checkpoint.save_every_n_steps,
            "save_total_limit": self.config.save_total_limit,
            "bf16": self.config.bf16,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            # DAPO: no KL
            "kl_coef": None,
            "beta": 0.0,
            "remove_unused_columns": self.config.remove_unused_columns,
        }
 
        # DAPO Clip-Higher — forwarded only if TRL supports them
        dapo_extra = {
            "epsilon":      getattr(self.config, "dapo_epsilon_low", 0.2),
            "epsilon_high": getattr(self.config, "dapo_epsilon_high", 0.28),
        }
        # token-level loss normalisation (TRL >= 0.15) — only if supported
        _loss_type_param = inspect.signature(GRPOConfig).parameters.get("loss_type")
        _valid_loss_types = getattr(_loss_type_param.annotation, "__args__", ()) if _loss_type_param else ()
        if "token" in _valid_loss_types:
            dapo_extra["loss_type"] = "token"
        config_kwargs.update(dapo_extra)
 
        if "report_to" in allowed:
            config_kwargs["report_to"] = (
                "wandb" if self.config.use_wandb else "none"
            )
 
        filtered = {
            k: v for k, v in config_kwargs.items()
            if k in allowed and v is not None
        }
 
        grpo_config = GRPOConfig(**filtered)
        reward_fn = self._create_reward_function()
 
        self._trl_trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            reward_funcs=reward_fn,
        )
 
    def _checkpoint_manager_run_dir(self):
        """Return the current run directory from checkpoint manager."""
        return self.checkpoint_manager.run_dir
 
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        DAPO advantage computation with dynamic sampling.
 
        If all rewards in the group are identical (std ≈ 0), the group
        is "uninformative" — return all-zeros to skip the update.
        Otherwise normalise to zero mean / unit variance (same as GRPO).
        """
        if not rewards:
            return rewards
 
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = var_r ** 0.5
 
        # Dynamic sampling: homogeneous group → zero advantages
        if std_r < 1e-6:
            return [0.0] * len(rewards)
 
        return [(r - mean_r) / std_r for r in rewards]