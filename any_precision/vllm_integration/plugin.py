"""
vLLM plugin registration for AnyPrecisionQwen3ForCausalLM.

vLLM auto-discovers this via the entry point in pyproject.toml:

  [project.entry-points."vllm.general_plugins"]
  any_precision_vllm = "any_precision.vllm_integration.plugin:register"

When vllm is started it calls register(), which maps the HuggingFace
architecture name "Qwen3ForCausalLM" to our custom model class.
"""


def register() -> None:
    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "any_precision.vllm_integration.model:AnyPrecisionQwen3ForCausalLM",
    )
