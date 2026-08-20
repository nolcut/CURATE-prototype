from __future__ import annotations
import os
import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse


MAX_RETRIES = int(os.environ.get("FAASR_LLM_MAX_RETRIES", "8"))

SONNET_MODEL_BEDROCK = os.environ.get(
    "FAASR_SONNET_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"
)

OPUS_MODEL_BEDROCK = os.environ.get(
    "FAASR_OPUS_BEDROCK_MODEL", "us.anthropic.claude-opus-4-5-20251101-v1:0"
)

SONNET_MODEL_ANTHROPIC = "claude-sonnet-4-6"
OPUS_MODEL_ANTHROPIC = "claude-opus-4-8"
OPENAI_MODEL = os.environ.get("FAASR_OPENAI_MODEL", "gpt-5")

MODELS = {
    "bedrock": {
        "sonnet": SONNET_MODEL_BEDROCK,
        "opus":   OPUS_MODEL_BEDROCK,
    },
    "anthropic": {
        "sonnet": SONNET_MODEL_ANTHROPIC,
        "opus":   OPUS_MODEL_ANTHROPIC,
    },
    "openai": {
        "sonnet": OPENAI_MODEL,
        "opus":   OPENAI_MODEL,
    },
}

selected_tier = "sonnet"

# Provider defaults to Bedrock. Direct provider APIs are opt-in ONLY via CLI
# flags; keys in the env are never auto-detected into a provider switch.
selected_provider = "bedrock"


def set_model_tier(tier: str) -> None:
    """Select the model tier ('opus' or 'sonnet') for subsequent get_llm() calls."""
    if tier not in MODELS["bedrock"]:
        raise ValueError(
            f"Unknown model tier '{tier}'; expected one of {sorted(MODELS['bedrock'])}"
        )
    global selected_tier
    selected_tier = tier


def set_provider(provider: str) -> None:
    """Select the LLM provider for subsequent get_llm() calls."""
    if provider not in MODELS:
        raise ValueError(
            f"Unknown provider '{provider}'; expected one of {sorted(MODELS)}"
        )
    global selected_provider
    selected_provider = provider


def using_anthropic() -> bool:
    """True when the CLI opted into the Anthropic API via --anthropic-api."""
    return selected_provider == "anthropic"


def using_openai() -> bool:
    """True when the CLI opted into the OpenAI API via --openai-api."""
    return selected_provider == "openai"


def get_default_model() -> str:
    """Resolve the model ID for the currently selected provider + tier."""
    return MODELS[selected_provider][selected_tier]


def make_bedrock_llm(model: str):
    api_key = os.environ.get("BEDROCK_API_KEY")
    if api_key and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

    if not (
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
    ):
        raise RuntimeError(
            "No AWS Bedrock credentials found. Set BEDROCK_API_KEY (BedrockAPIKey-…) "
            "or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (or AWS_PROFILE) in .env. "
        )

    region = os.environ.get("AWS_REGION", "us-east-1")
    key_id  = os.environ.get("AWS_ACCESS_KEY_ID")
    secret  = os.environ.get("AWS_SECRET_ACCESS_KEY")
    token   = os.environ.get("AWS_SESSION_TOKEN") or None

    session = boto3.Session(
        aws_access_key_id=key_id or None,
        aws_secret_access_key=secret or None,
        aws_session_token=token,
        region_name=region,
    )
    # Adaptive retries
    client_config = Config(retries={"max_attempts": MAX_RETRIES, "mode": "adaptive"})
    return ChatBedrockConverse(
        model=model,
        client=session.client("bedrock-runtime", config=client_config),
        max_tokens=8192,
    )


def make_anthropic_llm(model: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "--anthropic-api was passed but ANTHROPIC_API_KEY is not set. "
            "Add it to .env (sk-ant-… from console.anthropic.com)."
        )
    # Imported lazily so Bedrock-only environments don't need langchain-anthropic.
    from langchain_anthropic import ChatAnthropic

    # max_retries → the Anthropic SDK retries 429/500/529 (overloaded) with backoff.
    return ChatAnthropic(
        model=model, anthropic_api_key=api_key, max_tokens=8192,
        max_retries=MAX_RETRIES,
    )


def make_openai_llm(model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "--openai-api was passed but OPENAI_API_KEY is not set. "
            "Add it to .env (sk-proj-... from platform.openai.com)."
        )
    # Imported lazily so Bedrock/Anthropic-only environments don't need it.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        max_retries=MAX_RETRIES,
    )


def get_llm(model: str | None = None):
    if model is None:
        model = get_default_model()
    if using_anthropic():
        return make_anthropic_llm(model)
    if using_openai():
        return make_openai_llm(model)
    return make_bedrock_llm(model)
