"""
Shared Groq client for the AIDER reviewer agents.

Groq's free tier enforces a tokens-per-minute (TPM) limit. A single request
whose (input + reserved output) tokens exceed the limit is rejected with HTTP
413 ("Request too large for model ... on tokens per minute"); bursts of several
requests inside one minute are rejected with HTTP 429. Both broke the
pre-screening pipeline (the real submission produced a ~14.6k-token request
against a 12k limit).

This module centralises the fix so review.py and deep_review.py behave
identically:
  * the prompt is truncated to a hard token budget so a request can never
    exceed the TPM limit on its own, no matter how large the submission is;
  * transient 429 bursts are retried with backoff (the two agents fire within
    the same minute, so their combined usage can trip the per-minute window);
  * a 413 that slips through (estimate too optimistic for very dense content)
    halves the budget and retries.

All limits are overridable via environment variables so the budget can be
raised after a paid-tier upgrade without code changes.
"""

import os
import time

# Model + TPM configuration (env-overridable).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "12000"))
DEFAULT_OUTPUT_TOKENS = int(os.environ.get("GROQ_OUTPUT_TOKENS", "2500"))
SAFETY_MARGIN_TOKENS = int(os.environ.get("GROQ_SAFETY_MARGIN", "800"))
# Real submissions measure ~3.7 chars/token; we under-pack at 3.3 so dense
# LaTeX/code never blows the estimate.
CHARS_PER_TOKEN = float(os.environ.get("GROQ_CHARS_PER_TOKEN", "3.3"))
MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "4"))


def input_token_budget(output_tokens: int) -> int:
    """Tokens available for the prompt after reserving output + safety margin."""
    return max(1000, TPM_LIMIT - output_tokens - SAFETY_MARGIN_TOKENS)


def truncate_to_token_budget(prompt: str, token_budget: int) -> str:
    """Truncate the prompt so its estimated token count fits token_budget.

    Instruction/format directives live at the START of every prompt these
    agents build, so we keep the head and drop from the tail (the
    least-critical trailing sections).
    """
    char_budget = int(token_budget * CHARS_PER_TOKEN)
    if len(prompt) <= char_budget:
        return prompt
    note = "\n\n[... content truncated to fit the model token budget ...]"
    return prompt[: max(0, char_budget - len(note))] + note


def call_groq(prompt: str, max_tokens: int = None, temperature: float = 0.3) -> str:
    """Send a prompt to Groq, keeping the request within the TPM limit.

    Truncates the prompt to a safe budget, then retries on transient
    rate-limit bursts (429) and shrinks-and-retries on an over-budget 413.
    """
    from groq import Groq
    from groq import APIStatusError

    output_tokens = DEFAULT_OUTPUT_TOKENS if max_tokens is None else max_tokens
    budget = input_token_budget(output_tokens)
    client = Groq()

    last_err = None
    for attempt in range(MAX_RETRIES):
        bounded = truncate_to_token_budget(prompt, budget)
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": bounded}],
                max_tokens=output_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except APIStatusError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status == 413:
                budget = max(1000, budget // 2)
                print(f"413 too large; shrinking prompt budget to {budget} tokens and retrying")
                continue
            if status == 429:
                wait = 20 * (attempt + 1)
                print(f"429 rate limit; backing off {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise last_err
