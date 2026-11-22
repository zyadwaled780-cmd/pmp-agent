"""SDK-native guardrails (InputGuardrail / OutputGuardrail) that wrap the partner's core checks."""
import json
import re

from agents import GuardrailFunctionOutput, input_guardrail, output_guardrail

from pipeline.adapters import check_output_text
from pipeline.adapters import input_guardrail as core_input_guardrail

_MESSAGE = re.compile(r"<message>\n(.*?)\n</message>", re.S)


def _flatten(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)


@input_guardrail
async def pmp_input_guardrail(ctx, agent, input) -> GuardrailFunctionOutput:
    """Trips when the user's message tries to override the rules (checked on every <message> block)."""
    text = _flatten(input)
    bodies = _MESSAGE.findall(text) or [text]
    for body in bodies:
        r = core_input_guardrail(body)
        if not r.ok:
            return GuardrailFunctionOutput(output_info=r.model_dump(), tripwire_triggered=True)
    return GuardrailFunctionOutput(output_info={"ok": True}, tripwire_triggered=False)


@output_guardrail
async def pmp_output_guardrail(ctx, agent, output) -> GuardrailFunctionOutput:
    """Trips when the writer's draft promises dates or claims unproven completion."""
    parts = [getattr(output, "summary", "") or ""]
    parts += [a.comment or "" for a in getattr(output, "proposed_actions", [])]
    r = check_output_text("\n".join(parts))
    return GuardrailFunctionOutput(output_info=r.model_dump(), tripwire_triggered=not r.ok)


def guardrails_for(agent_name: str):
    """(input_guardrails, output_guardrails) for a given agent."""
    if agent_name == "Extraction Agent":
        return [pmp_input_guardrail], []
    if agent_name == "Status Writer":
        return [], [pmp_output_guardrail]
    return [], []
