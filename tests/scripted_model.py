"""A scripted model for the OpenAI Agents SDK: lets tests drive the REAL Runner (tools, context,
guardrails, sessions) without an API key. It is not a language model; it answers from a function."""
import json
import uuid

from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText


def tool_call(name, args=None):
    return ResponseFunctionToolCall(arguments=json.dumps(args or {}), call_id=uuid.uuid4().hex[:8],
                                    name=name, type="function_call")


def message(text):
    return ResponseOutputMessage(id="msg_" + uuid.uuid4().hex[:8], role="assistant", status="completed", type="message",
                                 content=[ResponseOutputText(text=text, type="output_text", annotations=[])])


def _get(item, key):
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def tool_outputs(input_items):
    if isinstance(input_items, str):
        return []
    return [str(_get(i, "output")) for i in input_items if _get(i, "type") == "function_call_output"]


def input_text(input_items):
    return input_items if isinstance(input_items, str) else json.dumps(input_items, default=str)


class ScriptedModel(Model):
    """respond(system_instructions, input, tools) -> list of output items."""

    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing,
                           *, previous_response_id=None, conversation_id=None, prompt=None):
        self.requests.append({"instructions": system_instructions or "", "input": input})
        out = self.respond(system_instructions or "", input, tools)
        if isinstance(out, Exception):
            raise out
        return ModelResponse(output=out, usage=Usage(), response_id=None)

    async def stream_response(self, *a, **k):
        raise NotImplementedError
        yield  # pragma: no cover
