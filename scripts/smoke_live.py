"""Live check with a REAL model. Reads credentials only from environment variables (never hardcoded).

Run (from pmp_agent/):
  OPENAI_API_KEY=your-openai-key python -m scripts.smoke_live
or
  GROQ_API_KEY=your-groq-key python -m scripts.smoke_live
"""
import asyncio
import sys
from datetime import date

from data.sample_project import build_sample_store
from domain.models import InputMessage, SourceType
from pipeline.approvals import ApprovalQueue
from pipeline.runner import run_update
from pmp_config import ConfigError, load_config
from project_agents.llm import call_llm, configure_runtime, get_runtime
from project_agents.workers import sdk_worker_runner


async def main():
    try:
        cfg = load_config()
    except ConfigError as exc:
        sys.exit(f"Configuration error: {exc}")
    if not cfg.live:
        sys.exit("Set OPENAI_API_KEY or GROQ_API_KEY first (no live credentials configured).")

    configure_runtime(cfg)
    store, queue = build_sample_store(date.today()), ApprovalQueue()
    msg = InputMessage(
        source=SourceType.EMPLOYEE,
        sender="Ahmed",
        text="I finished the Mobile layout, but the Payment API is still blocked by the vendor.",
    )
    print(f"Starting PMP live run with {cfg.describe()}...", flush=True)
    r = await run_update(msg, store, queue, llm=call_llm, worker_runner=sdk_worker_runner)
    print("\n--- Summary ---", flush=True)
    print("status:", r.status, "| message:", r.message, flush=True)
    print("workers:", list(r.worker_reports), "| tools:", r.tool_audit, flush=True)
    print("pending approvals:", len(queue.pending()), flush=True)
    rt = get_runtime()
    if rt.metrics:
        total_in = sum(m.input_tokens for m in rt.metrics)
        total_out = sum(m.output_tokens for m in rt.metrics)
        total_req = sum(m.requests for m in rt.metrics)
        print(f"stage usage (requests={total_req}, in={total_in}, out={total_out}):", rt.metrics_summary(), flush=True)
    if rt.events:
        print("provider diagnostics:", list(rt.events), flush=True)
    if rt.client is not None:
        await rt.client.close()


if __name__ == "__main__":
    asyncio.run(main())
