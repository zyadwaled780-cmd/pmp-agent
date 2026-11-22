"""End-to-end headless-browser integration tests for the Gradio UI (100% offline)."""
from __future__ import annotations

import os
import re
import socket
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import date

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, expect, sync_playwright

import app as pmp_app
import pmp_config
from domain.models import (
    ExtractedUpdate,
    ProposedAction,
    SourceType,
    TaskItem,
)
from project_agents.evaluator_agent import EvalResult
from project_agents.llm import ProviderCallError
from project_agents.orchestrator import OrchestrationPlan
from project_agents.risk_analyzer import RiskAnalysis
from project_agents.status_writer import ReportDraft


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ready(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(0.05)
    raise RuntimeError(f"Gradio server did not become ready at {url}: {last_err}")


def _launch_chromium(p) -> Browser:
    for kw in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    raise RuntimeError("No Chromium/Chrome/Edge browser executable available for Playwright")


@contextmanager
def running_gradio_app(**build_kwargs):
    port = _free_port()
    demo = pmp_app.build_ui(**build_kwargs)
    url = f"http://127.0.0.1:{port}"
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            quiet=True,
            show_error=True,
            inbrowser=False,
        )
        _wait_http_ready(url)
        yield url
    finally:
        try:
            demo.close()
        except Exception:
            pass


def _select_approval_id(page: Page, *, row_match: str | None = None) -> str:
    """Read a pending 6-char approval id from the Waiting for approval grid and select it."""
    grid = page.get_by_role("grid", name="Waiting for approval")
    row = grid.get_by_role("row").filter(has_text=row_match).first if row_match else grid.get_by_role("row").nth(1)
    row_text = row.inner_text()
    m = re.search(r"\b([a-f0-9]{6})\b", row_text)
    assert m is not None, f"Could not find a pending 6-char approval id in row:\n{row_text}"
    pid = m.group(1)
    dropdown = page.get_by_label("Pick an item (id)")
    dropdown.click()
    page.get_by_role("option", name=pid).click()
    return pid


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = _launch_chromium(p)
        try:
            yield b
        finally:
            b.close()


def _new_page(browser: Browser, url: str) -> tuple[BrowserContext, Page]:
    ctx = browser.new_context()
    page = ctx.new_page()
    page.set_default_timeout(15000)
    page.goto(url, wait_until="domcontentloaded")
    expect(page.get_by_role("heading", name="AI Project Operations Agent")).to_be_visible()
    # Wait for demo.load refresh to populate state tables.
    expect(page.get_by_text("Offline demo mode:")).to_be_visible()
    return ctx, page


def _new_page_live(browser: Browser, url: str, provider: str, model: str) -> tuple[BrowserContext, Page]:
    ctx = browser.new_context()
    page = ctx.new_page()
    page.set_default_timeout(45000)
    page.goto(url, wait_until="domcontentloaded")
    expect(page.get_by_role("heading", name="AI Project Operations Agent")).to_be_visible()
    expect(page.get_by_text(f"Live provider ready: {provider} / {model}")).to_be_visible()
    return ctx, page


_TERMINAL_HEADINGS = re.compile(
    r"^(?:🟠 Waiting for the manager|✅ Completed|⛔ Blocked \(suspicious message\)|"
    r"⛔ Refused|❓ Needs clarification|❌ Error|❌ Configuration error|❌ No live provider configured)$"
)


def _steps_text(page: Page) -> list[str]:
    return page.locator("details").all_inner_texts()


def _wait_for_run_outcome(
    page: Page,
    expected_heading: str,
    *,
    prev_steps: list[str] | None = None,
    timeout: int = 90000,
) -> None:
    """Wait for a fresh terminal run banner; fail fast on provider/pipeline error instead of timing out."""
    deadline = time.monotonic() + (timeout / 1000.0)
    if prev_steps is not None:
        while time.monotonic() < deadline:
            if _steps_text(page) != prev_steps:
                break
            page.wait_for_timeout(250)
        else:
            raise AssertionError("Timed out waiting for second run steps to update.")
    remaining_ms = max(1000, int((deadline - time.monotonic()) * 1000))
    banner_heading = page.locator("h2").filter(has_text=_TERMINAL_HEADINGS).first
    expect(banner_heading).to_be_visible(timeout=remaining_ms)
    actual = banner_heading.inner_text().strip()
    if actual != expected_heading:
        body = page.locator("body").inner_text()
        raise AssertionError(f"Expected banner {expected_heading!r}, got {actual!r}.\nPage snapshot:\n{body}")


def test_offline_startup_normal_critical_approval_rejection_repeat_isolation_reset(browser: Browser):
    clock = {"today": date(2026, 10, 16)}
    clock_lock = threading.Lock()

    def current_date() -> date:
        with clock_lock:
            return clock["today"]

    with running_gradio_app(env={}, today_provider=current_date) as url:
        ctx1, page1 = _new_page(browser, url)
        ctx2, page2 = _new_page(browser, url)
        try:
            # 1. App startup & offline-mode indication with no API keys
            offline_chk = page1.get_by_label("Use offline demo mode (no API key)")
            expect(offline_chk).to_be_checked()
            expect(offline_chk).to_be_disabled()
            expect(page1.get_by_text("auto: no provider credentials found")).to_be_visible()

            # 2. Normal update in Session 1 displays report & proposes update_task on T5
            page1.get_by_label("Sender name").fill("Ahmed")
            page1.get_by_label("Message").fill("I finished the Mobile layout.")
            page1.get_by_role("button", name="Run", exact=True).click()
            expect(page1.get_by_role("heading", name="🟠 Waiting for the manager")).to_be_visible()
            expect(page1.get_by_role("heading", name="Summary")).to_be_visible()
            expect(page1.get_by_text("waiting for manager")).to_be_visible()

            # 3. Session isolation: Session 2 has no report, no Mobile layout comment, and own budget run
            page2.get_by_role("tab", name="2. Manager approvals").click()
            expect(page2.get_by_text("No decisions yet.")).to_be_visible()
            page2.get_by_role("tab", name="3. Project state").click()
            expect(page2.get_by_text("Project deadline: 2026-10-20")).to_be_visible()
            expect(page2.get_by_text("No comments yet.")).to_be_visible()

            page2.get_by_role("tab", name="1. Send an update").click()
            page2.get_by_label("From").click()
            page2.get_by_role("option", name="accountant").click()
            page2.get_by_label("Sender name").fill("Nadia")
            page2.get_by_label("Message").fill("Spending passed 85% of the budget.")
            page2.get_by_role("button", name="Run", exact=True).click()
            expect(page2.get_by_text("Done. The report is ready")).to_be_visible()
            page2.get_by_role("tab", name="3. Project state").click()
            expect(page2.get_by_text("Budget used: 85%")).to_be_visible()

            # Session 1 must NOT see Session 2's 85% budget
            page1.get_by_role("tab", name="3. Project state").click()
            expect(page1.get_by_text("Budget used: not reported")).to_be_visible()

            # 4. Approve action in Session 1 & verify repeated click never executes twice
            page1.get_by_role("tab", name="2. Manager approvals").click()
            expect(page1.get_by_role("grid", name="Waiting for approval")).to_contain_text("update_task")
            pid1 = _select_approval_id(page1, row_match="update_task")
            approve_btn = page1.get_by_role("button", name="Approve and run", exact=True)
            approve_btn.click()
            expect(page1.get_by_text(re.compile(r"approved: Updated (?:task )?T5"))).to_be_visible()
            expect(page1.get_by_text(f"{pid1} update_task T5: approved")).to_be_visible()

            # Second click after approval cannot execute again (item cleared -> Pick an item first)
            approve_btn.click()
            expect(page1.get_by_text("Pick an item first.")).to_be_visible()
            expect(page1.locator("li").filter(has_text=f"{pid1} update_task T5: approved")).to_have_count(1)

            # Verify T5 is now done in Session 1 state
            page1.get_by_role("tab", name="3. Project state").click()
            expect(
                page1.get_by_role("grid", name="Tasks").get_by_role("row", name=re.compile(r"T5\s+Mobile layout"))
            ).to_contain_text("done")

            # 5. Short critical input ("system is down") takes the critical path
            page1.get_by_role("tab", name="1. Send an update").click()
            page1.get_by_label("Message").fill("system is down")
            page1.get_by_role("button", name="Run", exact=True).click()
            expect(page1.locator("blockquote").filter(has_text="Critical path: the manager is alerted")).to_be_visible()
            expect(page1.locator("div").filter(has_text="4. Router").filter(has_text="CRITICAL PATH").first).to_be_visible()

            # 6. Trigger another pending approval and Reject it
            page1.get_by_label("Sender name").fill("Sara")
            page1.get_by_label("Message").fill("Payment API is finished.")
            page1.get_by_role("button", name="Run", exact=True).click()
            expect(page1.get_by_role("heading", name="🟠 Waiting for the manager")).to_be_visible()

            page1.get_by_role("tab", name="2. Manager approvals").click()
            pid2 = _select_approval_id(page1, row_match="update_task")
            page1.get_by_role("button", name="Reject", exact=True).click()
            expect(page1.get_by_text("rejected: Rejected by manager")).to_be_visible()
            expect(page1.get_by_text(f"{pid2} update_task T2: rejected")).to_be_visible()

            # 7. Reset Session 1 clears only Session 1 and regenerates fresh sample dates (2026-10-25 + 4d = 2026-10-29)
            with clock_lock:
                clock["today"] = date(2026, 10, 25)
            page1.get_by_role("tab", name="3. Project state").click()
            page1.get_by_role("button", name="Reset to sample data", exact=True).click()
            expect(page1.get_by_text("Project deadline: 2026-10-29")).to_be_visible()
            expect(page1.get_by_text("No comments yet.")).to_be_visible()
            page1.get_by_role("tab", name="2. Manager approvals").click()
            expect(page1.get_by_text("No decisions yet.")).to_be_visible()

            # Session 2 remains completely untouched with old deadline & its 85% budget
            expect(page2.get_by_text("Project deadline: 2026-10-20")).to_be_visible()
            expect(page2.get_by_text("Budget used: 85%")).to_be_visible()
        finally:
            ctx1.close()
            ctx2.close()


def test_browser_withholds_unsafe_output_and_sanitizes_untrusted_text(browser: Browser):
    xss_sender = '<img src=x onerror="window.__xss=1">Ahmed'
    xss_message = 'Checking logs <script>window.__xss=2</script><b>no change</b>'
    xss_task_title = '<img src=x onerror="window.__xss=4">Fix XSS task'

    def unsafe_llm_factory(tasks):
        async def _llm(name, instructions, prompt, output_type):
            if output_type is ExtractedUpdate:
                if "Fix XSS task" in prompt:
                    return ExtractedUpdate(
                        source=SourceType.EMPLOYEE,
                        sender=xss_sender,
                        summary=f"Message from {xss_sender}: create task {xss_task_title}",
                        tasks=[TaskItem(title=xss_task_title, owner="Ahmed", status="todo")],
                        blockers=[],
                        risks=[],
                    )
                return ExtractedUpdate(
                    source=SourceType.EMPLOYEE,
                    sender=xss_sender,
                    summary=f"Message from {xss_sender}: checked logs",
                    tasks=[],
                    blockers=[],
                    risks=[],
                )
            if output_type is OrchestrationPlan:
                return OrchestrationPlan(workers=["tasks"], focus="Check logs")
            if output_type is RiskAnalysis:
                return RiskAnalysis(
                    risks=[],
                    data_gaps=[],
                )
            if output_type is EvalResult:
                return EvalResult(passed=True, issues=[])
            if output_type is ReportDraft:
                if "Fix XSS task" in prompt:
                    return ReportDraft(
                        summary=f"Queued safe task creation for {xss_task_title}",
                        proposed_actions=[
                            ProposedAction(
                                action="create_task",
                                title=xss_task_title,
                                owner="Ahmed",
                                estimate_days=1.0,
                                reason="Requested task",
                                evidence="Fix XSS task",
                            ),
                        ],
                    )
                return ReportDraft(
                    summary="Task T1 is completed and we guarantee shipping tomorrow. <script>window.__xss=3</script>",
                    proposed_actions=[
                        ProposedAction(
                            action="update_task",
                            task_id="T5",
                            status="done",
                            reason="Hallucinated completion",
                            evidence="none",
                        ),
                    ],
                )
            raise AssertionError(f"Unexpected output type: {output_type}")

        return _llm

    with running_gradio_app(env={}, llm_factory=unsafe_llm_factory) as url:
        ctx, page = _new_page(browser, url)
        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        try:
            page.get_by_label("Sender name").fill(xss_sender)
            page.get_by_label("Message").fill(xss_message)
            page.get_by_role("button", name="Run", exact=True).click()

            # Unsafe completion / promise output must be withheld and sent to manager; no automatic execution
            expect(page.get_by_role("heading", name="🟠 Waiting for the manager")).to_be_visible()
            expect(page.get_by_text("The draft report was withheld by the output guardrail")).to_be_visible()
            expect(page.get_by_text("we guarantee shipping tomorrow")).not_to_be_visible()
            expect(page.get_by_text("executed:")).to_have_count(0)

            # T5 must NOT be automatically updated to done
            page.get_by_role("tab", name="3. Project state").click()
            t5_row = page.get_by_role("grid", name="Tasks").get_by_role("row", name=re.compile(r"T5\s+Mobile layout"))
            expect(t5_row).to_contain_text("in_progress")
            expect(t5_row).not_to_contain_text("done")

            # Also create a task with an untrusted HTML title & verify Tasks table renders it without executing HTML
            page.get_by_role("tab", name="1. Send an update").click()
            page.get_by_label("Message").fill("Please add task: Fix XSS task")
            page.get_by_role("button", name="Run", exact=True).click()
            expect(page.locator("p").filter(has_text="Queued safe task creation")).to_be_visible()
            expect(page.get_by_text("executed: create_task")).to_be_visible()

            page.get_by_role("tab", name="3. Project state").click()
            expect(page.get_by_role("grid", name="Tasks")).to_contain_text("Fix XSS task")

            # Verify no XSS executed anywhere in DOM across all tabs
            assert not dialogs
            assert page.evaluate("Boolean(window.__xss)") is False
            assert page.evaluate("document.querySelectorAll('img[onerror]').length") == 0
        finally:
            ctx.close()


def test_browser_provider_error_shows_actionable_banner_and_preserves_state(browser: Browser):
    def failing_llm_factory(_tasks):
        async def _llm(name, instructions, prompt, output_type):
            raise ProviderCallError(
                "rate_limit",
                "Live provider rate limit reached (groq). Wait ~3s and try again. Nothing was executed.",
                retry_after=3.0,
            )

        return _llm

    with running_gradio_app(env={}, llm_factory=failing_llm_factory) as url:
        ctx, page = _new_page(browser, url)
        try:
            page.get_by_label("Message").fill("I finished the Mobile layout.")
            page.get_by_role("button", name="Run", exact=True).click()
            _wait_for_run_outcome(page, "❌ Error", timeout=15000)
            expect(page.get_by_role("paragraph").filter(has_text="Live provider rate limit reached (groq).")).to_be_visible()
            expect(page.get_by_text("I could not understand this message")).to_have_count(0)

            page.get_by_role("tab", name="2. Manager approvals").click()
            expect(page.get_by_text("No decisions yet.")).to_be_visible()
            page.get_by_role("tab", name="3. Project state").click()
            expect(page.get_by_text("No comments yet.")).to_be_visible()
            expect(
                page.get_by_role("grid", name="Tasks").get_by_role("row", name=re.compile(r"T5\s+Mobile layout"))
            ).to_contain_text("in_progress")
        finally:
            ctx.close()


@pytest.mark.skipif(
    os.getenv("PMP_RUN_LIVE_TESTS") != "1",
    reason="Live UI test is opt-in only (set PMP_RUN_LIVE_TESTS=1 plus provider key in env or .env).",
)
def test_optional_live_provider_ui_flow(browser: Browser):
    cfg = pmp_config.load_config()
    if not cfg.live:
        pytest.skip("No live provider credentials configured in env or .env.")

    with running_gradio_app() as url:
        ctx, page = _new_page_live(browser, url, cfg.provider, cfg.model or "")
        try:
            # 1. Confirm live-mode UI banner & unchecked offline toggle
            offline_chk = page.get_by_label("Use offline demo mode (no API key)")
            expect(offline_chk).not_to_be_checked()
            expect(offline_chk).to_be_enabled()

            # 2. Normal update through live provider (bounded single message proposing approval-gated actions)
            page.get_by_label("Sender name").fill("Ahmed")
            page.get_by_label("Message").fill(
                "I finished the Mobile layout, but the Payment API is still blocked by the vendor."
            )
            page.get_by_role("button", name="Run", exact=True).click()
            _wait_for_run_outcome(page, "🟠 Waiting for the manager", timeout=300000)
            expect(page.get_by_role("heading", name="Summary")).to_be_visible()
            expect(page.get_by_text(re.compile(r"Report ready\. [1-9]\d* action\(s\) are waiting for the manager\."))).to_be_visible()

            # 3. Approve first pending item on Tab 2 & verify repeated click never executes twice
            page.get_by_role("tab", name="2. Manager approvals").click()
            pid1 = _select_approval_id(page)
            approve_btn = page.get_by_role("button", name="Approve and run", exact=True)
            approve_btn.click()
            expect(page.get_by_text(re.compile(r"^approved:"))).to_be_visible()
            expect(page.get_by_text(re.compile(rf"{pid1}\s+.*approved"))).to_be_visible()

            approve_btn.click()
            expect(page.get_by_text("Pick an item first.")).to_be_visible()
            expect(page.locator("li").filter(has_text=re.compile(rf"{pid1}\s+.*approved"))).to_have_count(1)

            # 4. Reject a remaining pending item, or trigger one more bounded update if queue is now empty
            grid_rows = page.get_by_role("grid", name="Waiting for approval").get_by_role("row").count()
            if grid_rows <= 1:
                # Let the 60s sliding provider quota window reset cleanly before a second full run on low-OTPM tiers.
                page.wait_for_timeout(62000)
                page.get_by_role("tab", name="1. Send an update").click()
                prev_steps = _steps_text(page)
                page.get_by_label("From").click()
                page.get_by_role("option", name="supplier").click()
                page.get_by_label("Sender name").fill("Acme")
                page.get_by_label("Message").fill(
                    "The keys for the Payment API will be 5 days late; extend task T2 due date by 5 days."
                )
                page.get_by_role("button", name="Run", exact=True).click()
                _wait_for_run_outcome(page, "🟠 Waiting for the manager", prev_steps=prev_steps, timeout=300000)
                expect(
                    page.get_by_text(re.compile(r"Report ready\. [1-9]\d* action\(s\) are waiting for the manager\."))
                ).to_be_visible()
                page.get_by_role("tab", name="2. Manager approvals").click()

            pid2 = _select_approval_id(page)
            page.get_by_role("button", name="Reject", exact=True).click()
            expect(page.get_by_text("rejected: Rejected by manager")).to_be_visible()
            expect(page.get_by_text(re.compile(rf"{pid2}\s+.*rejected"))).to_be_visible()
        finally:
            ctx.close()
