import asyncio
import time
import json

from models.problem import Problem
from models.roles import LLMRolePreference
from llm.prompts import build_system_prompt, build_solver_user_prompt
from llm.client import generate_content_async


async def solve_problem(
    client,
    problem: Problem,
    solver_role: LLMRolePreference,
    timeout_sec: int = 2000,
    log_interval_sec: int = 5,
) -> str:
    system_prompt = build_system_prompt(solver_role)
    user_prompt = build_solver_user_prompt(problem)

    start_time = time.monotonic()

    async def log_progress():
        while True:
            elapsed = time.monotonic() - start_time
            print(
                f"[thinking] problem={problem.id} "
                f"elapsed={elapsed:.1f}s"
            )
            await asyncio.sleep(log_interval_sec)

    progress_task = asyncio.create_task(log_progress())

    try:
        raw_text = await asyncio.wait_for(
            generate_content_async(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
            timeout=timeout_sec,
        )

        # 🔍 DEBUG: print raw solver output
        print(f"\n[RAW LLM OUTPUT] problem={problem.id}\n{raw_text}\n")

        return raw_text or ""

    except asyncio.TimeoutError:
        timeout_payload = {
            "problem_id": problem.id,
            "llm_model": "gemini",
            "answer": None,
            "reasoning": ["Solver timed out before producing an answer."]
        }

        # 🔍 DEBUG: timeout event
        print(f"\n[TIMEOUT] problem={problem.id} after {timeout_sec}s\n")

        return json.dumps(timeout_payload, ensure_ascii=False)

    finally:
        progress_task.cancel()