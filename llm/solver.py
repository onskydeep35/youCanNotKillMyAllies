import asyncio
import time

from models.problem import Problem
from models.roles import LLMRolePreference
from models.solver_output import SolverOutput
from llm.prompts import build_system_prompt, build_solver_user_prompt


async def solve_problem(
    client,
    problem: Problem,
    solver_role: LLMRolePreference,
    timeout_sec: int = 2000,
    log_interval_sec: int = 5,
) -> SolverOutput:
    system_prompt = build_system_prompt(solver_role)
    user_prompt = build_solver_user_prompt(problem)

    start_time = time.monotonic()

    async def log_progress():
        try:
            while True:
                elapsed = time.monotonic() - start_time
                print(
                    f"[thinking] problem={problem.id} "
                    f"elapsed={elapsed:.1f}s"
                )
                await asyncio.sleep(log_interval_sec)
        except asyncio.CancelledError:
            # Normal cancellation on completion / timeout
            pass

    progress_task = asyncio.create_task(log_progress())

    def _call_sync() -> SolverOutput:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                {"role": "system", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SolverOutput.model_json_schema(),
            },
        )

        print(f"\n[RAW LLM OUTPUT] problem={problem.id}\n{response.text}\n")

        return SolverOutput.model_validate_json(response.text)

    try:
        solver_output = await asyncio.wait_for(
            asyncio.to_thread(_call_sync),
            timeout=timeout_sec,
        )
        return solver_output

    except asyncio.TimeoutError:
        print(f"\n[TIMEOUT] problem={problem.id} after {timeout_sec}s\n")

        return SolverOutput(
            problem_id=problem.id,
            llm_model="gemini-3-flash-preview",
            answer="TIMEOUT",
            reasoning=[
                "Solver timed out before producing an answer."
            ],
        )

    finally:
        progress_task.cancel()
