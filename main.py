import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from models.problem import Problem
from models.roles import LLMRolePreference
from models.solver_output import SolverOutput
from llm.client import create_gemini_client
from llm.solver import solve_problem


# -------------------------
# Helpers
# -------------------------
def load_problems(path: str) -> list[Problem]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Problem.from_dict(p) for p in raw]


def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_solution(
    output_dir: str,
    index: int,
    problem: Problem,
    solver_output: SolverOutput,
):
    payload = {
        "problem_index": index,
        "problem_id": problem.id,
        "problem": problem.__dict__,
        "solver_output": solver_output.model_dump(),
    }

    out_path = Path(output_dir) / f"solution_{index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -------------------------
# Async runner
# -------------------------
async def run():
    load_dotenv()

    problems = load_problems("data/problems.json")
    ensure_output_dir("data/temp")

    client = create_gemini_client()

    solver_role = LLMRolePreference(
        role_preferences=["Solver"],
        confidence_by_role={"Solver": 0.9},
        reasoning="Solve the problem using precise logical reasoning.",
    )

    semaphore = asyncio.Semaphore(4)

    async def sem_solve(problem: Problem):
        async with semaphore:
            return await solve_problem(client, problem, solver_role)

    tasks = [sem_solve(problem) for problem in problems]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, (problem, result) in enumerate(zip(problems, results)):
        if isinstance(result, Exception):
            print(f"[FAIL] problem={problem.id} error={result}")
            continue

        write_solution("data/temp", idx, problem, result)
        print(f"[OK] solution_{idx}.json written")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
