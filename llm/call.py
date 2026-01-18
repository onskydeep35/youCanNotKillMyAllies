from models.problem import Problem
from models.roles import LLMRolePreference
from llm.prompts import build_system_prompt, build_solver_user_prompt


def call_solver_llm(
    problem: Problem,
    role: LLMRolePreference,
    model_name: str
) -> str:
    system_prompt = build_system_prompt(role.reasoning)
    user_prompt = build_solver_user_prompt(problem.statement)

    # TODO:
    # Call actual LLM here (Gemini, LLaMA, etc.)
    # Return raw TEXT response (JSON string expected)

    raise NotImplementedError("LLM call not implemented yet")
