def build_system_prompt(role_reasoning: str) -> str:
    return f"""
You are an AI assistant participating in a collaborative problem-solving system.

Your assigned role reasoning is:
{role_reasoning}

You must strictly follow your role.
You must output valid JSON only.
"""


def build_solver_user_prompt(problem_statement: str) -> str:
    return f"""
Solve the following problem step by step and produce a structured JSON output.

Problem:
{problem_statement}

Your output MUST follow this schema:

{{
  "problem_id": "...",
  "llm_model": "...",
  "answer": "...",
  "reasoning": ["step 1", "step 2", "..."]
}}
"""
