from typing import List
from pydantic import BaseModel, Field


class SolverOutput(BaseModel):
    problem_id: str = Field(..., description="ID of the solved problem")
    llm_model: str = Field(..., description="LLM model used to solve the problem")
    answer: str = Field(..., description="Final answer to the problem")
    reasoning: List[str] = Field(
        ..., description="Step-by-step reasoning leading to the answer"
    )
