# Architecture Notes

This document complements the README by summarizing the concrete components and data artifacts in the multi‑LLM debate system.

## Core Components

- **Orchestrator**: `ProblemSolvingApp` loads the dataset, initializes agents once, and runs a `ProblemSolvingSession` per problem with concurrency controls. 
- **Session pipeline**: `ProblemSolvingSession` drives the staged workflow and persists every stage to Firestore and local JSON files.
- **Agent contexts**:
  - `SolverAgentContext` encapsulates role assessment, solution generation, peer review, and refinement. 
  - `JudgeAgentContext` aggregates solver artifacts and produces the final judgement. 
- **Provider abstraction**: `AgentFactory` selects provider‑specific agents based on configuration, enabling multi‑provider experiments without changing orchestration logic. 

## Data Artifacts

### Firestore collections

The system persists every stage in Firestore using standard collections: Runs, RoleAssessments, Solutions, SolutionReviews, RefinedSolutions, FinalJudgements, and Metrics. 

### Local JSON outputs

Every stage also writes JSON files to `data/output/` for offline analysis, mirroring Firestore documents for reproducibility.

## Stage-by-Stage Inputs/Outputs

1. **Role assessment**: inputs `Problem`, outputs `RoleAssessment`.
2. **Solve**: inputs `Problem`, outputs `ProblemSolution`. 
3. **Peer review**: inputs `Problem` + `ProblemSolution`, outputs `ProblemSolutionReview`. 
4. **Refinement**: inputs `Problem` + original solution + reviews, outputs `RefinedProblemSolution`. 
5. **Final judgement**: inputs problem + all solver contexts, outputs `FinalJudgement` with winner metadata. 

## Configuration Touchpoints

- Default output paths and problem selection are centralized in `config.py`.
- Agent provider keys are pulled from environment variables via provider clients. 
- Firestore credentials are provided through `FIREBASE_CREDENTIALS`.
