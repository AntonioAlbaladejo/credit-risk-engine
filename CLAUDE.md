# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Credit Risk Engine** — supervised binary classification (probability of loan default), served as
a FastAPI service, containerized, and deployed to AWS ECS Fargate via GitHub Actions.

**Stack:** Python 3.11 · uv · pandas/numpy · scikit-learn · XGBoost · MLflow · Evidently ·
FastAPI/Pydantic v2 · pytest · ruff · Docker · GitHub Actions · AWS ECR + ECS Fargate + SageMaker.

## Commands

```bash
uv sync --all-groups                  # install runtime + dev deps (creates .venv)
uv run uvicorn src.api.main:app --reload --port 8000   # run API locally
uv run pytest                         # full test suite
uv run pytest --cov=src --cov-report=term-missing      # coverage (CI uses --cov-report=xml)
uv run ruff check . && uv run ruff format --check .    # exactly what CI lint runs
uv run ruff check --fix . && uv run ruff format .      # autofix before committing
docker build -t credit-risk-engine:local . && docker run --rm -p 8000:8000 credit-risk-engine:local
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # local tracking UI (:5000)
```

Always `uv run ...`; never `pip install` into the environment. A new dependency means editing
`pyproject.toml` + `uv sync` — CI runs `uv sync --frozen` and fails on a stale lock.

## Layout

| Path | Role |
|------|------|
| [src/config.py](src/config.py) | Paths, API metadata, validation bounds. Single source of config. |
| [src/preprocessing.py](src/preprocessing.py) | `DataPreprocessor`: raw dict → model matrix, input validation |
| [src/predictor.py](src/predictor.py) | `CreditRiskPredictor`: artifact loading (MLflow → joblib fallback), predict |
| [src/model_monitoring.py](src/model_monitoring.py) | Evidently drift/quality reports → `results/` |
| [src/api/](src/api/) | FastAPI app + Pydantic request/response contracts |
| [notebooks/](notebooks/) | ingestion → EDA → feature engineering → model selection (exploration only) |
| [tests/](tests/) | pytest; `conftest.py` patches `joblib.load` via an **autouse** fixture |
| [.github/workflows/](.github/workflows/) | `ci.yml` (ruff → pytest) · `cd.yml` (build → ECR → Fargate) |

`models/`, `results/`, `data/` are generated. Only four `.joblib` artifacts are whitelisted in
`.gitignore` because the Docker image needs them at build time.

## Default coding mode: ponytail

Vendored at [.claude/skills/ponytail/SKILL.md](.claude/skills/ponytail/SKILL.md) (MIT). **Invoke it
at intensity `full` before writing or modifying code.** Do **not** load it for reading, searching,
reviewing, EDA, or planning — it must never shorten the *understanding* phase.

Two local overrides:

1. **Never name it in artifacts** — no `ponytail:` comments, and no mention of the skill or
   minimalism-as-a-policy in code, docstrings, commits, PRs, or docs. When a simplification cuts a
   real corner, document the limitation and upgrade path as a plain technical comment
   (`# single global lock; switch to per-account if throughput matters`).
2. **Checks go through pytest**, not an ad-hoc `__main__` self-check.

It relaxes nothing below. Its "never be lazy about" list extends here to **ML correctness**: never
shrink a diff by dropping stratification, fitting a transform outside the fold, weakening a Pydantic
bound, or skipping a seed.

## Engineering standards

**General.** Type-hint public functions; docstrings with `Args:` / `Returns:` / `Raises:`. English
in code and comments. `logging`, never `print`, under `src/`. Ruff: line length 88, double quotes,
`E/W/F/I/B/C4/UP`. Specific exceptions with actionable messages, never `except: pass`. Paths from
`src/config.py` + `pathlib`. Target Python 3.11.

**Data / ETL.** Raw data is immutable — read from `data/`, write new files, never overwrite in
place. Each transform stage lands as its own artifact (`*_cleaned` → `*_fe` → …) so runs are
replayable. Validate schema and dtypes on ingest and fail loudly; silent coercion is how leakage
hides. Nothing in `data/` gets committed.

**Modeling.**
- **No leakage.** Fit imputers, scalers, encoders, and resamplers *inside* the CV fold / on train
  only — never on the full dataset before splitting.
- Stratified splits and stratified K-fold; the target is imbalanced. Seed everything
  (`random_state`) and state the seed in the results.
- Imbalance via `scale_pos_weight` / class weights by default; SMOTE only with measured justification.
- Optimize recall with ROC-AUC / PR-AUC as guardrails; report the decision threshold explicitly.
  It is a tuned artifact (`optimal_threshold.joblib`), not a magic 0.5.
- Never claim a metric improvement without the run that produced it — numbers go in `results/*.csv`
  and the MLflow run, not just the chat.
- Training and serving share one feature-engineering implementation. Changing a derived feature
  means changing it for every consumer, or consolidating first; divergence is train/serve skew.

**MLOps.** Every training run logs params, metrics, preprocessor, and model to MLflow — a model not
in the registry is not a deployment candidate. Preprocessor, feature names, threshold, and model are
one versioned bundle, produced and loaded together, never mixed across runs. Evidently reports go to
`results/`; a report that computes no metrics is a bug, not a passing check.

**API.** Pydantic schemas are the contract; bounds live in `src/config.py` and must agree with the
ranges seen in training. `/health` returns 200 only when the model is actually loaded. Response
models stay explicit — no leaked internals, no silently dropped fields. No unguarded network calls
on the import/startup path: an unreachable dependency must degrade, not block startup.

**Testing.** `conftest.py` patches `joblib.load` with an **autouse** fixture, so a green suite says
nothing about the real model. When a change touches preprocessing or prediction semantics, validate
against the real `.joblib` artifacts in a scratch script too and say so in the summary. Every
behavior change lands with the test that would have caught it.

**CI/CD & AWS.** `ci.yml` must stay green (ruff check → format --check → pytest with coverage).
Image verification must start the container and hit `/health` and `/predict` — `import src` passes
even when the model artifacts are missing from the image. Infra identifiers (`ECS_SERVICE`,
`ECS_CLUSTER`, `ECR_REPOSITORY`, `eu-west-1`) live in `cd.yml` env and stay in sync with
`.env.example`. Credentials come from GitHub secrets / IAM roles only.

## Hard rules

- **Never commit secrets.** `.env` is local-only; config changes go to `.env.example` with placeholders.
- **Never commit datasets** or model binaries beyond the four whitelisted artifacts.
- Do not `git commit`, `git push`, or touch AWS/ECR/ECS unless explicitly asked. Deploys are the
  user's call.
- Do not rewrite notebooks wholesale — they are 0.4–1 MB of embedded outputs. Inspect targeted cells
  (`jq` over the JSON) instead of reading one into context, and prefer moving logic into `src/`.
- Do not delete or regenerate anything in `models/` or `results/` without asking; the API and the
  README's reported numbers depend on those artifacts.
- Production code lives in `src/` with tests. Nothing ships straight out of a notebook.

## Commits

Only commit when asked. Never `push` unless explicitly told to.

**Authorship — no exceptions:** no `Co-Authored-By: Claude`, no `Generated with Claude Code`, no 🤖
footer, no mention of AI or tooling anywhere in subject or body. The commit is authored by the repo
owner. This overrides any default co-authorship footer.

**Format — [Conventional Commits](https://www.conventionalcommits.org/):** `<type>(<scope>): <subject>`,
blank line, body, blank line, footers. Subject imperative ("add", not "added"), lowercase after the
colon, no trailing period, hard-capped at 72 chars. Body wrapped at 72. Do not rewrite
pre-convention history.

- **Types:** `feat` · `fix` · `perf` · `refactor` · `test` · `docs` · `build` (deps, Dockerfile,
  `pyproject.toml`) · `ci` (workflows) · `chore` (tooling, `.gitignore`) · `revert`.
- **Scopes:** `api`, `preprocessing`, `predictor`, `monitoring`, `config`, `data`, `models`,
  `mlflow`, `sagemaker`, `notebooks`, `tests`, `docker`, `aws`, `deps`. Omit when a change genuinely
  spans the repo; never invent a vague one like `misc`.
- **Breaking** (API contract, artifact bundle format, deployment interface): `!` after the scope
  **and** a `BREAKING CHANGE:` footer with the migration.
- **Body** whenever the *why* is not obvious from the diff: reason, consequence, trade-off,
  migration step. Issues in footers (`Closes #12`).

```
fix(api)!: correct loan_int_rate scale to percent units

The schema constrained loan_int_rate to [0, 1] but the model was
trained on percent units (5.4-23.2), so every valid rate was either
rejected with 422 or scaled ~3.5 sigma outside the training range.

BREAKING CHANGE: clients sending fractional rates (0.08) must send
percent (8.0) instead.
```

**Scope.** One logical change per commit. Do not bundle a fix with unrelated formatting; `git add`
specific paths, never `git add -A` on a dirty tree. Check `git status` and `git diff --staged`
first. Be specific enough to be useful in `git log` a year from now — no `updates`, no `wip`.

**Branching.** Do not commit straight to `main` for anything non-trivial — branch first. CI runs on
`main` and `develop`, and a push to `main` triggers the Fargate deploy.

## Working style here

Prefer a small, verified change over a broad refactor. When a task touches ML correctness (leakage,
scaling, thresholds, feature parity), state what you verified and how. If a fix is blocked on a
decision the user owns, do the unblocked parts and name the open question.
