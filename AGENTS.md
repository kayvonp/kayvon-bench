# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a clean scaffold. Use the structure below for all new contributions:
- `src/` - application or library source code, organized by feature/module.
- `tests/` - automated tests mirroring `src/` paths (for example, `tests/auth/test_login.*` for `src/auth/login.*`).
- `assets/` - static files such as images, sample data, or fixtures.
- `docs/` - design notes, architecture decisions, and usage docs.

Keep modules focused and small. Prefer feature-based folders over large shared utility files.

## Build, Test, and Development Commands
No build system is configured yet. When adding tooling, expose commands through a single entry point (for example, `Makefile` or package scripts) and document them here.

Recommended baseline commands to add:
- `make setup` - install dependencies and prepare local environment.
- `make test` - run full test suite.
- `make lint` - run formatters and linters.
- `make run` - start local app/service.

## Coding Style & Naming Conventions
- Indentation: 2 spaces for YAML/JSON/Markdown, 4 spaces for Python, language defaults elsewhere.
- Naming: `snake_case` for files and functions unless language conventions require otherwise; `PascalCase` for classes/types.
- Keep functions single-purpose and avoid deeply nested logic.
- Add and enforce formatter/linter configs early (for example, Prettier, ESLint, Ruff, Black).

## Testing Guidelines
- Place tests under `tests/` and mirror source layout.
- Test files should be named `test_<unit>.*` or `<unit>.test.*` (pick one style and stay consistent).
- Include at least one happy-path and one failure-path test per feature.
- Target meaningful coverage for changed code; add regression tests for bug fixes.

## Commit & Pull Request Guidelines
Because Git history is not initialized here yet, use Conventional Commits from the start:
- `feat: add user authentication module`
- `fix: handle missing config file`
- `docs: add setup instructions`

For pull requests:
- Write a short summary of what changed and why.
- Link related issue(s).
- Include test evidence (command + result).
- Add screenshots/log snippets for UI or behavior changes where relevant.

## Security & Configuration Tips
- Never commit secrets. Use `.env` files locally and provide `.env.example`.
- Keep dependencies minimal and pinned where possible.
- Review external inputs and fail safely with clear error messages.
