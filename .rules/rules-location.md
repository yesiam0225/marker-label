# Rules location

This directory is the **single source of truth** for all AI coding assistant rules in this repository.

## Policy

- **Create and edit rules here only** (`.rules/`). Do not add standalone rule content under `.cursor/rules/`, `.claude/rules/`, or other tool-specific paths.
- Tool-specific entry points (`.cursor/rules/*.mdc`, `CLAUDE.md`, `AGENTS.md`) are thin wrappers that reference files in this directory via `@` imports.
- Write all rule content in **English**.

## Adding a new rule

1. Add `your-topic.md` in `.rules/` with the full rule content.
2. Wire tool entry points:
   - **Cursor**: add `.cursor/rules/your-topic.mdc` with YAML frontmatter and `@.rules/your-topic.md`.
   - **Claude Code**: add `@.rules/your-topic.md` to `CLAUDE.md`.
   - **Other agents**: add `@.rules/your-topic.md` to `AGENTS.md` if applicable.
