# Git Rules

## Branch Rules

When performing git operations, refer to the [Branch Rules](../README.md#branch-rules) and [Version Management](../README.md#version-management) sections in the README.md file for complete information about branch workflows and version management.

## Commit Guidelines

### Logical Commit Separation

**IMPORTANT: Commits MUST be separated logically by functionality or feature.**

- Each commit should represent a single, logical change or feature
- Related changes should be grouped together in one commit
- Unrelated changes should be in separate commits
- This makes the git history easier to understand, review, and revert if needed

**Examples of good commit separation:**

- Separate API changes from UI changes
- Separate feature additions from bug fixes
- Separate configuration changes from code changes
- Separate different features into separate commits

## Commit Message Guidelines

**IMPORTANT: All commit messages MUST be written in English.**

When writing commit messages, use [gitmoji](https://gitmoji.dev/) emojis at the beginning of the commit message to indicate the type of change.

### Common Gitmoji Examples

- `:bug:` 🐛 - Fix a bug
- `:sparkles:` ✨ - Introduce new features
- `:memo:` 📝 - Add or update documentation
- `:art:` 🎨 - Improve structure / format of the code
- `:zap:` ⚡️ - Improve performance
- `:fire:` 🔥 - Remove code or files
- `:ambulance:` 🚑️ - Critical hotfix
- `:rocket:` 🚀 - Deploy stuff
- `:recycle:` ♻️ - Refactor code
- `:wrench:` 🔧 - Add or update configuration files
- `:lipstick:` 💄 - Add or update the UI and style files
- `:boom:` 💥 - Introduce breaking changes
- `:white_check_mark:` ✅ - Add, update, or pass tests
- `:lock:` 🔒️ - Fix security or privacy issues
- `:arrow_up:` ⬆️ - Upgrade dependencies
- `:arrow_down:` ⬇️ - Downgrade dependencies

### Format

```
:emoji: Short description of the change

Optional longer description explaining what and why
```

**Example:**

```
:sparkles: Add item image upload and delete functionality

- Implement image file selection with accept attribute for image types only
- Add preview functionality for selected images
- Send selected image file to server on save/update
- Add deleteImage flag when image is removed
```

For the complete list of available gitmoji, refer to [gitmoji.dev](https://gitmoji.dev/).

## AI Workflow for Git Operations

**IMPORTANT: Do NOT perform `git commit` or `git push` unless explicitly requested by the user.**

### Before EVERY commit (mandatory)

1. **Analyze Changes**: List all pending changes (staged and unstaged). Do not skip this step.
2. **Group Logically**: Split the list into distinct logical groups (e.g. source code vs config vs documentation vs tests). If you have more than one logical group, you must make more than one commit.
3. **Commit Sequentially**: For each group, stage only that group, then commit with a message following the [Commit Message Guidelines](#commit-message-guidelines). Repeat for the next group.
4. **Push Once**: Only after *all* logical groups have been committed, perform a `git push`.

### Wrong (do not do this)

- **One commit that mixes logical groups**: e.g. one commit containing `src/`, `README.md`, `pyproject.toml`, and `tests/` together with a single message like "Add feature, README, and tests". This violates logical separation. Use separate commits (e.g. code+config, then docs, then tests).

### Example scenario

You fixed a bug in `auth.ts` and also updated the documentation in `README.md`.

1. `git add src/auth.ts`
2. `git commit -m ":bug: Fix authentication token expiry issue"`
3. `git add README.md`
4. `git commit -m ":memo: Update auth documentation"`
5. `git push origin <branch>`
