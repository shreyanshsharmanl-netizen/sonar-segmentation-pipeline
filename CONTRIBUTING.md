# Contributing

1. Branch from `main` as `feature/<name>` or `fix/<name>`.
2. Run `make check` before pushing.
3. Open a pull request. CI must pass before merging.
4. Add a line to `CHANGELOG.md` for any user-visible change.
5. Record significant design decisions in `docs/adr/`.

Releases follow semantic versioning. Tagging `vX.Y.Z` on `main` publishes a Docker image.
