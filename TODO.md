# TODO

- Add an end-to-end fixture that launches an isolated `llm-router` with fake provider
  adapters and proves cascade provenance through the real HTTP boundary.
- Add router request attribution (`job`, `run`, and `operation`) to planning rounds.
- Rename the compatibility `--model` option to `--route` in a future major release.
