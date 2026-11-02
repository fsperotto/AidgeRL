# Contributor instructions for coding assistants

Read `docs/requirements.md` before modifying architecture. Keep README examples
and requirements aligned with the implemented API; distinguish implemented
features from future work. Preserve user edits and keep changes scoped.

## Design

- Prefer composition and small structural contracts; avoid deep inheritance.
- Use `Agent` and `Trainer` in common workflows. Keep backend-specific dispatch
  and training lifecycle inside adapters. Runner and Experiment are backend agnostic.
- Import external frameworks lazily. The core must work without SB3 or Gymnasium.
- Preserve native observation/action representations and heterogeneous agents.
- Reset execution state independently of learned state. Keep stream identities
  separate when sharing policies. Never reconstruct a trained model on episode reset.
- Preserve final terminated/truncated flags before experience ingestion.
- Keep training/evaluation environments and independent repetitions separate.
- Document buffer ownership, resource ownership, requested/actual step counts,
  and limitations. Avoid unmeasured performance claims.
- Use docstrings for modules, classes, functions, and methods. Explain non-obvious
  state/lifecycle decisions in comments. Keep optional integrations lightweight.

## Validation

From the repository root:

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
PYTHONPATH=src python examples/evaluation.py
PYTHONPATH=src python examples/heterogeneous.py
git diff --check
```

Add behavioral tests for changed lifecycle/training contracts. Use a real backend
smoke test when available; explicitly report fixture-only verification when it is
not. Never claim a Colab or real SB3 run was verified unless it actually ran.
