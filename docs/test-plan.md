# Test plan

| Level | What | Where | When |
|-------|------|-------|------|
| Unit | Simulator invariants, IoU, gate logic | `tests/` | Every push |
| API | Endpoints and bad input | `tests/test_pipeline.py` | Every push |
| Container | Image builds and reaches `/readyz` | CI `image` job | Every push |
| Model | Held-out metrics and gate | `make pipeline` | Each new model |
| Manual | Compare prediction and truth across noise levels | Demo | Each release |
