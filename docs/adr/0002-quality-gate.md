# ADR 0002: Gate model release in the pipeline

**Status:** accepted

**Context:** A model that trains without errors can still be worse than the last one.

**Decision:** Write the model file only after it passes metric thresholds and beats a baseline.

**Consequences:** Bad models fail loudly. Thresholds need revisiting when the simulator changes.
