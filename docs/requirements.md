# Requirements

## Problem
Build a repeatable pipeline that trains, evaluates and gates a seabed segmentation model, so a weak model cannot be released by accident.

## Data scope
All training and test data is simulated by `src/sonarseg/simulate.py`. This project uses no proprietary or company data. Results describe performance on this simulator only.

## Functional requirements
| ID | Requirement | Verified by |
|----|-------------|-------------|
| F1 | Same seed produces the same sample | `test_same_seed_same_sample` |
| F2 | Labels six classes per pixel | `test_shapes_and_ranges` |
| F3 | Log each run's parameters and metrics | MLflow run in `pipeline.py` |
| F4 | Block models under the quality floor or too close to baseline | `test_gate_blocks_weak_model` |
| F5 | Demo shows input, prediction and truth side by side | Manual check |

## Non-functional requirements
| ID | Requirement | Status |
|----|-------------|--------|
| N1 | Train and test seeds never overlap | Disjoint seed ranges in `pipeline.py` |
| N2 | Report robustness as noise increases | `miou_by_speckle_looks` in the report |
| N3 | Runs on CPU | Latency in `reports/evaluation.json` |

## Limits
A simulator is easier than a real seabed. This model is not validated on real sonar, and the upload endpoint says so in every response.
