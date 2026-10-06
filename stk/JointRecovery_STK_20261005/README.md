# JointRecovery: STK data and completed experiments

This directory contains the new **STK 11 satellite–ground contact dataset**, frozen scientific code, four fitted checkpoints, and completed experimental reports. The earlier manuscript is unchanged. This is a weighted scheduling dataset constructed from simulated physical access windows; it is not an existing public unit-weight MIS benchmark.

Start with the [final quality findings](reports/JOINTRECOVERY_QUALITY_RESULTS/FINAL_QUALITY_SIGNOFF_ZH.md), [full quality report](reports/JOINTRECOVERY_QUALITY_RESULTS/JOINTRECOVERY_QUALITY_RESULTS_ZH.md), and [reproduction guide](reports/PUBLICATION_REPRODUCTION_GUIDE_ZH.md). These detailed reports are in Chinese; the tables, machine-readable CSV/JSON, code, and figures expose the underlying definitions and results.

## Dataset and split

- **12 physical sources; 72 weighted conflict graphs.** Each source supplies six configurations: R12/R8 station sets × ground switching gaps of 170/340/680 seconds. Satellites retain their physical owner identities.
- A vertex is a complete contact window. Its reward is its original duration in seconds; graph edges enforce ground-station and satellite resource conflicts. The R8 configurations remove complete windows and preserve retained endpoints and rewards.
- Sources r000–r003 are used for fitting; r004–r005 for internal development. Independent validation uses r006–r007, and independent test uses r008–r011. Configurations, states, and requests within one source are not additional independent physical samples.
- Native microsecond integer rewards are a solver representation, not a claim of physical timing precision. Outputs are checked and rescored against the original graph and duration objective.

See the [dataset catalog](reports/DATASET_CATALOG_ZH.md) and [delivery description](reports/DATASET_DELIVERY_ZH.md) for physical parameters, provenance, contact fields, and graph contracts.

## Main completed result

The quality matrix contains **480 validation and 960 test executions**, all sealed. Each allowed time of **10/30/60/120/300 seconds** was executed separately. The primary table uses the 300-second allowance selected by the development completion rule, rather than by validation/test gain. An allowance is not the actual runtime.

The table below uses four equally weighted test sources, with six configurations per source. JR and JR-CheapSummary average two fitted seeds. Values are improvements over the same original feasible schedule; arrows indicate the preferred direction.

| Method | Gain, seconds ↑ | Source-equal gain, % ↑ | Actual caller time, seconds ↓ | Complete / all runs |
|---|---:|---:|---:|---:|
| **JointRecovery (JR; two-seed mean)** | **2667.282** | **0.259** | **3.255** | **48/48** |
| JR-CheapSummary (representation ablation; two-seed mean) | 2656.301 | 0.258 | 3.138 | 48/48 |
| Greedy-rank+CHILS (internal ranking control) | 2655.973 | 0.258 | 1.630 | 24/24 |
| Independent-replacement+CHILS (internal control) | 1823.578 | 0.176 | 1.411 | 24/24 |
| [CHILS-p1 (SEA 2025)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2025.22) | 89277.224 | 8.770 | 210.183 | 24/24 |
| [HiGHS-MILP](https://highs.dev/) (generic solver reference) | N/A | N/A | 248.562 | 15/24 |

**JR does not outperform CHILS.** Its mean improvement over the strong internal greedy control is only 11.308 seconds, while caller time is higher. Seed17 ties the greedy control; seed29 gives a small improvement. HiGHS has incomplete/late outputs, so its complete full-matrix quality is N/A; neither successful-only averages nor fallback schedules replace missing native results.

The five-point curves include actual caller cost and completion rate, not just allowed time. JR's fixed request procedure finishes early and its gain is flat across these allowances. CHILS already has much higher gain at the 10-second point. These observations do not establish the manuscript's claim of preserving explicit lookahead quality at lower cost.

The current STK transfer uses **remaining-head = 0**, so remaining-time learning is not validated. It fixes cap256, at most eight requests, and four native workpoints. CHILS and HiGHS search the whole graph; the differences in search scope and termination are disclosed. Eight raw policy configurations include two fitted seeds, ablations, and internal controls: they are not eight independent published algorithms. Two-seed ranges in figures are observed min–max ranges, not confidence intervals.

## Reading the evidence

| Material | Entry |
|---|---|
| Main complete-quality table and missing-output handling | [Quality report](reports/JOINTRECOVERY_QUALITY_RESULTS/JOINTRECOVERY_QUALITY_RESULTS_ZH.md) and [source-equal CSV](reports/JOINTRECOVERY_QUALITY_RESULTS/quality_source_equal.csv) |
| Five-point quality, actual-cost, comparator-difference and completion panels | [PDF](reports/JOINTRECOVERY_QUALITY_FIGURES/test/quality_curves_test.pdf), [PNG](reports/JOINTRECOVERY_QUALITY_FIGURES/test/quality_curves_test.png), [figure data](reports/JOINTRECOVERY_QUALITY_FIGURES/test/QUALITY_FIGURE_DATA.json), [manifest](reports/JOINTRECOVERY_QUALITY_FIGURES/test/QUALITY_FIGURE_MANIFEST.json) |
| Short-deadline efficiency pressure supplement | [Report](reports/SUBMITTED_METHOD_RESULTS/SUBMITTED_METHOD_RESULTS_ZH.md) and [findings](reports/SUBMITTED_METHOD_RESULTS/SHORT_EFFICIENCY_SIGNOFF_ZH.md) |
| Internal P1 fixed-call replay and original P2 confirmation | [P1 report](reports/P1_RESULTS/P1_RESULTS_ZH.md) |
| Method names, protocol revision and interpretation | [Evaluation scope](reports/METHOD_NAMES_AND_QUALITY_EVALUATION_ZH.md) |
| Existing-result regeneration and fresh execution | [Reproduction guide](reports/PUBLICATION_REPRODUCTION_GUIDE_ZH.md) |

Reports retain unsuccessful, late, zero-gain, and unavailable results. Total recovery gain is not attributed wholly to learning: methods share a paid greedy prefix and native recovery. Full receipts and checkpoint/source hashes accompany the relevant stages. Figure manifests bind inputs, chart data, generators, and rendered outputs; new renderings can differ in fonts or bytes without changing the experimental values.

## Git contents and full-result Release

Git directly exposes the readable code, protocol and reports, all 72 graph files, contact tables, four final checkpoints, the frozen scientific runtime, and vendored CHILS sources. Larger raw geometry and execution records are distributed as ten archives under Release tag **`stk-jointrecovery-20261006`**:

[Release assets and checksums](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/tag/stk-jointrecovery-20261006)

| Archive | Contents |
|---|---|
| `01-dataset-r000-r003.zip` | Dataset sources r000–r003 |
| `02-dataset-r004-r007.zip` | Dataset sources r004–r007 |
| `03-dataset-r008-r011.zip` | Dataset sources r008–r011 |
| `04-p0-results.zip` | P0 scope probes and evidence |
| `05-train-labels.zip` | Frozen training/development execution labels |
| `06-fit-models-and-features.zip` | Fitted models, histories and features |
| `07-p1-followup-p2-results.zip` | P1 follow-up and original P2 records |
| `08-short-comparison-and-quality-development.zip` | Short comparison and quality-development records |
| `09-quality-comparison-results.zip` | Complete validation/test quality execution records |
| `10-code-protocol-reports-frozen-sources.zip` | Code, protocol, reports and frozen sources |

**Every ZIP stores paths relative to this dataset root.** Clone/download the repository, enter `stk/JointRecovery_STK_20261005`, and extract all ten archives into that same directory. Do not create an extra nested `JointRecovery_STK_20261005` folder. Merge the archive directory trees and retain their recorded contents. Exact file sizes and SHA256 values come from the release asset inventory, not from estimates in this README.

For example, extracted `execution/quality_comparison/out/...` must be under this directory. Downloading only Git's source ZIP does not include the large Release execution archives. Published absolute paths inside original receipts are historical provenance; use the explicit local roots in the reproduction guide rather than editing frozen evidence.

## Reproduction requirements

- **Read results / regenerate reports:** Python, NumPy and Matplotlib; no STK or native solver run is required. The guide provides the real P1 five-root command, the short comparison's original budget bindings, and the quality report/figure commands.
- **Rebuild physical data:** licensed Windows STK 11, registered `STK11.Application` COM, compatible Python with `pywin32`, and NumPy. Preserve complete windows and the frozen physical/graph protocol.
- **Re-execute scientific algorithms:** the frozen Linux runtime, checkpoints and original solver bindings. Recorded versions are Python 3.8.10, NumPy 1.22.4, SciPy 1.10.1 and PyTorch 1.11.0+cu113; receipt-specific environment details remain authoritative. The recorded fit uses CUDA, whereas evaluation uses CPU.
- **CHILS:** official commit `515952724cd3dcc6c4365a340ecf0f1da782119a` from [KarlsruheMIS/CHILS](https://github.com/KarlsruheMIS/CHILS), with source/binary hashes and the whole-graph numeric gate. This experiment uses the single-thread p1 configuration, not the publication's strongest multithreaded configuration. A fresh build must not silently bypass the frozen binding checks.

Run fresh experiments in a separate output directory. The guide documents checkpoint locations, Linux entry points, graph validation and the source/binary receipts; no cloud authentication or original machine directory is needed to inspect this release.

## Exact downloadable asset inventory

| Asset | Compressed MB | Files |
|---|---:|---:|
| [01-dataset-r000-r003.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/01-dataset-r000-r003.zip) | 429.040 | 1583 |
| [02-dataset-r004-r007.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/02-dataset-r004-r007.zip) | 429.041 | 1580 |
| [03-dataset-r008-r011.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/03-dataset-r008-r011.zip) | 428.903 | 1580 |
| [04-p0-results.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/04-p0-results.zip) | 122.246 | 557 |
| [05-train-labels.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/05-train-labels.zip) | 228.426 | 709 |
| [06-fit-models-and-features.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/06-fit-models-and-features.zip) | 174.040 | 1310 |
| [07-p1-followup-p2-results.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/07-p1-followup-p2-results.zip) | 67.663 | 3019 |
| [08-short-comparison-and-quality-development.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/08-short-comparison-and-quality-development.zip) | 187.611 | 4262 |
| [09-quality-comparison-results.zip](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/download/stk-jointrecovery-20261006/09-quality-comparison-results.zip) | 369.185 | 6747 |

The code/report archive and every exact byte size/SHA256 are listed in [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) and [SHA256SUMS.txt](SHA256SUMS.txt). Checksums refer to the Release assets, not a Git-generated source ZIP.
