# MPCF Baseline and Fair-Comparison Protocol

## 1. Comparison Principle

Classical network-dismantling methods remove nodes to fragment an undirected
structural graph. MPCF protects nodes to increase the adaptive minimum cut of a
directed typed mission network. Their objectives are different, so the comparison
uses a preregistered transfer protocol rather than relabeling a dismantling
algorithm as a fortification optimizer.

For each frozen instance, every structural method receives the same undirected
equipment projection and produces its native removal ranking. That ranking is
kept unchanged and scanned as a protection ranking until the heterogeneous
protection budget is exhausted. Unaffordable nodes are skipped without
reordering later nodes. The selected set is then evaluated by the same exact
adaptive PathCut attacker used for MPCF on the original directed typed graph.

This protocol tests whether a structurally critical ranking is also useful for
protecting mission paths. It does not claim that the baseline originally optimized
the MPCF objective.

## 2. Nature Review Table 1 Panel

The ten dismantling families summarized in Table 1 of Artime et al., *Nature
Reviews Physics* 6, 114-131 (2024), are all represented:

| Family | Canonical implementation in this release |
|---|---|
| Collective Influence | `OfficialCI_L2-Protect` |
| Belief Propagation-guided Decimation | `BPDReference-Protect` |
| Min-Sum | `OfficialMinSumR-Protect` |
| Generalized Network Dismantling | `OfficialGNDR-Protect` |
| Ensemble GND | `OfficialEGND-Protect` |
| CoreHD | `OfficialCoreHD-Protect` |
| Explosive Immunization | `OfficialEI_S1-Protect` |
| Graph Dismantling Machine | `OfficialGDMR-Protect` |
| CoreGDM | `OfficialCoreGDM-Protect` |
| FINDER | `OfficialFINDER_R-Protect` |

All locally available parameter and reinsertion variants from the review
repository are retained as supplementary comparisons. The method catalog
records family, canonical status, implementation source, and conditional
applicability.

## 3. Full 45-Method Accounting

The `all` panel contains exactly 45 method labels.

### Proposed methods (3)

```text
MPCF-Exact
MPCF-CG
MPCF-Greedy
```

### Local protection baselines (9)

```text
NoProtection
RandomProtect
DegreeProtect
BetweennessProtect
KCoreProtect
PageRankProtect
PathFrequencyProtect
InitialPathCutProtect
BPDReference-Protect
```

### Official and derived review rankings (33)

```text
ReviewBruteForceFrequencyRank-Protect
OfficialCI_L1-Protect
OfficialCI_L2-Protect
OfficialCI_L3-Protect
OfficialCoreHD-Protect
OfficialGND-Protect
OfficialGNDR-Protect
OfficialEI_S1-Protect
OfficialEI_S2-Protect
OfficialMinSum-Protect
OfficialMinSumR-Protect
OfficialNetworkEntanglementSmall-Protect
OfficialNetworkEntanglementSmallR-Protect
OfficialNetworkEntanglementMid-Protect
OfficialNetworkEntanglementMidR-Protect
OfficialNetworkEntanglementLarge-Protect
OfficialNetworkEntanglementLargeR-Protect
OfficialVertexEntanglement-Protect
OfficialVertexEntanglementR-Protect
OfficialGDM-Protect
OfficialGDMR-Protect
OfficialCoreGDM-Protect
OfficialEGND-Protect
OfficialFINDER_R-Protect
OfficialDegreeStatic-Protect
OfficialDegreeDynamic-Protect
OfficialBetweennessStatic-Protect
OfficialBetweennessDynamic-Protect
OfficialEigenvectorStatic-Protect
OfficialEigenvectorDynamic-Protect
OfficialPageRankStatic-Protect
OfficialPageRankDynamic-Protect
OfficialRandomStatic-Protect
```

`ReviewBruteForceFrequencyRank` is a transparent derived diagnostic included in
the upstream adapter registry. It is marked as derived rather than official in
the output metadata.

## 4. Frozen Experimental Controls

Within each statistical comparison, all methods share:

* identical serialized graph and SHA-256 graph fingerprint;
* identical directed mission-path family;
* identical attack and protection costs;
* identical protection budget and finite uplift;
* identical exact adaptive PathCut evaluator;
* identical random seed where a method is stochastic.

The main server panel contains 45 synthetic instances: three graph sizes,
three topologies, and five seeds. The default protection budgets are 2%, 5%,
and 10% of total protection cost. Attack and protection costs use the frozen
balanced heterogeneous profile.

## 5. Applicability and Failure Policy

Some official algorithms require connected projections, non-empty 2-cores,
legacy model files, or isolated dependency environments. Every requested method
remains in `method_coverage.csv` even when it is unavailable, times out, or is
not applicable.

* `main_panel_summary.csv` contains only methods complete on the same frozen
  instance-budget keys.
* `conditional_panel_summary.csv` contains incomplete, failed, timed-out, or
  conditionally applicable methods with explicit reasons.
* No method is silently removed and no successful subset is substituted for
  the preregistered panel.

## 6. Primary Endpoints

The primary endpoint is the certified defended PathCut cost. Supporting fields
include absolute and relative margin gain, protection cost utilization,
adaptive cut size and composition, runtime, optimality status, and gap.
Paired comparisons against `MPCF-Exact` use identical instance-budget keys and
report Wilcoxon signed-rank statistics plus rank-biserial effect size. Ties are
retained.

