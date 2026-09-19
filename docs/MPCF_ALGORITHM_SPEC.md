# MPCF Algorithm Specification

## 1. Research Question and Boundary

MPCF answers one question:

> Given a finite protection budget, which equipment nodes should be fortified
> to maximize the minimum adversarial cost required to break every complete
> kill-network mission path?

The resulting margin is a resistance threshold for complete mission-path
disconnection. In a resilience interpretation, it is one pre-event design
dimension of absorptive capacity; it is not a complete resilience curve and
does not by itself quantify partial service loss, degradation speed, or
recovery.

The modeled object is the directed physical system graph available to the
friendly planner for one bounded task. Removable nodes are implemented component
instances, directed edges are source-backed contributions to task completion,
and initiation/completion terminals are fixed non-removable boundaries. The
mission-path family is the complete set of directed paths between those boundaries.
The four-stage `S-C-L-E` kill-network graph is the primary synthetic special
case, not a universal admission requirement. MPCF does not
model deception, information age, weapon scheduling, repair, or online role
inference. Those mechanisms are outside this chapter.

For `S-C-L-E` experiments, the role sequence remains an interface constraint
used to define feasible mission paths. For independent testbeds, descriptive stage
labels do not generate paths: only audited directed adjacency and the declared
task boundaries do. Neither representation changes during one fortification
decision.

## 2. Defender-Attacker Model

Let `G=(V,E)` be the directed typed equipment graph. For every removable node
`v`, let:

* `a_v > 0` be its attack cost;
* `b_v > 0` be its protection cost;
* `delta_v >= 0` be the finite attack-cost increase caused by fortification.

The default is `delta_v = a_v`. Protection therefore doubles the removal cost;
it never makes a node invulnerable. For a protected set `P`, define

```text
a_v(P) = a_v + delta_v * 1[v in P].
```

Let `C(G)` be the family of removable-component cuts that intersect every
complete directed mission path between the fixed boundaries. The adaptive
post-protection dismantling margin is

```text
kappa(P) = min_{C in C(G)} sum_{v in C} a_v(P).
```

MPCF solves

```text
max_P kappa(P)
subject to sum_{v in P} b_v <= B.
```

The primary output is a protected set and its certified defended margin. The
attacker is recomputed after every candidate protection decision; a fixed
pre-protection attack sequence is never used to score MPCF.

The attacker is adaptive only in the static Stackelberg sense: it observes the
committed protection set and then chooses one best-response cut. The model has
no sequential probing, online learning, cascade dynamics, or incomplete
information.

## 3. Relation to Established Optimization Models

MPCF belongs to the established family of budget-constrained network
improvement, minimum-cut interdiction, and fortification models. Prior work has
considered budgeted arc-capacity improvement, maximization of a minimum
source-sink cut, and generic exact fortification frameworks.

The contribution claimed here is therefore not the first max-min cut model,
the first defender-attacker coupling, or the first use of cut generation. The
chapter specializes this optimization structure to one source-audited directed
task thread, selectable implemented-component nodes, finite (rather than
invulnerable) attack-cost uplift, and a common certificate/evaluation protocol
for transferred dismantling rankings. The closest optimization references
include:

* Schwarz and Krumke (1998), *On budget-constrained flow improvement*,
  doi:10.1016/S0020-0190(98)00070-2;
* Abdolahzadeh, Aman and Tayyebi (2020), *Minimum st-cut interdiction
  problem*, doi:10.1016/j.cie.2020.106708;
* Gruettemeier et al. (2021), *Preventing Small (s,t)-Cuts by Protecting
  Edges*, doi:10.1007/978-3-030-86838-3_11;
* Leitner et al. (2023), *An exact method for binary fortification games*,
  doi:10.1016/j.ejor.2022.10.038.

## 4. Exact Compact Formulation

Each removable equipment node is split into an input and output vertex. Its
internal arc has capacity

```text
a_v + delta_v y_v,
```

where binary `y_v` indicates protection. Original graph edges and the
super-source/super-sink interface arcs receive the finite capacity

```text
M = sum_v (a_v + delta_v) + min_v (a_v + delta_v),
```

over removable nodes (and `M=1` when none exist). Since every removable-node
cut costs at most the first sum and all terms are positive, `M` is strictly
larger than every feasible node-cut cost. A finite optimum therefore cannot
cut a task-interface arc. Let `x_e` be a feasible source-sink flow and `F` its
value. `MPCF-Exact` maximizes `F` subject to:

```text
sum_v b_v y_v <= B,
0 <= x_(v_in,v_out) <= a_v + delta_v y_v,
flow conservation at all internal split vertices,
y_v in {0,1}.
```

For every fixed binary protection vector, max-flow/min-cut duality gives
`F=kappa(P)`. Maximizing `F` jointly over feasible `y` therefore gives the
global MPCF optimum. A second lexicographic solve minimizes used protection
cost while preserving the optimal defended margin. This tie-break prevents an
arbitrary, unnecessary use of budget.

The flow in this formulation is an attack-cost dual certificate. It is not a
physical communication rate, data throughput, or concurrent mission-service
flow.

## 5. Exact Cut Generation

`MPCF-CG` is an independently structured exact solver. For an observed cut
`C`, its master problem adds

```text
theta <= sum_{v in C} a_v + sum_{v in C} delta_v y_v.
```

The master maximizes `theta` under the protection budget. Its candidate set is
then evaluated by an exact weighted PathCut oracle on the fortified graph.
When the oracle finds a cheaper adaptive cut, that cut is added and the master
is solved again.

The restricted master is an upper bound and the adaptive PathCut value is a
feasible lower bound. If they agree within tolerance, the incumbent is globally
optimal. Because the finite graph has finitely many node cuts and every open
iteration adds a previously unseen violated cut, the algorithm terminates in a
finite number of iterations unless a declared solver or iteration limit is
reached. An incomplete run is reported as non-optimal with its open bounds.

## 6. Greedy Ablation

`MPCF-Greedy` repeatedly selects the affordable node with maximum

```text
[kappa(P union {v}) - kappa(P)] / b_v,
```

where each marginal gain is computed by the exact adaptive PathCut oracle.
It is deterministic after stable tie-breaking and is reported as a heuristic;
it carries no global-optimality claim. Because it repeatedly calls the exact
PathCut oracle for many candidates, the current implementation is an ablation
of one-step marginal selection, not the scalable solver claimed for large
networks.

## 7. Set-Function Properties

For feasible protection sets, `kappa` has the following properties.

**Proposition (monotonicity).** If `P` is a subset of `Q`, then
`kappa(P) <= kappa(Q)`.

**Proof.** Every node has weakly larger fortified attack cost under `Q` than
under `P`. Hence every feasible cut has weakly larger cost, and taking the
minimum over the same cut family preserves the inequality.

**Proposition (non-submodularity in general).** `kappa` is not generally a
submodular set function.

**Proof by counterexample.** Consider the unique mission path
`s -> u -> v -> t`, with `a_u=a_v=1` and `delta_u=delta_v=1`. Then
`kappa(empty)=kappa({u})=kappa({v})=1`, while `kappa({u,v})=2`. The marginal
gain of protecting `v` is zero at the empty set and one after protecting `u`,
which violates diminishing returns.

This complementarity explains why a fixed scalar ranking and a one-step
greedy rule can miss the best protection set. It also motivates the minimum
effective budget

```text
B_crit = min { sum_v b_v y_v : kappa(P) > kappa(empty) }.
```

Below `B_crit`, no feasible protection set raises the complete-disconnection
threshold even though it may change which cut is cheapest.

## 8. Objective-Lattice Optimality Certificate

Suppose every removable-node attack cost and fortification uplift is rational.
Let `q>0` be the greatest common rational quantum of these values, after they
are represented on a common denominator. Every fortified cut cost, every
adaptive PathCut value `kappa(P)`, and the global MPCF optimum then belongs to
the lattice `q Z`.

**Proposition (lattice closure).** Let `L` be the value of a replay-verified
feasible protection set and let `U` be a mathematically valid upper bound on
the MPCF optimum. If

```text
floor((U + tolerance) / q) <= round(L / q),
```

then `L` is globally optimal.

**Proof.** Feasibility gives `L <= OPT`, while validity of the dual or master
bound gives `OPT <= U`. Because `OPT` is an element of `q Z`, the displayed
inequality leaves no attainable lattice value strictly above `L` and no larger
than `U`. Hence `OPT=L`.

This certificate does not turn an incumbent or an unverified solver estimate
into an upper bound. It is used only after the following have been audited:

* the frozen graph fingerprint and protected-node rows agree;
* the protection set satisfies its budget and its reported cost;
* an independent exact PathCut replay reproduces the defended margin and cut;
* `U` comes from a valid compact-MILP dual bound, a valid restricted-master
  bound, or the minimum of such bounds across solvers;
* all cost generators share one exact rational quantum.

Thus an interval such as `[176,176.932]` is closed when `q=1`, whereas
`[182,185]` remains open. Exact and cut-generation runs may contribute their
best feasible lower bound and valid upper bound to one instance-level ledger,
but solver disagreement is retained rather than silently resolved.

## 9. Certificates and Reported Quantities

For `MPCF-Exact` and `MPCF-CG`, the implementation records:

* protection set and protection cost;
* undefended and defended minimum PathCut costs;
* the adaptive post-protection cut;
* primal value, upper bound, absolute gap, and solver status;
* cut-generation trace where applicable;
* graph fingerprint, budget, costs, and uplift rule.

`verify_mpcf_certificate` independently checks budget feasibility, protection
cost, adaptive PathCut value, and selected cut cost.
`verify_mpcf_objective_certificate` separately checks raw-bound or lattice
closure. The solver-level and instance-level ledgers preserve certificate
source, cost quantum, replay status, open gaps, and cross-solver conflicts.

## 10. Interpretation

MPCF identifies a collective protection set, not a universal scalar node
centrality. Its contribution is the task-thread-specific finite node-uplift
model, the source and path semantics required to instantiate it, and the
auditable exact solution protocol. The defender-attacker max-min structure is
positioned within established network improvement and fortification research.
The method is polynomial for a fixed protection set but the outer discrete
budget allocation is solved as a mixed-integer optimization problem. Claims of
global optimality are made only when an independently replayed feasible
solution and a valid upper bound close directly or through the objective
lattice.

## 11. Source-Backed Legal Path Families

For a generic directed task thread, edge provenance alone is insufficient:
the union of individually documented edges can enable cross-stitched paths
that are not legal under one operating condition. Schema v2 therefore requires
a source-backed list `P_declared` of complete legal paths for one task mode and
one operating condition. The adapter independently enumerates every complete
simple path `P_graph` enabled by the graph, and formal admission requires

```text
P_declared = P_graph.
```

An element of `P_graph - P_declared` is an undeclared cross-stitched path; an
element of `P_declared - P_graph` is a declared but unrealizable path. Either
case fails the semantic gate before any optimization is run. Version 1
mappings remain replayable but cannot enter formal performance tables.

## 12. Independent Capacity-Threshold Evaluation

The primary MPCF objective remains the minimum complete mission-path cut cost.
After a protection set has been frozen, a separate evaluation computes the
minimum adaptive attack cost required to leave no more than
`floor(r * Omega_0)` concurrent S-C-L-E role motifs, for preregistered
fractions `r`. The evaluator uses the same finite attack-cost uplift as MPCF,
deduplicates integer threshold aliases, verifies flow certificates and checks
the `r=0` point against fortified PathCut.

This target-out-of-objective test is evidence about robustness of the selected
set, not another selection criterion. Its first implementation is deliberately
limited to the fixed four-role capacity model; arbitrary directed task threads
are not assigned synthetic capacities merely to broaden the claim.
