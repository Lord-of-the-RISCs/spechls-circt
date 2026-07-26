# Speculative FSM Construction

This document describes the Java/EMF construction of the finite-state machine
used to recover from control-flow speculation in a SpecHLS SCC. It is the
semantic reference for an equivalent implementation in MLIR or xDSL.

The FSM is constructed by `ExposeControlFlowSpeculation` through
`SpecFSMFactory` and `FSMConstructor`, not by `SynthesizeControlLogic`.
`SynthesizeControlLogic` only resynthesizes already-existing combinational
control calls with Yosys.

Primary sources:

- `idg/transforms/ExposeControlFlowSpeculation.xtend`
- `idg/fsm/SpecFSMFactory.xtend`
- `idg/fsm/FSMConstructor.xtend`
- `idg/fsm/FSMBuilder.xtend`
- `idg/fsm/transforms/FSMToGecosASTLowering.xtend`

## Purpose

For every speculated gamma, the datapath initially executes a selected fast
input. The real predicate arrives later. If it selects a different input, the
FSM must:

1. stop accepting new input while replay is required;
2. wait until the selected slow result is usable;
3. restore scalar, array, and dependent-gamma state to a safe point;
4. rewind inputs by the matching amount;
5. select the real slow input;
6. refill the pipeline and resume committing only valid iterations.

The generated FSM is a controller for an already transformed datapath. It does
not compute predicates or values itself.

## Inputs From Scheduling

`SpeculationControlBuilder` creates a `GammaSpeculation` annotation for each
speculated gamma. The relevant values are:

| Value | Meaning |
|---|---|
| `fastIndices[s]` | selected fast input for speculation `s` |
| `inputLatencies[s][p]` | latency of input/path `p` |
| `condDelays[s]` | cycles until the real predicate resolves |
| `resolvedStage[s]` | pipeline stage at predicate resolution |
| rollback targets | prior state/gammas invalidated by recovery |

The builder requires the fast configuration to schedule with II <= 1. It
schedules non-speculative variants to obtain condition timing, then adds
hypothetical delays to candidate inputs until II=1 and records the required
latencies.

The factory derives:

```text
fillDelay[s]      = condDelay[s] - 1
stallDelay[s][p]  = inputLatency[s][p] - condDelay[s] - 2
pipelineDepth     = max(resolvedStage)
```

Negative stall delays mean no explicit stall state is needed.

## Datapath Rewriting

`ExposeControlFlowSpeculation` prepares and connects the controller as follows.

1. Delay slow paths so they are available when recovery selects them.
2. Delay each predicate/control value to its resolve stage.
3. Pack real gamma controls into the FSM `mispec` input record.
4. Create an `FSMNode` and a feedback `MuNode` holding its state record.
5. Create an `FSMCommandNode` that decodes the current registered state.
6. Feed `selSlowPath_s` commands to the speculated gamma controls.
7. Insert rewind nodes on SCC inputs using `nextInput` and `rewind`.
8. Insert rollback nodes for scalar Mu state, arrays, and dependent gamma
   outputs using rollback commands.
9. Replace the original commit enable with the FSM commit command after the
   required commit delay alignment.

The state record contains a state code, a rewind counter, delayed commit bits,
and all controller outputs/internal variables. The command record contains at
least:

```text
nextInput, commit, muRollBack, arrayRollBack, rewind, rbwe,
gammaRollBack_s*, startStall_s*, selSlowPath_s*
```

`nextInput` is derived as `rewindCpt == 0`. This is the mechanism that blocks
input retirement while previously consumed inputs must be replayed.

## Symbolic FSM Model

`FSMConstructor` first constructs a symbolic FSM, then assigns compact binary
state encodings. Its exploration state tracks:

- physical cycle and logical iteration;
- currently active speculation/path;
- start and rollback cycle;
- invalid or poisoned `(speculation, iteration)` predicate results;
- prior rewind actions;
- generated recovery paths;
- path class: initial, combined, new-mispeculation, or canceled.

The normal state is `Proceed`. It sets `rbwe=1`, enables commits, selects all
fast inputs, and self-holds unless a real predicate reports a slow path.

For every non-fast path `p` of gamma `s`, the triggering predicate is:

```text
mispec_s == p
```

The transition away from normal execution clears `commit_s`, records the
corrected rewind depth, records `slowPath_s = p`, disables rollback writes,
and asserts `startStall_s`.

## Single Mis-speculation Recovery

For speculation `s` and slow path `p`, let:

```text
Nstall = max(inputLatency[s][p] - condDelay[s] - 1, 0)
rewind = max(condDelay[s], inputLatency[s][p] - 1)
```

The generated recovery sequence is:

```text
Proceed
  -> detect(s, p)
  -> Stall_0 ... Stall_(Nstall-1)      if Nstall > 0
  -> Rollback
  -> Fill_0 ... Fill_(condDelay[s]-2)
  -> Proceed
```

At `Rollback`, the controller selects the stored slow path, re-enables rollback
writes, emits array rollback `rewind`, emits scalar Mu rollback approximately
`condDelay[s]` cycles (with a correction for already retired work), emits the
stored input rewind amount, and emits rollback commands for affected gammas.

The fill sequence lets state and delayed values become consistent with replayed
input before the controller returns to normal commit behavior.

Initial states `Init_0 ... Init_(maxCond-2)` are prepended to fill condition
latency before steady-state execution. They force fast paths and only enable a
speculation's commit when its resolution horizon has matured.

## Multiple Mis-speculations

The constructor explores recovery paths with a worklist. A later detected
mis-speculation is classified as one of:

| Case | Handling |
|---|---|
| Initial | create a new recovery path |
| Combined | co-recover compatible signals from the same logical iteration |
| New mis-spec | create a recovery path for a later iteration, truncated to remaining recovery time |
| Canceled | abandon an incompatible current path and jump to an existing older-iteration path |

Combination order is dependency aware: poison dependencies take priority, then
smaller condition delay, then smaller speculation index. Poison information
prevents stale condition results from being interpreted after rollback changed
the iteration to which they refer.

The canceled case has an explicit implementation limitation: if no compatible
already-created destination path exists, construction throws `Not yet
supported` rather than generating an unsound controller.

### Native xDSL Recovery Metadata

`#spechls.speculation_entry` carries `poison_speculation_ids`: the configured
speculations whose predicate observations are invalidated by this
speculation's rollback.  The relation is explicit because the Java exporter
does not export its complete symbolic poison map reliably.

FSM inference validates nonzero condition/path latencies, selectors, boolean
flags, rollback data, dependency references, and that the poison relation is
acyclic. Invalid or excessively large scenario spaces are rejected; they are
not reduced to independent recovery chains. For each supported selection,
`spechls.recovery_scenarios` on the generated xDSL FSM contains typed
`#spechls.speculation_recovery` attributes with:

- an `initial`, dependency-ordered `combined`, `new_mispec`, or `canceled`
  kind;
- selected speculation IDs and exact slow-path selectors;
- the Java `RELEASE_DELAY` window for each selected path, poisoned speculation
  IDs, remaining recovery cycles, and exact destination IDs/selectors.

The release window is `max(inputLatency - condDelay - 1, 0) + condDelay`.
Combination ordering is poison dependency first, then smaller condition
latency, then smaller speculation index, matching `isGreaterForCombined`.
`canceled` denotes the canceled-path rule: a late higher-priority recovery
replaces the incompatible active suffix with the canonical already-created
prefix. If that canonical destination is absent, inference rejects the
configuration with the Java-compatible `unsupported CANCELED destination
fallback` error; it never falls back to a newly invented recovery path.

### Configured Task Integration

Each native configuration entry also names its target `spechls.gamma` with
`gamma_id` and declares its `fast_selector`. These are required, rather than
being inferred from operation order. A configured task must contain exactly the
named gamma operations, their resolved selector inputs must be `i64`, and its
commit record must begin with an `i1` enable. The pass rejects a missing target,
an ambiguous selector width, duplicate gamma IDs, or a slow selector equal to
the fast selector.

Inference appends the machine at module scope and places an `fsm.instance` and
`fsm.trigger` in the task body. One `spechls.fsm_mispec_input` captures each
actual named gamma selector and supplies the trigger's corresponding `i64`
input. One `spechls.fsm_gamma_control` replaces each target gamma selector: it
selects the configured fast selector during normal execution and the FSM's
selected slow path during recovery. `spechls.fsm_commit_gate` ANDs the original
task enable with the controller `commit` output before `spechls.commit`.

Each slow path now has separate `rollback_mu_ids`, `rollback_array_ids`, and
`rollback_gamma_ids`. FSM outputs are target named: `muRollback_<id>`,
`arrayRollback_<id>`, `gammaRollback_<id>`, and `slowPath_<gamma-id>`. The
per-gamma slow-path outputs replace the previous shared `slowPath` command, so
a gamma never observes a different gamma's recovery selector.

The current generic `spechls.rewind`, `spechls.rollback`, and
`spechls.rollbackableDelay` operations cannot address the configured task
input buffers, Mu state, or arrays by these IDs. Emitting them here would
silently invent behavior. Instead, the pass emits `spechls.fsm_recovery_boundary`
with an ordered variadic command payload and a `command_names` attribute. Its
`speculation_config` and `recovery_scenarios` attributes retain every path,
typed target set, depth, release window, poison relation, and replacement rule.
A stateful lowering must apply the commands to the named resources; until then
the boundary is serializable metadata, not executable rewind/rollback behavior.

The current xDSL `fsm.MachineOp` has named per-gamma slow-path outputs but does
not expose Java's symbolic iteration/poison state. Consequently the machine
constructs dependency-ordered combined chains and explicit
`NewMispec_*` interrupt states. Every edge retains an exact `mispec_s == path`
guard; combined entries use the conjunction of their exact selectors. The typed
scenarios remain required by a stateful multi-recovery command lowering because
the machine command shape has one `slowPath` output and no iteration tag.

## Generated Implementation

`FSMBuilder` emits a pure next-state function and a command decoder.

The next-state function clears outputs, executes the symbolic state switch,
decrements a nonzero rewind counter, adds a newly issued rewind, and shifts
delayed commit bits. The command function exposes commands from the current
registered state.

`FSMToGecosASTLowering` lowers this model to a switch statement. It emits
default transitions after guarded transitions, preserving transition priority.
Later lowering replaces `FSMNode` and `FSMCommandNode` with calls to the
generated functions.

The MLIR exporter represents metadata as `spechls.fsm` and
`spechls.fsm_command`, but it does not export the complete symbolic transition
graph, poison map, or all derived rollback data. A native MLIR implementation
must either reconstruct the controller from the exported timing metadata or
introduce an explicit FSM transition representation.

## Correctness Sketch

The desired theorem is a cycle-indexed refinement statement: observable commits
and committed values of the transformed speculative SCC equal those of the
non-speculative SCC, modulo bounded recovery latency.

Define an architectural state as the state after the latest non-squashed input
iteration. Define a transformed state as the FSM state record, Mu/array
rollback history, rewind buffers, and datapath pipeline contents.

Maintain this simulation invariant at each controller cycle:

1. The transformed architectural state equals the reference state at the most
   recently valid logical iteration.
2. Any younger transformed work is either valid speculative work or is covered
   by a pending rollback/rewind recovery plan.
3. `rewindCpt > 0` iff input advancement would retire an input that recovery
   still needs to replay.
4. A visible commit corresponds only to an iteration outside every relevant
   predicate-resolution and recovery horizon.

### Base case

Initialization sets the FSM to its initial path, uses fast selectors, enables
rollback storage, and starts with no outstanding rewind. Therefore the
transformed architectural state equals the reference initial state.

### Fast-path step

In `Proceed`, each gamma selects its configured fast path. If every delayed
predicate agrees, no rollback command is issued and the state self-holds.
Scheduling guarantees II=1 for this configuration, so the next transformed
architectural state is exactly the reference state for the next iteration.

### Detection alignment

Predicate delays ensure that a signal examined at FSM cycle `t` belongs to the
logical iteration whose speculative value is at its resolution stage. Therefore
`mispec_s == p` detects precisely a divergence between the fast execution and
the real branch choice, not an unrelated earlier/later iteration.

### Recovery step

For slow path latency `L` and condition latency `C`, the FSM waits
`max(L-C-1, 0)` positions before rollback. Thus selecting the slow path cannot
use a value before it exists. The rewind depth `max(C, L-1)` covers both input
positions whose control was issued before resolution and positions needed to
align the slow result. Rollback restores every state component listed in the
rollback dependency set. Replaying rewound inputs under the saved slow selector
therefore reconstructs the reference execution from the recovery point.

### Input safety

The recurrence for `rewindCpt` decrements outstanding rewind each cycle and
adds a newly issued rewind. Since `nextInput` is true exactly when it is zero,
no input is discarded while recovery requires it. Induction on this recurrence
proves all replay inputs are supplied in order before normal advancement resumes.

### Commit safety

`commit_s` is suppressed from detection through rollback and the associated
resolution horizon. Delayed commit shifting aligns per-speculation validity to
the task commit interface. Consequently a committed value is never derived
from a later-squashed iteration.

### Multiple signals

Poison windows exclude condition values invalidated by a preceding rollback.
Dependency-aware combination order makes simultaneously recoverable signals
deterministic. The worklist exhausts supported recovery combinations, so the
same single-signal recovery argument applies to each compatible combined path.

## Assumptions and Known Limits

- At least one valid speculation/timing record exists; the constructor uses
  nonempty maxima/minima and `.head` values.
- The fast configuration has II <= 1.
- Rollback dependencies supplied by `SpeculationControlBuilder` are complete.
- The canceled-path fallback described above is unsupported.
- `SpecFSMFactory` mutates its poison map after construction; a reimplementation
  should define one authoritative poison relation rather than reproduce this
  ordering accidentally.
- `SynthesizeControlLogic` delay is a topological node count, not a proven
  hardware timing bound; it must not be used as a substitute for scheduler
  latency metadata.
