# Embedded Swift Support — Implementation Plan

Target: the **traitless** `EdgeTools` core compiles and runs as Embedded Swift on **Swift 6.4**.

6.4 is the floor because it lifts non-class existentials, `Any`/`AnyObject`, metatypes, and untyped
throwing. 6.3 would force typed throws through the entire engine/session API and a callback-only
session stream; on 6.4 both become optional, so the public API can stay feature-complete rather than
degraded.

Out of scope: every trait-gated target (MLX, ONNX, CoreML, CoreAI, Transformers, Foundation, JS,
XGrammar). Bare-metal XGrammar is a sysroot problem, not a code problem — it needs a full STL and
heap the toolchain does not ship for freestanding targets.

## Verified constraints

Probed against Swift 6.3.2's embedded stdlib and cross-checked with the 6.4 status docs
(`docs.swift.org/embedded`). Items marked **permanent** are documented as intentionally unsupported
long-term.

| Constraint | Status on 6.4 |
| --- | --- |
| `Codable` / `Encodable` / `Decodable` | Unsupported (**permanent**) |
| Key paths | Partial only — compile-time-constant, stored properties, `MemoryLayout`/`UnsafePointer` APIs |
| `Observation` module | Not shipped (needs key paths) |
| `weak` / `unowned` | Unsupported ("not yet", no timeline) |
| Generic initializers on classes | Unsupported (**permanent**) — an init cannot be `final` |
| Opening an existential into a generic context | Unsupported (**permanent**) — consequence of full specialization |
| `Mutex` | Absent; embedded `Synchronization` ships only `Atomic` |
| `ContinuousClock` | Absent (`Duration` itself is available) |
| Swift Concurrency | Partial, experimental, single-threaded mode |
| Executor | No default global executor — supplied by the consumer, not by EdgeTools |

Compiler bug to keep in mind: `is`/`as?` against a generic final class OOM-crashes the 6.3.2
frontend when that specialization is never instantiated in the module. Re-check on 6.4.

## Phase 0 — Verification harness

Nothing else is verifiable without this, so it lands first.

1. `scripts/lint-embedded.sh` — **done**. Builds the traitless core with the `EmbeddedRestrictions`
   diagnostic group and fails on any violation. Fast, needs no embedded SDK or cross-compilation,
   runs on any host. Baseline was 14 violations; now 0.

   Two toolchain quirks it has to work around, both verified on Apple Swift 6.4:
   - `-Werror EmbeddedRestrictions` does **not** promote the group to errors — diagnostics stay
     warnings and the build exits 0. The script therefore greps output rather than trusting the
     exit code.
   - The default (swiftbuild) build system passes `-suppress-warnings`, hiding the diagnostics
     entirely, so `--build-system native` is required despite being deprecated.

   Verified to fail on an injected violation, not just to pass when clean.

   This is necessary but *not sufficient*: it does not catch missing modules (`Observation`),
   key paths, `Mutex`, or **generic initializers on classes** — that last one is diagnosed only in a
   real embedded compile, and the `AnyEdgeToolCall` erasure design depends on it, so it is
   currently unverified by CI. Treat the lint as the fast signal, not the proof.

2. `scripts/build-embedded.sh` — the real compile-only cross build, proving the module actually
   forms under `-enable-experimental-feature Embedded -wmo`.

   Known blocker: `swift build -Xswiftc` leaks the embedded flag into the host macro-plugin build
   and breaks swift-syntax. Approaches, in order of preference:
   - a Swift SDK for a bare-metal triple, so host tools build with host flags (verify `-Xswiftc`
     scoping when cross-compiling);
   - a direct `swiftc` invocation over the core plus its dependency sources, with the macro plugin
     built separately for the host and passed via `-load-plugin-executable`. Note swift-collections
     needs its per-module layout preserved — a naive single-module `-wmo` build breaks on duplicate
     basenames and `fileprivate` scoping.

   Spike this before committing to a shape.

3. CI job (see below) running both scripts on a 6.4 development snapshot.

## Phase 1 — Toolchain-agnostic work

None of this depends on the embedded toolchain, all of it is testable in the existing suite, and
most is an improvement regardless of embedded.

### 1a. Decouple serialization from `Codable` — **done**

- Add a direct yyjson → `EdgeToolsValue` parser with no `Decoder` involvement.
- Make `OrderedKeyJSONWriter` the sole encode path.
- Add symmetric JSON entry points as extension methods on the existing conversion protocols:
  `jsonString` on `ConvertibleToEdgeToolsValue`, an `init(edgeToolsJSON:)`-style entry on
  `ConvertibleFromEdgeToolsValue`. No new conformance burden.
- `EdgeToolsGenerationSchema` needs a `ConvertibleFromEdgeToolsValue` conformance; port the
  hand-written `Decodable` init at `EdgeToolsGenerationSchema.swift:94`.
- Demote to `#if !$Embedded`: the whole `EdgeToolsJSONDecoder` `Decoder` conformance, plus the
  `Codable` conformances on `EdgeToolsGenerationSchema`, `EdgeToolDefinition`, `EdgeRawToolCall`,
  and `EdgeToolCallID`.

### 1b. `Internal/_ObservationRegistrar`

Token-based API (`access(.tools)` with a private per-type enum), *not* key-path-shaped, so the
`#if` lives entirely inside the shim and every call site stays uniform. Maps tokens back to key
paths for the real `ObservationRegistrar` on Apple platforms. Three types, ~6 properties.

Leaves room to add an elementary-ui `Reactivity` backend later, since the shim is registrar-shaped
rather than key-path-shaped.

### 1c. Tool erasure via class boxes — **done**

Replaces the existential open at `EdgeToolsSession.swift:396`. Reference sketch verified compiling
under embedded 6.3.2 (`/tmp/embprobe/boxing/Boxing.swift`, to be moved into the repo as a test).

- `private protocol _AnyEdgeTool: AnyObject` with **only non-generic requirements** —
  `erasedDefinition`, `makeCall(rawInput:)`, `invokeDiscardingOutput(_:)`.
- `private final class EdgeToolBox<Tool: EdgeTool>: _AnyEdgeTool` — `Tool` is concrete inside, so
  the generic work happens without opening an existential. Auto-invocation survives because the
  failure and output stay concrete and are swallowed inside the specialization.
- `public struct AnyEdgeTool` wrapping `any _AnyEdgeTool` — a struct, because a generic init on a
  class is banned even when the class is final.
- Same treatment for `AnyEdgeToolCall`: make it a struct, constrain `_AnyEdgeToolCall` to
  `AnyObject`.
- Because opening is banned on 6.4 too, `[any EdgeTool]` is unusable as an *input* type. Tools must
  be acquired where the concrete type is static — an `@EdgeToolsBuilder` result builder
  (`buildExpression(_ tool: some EdgeTool) -> AnyEdgeTool`, varargs `buildBlock`, **no parameter
  packs**) or a generic `addTool(_ tool: some EdgeTool)`.
- Gate to `#if !$Embedded`: the erased *reads* `AnyEdgeToolCall.tool` / `.input` / `.output` /
  `.status`, which surface `any Sendable` / `any Error`. Embedded consumers use `as(Tool.self)`.
- On 6.4 the box can expose `var erasedTool: any EdgeTool`, so `session.tools` keeps returning
  `[any EdgeTool]` — the erasure stays an implementation detail and this is not a breaking change.

### 1d. Remove `weak` from the traitless core — **done**

Every `weak` in the core is redundant: `EdgeToolsSessionStream.start` already retains both the
stream and the session strongly in the generation `Task` for the whole generation.

- `:308-309` channel callbacks → strong. Semantically identical; also fixes the latent case where a
  released stream silently discards output while generation continues.
- `:412` `setOnFinish` → strong, **and** clear `state.onFinish = nil` inside `finish()` alongside
  the existing `state.stop = nil`. That is the only genuine cycle, and clearing makes it
  deterministic. `finish()` is guaranteed to run once on success, failure, or stop.
- `:238`, `:283` `continuation.onTermination` → replace with explicit subscription handles. This is
  the only cleanup driven by consumer abandonment rather than generation completion:

  ```swift
  public final class EdgeToolsSubscription {
    private let stream: EdgeToolsSessionStream  // strong: ownership points one way
    private let id: Int
    public func cancel() { self.stream.removeSubscriber(self.id) }
    deinit { self.cancel() }
  }
  ```

  No cycle — the stream holds only a callback keyed by ID. Strictly more deterministic than
  `onTermination`, which only fires on `.cancelled`.

**Behavior change to sign off on:** with strong captures, an engine that never calls back leaks the
session, stream, and engine rather than just the engine. Today's `weak` partially masks that. A
non-returning engine is already a fatal bug, but this should be a deliberate decision.

### 1e. Metadata boxing

Replace `[EdgeToolsMetadataKey: any Sendable]` with a generic typed key (`EdgeToolsMetadataKey<Value>`)
over a final-class box. Removes the `as?` casts at every call site as a side effect.

### 1f. Mechanical cleanups — **partially done** (`Task.checkCancellation` and the `any Error`
interpolation are fixed; key-path rewrites remain outside the session)

- Rewrite the ~21 key-path sites as closures (`map(\.definition)` → `map { $0.definition }`).
- `Task.checkCancellation()` at `EdgeToolsSession.swift:324` → `if Task.isCancelled` returning a
  `wasStopped` generation.
- Fix the `any Error` interpolation in `Internal/Task+CancellableValue.swift:12`.
- Confirm `EdgeToolCall.init(id:tool:rawInput:)` stays legal — it uses only the class's own generic
  parameter, so it is not a generic method.

## Phase 2 — Embedded-only shims

- `Lock`: needs a third branch. Preference order — (1) pthread-backed, selected by a **trait**,
  since no `canImport` can distinguish "sysroot has pthreads"; (2) a consumer-supplied C shim;
  (3) `Atomic` spinlock last, documented as unsafe under priority-based preemption. A trait-gated
  no-op is legitimate for explicitly single-threaded targets — that is a user assertion, not a
  framework assumption.
- Timing: `Duration` works, `ContinuousClock` does not. Engines are trait-gated so the core is
  fine, but the consumer-supplied clock needs documenting.
- Executor: document prominently that the consumer provides the global executor. A session whose
  tasks never run is a miserable first experience.
- Audit remaining `#if $Embedded` gaps once `build-embedded.sh` runs green.

## Phase 3 — CI

New job, added to `.github/workflows/ci.yml`. Note most jobs there are currently commented out;
this one ships enabled.

```yaml
  embedded:
    name: Embedded Build
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        with:
          persist-credentials: false
          submodules: recursive
      - name: Install swiftly
        run: |
          curl -O https://download.swift.org/swiftly/linux/swiftly-$(uname -m).tar.gz
          tar zxf swiftly-$(uname -m).tar.gz
          ./swiftly init --quiet-shell-followup --assume-yes
          echo "$HOME/.local/share/swiftly/bin" >> "$GITHUB_PATH"
      - name: Install Swift 6.4 development snapshot
        run: swiftly install --use 6.4-snapshot
      - name: Lint embedded restrictions
        run: scripts/lint-embedded.sh
      - name: Build traitless core in embedded mode
        run: scripts/build-embedded.sh
```

Pin the snapshot date once one is known good; a floating `6.4-snapshot` will break unpredictably.
Both steps must be gating, not advisory — the lint alone does not prove the module forms.

## Sequencing

Phase 0 first (nothing is verifiable without it), then 1a → 1b → 1c → 1d, then 1e/1f in any order,
then Phase 2 once the embedded build runs. Phase 3's lint step can land with Phase 0; the full
embedded build step turns on when Phase 2 completes.

## Open questions

- Does the erasure `as?` OOM bug persist on 6.4? Blocks 1c if so.
- Does `-Xswiftc` scope correctly to target-only when cross-compiling with a Swift SDK? Determines
  the shape of `build-embedded.sh`.
- Is the strong-capture leak profile (1d) acceptable?
- Is `AsyncThrowingStream` actually usable on 6.4 embedded given concurrency is still "partial,
  experimental, single-threaded"? If not, the callback stream returns as a requirement rather than
  a back-deployment nicety.
