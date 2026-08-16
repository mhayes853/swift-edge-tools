# Llama Engine

Working notes, findings, and architectural decisions for adding a llama.cpp engine to
EdgeTools. Updated as implementation progresses.

## Goal

A `LlamaEngine` for GGUF models on Apple platforms and non-WASI, non-Apple platforms
(Linux, Windows, Android). The engine mirrors `MLXEngine`'s public surface: forkable
observable contexts, the shared `EdgeToolsGenerationLoop`, `EdgeToolsModelProfile`-based
model integration, true prefill via `EdgeToolsPrefillableEngine`, and XGrammar constraints.

## Cactus Hybrid Findings

https://github.com/cactus-compute/cactus-hybrid

- "Hybrid inference" means confidence-based routing between an on-device model and a cloud
  model, not hardware splitting. Models carry a trained probe head (extra GGUF tensors)
  that scores response reliability so low-confidence queries can be handed off.
- The repo pins llama.cpp tag `b10076` (`PIN` file) and ships a six-patch
  `git format-patch` series:
  1. GGUF Python tooling: probe tensor + KV key support.
  2. HF → GGUF conversion for `Gemma4E2BItHybridForCausalLM` (probe weights kept F32).
  3. C++ architecture: gemma4-derived model with probe tensor loading and quantization
     exclusions.
  4. Probe runtime + `llama_probe_*` C API.
  5. llama-server `"confidence"` response field (not needed by us).
  6. Golden tests against a numpy reference (not needed by us).
- Patches 1–4 matter for the Swift engine and artifact build. Patches 1–2 also inform the
  `python` side for export tooling.
- This dovetails with existing framework behavior: the MLX engine already computes a
  top-2-logit heuristic confidence per token (`ConfidenceState`) surfaced as
  `generationConfidence` / `perTokenConfidences` metadata. The probe is a second, learned
  confidence source. The Llama engine emits both: heuristic confidence for any GGUF
  (computed from the CPU logits at sampling time) and probe scores as a new metadata key
  when the loaded model has probe tensors.

## Decisions

### Packaging: prebuilt artifactbundle + bring-your-own-build traits

Two traits, following the removed ONNX engine's precedent (`ONNXCore` / `ONNX`, see
`02f0257^`):

- `LlamaCore`: engine, protocols, and a headers-only `CLlama` target (vendored `llama.h`
  et al., pinned to the `b10076` ABI). No binaries. Users link their own llama.cpp build.
- `Llama`: enables `LlamaCore` and adds our vendored, cactus-patched binary artifactbundle.

Unlike ONNX Runtime, llama.cpp has no `OrtApi`-style function-pointer table; `llama_*`
symbols bind at link time. So the engine never references `llama_*` symbols directly.
Instead a `LlamaApi` struct of closures covers the exact C surface the engine uses
(model/context lifecycle, vocab/tokenize/detokenize, batch/decode/logits,
`llama_memory_seq_*`, `llama_state_seq_*`, chat-template metadata lookup, probe). Under
the `Llama` trait we provide a default instance wired to the linked patched symbols. The
`llama_probe_*` entries are optional closures — `nil` on stock builds — so
bring-your-own users without the cactus patches lose only probe confidence.

Artifact build: `scripts/llama/build-artifact.sh` mirroring `scripts/needle2` and
`scripts/tokenizers` — clone `b10076`, `git am` the cactus series, CMake per target
triple, zip a static-library artifactbundle. Backends: Metal + Accelerate on Apple,
CPU-only elsewhere for v1 (Vulkan deferred; llama.cpp's CPU backend is strong for the
sub-1B models this framework targets). Binary target conditioned on every platform except
`.wasi`.

### KV cache and forking: fork-family ownership

Constraint: the KV cache lives inside `llama_context`. Cheap copy-on-write forking
(`llama_memory_seq_cp`, cells shared between sequence ids until divergence with a unified
KV cache) only works *within* one `llama_context`. Across two contexts the only mechanism
is `llama_state_seq_get_data` / `set_data` — a full serialize/copy, into a destination
that has its own full-size KV allocation.

Rejected options:

- Exclusive `llama_context` per EdgeTools context: every fork is a blob copy plus a full
  KV allocation. Least efficient.
- Single engine-owned `llama_context`: unrelated root contexts share one `n_ctx` budget
  and all decoding serializes globally. Also mismatches MLX, where fresh contexts are
  independent.

Chosen: contexts own the state, jointly within a fork family. `engine.context()` creates
a root context whose `LlamaModelState` allocates a `llama_context` inside a shared
reference-counted, lock-protected holder that manages sequence-id allocation. `fork()`
hands the child the same holder plus a fresh sequence id via `llama_memory_seq_cp` (COW).
The engine holds only the `llama_model` — mirroring MLX, where the engine shares immutable
weights and contexts own mutable cache state. Prefix reuse (MLX's `CachedPrefill`
`starts(with:)` check) maps to tracking token ids per sequence, trimming the divergent
tail with `llama_memory_seq_rm`, and decoding only the suffix.

Notes:

- Requires `kv_unified = true` so sequences share the cell pool (verify against
  `b10076` during implementation).
- `n_ctx` and `n_seq_max` are fixed at `llama_context` creation → they become
  `LlamaContextParameters` on root-context creation. Exceeding `n_seq_max` on fork either
  errors (v1) or falls back to a blob-copy migration into a fresh family.
- Sequence ids are recycled through the holder when forked contexts deinit.
- Decode serializes within a family (holder lock), matching MLX's per-context locking;
  separate families can decode in parallel.
- KV quantization differs from MLX: llama.cpp fixes the KV cache type (`q8_0`, …) at
  context creation, so it lives on `LlamaContextParameters`, not per-step
  `GenerateParameters` (no analogue of MLX's dynamic `kvCacheQuantizationBits`).

The `MLXContext` transcript/revision/`isResponding`/fork bookkeeping (~350 lines) is
engine-agnostic except for the model-state type; extract a shared generic context so
`LlamaContext` is not a copy-paste. `LlamaModelState` (holder reference + seq id + cached
token ids) slots into the same shape as `MLXModelState`.

### Decode loop

Logits arrive as a CPU `float *`, so:

- XGrammar bitmask applies with a plain CPU loop (no `applyBitmaskMLX` analogue needed).
- A CPU fused sampler honors `EdgeToolsFusedSamplingParameters` (counterpart to
  `MLXFusedSampler`).
- Top-2 heuristic confidence falls out of the same pass for every model; probe confidence
  (Gemma 4 hybrid GGUFs only) queried through the optional `LlamaApi` probe closures.

Model profiles follow the existing per-engine pattern (`Qwen3LlamaProfile` alongside
`Qwen3MLXProfile`), reusing the engine-agnostic tool-calling parsers and grammars in
`Models/`.

### Tokenizer: llama.cpp native

The GGUF-embedded vocab is ground truth for what the model was quantized against. A
`LlamaTokenizer` over `llama_vocab` (through `LlamaApi`) conforms to:

- `EdgeToolsTokenizer`: `llama_tokenize` / `llama_detokenize` / vocab getters.
- `XGRTokenizer`: `XGRTokenizerInfo` needs the encoded vocabulary, vocabulary type, and
  `addPrefixSpace` — derivable from `llama_vocab_type` (SPM → `byteFallback` with prefix
  space, GPT-2-style BPE → `byteLevel`).
- `EdgeToolsChatTokenizer`: renders the GGUF-embedded `tokenizer.chat_template` (see
  below).

### Chat templating: vendored minja, decoupled from HuggingFaceTokenizers

llama.cpp's in-library `llama_chat_apply_template` is a hardcoded template lookup with no
tools support — disqualifying for a tool-calling framework. Real Jinja is required, fed by
the GGUF-embedded template string.

History that constrains the choice: swift-jinja was used previously and dropped
(`93ef0ca`, 2026-08-15) for fidelity — its `tojson` sorted keys and emitted compact JSON,
so tool schemas reached models in neither the order nor the spacing they were trained on,
and it dropped a hyphen from literal text. The Rust `edge_template_render` (minijinja +
pycompat, transformers-compatible `tojson`, `raise_exception`, `strftime_now` pinnable via
`edge_tools_now`, `{% generation %}` rewrite) renders byte-identical to transformers'
jinja2 (`3ff4f13`).

The renderer (`hf_template_render`, `rust/tokenizers/src/lib.rs`) only uses minijinja,
minijinja-contrib, serde_json, and chrono — the heavy `tokenizers` crate (fancy-regex,
unicode tables) never enters that path, so templating does not require bundling the HF
tokenizer code. Relying on linker stripping of the single combined artifact was rejected:
dead-stripping happens at the *final app* link under the user's toolchain flags
(`-dead_strip` / `--gc-sections` / `/OPT:REF`, not guaranteed outside Xcode), Rust
codegen-unit boundaries don't follow semantic boundaries, and stripping never shrinks the
shipped artifactbundle (currently 22–45 MB per slice).

Reviving swift-jinja was rejected: its failures live inside the engine, not at the API
boundary. Key-order loss happens when the context converts into the engine's value model
(no custom filter can recover it), the dropped-hyphen bug is in the lexer/parser, and the
`tojson` formatting and pycompat surface are engine-fixed. "Using it differently" would
mean forking and maintaining a Jinja engine in Swift and re-doing validation that prompts
stay byte-identical — churn that the sub-1B models this framework targets are sensitive
to.

Chosen — vendored minja (google/minja): a header-only C++17 Jinja engine built
specifically for HF chat templates (~2,500 LOC + nlohmann::json, insertion-ordered via
`ordered_json`), tested upstream byte-for-byte against Python jinja2 on real HF templates,
and already used by llama.cpp itself. Vendor it as a C++ source target (the `CXGrammar`
pattern; PatchPlugin available for tweaks) behind a `ChatTemplates` trait/target that both
`HuggingFaceTokenizers` and `Llama` enable. If it also replaces the HF tokenizer's Rust
renderer, minijinja/chrono drop out of `rust/tokenizers` entirely: no new artifact, a
smaller `CTokenizers`, and one renderer guaranteeing identical prompt bytes across
engines.

Acceptance gate: the existing chat-template snapshots (byte-identical to transformers as
of `93ef0ca`) must pass unchanged with minja. Known deltas to handle: `{% generation %}`
via the existing pre-rewrite (a string transform, movable to Swift), `strftime_now`
pinning (global injection or a small patch), exceptions are fine (templating platforms
exclude WASI). Minja omits `{% raw %}`, template inheritance, and most non-chat filters —
acceptable for chat templates, but the snapshots decide.

Fallback if minja fails byte-identity in unpatchable ways: split the Rust layer into a
workspace (`rust/templates` → small `CTemplates` artifactbundle with the minijinja
renderer as-is; `rust/tokenizers` drops it), with the same `ChatTemplates` trait shape.

### Naming

`Llama` / `LlamaCore` traits; `LlamaEngine`, `LlamaModelProfile`, `LlamaContext`,
`LlamaTokenizer`, `LlamaApi` types. "Llama" is understood as llama.cpp, not the model
family.

## Implementation Plan

Two parallel early tracks — generic library improvements (A) and llama vendoring (B) —
then the engine (C) on top. Nothing in track A references llama.cpp; each A phase lands
independently against the existing MLX/tokenizer test suites.

### Track A: generic library improvements (parallel with B)

- **A1 — minja chat templating. [DONE — artifact rebuild pending]** Vendor minja + nlohmann::json as a C++ source target
  (`CMinja`, the `CXGrammar` pattern, PatchPlugin available) with a small C shim, behind a
  `ChatTemplates` trait/target. Move the `{% generation %}` pre-rewrite to Swift; keep
  `strftime_now` pinning (`edge_tools_now`-equivalent context key). Swap
  `HuggingFaceTokenizer.renderChatTemplate` onto it. Gate: existing chat-template
  snapshots pass byte-identical. Then drop minijinja/chrono from `rust/tokenizers` and
  rebuild a slimmer `CTokenizers` artifact. Fallback if the gate fails unpatchably: Rust
  workspace split (see Chat Templating section).
- **A2 — generic fork context. [DONE]** Extract `MLXContext`'s engine-agnostic bookkeeping
  (transcript/revision/`isResponding`, `begin`/`finish`, `fork`) into a public generic
  context parameterized over a forkable model-state requirement
  (`forkedContextState(copyingCache:)`). `MLXContext` becomes a thin wrapper; evaluate
  also lifting the `generationTask`/`finalize` wiring shared by transcript engines.
  Gate: MLX engine tests and snapshots unchanged.
- **A3 — CPU sampling + masking (SIMD). [DONE]** A CPU fused sampler honoring
  `EdgeToolsFusedSamplingParameters` over a logits buffer (counterpart to
  `MLXFusedSampler`), CPU grammar-bitmask application, and top-2 per-token confidence
  extraction in the same pass. Vectorized following the removed ONNX-era pattern
  (`EdgeToolsSampling.swift` at `02f0257^`): Accelerate/vDSP fast path under
  `canImport(Accelerate)`, portable `SIMD16<Float>` path elsewhere (lane-offset index
  tracking, masked replace, scalar tail) — resurrect and extend the deleted
  `argmaxContiguous`/`argmaxSIMD` helpers into softmax/top-k/top-p/min-p/penalties.
  ggml was considered and rejected: it would tie the generic layer to llama linkage
  (breaking track independence and requiring `LlamaApi` closures for non-public ggml
  symbols) and its vector kernels aren't exported as stable public API. Engine-agnostic,
  unit-tested directly.

### Track B: llama vendoring (parallel with A)

- **B1 — patched artifact. [DONE — Apple slices; linux/android/windows are release
  chores]** `scripts/llama/build-artifact.sh` mirroring
  `scripts/needle2` conventions: clone `b10076`, `git am` cactus patches 1–4, CMake static
  builds per triple (Apple: Metal + Accelerate; Linux/Windows/Android: CPU), assemble
  headers + modulemap + checksums into `bin/llama-<version>.artifactbundle.zip`.
- **B2 — traits + `LlamaApi`. [DONE]** `LlamaCore` trait: headers-only `CLlama` target (vendored
  `b10076` headers) plus the `LlamaApi` struct of closures over the C surface the engine
  uses (lifecycle, vocab/tokenize, batch/decode/logits, `llama_memory_seq_*`,
  `llama_state_seq_*`, GGUF metadata, optional probe entries). `Llama` trait: enables
  `LlamaCore` + `ChatTemplates`, links the B1 binary, provides the vendored `LlamaApi`
  default.

### Track C: the engine (after A + B)

Per-token (top-2) confidence is part of the decode step (C2), not a separate phase —
every model gets it. Probe ("hybrid") confidence applies only to Gemma 4 hybrid GGUFs
(Needle 2 handles its own confidence), so it lives with the Gemma 4 profile work in the
C5 fan-out rather than on the critical path.

- **C1 — `LlamaTokenizer`. [DONE — real-GGUF validation lands with C5a]** Core (needs B2 only): `EdgeToolsTokenizer` over
  `llama_vocab` through `LlamaApi`; `XGRTokenizer` via vocab-type mapping
  (SPM → `byteFallback` + prefix space, BPE → `byteLevel`); stop tokens from GGUF
  metadata. Chat conformance (adds A1): `EdgeToolsChatTokenizer` rendering the
  GGUF-embedded template through `ChatTemplates`.
- **C2 — single-sequence model state. [DONE]** (needs A3, B2; parallel with C1).
  `LlamaModelState` over one `llama_context`/sequence: prefill, decode step
  (bitmask → CPU sampler → per-token confidence), commit/reset mirroring
  `MLXModelState`. Enough to bring the engine up before forking exists.
- **C3 — `LlamaEngine`. [DONE]** (needs A2, C1, C2). `LlamaContext` from the generic context,
  `LlamaContextParameters` (`n_ctx`, `n_seq_max`, KV type, threads, flash attention),
  `LlamaGenerateParameters`, engine conformances (`EdgeToolsEngine`,
  `EdgeToolsPrefillableEngine`, `EdgeToolsTokenizingEngine`) wired through
  `EdgeToolsGenerationLoop`.

After C3, the remaining phases run in parallel:

- **C4 — fork family + prefix reuse.** The shared lock-protected holder (seq-id
  allocation/recycling), `fork()` via `llama_memory_seq_cp`, prefix reuse via
  `llama_memory_seq_rm` + suffix decode. Verify `kv_unified` `seq_cp` COW behavior at
  `b10076` here.
- **C5a — test harness + first profile.** `scripts/test-llama.sh` with GGUF fixtures,
  engine generation snapshot tests (`withKnownIssue` recording), and a first profile
  (Qwen3) reusing the engine-agnostic tool-calling parsers/grammars in `Models/`.
- **C5b — profile fan-out** (after C5a; parallel per model). Remaining llama profiles
  (Gemma 4, LFM2.5, Granite, …). The Gemma 4 profile additionally wires probe
  confidence: a probe metadata key emitted when the `LlamaApi` probe closures are present
  and the model carries probe tensors (needs the track D hybrid GGUF for validation).
- **C5c — CLI.** `.gguf` model detection in `EdgeToolsCLI`.

### Track D: python export (independent, anytime)

Fold cactus conversion patches 1–2 into the python export tooling and produce a hybrid
probe GGUF. No engine dependency; only C5b's Gemma 4 probe validation consumes its
output.

## Progress Log

- **2026-08-16 — A3 complete.** `EdgeToolsCPUSampling.swift`: `EdgeToolsCPUFusedSampler`
  (+ `EdgeToolsCPUTokenHistory`, `EdgeToolsCPUSample`), `applyBitmaskCPU`, and
  internal SIMD reductions (`maxContiguous`, `argmaxContiguous`, `topTwoContiguous`,
  `sumExpContiguous`). Semantics mirror `MLXFusedSampler` exactly (selection at
  temperature 1 over full-vocab log-probabilities, temperature applied in the categorical
  draw, default temperature 0.6, history ring of `repetitionContextSize ?? 20`,
  confidence post-mask/pre-penalty); the test suite is a 1:1 port of
  `MLXFusedSamplerTests` plus bitmask/confidence/large-vocab cases. Implementation notes:
  vDSP paths under `canImport(Accelerate) && !$Embedded`, portable `SIMD16<Float>`
  otherwise; top-k via `HeapModule.Heap`; top-p without top-k uses progressive top-M
  (256, ×4) until the nucleus mass is covered; full-vocab log-sum-exp skips SIMD blocks
  below `max - 20`; RNG is SplitMix64 (embedded gets a fixed default seed since
  `SystemRandomNumberGenerator` may be unavailable); key-path literals avoided for
  embedded compatibility. `validateBitmaskCoverage` promoted from `private` to internal
  for reuse.

- **2026-08-16 — A2 complete.** `Engines/EdgeToolsTranscriptContext.swift`:
  `EdgeToolsTranscriptContext<ModelState>` extracted from `MLXContext` (state lock,
  transcript/revision/`isResponding` observation, `begin`/`finish`, lazy-copy `fork`),
  parameterized over a new `EdgeToolsForkableModelState` protocol
  (`forkedContextState(copyingCache:)`), plus public `EdgeToolsEngineIdentity` for the
  context–engine ownership check. `MLXContext` and `MLXContextParameters` are now
  typealiases; `MLXModelState` conforms to the protocol. Because the generic context
  lives in the traitless core, it uses the session's embedded-safe observation pattern
  (fileprivate `ObservedProperty` enum, `#if !$Embedded` around registrar/keypath use and
  the `Observable` conformance) instead of `MLXContext`'s direct keypath calls. Gate held:
  full MLX suite green (46 tests, context/fork tests unchanged); generation snapshots
  re-recorded with expected phrasing variance only (tool calls unchanged, validated).
  The `generationTask`/`finalize` wiring was left in `MLXEngine` for now — it depends on
  profile/parameter types the llama engine won't share; revisit during C3 if a shared
  shape emerges.

- **2026-08-16 — A1 complete (CTokenizers rebuild pending).** `CMinja` target vendors
  minja `021c229` + nlohmann v3.12.0 (PIN files record versions) with an
  `edge_template_render` C shim; `EdgeToolsChatTemplates` wraps it behind the new
  `ChatTemplates` trait, which `HuggingFaceTokenizers` enables. Findings:
  - As predicted, minja's builtin `tojson` rejects `ensure_ascii` kwargs (Qwen-family
    templates pass `tojson(ensure_ascii=False)`) and its output diverges from
    json.dumps. Solved without patching minja: the shim shadows `tojson` (and injects
    pinned-UTC `strftime_now`) via context values, which resolve before the builtin
    parent scope — the same mechanism minja's own `chat-template.hpp` uses.
  - The `{% generation %}` neutralization and `edge_tools_now` pinning ported into the
    shim 1:1 from the Rust renderer.
  - **Gate held**: the 9-profile model chat-template snapshot passed byte-identical with
    no re-record; full suites green on macOS and Linux (docker).
  - Rust crate slimmed to pure tokenization (minijinja/minijinja-contrib/chrono/serde_json
    dropped); artifact version bumped to 0.2.0 in the build script. The shipped 0.1.0
    bundle still contains the dead `hf_template_render` symbol until a multi-toolchain
    rebuild — harmless, nothing references it.

- **2026-08-16 — B1/B2 complete; C1 complete.** The committed bundle links and runs from
  Swift (backend smoke test passes; probe closures present). `LlamaApi` is a closure
  struct over semantic operations (handles as `@unchecked Sendable` pointer wrappers,
  probe entries optional so stock builds work); `LlamaApi.vendored` binds the patched
  symbols under the `Llama` trait. A key testability payoff: tests construct mock
  `LlamaApi` values (`MockLlamaApi.swift`), so `LlamaTokenizer` — `EdgeToolsTokenizer` +
  `XGRTokenizer` (SPM → `byteFallback` + prefix space, BPE → `byteLevel`) +
  `EdgeToolsChatTokenizer` (GGUF `tool_use`/default template through minja) — is fully
  unit-tested without model files. SwiftPM gotchas hit: the artifact dict in `info.json`
  requires a `version` field (opaque "does not contain a binary artifact" otherwise),
  and a deleted `.build/artifacts` needs `workspace-state.json` cleared too or extraction
  silently never re-runs. Findings from bringing up the patched build:
  - The cactus patch series (six patches at cactus `cfea89c`) applies cleanly onto tag
    `b10076` and builds statically with `LLAMA_BUILD_COMMON/APP/MTMD/TOOLS/...=OFF`.
    The minimal library set is `libllama.a` + `libggml{,-base,-cpu,-metal}.a`
    (`llama-common-base` and `cpp-httplib` are not needed); a merged `libllama.a` links
    and runs against Metal/MetalKit/Accelerate/c++ with the embedded Metal shader
    library (`GGML_METAL_EMBED_LIBRARY=ON`).
  - The probe API (`llama_model_has_probe`, `llama_probe_confidence(ctx, seq_id)`,
    `llama_probe_reset(ctx, seq_id)`) is declared in `src/llama-ext.h` — an internal
    staging header outside any `extern "C"` block — so the symbols were C++-mangled and
    unreachable from Swift. Fixed with our own patch
    (`patches/Llama/0001-llama-export-the-handoff-probe-API-with-C-linkage.patch`)
    applied after the cactus series by the artifact build. Probe confidence is
    **per sequence id**, which meshes exactly with the fork-family design.
  - `b10076` confirms the KV design inputs: `llama_context_params.kv_unified` documents
    the exact tradeoff (unified helps "when the sequences share a large prefix" — our
    case), and `llama_model_chat_template(model, name)` reads named GGUF templates
    (default/tool-use) directly.
  - `scripts/llama/build-artifact.sh` mirrors the needle2/tokenizers conventions
    (target matrix with `LLAMA_TARGETS` subsetting, merged static lib per slice via
    libtool/ar-MRI, generated `info.json`, deterministic zip). Initial bundle carries
    the Apple slices (macos-arm64/x86_64, ios-arm64, ios-sim-arm64; Metal off on
    x86_64); linux (docker), android (NDK), and windows slices are release chores on
    their host toolchains.
  - Package: `LlamaCore` trait (enables `ChatTemplates`) and `Llama` trait (enables
    `LlamaCore`, links the `CLlama` binary target + Metal/MetalKit/Accelerate/c++).
    `CLlama` dependency currently conditioned to macOS/iOS until the other slices ship.

- **2026-08-16 — C2/C3 complete.** `LlamaEngine<Profile>` generates end-to-end through
  the shared loop against mock scripted logits (session integration, grammar-constrained
  sampling, per-token confidence metadata). Design notes from implementation:
  - `LlamaSequence` owns one lazily created `llama_context` (lazy so `context()` and
    `fork()` stay non-throwing; creation errors surface at generation). Cross-turn KV
    reuse = trim to longest common token prefix (`memoryRemove`) + decode the suffix in
    512-token chunks, with a re-decode-last-token fallback when the prompt fully matches
    the cache (logits would otherwise be stale).
  - `EdgeToolsForkableModelState` gained a `generationState()` requirement (defaulted to
    the fork path for MLX): `begin` reusing `forkedContextState(copyingCache: false)`
    would have handed generations a fresh llama context and severed KV continuity.
    Generations share the context's sequence; exclusivity comes from `isResponding`, and
    the sequence serializes all llama calls internally.
  - User forks currently allocate a fresh llama context (correct, cache-cold); C4
    replaces this with `seq_cp` copy-on-write inside a shared family.
  - Chat-template message/tool conversions moved from `MLXEngine.swift` to a shared
    `EdgeToolsChatTemplateValues.swift`, so both engines produce identical prompt values.
  - The CPU sampler samples llama's logits buffer in place (no scratch copy).
  - Testing: the closure-struct `LlamaApi` lets tests script logits per decode step, so
    prefill→generate KV reuse is asserted at the decode-batch level (positions and token
    payloads) without any model file.

## Open Questions

- `n_seq_max` fork-overflow behavior: error in v1, or blob-copy migration fallback.
- Exact `kv_unified` / sequence-stream semantics at `b10076` — confirm `seq_cp` is COW
  under the chosen context params.
- Whether `LlamaApi` closures take C types by value (pins users to `b10076`-era ABI for
  `llama_batch` etc.) or we shrink the by-value surface. Document the pin either way.

## Deferred

- Vulkan (Windows/Linux/Android GPU) backend in the vendored artifact.
- Blob-copy family migration if `n_seq_max` proves limiting.
- `llama_decode` blocks a cooperative-pool thread during decode (MLX has the same
  behavior); a dedicated executor is a possible future refinement.
- CTokenizers 0.2.0 artifact rebuild (drops the dead renderer symbol) and the
  linux/android/windows CLlama slices — release chores on their host toolchains.
