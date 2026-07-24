# Needle ONNX Engine Implementation Plan

## Implementation Status

The Apple implementation and proof gate are complete, including CPU and CoreML execution, Float32/INT4/INT8 models, `ONNXCore` custom backends, and macOS/iOS validation. The requested 1.27.1 Apple archive was not published, so the binary target currently pins Microsoft's official ONNX Runtime 1.27.0 archive. Upgrade the binary target and vendored C headers together when a 1.27.1-compatible artifact is available.

The initial implementation will prove a concrete ONNX Runtime engine on Apple platforms before extracting any runtime abstraction. All Needle-specific ONNX Swift code will remain in a single file:

```text
Sources/EdgeTools/Models/Needle/Engines/NeedleONNXEngine.swift
```

## Phase 1: Prepare the export

Make ONNX export reliably Float32:

- Add explicit Float32 export configuration.
- Ensure `configuration.json` records Float32.
- Verify unquantized, INT4, and INT8 models with ONNX Runtime's CPU Execution Provider.
- Preserve external-data support.
- Keep the `MatMulNBits` contrib operators required by quantized exports.

## Phase 2: Integrate the GPU-capable XCFramework

Use Microsoft's official ONNX Runtime 1.27.0 archive, which includes:

- Default CPU Execution Provider.
- CoreML Execution Provider.
- Full standard and contrib operator sets.
- Static linkage.
- No XNNPACK or training APIs.

Include these slices:

- macOS arm64.
- macOS x86_64.
- iOS arm64.
- iOS Simulator arm64.
- iOS Simulator x86_64.

Reference the official archive directly from the SwiftPM binary target and pin its checksum.

## Phase 3: Add traits without abstraction

Declare:

- `ONNXCore`.
- `ONNX`, enabling `ONNXCore`.

During the proof phase:

- `NeedleONNXEngine` temporarily requires `ONNX`.
- No ONNX provider protocols exist.
- The implementation calls ONNX Runtime directly.
- `ONNXCore` receives the engine only after the proof gate and protocol extraction.

Neither trait is enabled by default.

## Phase 4: Concrete execution configuration

The concrete engine should support:

```swift
extension NeedleONNXEngine {
  public struct RuntimeConfiguration: Hashable, Sendable {
    public var executionProvider: ExecutionProvider
  }

  public enum ExecutionProvider: Hashable, Sendable {
    case cpu
    case coreML(computeUnits: CoreMLComputeUnits)
  }

  public enum CoreMLComputeUnits: String, Hashable, Sendable {
    case all = "ALL"
    case cpuAndGPU = "CPUAndGPU"
    case cpuAndNeuralEngine = "CPUAndNeuralEngine"
    case cpuOnly = "CPUOnly"
  }
}
```

CPU mode uses ONNX Runtime's default CPU EP.

CoreML mode appends the CoreML EP before session creation, with options such as:

- `MLComputeUnits`.
- `ModelFormat = MLProgram`.
- `RequireStaticInputShapes = 1`.
- `EnableOnSubgraphs = 1`.

ORT's CPU EP remains available as fallback for unsupported graph nodes. `.coreML(.cpuAndGPU)` makes GPU execution eligible, but Core ML still chooses the actual hardware for individual operations.

Start with `.cpu` as the default. Consider an accelerated default only after correctness and performance are measured.

## Phase 5: First end-to-end CPU test

Create:

```text
Tests/EdgeToolsTests/Models/Needle/Engines/NeedleONNXEngineTests.swift
```

Add only `Generate Basics With CPU Execution Provider` first. It should use the real Float32 Needle export and assert:

- Generation completes.
- It was not stopped.
- The response matches the verified deterministic ONNX response.
- Tokens are nonempty.
- Metrics match token counts.
- Confidence values are finite and valid.

Implement only enough concrete engine behavior to make this pass.

## Phase 6: Prove CoreML/GPU eligibility

Add `Generate Basics With Core ML CPU And GPU` using `.coreML(.cpuAndGPU)`.

Assert:

- Provider registration and session creation succeed.
- Generation completes.
- Response matches the separately verified CoreML response.
- Token counts and confidence values are valid.

If practical, use ONNX Runtime graph-assignment diagnostics to confirm CoreML receives at least one subgraph. This proves CoreML participation, though not the exact hardware used for every Core ML operation.

## Phase 7: Grow the real test suite vertically

Add one failing test and make it pass before adding the next.

### State and concurrency

1. `Sequential Generations With CPU`.
2. `Concurrent Generations With CPU`.
3. `Sequential Generations With Core ML`.
4. `Concurrent Generations With Core ML`.

### Generation behavior

1. `Generate Streamed Response Matches Final Response`.
2. `Generate Invokes Custom Logit Processor`.
3. `Generate Stops And Returns Stopped Generation`.
4. `Generate Cancels And Throws Cancellation Error`.
5. `Generate Through EdgeToolsSession`.
6. `Generate Throws When Prompt Exceeds Context Length`.

Use CPU for the main behavior suite. Give CoreML dedicated basic, sequential, concurrent, and cancellation coverage as needed.

### Quantization

1. `Generate Basics With INT4 Export`.
2. `Generate Basics With INT8 Export`.

Run quantized tests with CPU initially. Add accelerated variants only if CoreML accepts meaningful portions of those graphs.

Avoid snapshots containing exact timings or complete confidence arrays.

## Phase 8: Error tests

After happy paths work, add these one at a time:

1. `Initialization Throws For Missing Encoder Model`.
2. `Initialization Throws For Missing Decoder Model`.
3. `Initialization Throws For Missing External Data`.
4. `Initialization Throws For Invalid Model Signature`.
5. `Runtime Error Preserves ONNX Status And Message`.
6. `Initialization Throws When Core ML Provider Cannot Be Registered`.

Explicit CoreML selection should throw if provider registration fails. Unsupported graph nodes falling back to CPU after successful registration are expected.

## Phase 9: Proof gate

Do not design protocols until all of these pass:

- Float32 CPU generation.
- Float32 CoreML CPU-and-GPU-eligible generation.
- Evidence that CoreML accepts at least part of the graph.
- Sequential and concurrent generation on both providers.
- Stop and cancellation.
- Custom logits processing.
- `EdgeToolsSession` integration.
- INT4 and INT8 CPU generation.
- Focused error handling.
- macOS execution.
- iOS device and simulator linking.

## Phase 10: Test-drive the abstraction

Only after the proof gate, add `Generate Using Custom ONNX Backend`. It should supply a backend without importing ONNX Runtime and initially fail because no abstraction exists.

Extract the smallest protocol demonstrated by the concrete implementation. The expected shape is:

```swift
public protocol NeedleONNXBackend: Sendable {
  var configuration: NeedleModelConfiguration { get }

  func prefill(
    tokenIDs: [EdgeToolsToken.ID]
  ) async throws -> any NeedleONNXGeneration
}

public protocol NeedleONNXGeneration: Sendable {
  func decode(
    tokenID: EdgeToolsToken.ID,
    position: Int
  ) async throws -> [Float]
}
```

Execution-provider selection remains part of the concrete ONNX Runtime backend, not the abstract protocol.

After extraction:

- `NeedleONNXEngine` moves under `ONNXCore`.
- Concrete ONNX Runtime support remains under `ONNX`.
- The complete real CPU/CoreML suite passes unchanged.
- All Needle-specific ONNX implementation remains in `NeedleONNXEngine.swift`.

## Non-Apple static artifact bundles

Build tooling lives at:

```text
bin/build_onnxruntime_artifactbundles.py
```

The script pins ONNX Runtime 1.27.0 at commit
`8f0278c77bf44b0cc83c098c6c722b92a36ac4b5`. It builds one host-compatible
variant at a time, consolidates ONNX Runtime's component archives, and explicitly
builds static dependencies such as RE2 that the default build omits. It generates
and verifies deterministic artifact manifests and ZIPs.

The native WebGPU bundle uses Dawn's Vulkan backend without WebAssembly or NNAPI
support. Its matrix is:

- Linux x86_64 and ARM64.
- Android ARM64 and x86_64, built for API 28 and declared compatible through API 36.

Build Linux variants in architecture-matched Docker containers and Android variants with the
local NDK, then assemble them with:

```console
python3 \
  bin/build_onnxruntime_artifactbundles.py \
  container-build \
  --variant linux-x86_64
python3 \
  bin/build_onnxruntime_artifactbundles.py \
  container-build \
  --variant linux-aarch64
python3 \
  bin/build_onnxruntime_artifactbundles.py \
  build \
  --variant android-arm64 \
  --android-sdk "$ANDROID_HOME" \
  --android-ndk "$ANDROID_NDK_HOME"
python3 \
  bin/build_onnxruntime_artifactbundles.py \
  build \
  --variant android-x86_64 \
  --android-sdk "$ANDROID_HOME" \
  --android-ndk "$ANDROID_NDK_HOME"
python3 bin/build_onnxruntime_artifactbundles.py assemble
```

The completed deterministic archive is checked in through Git LFS at:

```text
bin/onnxruntime-webgpu-1.27.0.artifactbundle.zip
```

Its SwiftPM checksum is
`536f66970b7c9d44125763e8f8eba8af3d625f3f48bead531e38792258593eaa`.
The bundle's four variants have been verified as static archives, and both Linux
variants have successfully linked into Swift executables.

## Deferred work

- Reduced-operator builds.
- WASI and other WebAssembly targets.
- Dynamic library loading.
- CI integration.
