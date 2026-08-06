import EdgeTools
import Foundation

// MARK: - Needle CoreAI Loading

/// CoreAI support is experimental, needs Swift 6.4 to build, and needs OS 27 to run, so the runner
/// only exists when all three hold.
func makeNeedleCoreAIRunner(from directory: URL) async throws -> any EdgeRunner {
  #if swift(>=6.4) && canImport(CoreAI)
    guard #available(macOS 27.0, *) else {
      throw EdgeCLIError("The coreai engine needs macOS 27 or newer.")
    }
    let engine = try await NeedleCoreAIModelEngine(modelDirectoryURL: directory)
    return EngineRunner(
      engine: engine,
      clearCaches: { await engine.clearCaches() },
      supportsCustomGrammar: false,
      supportsSampling: false,
      makePrompt: { NeedlePrompt(system: $0.system, user: $0.user) },
      makeParameters: { request in
        NeedleCoreAIModel.GenerateParameters(
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
        )
      }
    )
  #else
    throw EdgeCLIError("This build of edge has no coreai engine; it requires Swift 6.4 or newer.")
  #endif
}
