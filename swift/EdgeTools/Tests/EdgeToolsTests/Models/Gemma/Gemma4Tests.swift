import EdgeTools
import Testing

#if !os(WASI)
  import Foundation
  import SnapshotTesting
#endif

@Suite
struct `Gemma4 tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `Gemma4MLXModelEngine tests` {
      @Test
      func `Completes Text Tool Turn Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Generates Reasoning Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }

      @Test
      func `Describes Image Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let response = try await describeRedImage(using: engine)

        withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
      }

      @Test
      func `Completes Image Conditioned Tool Turn Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let result = try await completeImageColorTurn(using: engine)

        withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
      }
    }
  #endif

  #if HuggingFaceTokenizers && Llama && XGrammar && canImport(CLlama) && !os(WASI)
    @Suite(.serialized)
    struct `Gemma4LlamaModelEngine tests` {
      @Test
      func `Llama Completes Tool Turn Snapshot`() async throws {
        let engine = try Gemma4LlamaModelEngine(
          modelPath: (try await downloadGGUFModel(id: .gemma4E2BHybrid)).path()
        )
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Llama Generates Reasoning Snapshot`() async throws {
        let engine = try Gemma4LlamaModelEngine(
          modelPath: (try await downloadGGUFModel(id: .gemma4E2BHybrid)).path()
        )
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }

      @Test
      func `Llama Describes Image Snapshot`() async throws {
        let engine = try await self.multimodalEngine()
        let response = try await describeRedImage(using: engine)

        withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
      }

      @Test
      func `Llama Completes Image Conditioned Tool Turn Snapshot`() async throws {
        let engine = try await self.multimodalEngine()
        let result = try await completeImageColorTurn(using: engine)

        withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
      }

      @Test
      func `Llama Completes Audio Conditioned Tool Turn Snapshot`() async throws {
        let engine = try await self.multimodalEngine()
        let result = try await completeAudioToneTurn(using: engine)

        withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
      }

      @Test
      func `Llama Reuses Image Conditioned Prefix`() async throws {
        let engine = try await self.multimodalEngine()
        let context = engine.context(
          EdgeToolsTranscriptContextParameters(
            transcript: EdgeToolsTranscript(messages: [
              .user("What is the dominant color?", images: [try llamaRedImageAsset()])
            ])
          )
        )
        _ = try await engine.prefill(context: context)
        context.transcript.messages.append(.assistant([.text("The dominant color is red.")]))
        context.transcript.messages.append(.user("Confirm that briefly."))

        let cached = try await engine.prefill(context: context)

        let params = EdgeToolsTranscriptContextParameters(transcript: context.transcript)
        let fresh = try await engine.prefill(context: engine.context(params))

        expectNoDifference(
          try #require(cached.metrics.prefillTokens) < #require(fresh.metrics.prefillTokens),
          true
        )
      }

      @Test
      func `Llama Reuses Prepared Media For Text Only Continuations`() async throws {
        let source = try llamaRedImageAsset()
        guard case .bytes(let bytes) = source.content else {
          Issue.record("The image fixture did not contain bytes.")
          return
        }
        let directory = FileManager.default.temporaryDirectory
          .appending(path: "edge-tools-llama-media-\(UUID().uuidString)")
        let validImageURL = directory.appending(path: "valid.png")
        let imageURL = directory.appending(path: "image.png")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(bytes).write(to: validImageURL)
        try FileManager.default.createSymbolicLink(
          at: imageURL,
          withDestinationURL: validImageURL
        )

        let engine = try await self.multimodalEngine()
        let context = engine.context(
          EdgeToolsTranscriptContextParameters(
            transcript: EdgeToolsTranscript(messages: [
              .user(
                "What is the dominant color?",
                images: [EdgeToolsTranscript.Asset(path: imageURL.path())]
              )
            ])
          )
        )
        _ = try await engine.prefill(context: context)
        try FileManager.default.removeItem(at: validImageURL)
        context.transcript.messages.append(.assistant([.text("The dominant color is red.")]))
        context.transcript.messages.append(.user("Confirm that briefly."))

        _ = try await engine.prefill(context: context)
      }

      @Test
      func `Llama Reuses Audio Conditioned Prefix`() async throws {
        let engine = try await self.multimodalEngine()
        let context = engine.context(
          EdgeToolsTranscriptContextParameters(
            transcript: EdgeToolsTranscript(messages: [
              .user("Does this contain a tone?", audio: [llamaToneAudioAsset()])
            ])
          )
        )
        _ = try await engine.prefill(context: context)
        context.transcript.messages.append(.assistant([.text("The audio contains a tone.")]))
        context.transcript.messages.append(.user("Confirm that briefly."))

        let cached = try await engine.prefill(context: context)

        let params = EdgeToolsTranscriptContextParameters(transcript: context.transcript)
        let fresh = try await engine.prefill(context: engine.context(params))

        expectNoDifference(
          try #require(cached.metrics.prefillTokens) < #require(fresh.metrics.prefillTokens),
          true
        )
      }

      private func multimodalEngine() async throws -> Gemma4LlamaModelEngine {
        let model = try await downloadGGUFMultimodalModel(id: .gemma4E2BHybrid)
        return try Gemma4LlamaModelEngine(
          modelPath: model.model.path(),
          multimodalProjectorPath: model.projector.path(),
          contextParameters: LlamaContextParameters(maximumSequenceCount: 1),
          multimodalParameters: LlamaMultimodalParameters(warmUp: false)
        )
      }
    }
  #endif
}
