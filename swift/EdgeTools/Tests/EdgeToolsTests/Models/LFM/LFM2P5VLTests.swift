import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `LFM2P5VL tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `LFM2P5VLMLXModelEngine tests` {
      @Test
      func `Completes Text Tool Turn Snapshot`() async throws {
        let engine = try await LFM2P5VLMLXModelEngine(from: downloadLFM2P5VL())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Describes Image Snapshot`() async throws {
        let engine = try await LFM2P5VLMLXModelEngine(from: downloadLFM2P5VL())
        let response = try await describeRedImage(using: engine)

        withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
      }

      @Test
      func `Completes Image Conditioned Tool Turn Snapshot`() async throws {
        let engine = try await LFM2P5VLMLXModelEngine(from: downloadLFM2P5VL())
        let result = try await completeImageColorTurn(using: engine)

        withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
      }
    }
  #endif

  #if HuggingFaceTokenizers && Llama && XGrammar && canImport(CLlama) && !os(WASI)
    @Suite(.serialized)
    struct `LFM2P5VLLlamaModelEngine tests` {
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

      private func multimodalEngine() async throws -> LFM2P5VLLlamaModelEngine {
        let model = try await downloadGGUFMultimodalModel(id: .lfm2P5VL)
        return try LFM2P5VLLlamaModelEngine(
          modelPath: model.model.path(),
          multimodalProjectorPath: model.projector.path(),
          contextParameters: LlamaContextParameters(cacheForking: .isolated),
          multimodalParameters: LlamaMultimodalParameters(warmUp: false)
        )
      }
    }
  #endif
}
