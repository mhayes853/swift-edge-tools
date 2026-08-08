import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `Qwen3P5 tests` {
  @Test
  func `Qwen XML Preserves A Tool Call Closing Tag Inside A Parameter Value`() throws {
    var parser = QwenXMLToolCallParser()
    let source =
      "<tool_call><function=record_note><parameter=text>literal </tool_call> text</parameter></function></tool_call>"

    let call = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))

    let parsedCall = try #require(call.first)
    expectNoDifference(parsedCall.name, "record_note")
    expectNoDifference(parsedCall.arguments, ["text": "literal </tool_call> text"])
  }

  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `Qwen3P5MLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await Qwen3P5MLXModelEngine(from: downloadQwen3P5())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }
    }

    #if canImport(CoreImage) && canImport(MLXVLM)
      @Suite(.serialized, .enabledIfXcode())
      struct `Qwen3P5VLMLXModelEngine tests` {
        @Test
        func `Describes Video Snapshot`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let response = try await describeRedVideo(using: engine)

          withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
        }

        @Test
        func `Describes Image And Video Snapshot`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let response = try await describeRedImageAndVideo(using: engine)

          withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
        }

        @Test
        func `Completes Video Conditioned Tool Turn Snapshot`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let result = try await completeVideoColorTurn(using: engine)

          withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
        }
      }
    #endif
  #endif
}
