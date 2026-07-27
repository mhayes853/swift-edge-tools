import CustomDump
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

    let parsedCall = try #require(call)
    expectNoDifference(parsedCall.name, "record_note")
    expectNoDifference(parsedCall.arguments, ["text": "literal </tool_call> text"])
  }

  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `Qwen3P5 MLX engine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await Qwen3P5MLXEngine(from: downloadQwen3P5())
        let transcript = try await completeWeatherTurn(using: engine)

        assertSnapshot(of: transcript, as: .dump)
      }
    }
  #endif
}
