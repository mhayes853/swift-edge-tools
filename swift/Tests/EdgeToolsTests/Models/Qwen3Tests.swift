import CustomDump
import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `Qwen3 tests` {
  @Suite
  struct `Qwen3ToolCallParser tests` {
    @Test
    func `Decodes Escaped Unicode Surrogate Pairs`() throws {
      var parser = Qwen3ToolCallParser()
      let source = #"<tool_call>{"name":"emoji","arguments":{"value":"\uD83D\uDE00"}}</tool_call>"#
      let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
      let call = try #require(parsed)

      expectNoDifference(call.arguments, ["value": "😀"])
    }
  }

  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `Qwen3 MLX engine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await Qwen3MLXEngine(from: downloadQwen3())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }
    }
  #endif
}
