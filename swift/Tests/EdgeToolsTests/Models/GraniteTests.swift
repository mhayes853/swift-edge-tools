import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `Granite tests` {
  @Suite
  struct `GraniteToolCallParser tests` {
    @Test
    func `Decodes Escaped Unicode Surrogate Pairs`() throws {
      var parser = GraniteToolCallParser()
      let source = #"<tool_call>{"name":"emoji","arguments":{"value":"\uD83D\uDE00"}}</tool_call>"#
      let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
      let call = try #require(parsed.first)

      expectNoDifference(call.arguments, ["value": "😀"])
    }
  }

  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `GraniteMoeHybridMLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await GraniteMoeHybridMLXModelEngine(from: downloadGraniteMoeHybrid())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }
    }
  #endif
}
