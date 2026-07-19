import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsMetrics tests` {
  @Test
  func `Decode Metrics Calculates Tokens Per Second`() {
    let metrics = EdgeToolsDecodeMetrics(
      tokens: 10,
      duration: .milliseconds(2_500),
      durationToFirstToken: .milliseconds(250)
    )

    expectNoDifference(metrics.tokensPerSecond, 4)
  }

  @Test
  func `Prefill Metrics Calculates Tokens Per Second`() {
    let metrics = EdgeToolsPrefillMetrics(tokens: 100, duration: .seconds(4))

    expectNoDifference(metrics.tokensPerSecond, 25)
  }
}
