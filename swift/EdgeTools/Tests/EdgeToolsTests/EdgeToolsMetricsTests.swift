import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsMetrics tests` {
  @Test
  func `Decode Tokens Per Second Derives From Tokens And Duration`() {
    var metrics = EdgeToolsMetrics()
    metrics.decodeTokens = 10
    metrics.decodeDuration = .milliseconds(2_500)

    expectNoDifference(metrics.decodeTokensPerSecond, 4)
  }

  @Test
  func `Prefill Tokens Per Second Derives From Tokens And Duration`() {
    var metrics = EdgeToolsMetrics()
    metrics.prefillTokens = 100
    metrics.prefillDuration = .seconds(4)

    expectNoDifference(metrics.prefillTokensPerSecond, 25)
  }

  @Test
  func `Prefill Tokens Per Second Prefers A Directly Reported Rate`() {
    var metrics = EdgeToolsMetrics()
    metrics.prefillTokensPerSecond = 50

    expectNoDifference(metrics.prefillTokensPerSecond, 50)
  }
}
