import EdgeTools
import Testing

@Suite
struct `EdgeToolsSampling tests` {
  #if os(macOS) || os(linux) || os(windows)
    @Test
    func `Nonfinite Temperature Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        _ = EdgeToolsFusedSamplingParameters(temperature: .infinity)
      }
    }

    @Test
    func `Negative Top K Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        _ = EdgeToolsFusedSamplingParameters(topK: -1)
      }
    }

    @Test
    func `Zero Top P Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        _ = EdgeToolsFusedSamplingParameters(topP: 0)
      }
    }

    @Test
    func `Min P Above One Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        _ = EdgeToolsFusedSamplingParameters(minP: 2)
      }
    }

    @Test
    func `Nonfinite Repetition Penalty Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        _ = EdgeToolsFusedSamplingParameters(repetitionPenalty: .infinity)
      }
    }

    @Test
    func `Nonfinite Presence Penalty Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        _ = EdgeToolsFusedSamplingParameters(presencePenalty: .nan)
      }
    }

    @Test
    func `Zero Repetition Context Size Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        _ = EdgeToolsFusedSamplingParameters(repetitionContextSize: 0)
      }
    }

    @Test
    func `Mutating Temperature Below Zero Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        var parameters = EdgeToolsFusedSamplingParameters()
        parameters.temperature = -1
      }
    }

    @Test
    func `Mutating Top K Below Zero Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        var parameters = EdgeToolsFusedSamplingParameters()
        parameters.topK = -1
      }
    }

    @Test
    func `Mutating Top P Above One Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        var parameters = EdgeToolsFusedSamplingParameters()
        parameters.topP = 2
      }
    }

    @Test
    func `Mutating Min P Below Zero Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        var parameters = EdgeToolsFusedSamplingParameters()
        parameters.minP = -1
      }
    }

    @Test
    func `Mutating Repetition Penalty To Zero Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        var parameters = EdgeToolsFusedSamplingParameters()
        parameters.repetitionPenalty = 0
      }
    }

    @Test
    func `Mutating Presence Penalty To Nonfinite Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        var parameters = EdgeToolsFusedSamplingParameters()
        parameters.presencePenalty = .infinity
      }
    }

    @Test
    func `Mutating Repetition Context Size To Zero Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        var parameters = EdgeToolsFusedSamplingParameters()
        parameters.repetitionContextSize = 0
      }
    }
  #endif
}
