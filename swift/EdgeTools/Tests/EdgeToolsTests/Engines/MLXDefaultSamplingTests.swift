#if MLX && canImport(MLX) && !os(WASI)
  import CustomDump
  import EdgeTools
  import Foundation
  import Testing

  @Suite
  struct `MLXDefaultSampling tests` {
    @Test
    func `Reads Sampling Values From The Generation Configuration`() throws {
      let directory = try modelDirectory(
        generationConfiguration: """
          { "do_sample": true, "temperature": 0.9, "top_k": 50, "top_p": 0.95, \
          "repetition_penalty": 1.05 }
          """
      )

      let parameters = try directory.loadDefaultSampling()

      expectNoDifference(parameters?.temperature, 0.9)
      expectNoDifference(parameters?.topK, 50)
      expectNoDifference(parameters?.topP, 0.95)
      expectNoDifference(parameters?.repetitionPenalty, 1.05)
    }

    @Test
    func `Treats Disabled Sampling As Greedy`() throws {
      let directory = try modelDirectory(
        generationConfiguration: #"{ "do_sample": false, "temperature": 0.9, "top_p": 0.95 }"#
      )

      let parameters = try directory.loadDefaultSampling()

      expectNoDifference(parameters?.isGreedy, true)
      expectNoDifference(parameters?.topP, nil)
    }

    @Test
    func `Has No Values When The Configuration Omits Sampling`() throws {
      let directory = try modelDirectory(
        generationConfiguration: #"{ "bos_token_id": 1, "eos_token_id": 2 }"#
      )

      expectNoDifference(try directory.loadDefaultSampling(), nil)
    }

    @Test
    func `Has No Values When The Configuration Is Missing`() throws {
      let directory = try modelDirectory(generationConfiguration: nil)

      expectNoDifference(try directory.loadDefaultSampling(), nil)
    }
  }

  private func modelDirectory(generationConfiguration: String?) throws -> MLXModelDirectory {
    let url = URL.temporaryDirectory.appending(path: "edge-tools-sampling-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    if let generationConfiguration {
      try Data(generationConfiguration.utf8)
        .write(to: url.appending(path: "generation_config.json"))
    }
    return MLXModelDirectory(url: url)
  }
#endif
