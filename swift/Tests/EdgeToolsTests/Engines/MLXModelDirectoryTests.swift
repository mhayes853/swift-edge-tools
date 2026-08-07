#if MLX && canImport(MLX) && !os(WASI)
  import CustomDump
  import EdgeTools
  import Foundation
  import MLX
  import Testing

  @Suite
  struct `MLXModelDirectory tests` {
    @Test
    func `Generation Configuration Overrides Stop Token IDs`() throws {
      let fileManager = FileManager.default
      let directoryURL = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
      try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
      defer { try? fileManager.removeItem(at: directoryURL) }

      try Data(#"{"model_type":"test","eos_token_id":[1,2]}"#.utf8)
        .write(to: directoryURL.appending(path: "config.json"))
      try Data(#"{"eos_token_id":[3,4]}"#.utf8)
        .write(to: directoryURL.appending(path: "generation_config.json"))

      expectNoDifference(
        try MLXModelDirectory(url: directoryURL).loadStopTokenIds(),
        [3, 4]
      )
    }

    @Test
    func `Loads Metadata From Every Safetensor`() throws {
      let fileManager = FileManager.default
      let directoryURL = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
      try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
      defer { try? fileManager.removeItem(at: directoryURL) }

      let firstURL = directoryURL.appending(path: "model-00001.safetensors")
      let secondURL = directoryURL.appending(path: "model-00002.safetensors")
      try MLX.save(
        arrays: ["first": MLXArray([1])],
        metadata: ["first": "1", "shared": "first"],
        url: firstURL
      )
      try MLX.save(
        arrays: ["second": MLXArray([2])],
        metadata: ["second": "2", "shared": "second"],
        url: secondURL
      )

      let safetensors = try MLXModelDirectory(url: directoryURL).loadSafetensors()
      let metadataByFilename = Dictionary(
        uniqueKeysWithValues: safetensors.metadataByFile.map {
          ($0.key.lastPathComponent, $0.value)
        }
      )

      expectNoDifference(
        metadataByFilename,
        [
          "model-00001.safetensors": ["first": "1", "shared": "first"],
          "model-00002.safetensors": ["second": "2", "shared": "second"]
        ]
      )
      expectNoDifference(
        safetensors.mergedMetadata,
        ["first": "1", "second": "2", "shared": "second"]
      )
    }
  }
#endif
