#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLMCommon

  // MARK: - MLXModelDirectory

  public struct MLXModelDirectory: Sendable {
    public let url: URL

    public init(url: URL) {
      self.url = url
    }

    public func loadConfigurationData() throws -> Data {
      let configurationURLs = [
        self.url.appending(path: "configuration.json"),
        self.url.appending(path: "config.json")
      ]
      guard
        let configurationURL = configurationURLs.first(where: {
          FileManager.default.fileExists(atPath: $0.path())
        })
      else {
        throw EdgeToolsError.failedToLoadConfiguration
      }
      return try Data(contentsOf: configurationURL)
    }

    public func loadConfiguration<Configuration: Decodable>(
      _ type: Configuration.Type,
      decoder: JSONDecoder = .json5()
    ) throws -> Configuration {
      try decoder.decode(type, from: self.loadConfigurationData())
    }

    public func loadProcessorConfiguration<Configuration: Decodable>(
      _ type: Configuration.Type,
      decoder: JSONDecoder = .json5()
    ) throws -> Configuration {
      try decoder.decode(type, from: self.loadProcessorConfigurationData())
    }

    public func loadProcessorConfigurationData() throws -> Data {
      let preprocessorURL = self.url.appending(path: "preprocessor_config.json")
      let processorURL = self.url.appending(path: "processor_config.json")
      let url =
        FileManager.default.fileExists(atPath: preprocessorURL.path())
        ? preprocessorURL
        : processorURL
      guard FileManager.default.fileExists(atPath: url.path()) else {
        throw EdgeToolsError.failedToLoadConfiguration
      }
      return try Data(contentsOf: url)
    }

    public func loadGenerationConfiguration(
      decoder: JSONDecoder = .json5()
    ) throws -> GenerationConfigFile? {
      let configurationURL = self.url.appending(path: "generation_config.json")
      guard FileManager.default.fileExists(atPath: configurationURL.path()) else { return nil }
      return try decoder.decode(
        GenerationConfigFile.self,
        from: Data(contentsOf: configurationURL)
      )
    }

    public func loadStopTokenIds() throws -> Set<EdgeToolsToken.ID> {
      let baseConfiguration = try self.loadConfiguration(BaseConfiguration.self)
      var tokenIds = Set(baseConfiguration.eosTokenIds?.values ?? [])
      if let generationConfiguration = try self.loadGenerationConfiguration(),
        let values = generationConfiguration.eosTokenIds?.values
      {
        tokenIds = Set(values)
      }
      return tokenIds
    }

    public func safetensorURLs() throws -> [URL] {
      guard
        let enumerator = FileManager.default.enumerator(
          at: self.url,
          includingPropertiesForKeys: nil
        )
      else {
        throw EdgeToolsError.missingModelWeights
      }
      var urls = [URL]()
      for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        urls.append(url)
      }
      return urls.sorted { $0.path() < $1.path() }
    }

    public func loadSafetensors() throws -> MLXSafetensors {
      let urls = try self.safetensorURLs()
      guard !urls.isEmpty else { throw EdgeToolsError.missingModelWeights }

      var weights = [String: MLXArray]()
      var metadataByFile = [URL: [String: String]]()
      for url in urls {
        let (arrays, fileMetadata) = try MLX.loadArraysAndMetadata(url: url)
        weights.merge(arrays) { _, new in new }
        metadataByFile[url] = fileMetadata
      }
      return MLXSafetensors(weights: weights, metadataByFile: metadataByFile)
    }

    public func loadTokenizer() async throws -> sending any EdgeToolsTokenizer {
      try await EdgeToolsAutoTokenizer.from(modelDirectory: self.url)
    }
  }

  // MARK: - MLXSafetensors

  public struct MLXSafetensors {
    public var weights: [String: MLXArray]
    public var metadataByFile: [URL: [String: String]]

    public init(
      weights: [String: MLXArray],
      metadataByFile: [URL: [String: String]]
    ) {
      self.weights = weights
      self.metadataByFile = metadataByFile
    }

    public var mergedMetadata: [String: String] {
      var result = [String: String]()
      for url in self.metadataByFile.keys.sorted(by: { $0.path() < $1.path() }) {
        result.merge(self.metadataByFile[url] ?? [:]) { _, new in new }
      }
      return result
    }
  }
#endif
