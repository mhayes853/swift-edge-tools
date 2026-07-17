#if MLX && canImport(MLX)
  import Foundation
  import MLX
  import MLXNN
  import MLXLMCommon

  package func loadEdgeToolsMLXLanguageModel<
    Configuration: EdgeToolsMLXModelConfiguration
  >(
    _ configuration: Configuration.Type,
    from directoryURL: URL,
    editModelConfiguration: (inout Configuration.ModelConfiguration) -> Void = { _ in }
  ) throws -> sending Configuration.LanguageModel {
    guard
      var modelConfiguration = try decodeModelConfiguration(
        Configuration.ModelConfiguration.self,
        in: directoryURL
      )
    else {
      throw EdgeToolsMLXEngineError.failedToLoadConfiguration
    }
    editModelConfiguration(&modelConfiguration)

    let languageModel = configuration.languageModel(configuration: modelConfiguration)
    return try populateEdgeToolsMLXWeights(
      in: languageModel,
      from: directoryURL
    )
  }

  private func populateEdgeToolsMLXWeights<LanguageModel: MLXLMCommon.LanguageModel>(
    in languageModel: sending LanguageModel,
    from directoryURL: URL
  ) throws -> sending LanguageModel {
    let fileManager = FileManager.default
    guard let enumerator = fileManager.enumerator(
      at: directoryURL,
      includingPropertiesForKeys: nil
    ) else {
      throw EdgeToolsMLXEngineError.failedToLoadWeights
    }

    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for case let url as URL in enumerator where url.pathExtension == "safetensors" {
      let (arrays, fileMetadata) = try loadArraysAndMetadata(url: url)
      weights.merge(arrays) { _, replacement in replacement }
      if metadata.isEmpty {
        metadata = fileMetadata
      }
    }
    guard !weights.isEmpty else { throw EdgeToolsMLXEngineError.failedToLoadWeights }

    let sanitizedWeights = languageModel.sanitize(weights: weights, metadata: metadata)
    try languageModel.update(
      parameters: ModuleParameters.unflattened(sanitizedWeights),
      verify: .all
    )
    eval(languageModel)
    return languageModel
  }

#endif
