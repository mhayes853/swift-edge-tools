#if MLX && canImport(MLX)
  import Foundation
  import MLX
  import MLXNN
  import MLXLMCommon

  package func loadEdgeToolsLanguageModel<Model: EdgeToolsLanguageModel>(
    _ model: Model.Type,
    from directoryURL: URL,
    editModelConfiguration: (inout Model.ModelConfiguration) -> Void = { _ in }
  ) throws -> Model {
    guard
      var modelConfiguration = try decodeModelConfiguration(
        Model.ModelConfiguration.self,
        in: directoryURL
      )
    else {
      throw EdgeToolsMLXEngineError.failedToLoadConfiguration
    }
    editModelConfiguration(&modelConfiguration)

    return try populateEdgeToolsMLXWeights(
      in: Model(configuration: modelConfiguration),
      from: directoryURL
    )
  }

  private func populateEdgeToolsMLXWeights<LanguageModel: MLXLMCommon.LanguageModel>(
    in languageModel: LanguageModel,
    from directoryURL: URL
  ) throws -> LanguageModel {
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
