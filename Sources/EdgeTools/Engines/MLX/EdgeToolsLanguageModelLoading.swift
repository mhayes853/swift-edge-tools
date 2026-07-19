#if MLX && canImport(MLX)
  #if canImport(FoundationEssentials)
    import FoundationEssentials
  #else
    import Foundation
  #endif
  import MLXLMCommon

  package func loadEdgeToolsLanguageModel<Model: EdgeToolsLanguageModel>(
    _: Model.Type,
    from directoryURL: URL,
    model: (Model.ModelConfiguration) throws -> Model
  ) throws -> Model {
    let configuration = try decodeModelConfiguration(
      Model.ModelConfiguration.self,
      in: directoryURL
    )
    let baseConfiguration = try decodeModelConfiguration(
      BaseConfiguration.self,
      in: directoryURL
    )
    guard let configuration, let baseConfiguration else {
      throw EdgeToolsMLXEngineError.failedToLoadConfiguration
    }

    let model = try model(configuration)
    try loadWeights(
      modelDirectory: directoryURL,
      model: model,
      perLayerQuantization: baseConfiguration.perLayerQuantization
    )
    return model
  }
#endif
