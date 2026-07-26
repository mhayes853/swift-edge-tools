#if ONNXCore && JS
  import JavaScriptKit

  public typealias NeedleJSONNXEngine =
    EdgeToolsONNXEngine<JSONNXRuntime, NeedleONNXModel<JSONNXRuntime>>

  extension EdgeToolsONNXEngine
  where Runtime == JSONNXRuntime, Model == NeedleONNXModel<JSONNXRuntime> {
    public init(
      onnxRuntime: JSObject,
      configuration: NeedleModelConfiguration,
      tokenizer: sending NeedleSPTokenizer,
      encoderModel: JSONNXRuntime.ModelSource,
      decoderModel: JSONNXRuntime.ModelSource,
      encoderConfiguration: JSONNXRuntime.Configuration = JSONNXRuntime.Configuration(),
      decoderConfiguration: JSONNXRuntime.Configuration = JSONNXRuntime.Configuration()
    ) async throws {
      let runtime = try JSONNXRuntime(onnxRuntime: onnxRuntime)
      let encoderSession = try await runtime.session(
        model: encoderModel,
        configuration: encoderConfiguration
      )
      let decoderSession = try await runtime.session(
        model: decoderModel,
        configuration: decoderConfiguration
      )
      let model = NeedleONNXModel<JSONNXRuntime>(
        configuration: configuration,
        encoderSession: encoderSession,
        decoderSession: decoderSession
      )
      let components = EdgeToolsONNXEngineComponents(runtime: runtime, model: model)
      try self.init(components: components, tokenizer: tokenizer)
    }
  }
#endif
