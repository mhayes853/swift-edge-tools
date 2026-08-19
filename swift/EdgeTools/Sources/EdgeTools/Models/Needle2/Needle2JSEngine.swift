#if Needle2 && JS && canImport(JavaScriptKit)
  import EdgeToolsCore
  import _Concurrency
  import JavaScriptEventLoop
  import JavaScriptKit

  // MARK: - Needle2JSProvider

  @nonexhaustive
  public enum Needle2JSProvider: String, Hashable, Sendable {
    case worker
    case direct
  }

  // MARK: - Needle2JSBinarySource

  @nonexhaustive
  public enum Needle2JSBinarySource: Hashable, Sendable {
    case location(String)
    case bytes([UInt8])
  }

  // MARK: - Needle2JSRuntimeConfiguration

  public struct Needle2JSRuntimeConfiguration: Hashable, Sendable {
    public var provider: Needle2JSProvider
    public var wasm: Needle2JSBinarySource?
    public var weights: Needle2JSBinarySource?

    public init(
      provider: Needle2JSProvider = .worker,
      wasm: Needle2JSBinarySource? = nil,
      weights: Needle2JSBinarySource? = nil
    ) {
      self.provider = provider
      self.wasm = wasm
      self.weights = weights
    }
  }

  public typealias Needle2JSRuntimeFactory = JSObject

  // MARK: - Needle2JSEngine

  public final class Needle2JSEngine: EdgeToolsEngine {
    public typealias Context = Needle2Context
    public typealias ContextParameters = Needle2ContextParameters
    public typealias Prompt = Needle2Prompt
    public typealias GenerateParameters = Needle2GenerateParameters

    private let runtime: JSRemote<JSObject>
    private let defaultContext: Context
    private let activeContextIdentifier = Lock<UInt64?>(nil)

    public init(
      createRuntime: sending Needle2JSRuntimeFactory,
      configuration: Needle2JSRuntimeConfiguration = Needle2JSRuntimeConfiguration()
    ) async throws {
      let factory = JSRemote(createRuntime)
      self.runtime = try await factory.promiseValue(
        transform: Self.remoteRuntime,
        failure: Self.error
      ) { factory in
        factory.promisingCall(
          arguments: [Self.configurationObject(configuration).jsValue]
        )
      }
      self.defaultContext = Context(parameters: ContextParameters())
    }

    public init(runtime: sending JSObject) throws {
      self.runtime = JSRemote(try Needle2JSRuntimeObject(runtime).object)
      self.defaultContext = Context(parameters: ContextParameters())
    }

    deinit {
      let runtime = self.runtime
      Task { @concurrent in
        await runtime.withJSObject { runtime in
          let receiver = runtime["receiver"].object!
          let dispose = runtime["dispose"].object!
          JSPromise(unsafelyWrapping: dispose.promisingCall(this: receiver).object!)
            .catch { _ in .undefined }
        }
      }
    }

    public func context() -> Context {
      self.defaultContext
    }

    public func context(_ parameters: ContextParameters) -> Context {
      Context(parameters: parameters)
    }

    public func generate(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      parameters: sending GenerateParameters,
      context: Context,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      let maximumTokenCount = parameters.maxTokens ?? 256
      guard maximumTokenCount > 0 else {
        throw Needle2Error(
          code: .invalidGenerateParameters,
          message: "Needle 2 maxTokens must be greater than zero."
        )
      }
      guard parameters.outputCapacity > 0 else {
        throw Needle2Error(
          code: .invalidGenerateParameters,
          message: "Needle 2 outputCapacity must be greater than zero."
        )
      }

      return AnyGenerationTask { stopper in
        let contextParameters = try context.begin()
        defer { context.finish() }
        try self.claim(context)
        let request = Needle2JSRequest(
          prompt: prompt.needle2Input,
          systemPrompt: contextParameters.system.formatted(),
          tools: tools.needle2Value,
          toolIndexPath: contextParameters.toolIndexPath,
          maximumTokenCount: maximumTokenCount,
          outputCapacity: parameters.outputCapacity
        )
        let response = try await self.runtime.promiseValue(
          transform: Needle2Response.init(javaScriptValue:),
          failure: Self.error
        ) { runtime in
          Self.invoke(
            runtime: runtime,
            function: "generate",
            arguments: [Self.requestObject(request).jsValue]
          )
        }
        if !response.success {
          throw Needle2Error(
            code: response.errorCode.map(Needle2Error.Code.init(rawValue:)) ?? .generationFailed,
            message: response.error ?? "Needle 2 generation failed."
          )
        }

        let stopped = stopper.isStopped
        if !stopped {
          for part in response.parts {
            channel.emit(part: part)
          }
        }
        return response.generation(
          tokenCount: response.tokenCount ?? 0,
          wasStopped: stopped
        )
      }
    }

    public func load(_ source: Needle2JSBinarySource) async throws {
      guard self.activeContextIdentifier.withBorrowedLock({ $0 == nil }) else {
        throw Needle2Error(
          code: .activeContext,
          message: "Reset the active Needle 2 context before loading new weights."
        )
      }
      try await self.runtime.promiseValue(failure: Self.error) { runtime in
        Self.invoke(
          runtime: runtime,
          function: "load",
          arguments: [Self.jsValue(source)]
        )
      }
    }

    public func reset(_ context: Context) async throws {
      guard !context.isResponding else {
        throw EdgeToolsError.contextInUse
      }
      try self.activeContextIdentifier.withLock { activeContextIdentifier in
        guard let activeContextIdentifier else {
          return
        }
        guard activeContextIdentifier == context.identifier else {
          throw Needle2Error(
            code: .activeContext,
            message: "Another Needle 2 context has an active conversation."
          )
        }
      }
      try await self.runtime.promiseValue(failure: Self.error) { runtime in
        Self.invoke(runtime: runtime, function: "reset", arguments: [])
      }
      self.activeContextIdentifier.withLock { $0 = nil }
    }
  }

  extension Needle2JSEngine: Needle2SessionEngine {}

  // MARK: - JavaScript Bridge

  extension Needle2JSEngine {
    private func claim(_ context: Context) throws {
      try self.activeContextIdentifier.withLock { activeIdentifier in
        guard let activeIdentifier else {
          activeIdentifier = context.identifier
          return
        }
        guard activeIdentifier == context.identifier else {
          throw Needle2Error(
            code: .activeContext,
            message: "Another Needle 2 context has an active conversation. Reset it before using this context."
          )
        }
      }
    }

    private static func remoteRuntime(
      _ value: JSValue
    ) throws -> JSRemote<JSObject> {
      JSRemote(try Needle2JSRuntimeObject(value).object)
    }

    private static func error(_ value: JSValue?) -> Needle2Error {
      guard let value else {
        return Needle2Error(
          code: .invalidJavaScriptRuntime,
          message: "The Needle 2 JavaScript operation did not return a Promise."
        )
      }
      let object = value.object
      return Needle2Error(
        code: object?["code"].string.map(Needle2Error.Code.init(rawValue:))
          ?? .generationFailed,
        message: object?["message"].string ?? value.string ?? "Needle 2 JavaScript failed."
      )
    }

    private static func configurationObject(_ configuration: Needle2JSRuntimeConfiguration)
      -> JSObject
    {
      let object = JSObject()
      object["provider"] = .string(configuration.provider.rawValue)
      if let wasm = configuration.wasm {
        object["wasm"] = self.jsValue(wasm)
      }
      if let weights = configuration.weights {
        object["weights"] = self.jsValue(weights)
      }
      return object
    }

    private static func requestObject(_ request: Needle2JSRequest) -> JSObject {
      let initialization = JSObject()
      initialization["systemPrompt"] = .string(request.systemPrompt)
      initialization["tools"] = request.tools.jsValue
      if let toolIndexPath = request.toolIndexPath {
        initialization["toolIndexPath"] = .string(toolIndexPath)
      }

      let object = JSObject()
      object["prompt"] = .string(request.prompt)
      object["initialization"] = initialization.jsValue
      object["maxTokens"] = .number(Double(request.maximumTokenCount))
      object["outputCapacity"] = .number(Double(request.outputCapacity))
      return object
    }

    private static func jsValue(_ source: Needle2JSBinarySource) -> JSValue {
      switch source {
      case .location(let location): .string(location)
      case .bytes(let bytes): JSUint8Array(bytes).jsObject.jsValue
      }
    }

    private static func invoke(
      runtime: JSObject,
      function name: String,
      arguments: [JSValue]
    ) -> JSValue {
      runtime[name].object!
        .promisingCall(
          this: runtime["receiver"].object!,
          arguments: arguments
        )
    }
  }

  // MARK: - Request

  private struct Needle2JSRequest: Sendable {
    var prompt: String
    var systemPrompt: String
    var tools: EdgeToolsValue
    var toolIndexPath: String?
    var maximumTokenCount: Int
    var outputCapacity: Int
  }

  // MARK: - Runtime Object

  private struct Needle2JSRuntimeObject {
    var object: JSObject

    init(_ value: JSValue) throws {
      guard let object = value.object else {
        throw Needle2Error(
          code: .invalidJavaScriptRuntime,
          message: "The Needle 2 factory did not resolve to an object."
        )
      }
      try self.init(object)
    }

    init(_ runtime: JSObject) throws {
      let object = JSObject()
      object["receiver"] = runtime.jsValue
      for name in ["generate", "load", "reset", "dispose"] {
        guard let function = runtime[name].object else {
          throw Needle2Error(
            code: .invalidJavaScriptRuntime,
            message:
              "The Needle 2 JavaScript runtime must expose generate, load, and dispose functions."
          )
        }
        object[name] = function.jsValue
      }
      self.object = object
    }
  }

  // MARK: - Response

  extension Needle2Response {
    fileprivate init(javaScriptValue value: JSValue) throws {
      guard let object = value.object,
        let type = object["type"].string,
        let success = object["success"].boolean,
        let tokenCount = object["tokenCount"].number.flatMap(Int.init(exactly:)),
        let functionCalls = EdgeToolsValue.construct(from: object["functionCalls"])
      else {
        throw Needle2Error(
          code: .invalidResponse,
          message: "Needle 2 JavaScript returned an invalid generation result."
        )
      }

      let metrics = object["metrics"].object
      self.type = type
      self.success = success
      self.error = object["error"].string
      self.errorCode = object["errorCode"].string
      self.reasoning = object["reasoning"].string
      self.confidence = object["confidence"].number.map(Float.init)
      self.prefillTokensPerSecond = metrics?["prefillTokensPerSecond"].number
      self.decodeTokensPerSecond = metrics?["decodeTokensPerSecond"].number
      self.peakRAMMegabytes = metrics?["peakRAMMegabytes"].number
      self.tokenCount = tokenCount
      self.functionCalls = try functionCalls.needle2ToolCalls
    }

    fileprivate func generation(
      tokenCount: Int,
      wasStopped: Bool
    ) -> EdgeToolsEngineGeneration {
      var metadata = EdgeToolsMetadata()
      metadata.generationConfidence = self.confidence
      metadata.needle2ResponseType = self.type
      metadata.needle2PrefillTokensPerSecond = self.prefillTokensPerSecond
      metadata.needle2DecodeTokensPerSecond = self.decodeTokensPerSecond
      metadata.needle2PeakRAMMegabytes = self.peakRAMMegabytes
      return EdgeToolsEngineGeneration(
        prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: tokenCount,
          duration: Duration(
            needle2TokenCount: tokenCount,
            tokensPerSecond: self.decodeTokensPerSecond
          ),
          durationToFirstToken: .zero
        ),
        wasStopped: wasStopped,
        tokens: [],
        response: "",
        parts: self.parts,
        metadata: metadata
      )
    }
  }
#endif
