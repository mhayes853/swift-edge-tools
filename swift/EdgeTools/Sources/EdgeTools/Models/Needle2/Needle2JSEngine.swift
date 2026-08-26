#if Needle2 && JS && canImport(JavaScriptKit)
  import EdgeToolsCore
  import JavaScriptEventLoop
  import JavaScriptKit

  // MARK: - Needle2JSProvider

  public enum Needle2JSProvider: String, Hashable, Sendable {
    case worker
    case direct
  }

  public enum Needle2JSRuntimeEngine: String, Hashable, Sendable {
    case wasm
    case native
    case automatic = "auto"
  }

  // MARK: - Needle2JSBinarySource

  public enum Needle2JSBinarySource: Hashable, Sendable {
    case location(String)
    case bytes([UInt8])
  }

  // MARK: - Needle2JSRuntimeConfiguration

  public struct Needle2JSRuntimeConfiguration: Hashable, Sendable {
    public var provider: Needle2JSProvider
    public var engine: Needle2JSRuntimeEngine
    public var wasm: Needle2JSBinarySource?
    public var weights: Needle2JSBinarySource?

    public init(
      provider: Needle2JSProvider = .worker,
      engine: Needle2JSRuntimeEngine = .wasm,
      wasm: Needle2JSBinarySource? = nil,
      weights: Needle2JSBinarySource? = nil
    ) {
      self.provider = provider
      self.engine = engine
      self.wasm = wasm
      self.weights = weights
    }
  }

  // MARK: - Needle2JSEngine

  public final class Needle2JSEngine: EdgeToolsEngine {
    public typealias Context = Needle2Context
    public typealias ContextParameters = Needle2ContextParameters
    public typealias Prompt = Needle2Prompt
    public typealias GenerateParameters = Needle2GenerateParameters

    private struct State {
      var configuration: Needle2JSRuntimeConfiguration
      var activeContextIdentifier: UInt64?
      var contextParameters: Needle2ContextParameters?
      var runtime: JSRemote<JSObject>?
    }

    private let createRuntime: JSRemote<JSObject>
    private let defaultContext: Context
    private let state: Lock<State>

    public init(
      createRuntime: sending JSObject,
      configuration: Needle2JSRuntimeConfiguration = Needle2JSRuntimeConfiguration()
    ) async throws {
      self.createRuntime = JSRemote(createRuntime)
      self.defaultContext = Context(parameters: ContextParameters(), tools: [])
      self.state = Lock(State(configuration: configuration))
    }

    deinit {
      if let runtime = self.state.withBorrowedLock({ $0.runtime }) {
        Task {
          try? await Self.dispose(runtime)
        }
      }
    }

    public func context() -> Context {
      self.defaultContext
    }

    public func context(tools: [any EdgeTool]) -> Context {
      Context(parameters: ContextParameters(), tools: tools)
    }

    public func context(
      _ parameters: ContextParameters,
      tools: [any EdgeTool]
    ) -> Context {
      Context(parameters: parameters, tools: tools)
    }

    public func generate(
      prompt: Prompt,
      parameters: sending GenerateParameters,
      context: Context,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      let maximumTokenCount = parameters.maxTokens ?? 256
      return AnyGenerationTask { stopper in
        let contextParameters = try context.begin()
        let tools = context.tools.map { $0.definition }
        defer { context.finish() }
        let runtime = try await self.runtime(
          contextIdentifier: context.identifier,
          parameters: contextParameters,
          tools: tools
        )
        let response = try await runtime.promiseValue(
          transform: Needle2Response.init(javaScriptValue:),
          failure: Self.error
        ) { runtime in
          let request = JSObject()
          request["prompt"] = .string(prompt.needle2Input)
          request["maxTokens"] = .number(Double(maximumTokenCount))
          return runtime["generate"].object!
            .promisingCall(
              this: runtime,
              arguments: [request.jsValue]
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
      try self.state.withLock { state in
        guard state.activeContextIdentifier == nil else {
          throw Needle2Error(
            code: .activeContext,
            message: "Reset the active Needle 2 context before loading new weights."
          )
        }
        state.configuration.weights = source
      }
    }

    public func reset(_ context: Context) async throws {
      guard !context.isResponding else {
        throw EdgeToolsError.contextInUse
      }
      let runtime: JSRemote<JSObject>? = try self.state.withBorrowedLock { state in
        guard let activeContextIdentifier = state.activeContextIdentifier else {
          return nil
        }
        guard activeContextIdentifier == context.identifier else {
          throw Needle2Error(
            code: .activeContext,
            message: "Another Needle 2 context has an active conversation."
          )
        }
        return state.runtime
      }
      if let runtime {
        try await Self.dispose(runtime)
      }
      self.state.withLock { state in
        state.activeContextIdentifier = nil
        state.contextParameters = nil
        state.runtime = nil
      }
    }
  }

  extension Needle2JSEngine: Needle2SessionEngine {}

  // MARK: - JavaScript Bridge

  extension Needle2JSEngine {
    private func runtime(
      contextIdentifier: UInt64,
      parameters: Needle2ContextParameters,
      tools: [EdgeToolDefinition]
    ) async throws -> JSRemote<JSObject> {
      if let runtime = try self.state.withLock({ state -> JSRemote<JSObject>? in
        guard state.activeContextIdentifier.map({ $0 == contextIdentifier }) ?? true else {
          throw Needle2Error(
            code: .activeContext,
            message:
              "Another Needle 2 context has an active conversation. Reset it before using this context."
          )
        }
        guard state.contextParameters.map({ $0 == parameters }) ?? true else {
          throw Needle2Error(
            code: .initializationChanged,
            message: "Reset the Needle 2 context before changing its system values or tool index."
          )
        }
        state.activeContextIdentifier = contextIdentifier
        state.contextParameters = parameters
        return state.runtime
      }) {
        return runtime
      }

      do {
        let configuration = self.state.withBorrowedLock { $0.configuration }
        let runtime = try await self.createRuntime.promiseValue(
          transform: Self.remoteRuntime,
          failure: Self.error
        ) { createRuntime in
          createRuntime.promisingCall(
            arguments: [
              Self.configurationObject(
                configuration,
                parameters: parameters,
                tools: tools
              )
              .jsValue
            ]
          )
        }
        self.state.withLock { $0.runtime = runtime }
        return runtime
      } catch {
        self.state.withLock { state in
          state.activeContextIdentifier = nil
          state.contextParameters = nil
        }
        throw error
      }
    }

    private static func remoteRuntime(
      _ value: JSValue
    ) throws -> JSRemote<JSObject> {
      guard let runtime = value.object else {
        throw Needle2Error(
          code: .invalidJavaScriptRuntime,
          message: "The Needle 2 factory did not resolve to an object."
        )
      }
      return JSRemote(runtime)
    }

    private static func error(_ value: JSValue?) -> Needle2Error {
      let object = value?.object
      return Needle2Error(
        code: object?["code"].string.map(Needle2Error.Code.init(rawValue:))
          ?? .generationFailed,
        message: object?["message"].string ?? value?.string ?? "Needle 2 JavaScript failed."
      )
    }

    private static func configurationObject(
      _ configuration: Needle2JSRuntimeConfiguration,
      parameters: Needle2ContextParameters,
      tools: [EdgeToolDefinition]
    )
      -> JSObject
    {
      let object = JSObject()
      object["provider"] = .string(configuration.provider.rawValue)
      object["engine"] = .string(configuration.engine.rawValue)
      if let wasm = configuration.wasm {
        object["wasm"] = self.jsValue(wasm)
      }
      if let weights = configuration.weights {
        object["weights"] = self.jsValue(weights)
      }
      let systemValues = JSObject()
      for (key, value) in parameters.system.values {
        systemValues[key.rawValue] = .string(value)
      }
      object["systemValues"] = systemValues.jsValue
      object["tools"] = tools.needle2Value.jsValue
      if let toolIndexPath = parameters.toolIndexPath {
        object["toolIndexPath"] = .string(toolIndexPath)
      }
      return object
    }

    private static func jsValue(_ source: Needle2JSBinarySource) -> JSValue {
      switch source {
      case .location(let location): .string(location)
      case .bytes(let bytes): JSUint8Array(bytes).jsObject.jsValue
      }
    }

    private static func dispose(_ runtime: JSRemote<JSObject>) async throws {
      try await runtime.promiseValue(failure: Self.error) { runtime in
        runtime["dispose"].object!.promisingCall(this: runtime)
      }
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
      var metrics = EdgeToolsMetrics()
      metrics.generationConfidence = self.confidence
      metrics.needle2ResponseType = self.type
      metrics.prefillTokensPerSecond = self.prefillTokensPerSecond
      metrics.decodeTokens = tokenCount
      metrics.decodeTokensPerSecond = self.decodeTokensPerSecond
      metrics.needle2PeakRAMMegabytes = self.peakRAMMegabytes
      return EdgeToolsEngineGeneration(
        wasStopped: wasStopped,
        tokens: [],
        response: "",
        parts: self.parts,
        metrics: metrics
      )
    }
  }
#endif
