#if Needle2 && canImport(CNeedle2)
  import CNeedle2

  #if Foundation
    import _EdgeToolsFoundation
  #endif

  #if !$Embedded
    import Observation
  #endif

  // MARK: - Needle2ContextParameters

  public struct Needle2ContextParameters: Hashable, Sendable {
    public var system: Needle2System
    public var toolIndexPath: String?

    public init(
      system: Needle2System = [],
      toolIndexPath: String? = nil
    ) {
      self.system = system
      self.toolIndexPath = toolIndexPath
    }
  }

  // MARK: - Needle2Context

  public final class Needle2Context: Identifiable, Sendable {
    private struct State {
      var parameters: Needle2ContextParameters
      var isResponding = false
    }

    private let state: Lock<State>
    private let observationRegistrar = _ObservationRegistrar()

    public var system: Needle2System {
      get {
        self.observationRegistrar.access(self, keyPath: \.system)
        return self.state.withBorrowedLock { $0.parameters.system }
      }
      set {
        self.state.withLock { state in
          self.observationRegistrar.withMutation(of: self, keyPath: \.system) {
            state.parameters.system = newValue
          }
        }
      }
    }

    public var toolIndexPath: String? {
      get {
        self.observationRegistrar.access(self, keyPath: \.toolIndexPath)
        return self.state.withBorrowedLock { $0.parameters.toolIndexPath }
      }
      set {
        self.state.withLock { state in
          self.observationRegistrar.withMutation(of: self, keyPath: \.toolIndexPath) {
            state.parameters.toolIndexPath = newValue
          }
        }
      }
    }

    public var isResponding: Bool {
      self.observationRegistrar.access(self, keyPath: \.isResponding)
      return self.state.withBorrowedLock { $0.isResponding }
    }

    fileprivate init(parameters: Needle2ContextParameters) {
      self.state = Lock(State(parameters: parameters))
    }

    fileprivate func begin(
      prompt: String,
      toolsJSON: String,
      maximumTokenCount: Int32,
      outputCapacity: Int32
    ) throws -> Needle2Request {
      try self.state.withLock { state in
        guard !state.isResponding else {
          throw EdgeToolsError.contextInUse
        }
        var request: Needle2Request!
        self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
          state.isResponding = true
          request = Needle2Request(
            prompt: prompt,
            initialization: Needle2Initialization(
              systemPrompt: state.parameters.system.formatted(),
              toolsJSON: toolsJSON,
              toolIndexPath: state.parameters.toolIndexPath
            ),
            maximumTokenCount: maximumTokenCount,
            outputCapacity: outputCapacity
          )
        }
        return request
      }
    }

    fileprivate func finish() {
      self.state.withLock { state in
        self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
          state.isResponding = false
        }
      }
    }
  }

  #if !$Embedded
    extension Needle2Context: Observable {}
  #endif

  // MARK: - Needle2GenerateParameters

  public struct Needle2GenerateParameters: EdgeToolsEngineGenerateParameters, Hashable, Sendable {
    public static let `default` = Self()

    public var maxTokens: Int?
    public var outputCapacity: Int

    public init(
      maxTokens: Int? = 256,
      outputCapacity: Int = 65_536
    ) {
      self.maxTokens = maxTokens
      self.outputCapacity = outputCapacity
    }
  }

  // MARK: - Needle2Error

  public struct Needle2Error: Error, Hashable, Sendable {
    public struct Code: Hashable, RawRepresentable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let invalidGenerateParameters = Self(rawValue: "invalid-generate-parameters")
      public static let initializationFailed = Self(rawValue: "initialization-failed")
      public static let loadingFailed = Self(rawValue: "loading-failed")
      public static let generationFailed = Self(rawValue: "generation-failed")
      public static let invalidResponse = Self(rawValue: "invalid-response")
      public static let outputCapacityExceeded = Self(rawValue: "output-capacity-exceeded")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  // MARK: - Needle2Engine

  @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
  public struct Needle2Engine: EdgeToolsEngine {
    public typealias Context = Needle2Context
    public typealias ContextParameters = Needle2ContextParameters
    public typealias Prompt = String
    public typealias GenerateParameters = Needle2GenerateParameters

    private let defaultContext: Context

    public init() {
      self.defaultContext = Context(parameters: ContextParameters())
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
    ) throws -> some EdgeToolsEngineGenerationTask {
      let maximumTokenCount = parameters.maxTokens ?? 256
      guard maximumTokenCount > 0, let maximumTokenCount = Int32(exactly: maximumTokenCount) else {
        throw Needle2Error(
          code: .invalidGenerateParameters,
          message: "Needle 2 maxTokens must be between 1 and Int32.max."
        )
      }
      guard parameters.outputCapacity > 0,
        let outputCapacity = Int32(exactly: parameters.outputCapacity)
      else {
        throw Needle2Error(
          code: .invalidGenerateParameters,
          message: "Needle 2 outputCapacity must be between 1 and Int32.max."
        )
      }

      return AnyGenerationTask { stopper in
        let request = try context.begin(
          prompt: prompt,
          toolsJSON: tools.needle2JSON,
          maximumTokenCount: maximumTokenCount,
          outputCapacity: outputCapacity
        )
        defer { context.finish() }
        return try await self.runNeedle2Generation(
          request: request,
          channel: channel,
          wasStopped: { stopper.isStopped }
        )
      }
    }
    @concurrent
    private func runNeedle2Generation(
      request: Needle2Request,
      channel: EdgeToolsGenerationChannel,
      wasStopped: @Sendable () -> Bool
    ) async throws -> EdgeToolsEngineGeneration {
      let nativeResult = try Needle2Runtime.shared.complete(request: request)
      let response = try Needle2Response(json: nativeResult.response)
      if !response.success {
        throw Needle2Error(
          code: response.errorCode.map(Needle2Error.Code.init(rawValue:)) ?? .generationFailed,
          message: response.error ?? "Needle 2 generation failed."
        )
      }

      let parts = response.parts
      let stopped = wasStopped()
      if !stopped {
        for part in parts {
          channel.emit(part: part)
        }
      }

      var metadata = EdgeToolsMetadata()
      metadata.generationConfidence = response.confidence
      metadata.needle2ResponseType = response.type
      metadata.needle2PrefillTokensPerSecond = response.prefillTokensPerSecond
      metadata.needle2DecodeTokensPerSecond = response.decodeTokensPerSecond
      metadata.needle2PeakRAMMegabytes = response.peakRAMMegabytes
      return EdgeToolsEngineGeneration(
        prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: nativeResult.tokenCount,
          duration: Duration(
            needle2TokenCount: nativeResult.tokenCount,
            tokensPerSecond: response.decodeTokensPerSecond
          ),
          durationToFirstToken: .zero
        ),
        wasStopped: stopped,
        tokens: [],
        response: nativeResult.response,
        parts: parts,
        metadata: metadata
      )
    }

    public static func load(_ bytes: UnsafeBufferPointer<UInt8>) throws {
      try Needle2Runtime.shared.load(bytes)
    }

    public static func load(_ bytes: [UInt8]) throws {
      try bytes.withUnsafeBufferPointer(Self.load)
    }
  }

  #if Foundation
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    extension Needle2Engine {
      public static func load(_ data: Data) throws {
        try data.withUnsafeBytes { bytes in
          try self.load(bytes.bindMemory(to: UInt8.self))
        }
      }

      public static func load(contentsOf url: URL) throws {
        try self.load(Data(contentsOf: url))
      }
    }
  #endif

  // MARK: - Native Generation

  private final class Needle2Runtime: Sendable {
    private struct State {
      var initialization: Needle2Initialization?
    }

    static let shared = Needle2Runtime()

    private let state = Lock(State())

    func complete(request: Needle2Request) throws -> (response: String, tokenCount: Int) {
      try self.state.withLock { state in
        if state.initialization != request.initialization {
          try request.initialization.initialize()
          state.initialization = request.initialization
        }
        defer { needle_reset() }
        return try request.complete()
      }
    }

    func load(_ bytes: UnsafeBufferPointer<UInt8>) throws {
      let result = self.state.withLock { state in
        let result = needle_load(bytes.baseAddress, UInt64(bytes.count))
        state.initialization = nil
        return result
      }
      guard result >= 0 else {
        throw Needle2Error(
          code: .loadingFailed,
          message: "needle_load failed with status \(result)."
        )
      }
    }
  }

  private struct Needle2Initialization: Hashable, Sendable {
    var systemPrompt: String
    var toolsJSON: String
    var toolIndexPath: String?

    func initialize() throws {
      let result = self.systemPrompt.withCString { system in
        self.toolsJSON.withCString { tools in
          withOptionalCString(self.toolIndexPath) { toolIndexPath in
            needle_init(system, tools, toolIndexPath)
          }
        }
      }
      guard result >= 0 else {
        throw Needle2Error(
          code: .initializationFailed,
          message: "needle_init failed with status \(result)."
        )
      }
    }
  }

  private struct Needle2Request: Sendable {
    var prompt: String
    var initialization: Needle2Initialization
    var maximumTokenCount: Int32
    var outputCapacity: Int32

    func complete() throws -> (response: String, tokenCount: Int) {
      var output = [CChar](repeating: 0, count: Int(self.outputCapacity))
      let result = self.prompt.withCString { prompt in
        output.withUnsafeMutableBufferPointer { output in
          needle_complete(
            prompt,
            self.maximumTokenCount,
            output.baseAddress,
            self.outputCapacity
          )
        }
      }
      guard result >= 0 else {
        throw Needle2Error(
          code: .generationFailed,
          message: "needle_complete failed with status \(result)."
        )
      }
      guard let terminator = output.firstIndex(of: 0) else {
        throw Needle2Error(
          code: .outputCapacityExceeded,
          message: "Needle 2 filled the output buffer without terminating its response."
        )
      }
      let bytes = output[..<terminator].map(UInt8.init(bitPattern:))
      return (String(decoding: bytes, as: UTF8.self), Int(result))
    }
  }

  // MARK: - Response

  private struct Needle2Response {
    var type: String
    var success: Bool
    var error: String?
    var errorCode: String?
    var functionCalls: [EdgeRawToolCall]
    var reasoning: String?
    var confidence: Float?
    var prefillTokensPerSecond: Double?
    var decodeTokensPerSecond: Double?
    var peakRAMMegabytes: Double?

    var parts: [EdgeToolsGenerationPart] {
      let reasoning = self.reasoning.map { [EdgeToolsGenerationPart.reasoning($0)] } ?? []
      return reasoning + self.functionCalls.map(EdgeToolsGenerationPart.toolCall)
    }

    init(json: String) throws {
      let value: EdgeToolsValue
      do {
        value = try EdgeToolsValue(json: json)
      } catch {
        throw Needle2Error(
          code: .invalidResponse,
          message: "Needle 2 returned invalid JSON: \(error)"
        )
      }
      guard case .object(let object) = value,
        case .string(let type) = object["type"],
        case .boolean(let success) = object["success"]
      else {
        throw Needle2Error(
          code: .invalidResponse,
          message: "Needle 2 returned a response without a valid type and success value."
        )
      }

      self.type = type
      self.success = success
      self.error = object["error"]?.string
      self.errorCode = object["error_code"]?.string
      self.reasoning = object["reasoning"]?.string
      self.confidence = object["confidence"]?.double.map(Float.init)
      self.prefillTokensPerSecond = object["prefill_tps"]?.double
      self.decodeTokensPerSecond = object["decode_tps"]?.double
      self.peakRAMMegabytes = object["peak_ram_mb"]?.double
      self.functionCalls = try object["function_calls"]?.needle2ToolCalls ?? []
    }
  }

  // MARK: - Helpers

  extension Array where Element == EdgeToolDefinition {
    fileprivate var needle2JSON: String {
      EdgeToolsValue.array(
        self.map { tool in
          [
            "name": .string(tool.name.snakeCased()),
            "description": .string(tool.description),
            "parameters": tool.arguments.edgeToolsValue
          ]
        }
      )
      .orderedJSONString()
    }
  }

  extension Duration {
    fileprivate init(
      needle2TokenCount tokenCount: Int,
      tokensPerSecond: Double?
    ) {
      guard let tokensPerSecond, tokensPerSecond > 0 else {
        self = .zero
        return
      }
      let nanoseconds = Double(tokenCount) / tokensPerSecond * 1_000_000_000
      guard nanoseconds < Double(Int64.max) else {
        self = .zero
        return
      }
      self = .nanoseconds(Int64(nanoseconds.rounded()))
    }
  }

  extension EdgeToolsValue {
    fileprivate var needle2ToolCalls: [EdgeRawToolCall] {
      get throws {
        if self.isNull {
          return []
        }
        guard let values = self.array else {
          throw Needle2Error(
            code: .invalidResponse,
            message: "Needle 2 returned an invalid function_calls value."
          )
        }
        return try values.map { value in
          guard let call = EdgeRawToolCall(jsonValue: value) else {
            throw Needle2Error(
              code: .invalidResponse,
              message: "Needle 2 returned an invalid function call."
            )
          }
          return call
        }
      }
    }
  }
#endif
