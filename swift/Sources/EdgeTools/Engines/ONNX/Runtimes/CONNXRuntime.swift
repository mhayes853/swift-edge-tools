#if ONNXCore && canImport(COnnxRuntime)
  import COnnxRuntime

  #if Foundation
    import _EdgeToolsFoundation
  #endif

  // MARK: - CONNXRuntimeError

  public struct CONNXRuntimeError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let incompatibleSessionOptions = Self(rawValue: "incompatible-session-options")
      public static let invalidGraphOptimizationLevel = Self(
        rawValue: "invalid-graph-optimization-level"
      )
      public static let invalidThreadCount = Self(rawValue: "invalid-thread-count")
      public static let onnxRuntime = Self(rawValue: "onnx-runtime")
      public static let unsupportedAPIVersion = Self(rawValue: "unsupported-api-version")
    }

    public let code: Code
    public let message: String
    public let onnxRuntimeCode: Int?

    public init(code: Code, message: String, onnxRuntimeCode: Int? = nil) {
      self.code = code
      self.message = message
      self.onnxRuntimeCode = onnxRuntimeCode
    }
  }

  // MARK: - CONNXRuntime

  public final class CONNXRuntime: @unchecked Sendable {
    // The API and environment pointers are immutable, and ONNX Runtime supports concurrent sessions.
    private static let apiVersion = UInt32(ORT_API_VERSION)

    public let api: UnsafePointer<OrtApi>
    public let environment: OpaquePointer

    fileprivate let allocator: UnsafeMutablePointer<OrtAllocator>
    private let configuration: Configuration

    public init(api: OpaquePointer, configuration: Configuration = Configuration()) throws {
      let api = UnsafePointer<OrtApi>(api)
      let environment: OpaquePointer = try Self.output(api: api) { environment in
        configuration.logIdentifier.withCString { identifier in
          api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, identifier, environment)
        }
      }
      let allocator: UnsafeMutablePointer<OrtAllocator>
      do {
        allocator = try Self.output(api: api) {
          api.pointee.GetAllocatorWithDefaultOptions($0)
        }
      } catch {
        api.pointee.ReleaseEnv(environment)
        throw error
      }
      self.api = api
      self.environment = environment
      self.allocator = allocator
      self.configuration = configuration
    }

    #if ONNX
      public convenience init(configuration: Configuration = Configuration()) throws {
        guard
          let apiBase = OrtGetApiBase(),
          let api = apiBase.pointee.GetApi(Self.apiVersion)
        else {
          throw CONNXRuntimeError(
            code: .unsupportedAPIVersion,
            message: "ONNX Runtime does not support API version \(Self.apiVersion)."
          )
        }
        try self.init(api: OpaquePointer(api), configuration: configuration)
      }
    #endif

    deinit {
      self.api.pointee.ReleaseEnv(self.environment)
    }

    public func check(_ status: OpaquePointer?) throws {
      try Self.check(api: self.api, status: status)
    }

    public func sessionOptions() throws -> SessionOptions {
      try self.sessionOptions(configuration: self.configuration)
    }

    public func sessionOptions(configuration: Configuration) throws -> SessionOptions {
      let options = try SessionOptions(runtime: self)
      try options.setGraphOptimizationLevel(configuration.graphOptimizationLevel)
      try configuration.configureSessionOptions(options)
      return options
    }

    public func session(modelPath: String) throws -> CONNXRuntimeSession {
      let options = try self.sessionOptions()
      return try self.session(modelPath: modelPath, options: options)
    }

    public func session(
      modelPath: String,
      options: borrowing SessionOptions
    ) throws -> CONNXRuntimeSession {
      guard options.runtime === self else {
        throw CONNXRuntimeError(
          code: .incompatibleSessionOptions,
          message: "Session options must be created by the runtime creating the session."
        )
      }
      let handle: OpaquePointer = try modelPath.withCString { path in
        try Self.output(api: self.api) {
          self.api.pointee.CreateSession(self.environment, path, options.handle, $0)
        }
      }
      return try CONNXRuntimeSession.make(runtime: self, handle: handle)
    }

    #if Foundation
      public func session(modelURL: URL) throws -> CONNXRuntimeSession {
        try self.session(modelPath: modelURL.path())
      }
    #endif

    public func tensor<Values: Sequence>(
      values: Values,
      shape: [Int]
    ) throws -> CONNXRuntimeTensor where Values.Element: ONNXElement {
      let values = ContiguousArray(values)
      try edgeToolsONNXValidateValueCount(values.count, shape: shape)
      let dimensions = shape.map(Int64.init)
      let dtype = Values.Element.onnxDType
      let handle: OpaquePointer = try dimensions.withUnsafeBufferPointer { dimensions in
        try Self.output(api: self.api) {
          self.api.pointee.CreateTensorAsOrtValue(
            self.allocator,
            dimensions.baseAddress,
            dimensions.count,
            ONNXTensorElementDataType(rawValue: UInt32(dtype.rawValue)),
            $0
          )
        }
      }
      do {
        let destination: UnsafeMutableRawPointer = try Self.output(api: self.api) {
          self.api.pointee.GetTensorMutableData(handle, $0)
        }
        values.withUnsafeBytes {
          guard let source = $0.baseAddress else { return }
          destination.copyMemory(from: source, byteCount: $0.count)
        }
        return CONNXRuntimeTensor(runtime: self, handle: handle, dtype: dtype, shape: shape)
      } catch {
        self.api.pointee.ReleaseValue(handle)
        throw error
      }
    }

    private static func graphOptimizationLevel(
      _ level: ONNXGraphOptimizationLevel
    ) throws -> GraphOptimizationLevel {
      switch level {
      case .disabled: ORT_DISABLE_ALL
      case .basic: ORT_ENABLE_BASIC
      case .extended: ORT_ENABLE_EXTENDED
      case .layout: ORT_ENABLE_LAYOUT
      case .all: ORT_ENABLE_ALL
      default:
        throw CONNXRuntimeError(
          code: .invalidGraphOptimizationLevel,
          message: "Unsupported graph optimization level: \(level.rawValue)."
        )
      }
    }

    static func output<Value>(
      api: UnsafePointer<OrtApi>,
      _ operation: (UnsafeMutablePointer<Value?>) -> OpaquePointer?
    ) throws -> Value {
      var value: Value?
      try Self.check(api: api, status: operation(&value))
      return value!
    }

    static func check(api: UnsafePointer<OrtApi>, status: OpaquePointer?) throws {
      guard let status else { return }
      defer { api.pointee.ReleaseStatus(status) }
      let code = api.pointee.GetErrorCode(status)
      let message =
        api.pointee.GetErrorMessage(status).map(String.init(cString:))
        ?? "Unknown ONNX Runtime error."
      throw CONNXRuntimeError(
        code: .onnxRuntime,
        message: message,
        onnxRuntimeCode: Int(code.rawValue)
      )
    }
  }

  extension CONNXRuntime {
    public struct Configuration: Sendable {
      public var logIdentifier: String
      public var graphOptimizationLevel: ONNXGraphOptimizationLevel
      public var configureSessionOptions: @Sendable (borrowing SessionOptions) throws -> Void

      public init(
        logIdentifier: String = "swift-edge-tools",
        graphOptimizationLevel: ONNXGraphOptimizationLevel = .all,
        configureSessionOptions: @escaping @Sendable (borrowing SessionOptions) throws -> Void = { _ in }
      ) {
        self.logIdentifier = logIdentifier
        self.graphOptimizationLevel = graphOptimizationLevel
        self.configureSessionOptions = configureSessionOptions
      }
    }

    public struct SessionOptions: ~Copyable {
      let runtime: CONNXRuntime
      public let api: UnsafePointer<OrtApi>
      public let handle: OpaquePointer

      init(runtime: CONNXRuntime) throws {
        self.runtime = runtime
        self.api = runtime.api
        self.handle = try CONNXRuntime.output(api: runtime.api) {
          runtime.api.pointee.CreateSessionOptions($0)
        }
      }

      deinit {
        self.api.pointee.ReleaseSessionOptions(self.handle)
      }

      public borrowing func check(_ status: OpaquePointer?) throws {
        try CONNXRuntime.check(api: self.api, status: status)
      }

      public borrowing func setIntraOpThreadCount(_ count: Int) throws {
        guard let count = Int32(exactly: count), count > 0 else {
          throw CONNXRuntimeError(
            code: .invalidThreadCount,
            message: "The intra-op thread count must be between 1 and \(Int32.max)."
          )
        }
        try self.check(self.api.pointee.SetIntraOpNumThreads(self.handle, count))
      }

      public borrowing func configure(
        providerNamed name: String,
        options: [String: String] = [:]
      ) throws {
        guard name.lowercased() != "cpu" else { return }
        let keys = Array(options.keys)
        let values = keys.map { options[$0]! }
        try withCopiedCStringPointerBuffer(keys) { keys in
          try withCopiedCStringPointerBuffer(values) { values in
            try name.withCString {
              try self.check(
                self.api.pointee.SessionOptionsAppendExecutionProvider(
                  self.handle,
                  $0,
                  keys.baseAddress,
                  values.baseAddress,
                  keys.count
                )
              )
            }
          }
        }
      }

      borrowing func setGraphOptimizationLevel(
        _ level: ONNXGraphOptimizationLevel
      ) throws {
        try self.check(
          self.api.pointee.SetSessionGraphOptimizationLevel(
            self.handle,
            try CONNXRuntime.graphOptimizationLevel(level)
          )
        )
      }
    }
  }

  // MARK: - CONNXRuntimeSession

  public final class CONNXRuntimeSession: @unchecked Sendable {
    // ONNX Runtime sessions support concurrent inference through an immutable handle.
    private let runtime: CONNXRuntime

    public let handle: OpaquePointer
    public let inputNames: [String]
    public let outputNames: [String]

    private var api: UnsafePointer<OrtApi> { self.runtime.api }

    private init(
      runtime: CONNXRuntime,
      handle: OpaquePointer,
      inputNames: [String],
      outputNames: [String]
    ) {
      self.runtime = runtime
      self.handle = handle
      self.inputNames = inputNames
      self.outputNames = outputNames
    }

    deinit {
      self.api.pointee.ReleaseSession(self.handle)
    }

    static func make(runtime: CONNXRuntime, handle: OpaquePointer) throws -> CONNXRuntimeSession {
      do {
        return CONNXRuntimeSession(
          runtime: runtime,
          handle: handle,
          inputNames: try Self.names(
            runtime: runtime,
            handle: handle,
            count: runtime.api.pointee.SessionGetInputCount,
            name: runtime.api.pointee.SessionGetInputName
          ),
          outputNames: try Self.names(
            runtime: runtime,
            handle: handle,
            count: runtime.api.pointee.SessionGetOutputCount,
            name: runtime.api.pointee.SessionGetOutputName
          )
        )
      } catch {
        runtime.api.pointee.ReleaseSession(handle)
        throw error
      }
    }

    public func run(
      inputs: [String: CONNXRuntimeTensor],
      outputNames: [String]
    ) throws -> [String: CONNXRuntimeTensor] {
      try edgeToolsONNXValidateOutputNames(outputNames)
      let inputs = Array(inputs)
      let inputNames = inputs.map(\.key)
      let inputValues = inputs.map { Optional($0.value.handle) }
      var outputValues = [OpaquePointer?](repeating: nil, count: outputNames.count)
      do {
        try withCopiedCStringPointerBuffer(inputNames) { inputNames in
          try withCopiedCStringPointerBuffer(outputNames) { outputNames in
            try inputValues.withUnsafeBufferPointer { inputValues in
              try outputValues.withUnsafeMutableBufferPointer { outputValues in
                try CONNXRuntime.check(
                  api: self.api,
                  status: self.api.pointee.Run(
                    self.handle,
                    nil,
                    inputNames.baseAddress,
                    inputValues.baseAddress,
                    inputValues.count,
                    outputNames.baseAddress,
                    outputNames.count,
                    outputValues.baseAddress
                  )
                )
              }
            }
          }
        }
      } catch {
        outputValues.compactMap { $0 }.forEach { self.api.pointee.ReleaseValue($0) }
        throw error
      }

      let handles = outputValues.map { $0! }
      do {
        return try Dictionary(
          uniqueKeysWithValues: zip(outputNames, handles)
            .map { name, handle in
              (name, try CONNXRuntimeTensor(runtime: self.runtime, handle: handle))
            }
        )
      } catch {
        handles.forEach { self.api.pointee.ReleaseValue($0) }
        throw error
      }
    }

    private static func names(
      runtime: CONNXRuntime,
      handle: OpaquePointer,
      count getCount: (OpaquePointer?, UnsafeMutablePointer<Int>?) -> OpaquePointer?,
      name getName: (
        OpaquePointer?, Int, UnsafeMutablePointer<OrtAllocator>?,
        UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
      ) -> OpaquePointer?
    ) throws -> [String] {
      var count = 0
      try runtime.check(getCount(handle, &count))
      return try (0..<count)
        .map { index in
          let name: UnsafeMutablePointer<CChar> = try CONNXRuntime.output(api: runtime.api) {
            getName(handle, index, runtime.allocator, $0)
          }
          defer { runtime.allocator.pointee.Free(runtime.allocator, name) }
          return String(cString: name)
        }
    }
  }

  // MARK: - CONNXRuntimeTensor

  public final class CONNXRuntimeTensor: @unchecked Sendable {
    // @unchecked Sendable is safe for concurrent reads, but `withUnsafeMutableBufferPointer` exposes
    // the tensor's own buffer directly (no copy) - callers must not mutate it concurrently.
    public typealias DType = ONNXDType

    private let runtime: CONNXRuntime

    public let handle: OpaquePointer
    public let dtype: DType
    public let shape: [Int]

    init(runtime: CONNXRuntime, handle: OpaquePointer, dtype: DType, shape: [Int]) {
      self.runtime = runtime
      self.handle = handle
      self.dtype = dtype
      self.shape = shape
    }

    convenience init(runtime: CONNXRuntime, handle: OpaquePointer) throws {
      let metadata = try Self.metadata(runtime: runtime, handle: handle)
      self.init(runtime: runtime, handle: handle, dtype: metadata.dtype, shape: metadata.shape)
    }

    deinit {
      self.runtime.api.pointee.ReleaseValue(self.handle)
    }

    nonisolated(nonsending) public func withUnsafeMutableBufferPointer<Element: ONNXElement, Result>(
      as type: Element.Type,
      _ body: nonisolated(nonsending) (UnsafeMutableBufferPointer<Element>) async throws -> Result
    ) async throws -> Result {
      guard self.dtype == Element.onnxDType else {
        throw ONNXRuntimeError(
          code: .unexpectedTensorElementType,
          message:
            "Expected tensor element type \(Element.onnxDType.rawValue), got \(self.dtype.rawValue)."
        )
      }
      let count = try edgeToolsONNXElementCount(for: self.shape)
      var data: UnsafeMutableRawPointer?
      try self.runtime.check(self.runtime.api.pointee.GetTensorMutableData(self.handle, &data))
      let buffer = UnsafeMutableBufferPointer<Element>(
        start: count == 0 ? nil : data!.assumingMemoryBound(to: Element.self),
        count: count
      )
      return try await body(buffer)
    }

    private static func metadata(
      runtime: CONNXRuntime,
      handle: OpaquePointer
    ) throws -> (dtype: DType, shape: [Int]) {
      let info: OpaquePointer = try CONNXRuntime.output(api: runtime.api) {
        runtime.api.pointee.GetTensorTypeAndShape(handle, $0)
      }
      defer { runtime.api.pointee.ReleaseTensorTypeAndShapeInfo(info) }
      var elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
      try runtime.check(runtime.api.pointee.GetTensorElementType(info, &elementType))
      var count = 0
      try runtime.check(runtime.api.pointee.GetDimensionsCount(info, &count))
      var dimensions = [Int64](repeating: 0, count: count)
      try dimensions.withUnsafeMutableBufferPointer {
        try runtime.check(runtime.api.pointee.GetDimensions(info, $0.baseAddress, $0.count))
      }
      return (
        DType(rawValue: Int(elementType.rawValue)),
        dimensions.map(Int.init)
      )
    }
  }

  extension CONNXRuntimeTensor: ONNXTensor {}

  extension CONNXRuntimeSession: ONNXSession {
    public typealias Tensor = CONNXRuntimeTensor
  }

  extension CONNXRuntime: ONNXRuntime {
    public typealias ModelSource = String
    public typealias Session = CONNXRuntimeSession
    public typealias Tensor = CONNXRuntimeTensor

    public func session(
      model: String,
      configuration: Configuration
    ) throws -> CONNXRuntimeSession {
      let options = try self.sessionOptions(configuration: configuration)
      return try self.session(modelPath: model, options: options)
    }
  }
#endif
