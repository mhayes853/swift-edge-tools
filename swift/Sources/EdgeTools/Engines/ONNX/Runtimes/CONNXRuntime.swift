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

      public static let invalidGraphOptimizationLevel = Self(
        rawValue: "invalid-graph-optimization-level"
      )
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

    private let api: UnsafePointer<OrtApi>
    private let environment: OpaquePointer

    fileprivate let allocator: UnsafeMutablePointer<OrtAllocator>
    private let configuration: Configuration

    public init(api: OpaquePointer, configuration: Configuration = Configuration()) throws {
      let api = UnsafePointer<OrtApi>(api)
      let environment: OpaquePointer = try output(api: api) { environment in
        configuration.logIdentifier.withCString { identifier in
          api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, identifier, environment)
        }
      }
      let allocator: UnsafeMutablePointer<OrtAllocator>
      do {
        allocator = try output(api: api) {
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

    public func withUnsafeAPIPointer<Result, Failure: Error>(
      _ body: (UnsafePointer<OrtApi>) throws(Failure) -> Result
    ) throws(Failure) -> Result {
      try body(self.api)
    }

    public func withUnsafeEnvironmentPointer<Result, Failure: Error>(
      _ body: (OpaquePointer) throws(Failure) -> Result
    ) throws(Failure) -> Result {
      try body(self.environment)
    }

    public func session(modelPath: String) throws -> CONNXRuntimeSession {
      try self.session(modelPath: modelPath, configuration: self.configuration)
    }

    public func session(
      modelPath: String,
      configuration: Configuration
    ) throws -> CONNXRuntimeSession {
      let optionsHandle = try self.makeSessionOptionsHandle(configuration: configuration)
      defer { self.api.pointee.ReleaseSessionOptions(optionsHandle) }
      let handle: OpaquePointer = try modelPath.withCString { path in
        try output(api: self.api) {
          self.api.pointee.CreateSession(self.environment, path, optionsHandle, $0)
        }
      }
      return try CONNXRuntimeSession(runtime: self, handle: handle)
    }

    private func makeSessionOptionsHandle(configuration: Configuration) throws -> OpaquePointer {
      let handle: OpaquePointer = try output(api: self.api) {
        self.api.pointee.CreateSessionOptions($0)
      }
      do {
        try check(
          api: self.api,
          status: self.api.pointee.SetSessionGraphOptimizationLevel(
            handle,
            try Self.graphOptimizationLevel(configuration.graphOptimizationLevel)
          )
        )
        try configuration.configureSessionOptions(self, handle)
        return handle
      } catch {
        self.api.pointee.ReleaseSessionOptions(handle)
        throw error
      }
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
        try output(api: self.api) {
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
        let destination: UnsafeMutableRawPointer = try output(api: self.api) {
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

  }

  private func check(api: UnsafePointer<OrtApi>, status: OpaquePointer?) throws {
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

  // MARK: - Configuration

  extension CONNXRuntime {
    public struct Configuration: Sendable {
      public var logIdentifier: String
      public var graphOptimizationLevel: ONNXGraphOptimizationLevel
      public var configureSessionOptions: @Sendable (CONNXRuntime, OpaquePointer) throws -> Void

      public init(
        logIdentifier: String = "swift-edge-tools",
        graphOptimizationLevel: ONNXGraphOptimizationLevel = .all,
        configureSessionOptions: @escaping @Sendable (CONNXRuntime, OpaquePointer) throws -> Void =
          { _, _ in }
      ) {
        self.logIdentifier = logIdentifier
        self.graphOptimizationLevel = graphOptimizationLevel
        self.configureSessionOptions = configureSessionOptions
      }
    }
  }

  // MARK: - CONNXRuntimeSession

  public final class CONNXRuntimeSession: @unchecked Sendable {
    // ONNX Runtime sessions support concurrent inference through an immutable handle, so @unchecked Sendable is safe.

    public let runtime: CONNXRuntime
    private let handle: OpaquePointer

    public let inputNames: [String]
    public let outputNames: [String]

    fileprivate init(runtime: CONNXRuntime, handle: OpaquePointer) throws {
      self.runtime = runtime
      self.handle = handle
      do {
        self.inputNames = try runtime.withUnsafeAPIPointer { api in
          try Self.names(
            api: api,
            allocator: runtime.allocator,
            handle: handle,
            count: api.pointee.SessionGetInputCount,
            name: api.pointee.SessionGetInputName
          )
        }
        self.outputNames = try runtime.withUnsafeAPIPointer { api in
          try Self.names(
            api: api,
            allocator: runtime.allocator,
            handle: handle,
            count: api.pointee.SessionGetOutputCount,
            name: api.pointee.SessionGetOutputName
          )
        }
      } catch {
        runtime.withUnsafeAPIPointer { $0.pointee.ReleaseSession(handle) }
        throw error
      }
    }

    deinit {
      self.runtime.withUnsafeAPIPointer { $0.pointee.ReleaseSession(self.handle) }
    }

    public func withUnsafePointer<Result, Failure: Error>(
      _ body: (OpaquePointer) throws(Failure) -> Result
    ) throws(Failure) -> Result {
      try body(self.handle)
    }

    public func run(
      inputs: [String: CONNXRuntimeTensor],
      outputNames: [String]
    ) throws -> [String: CONNXRuntimeTensor] {
      try edgeToolsONNXValidateOutputNames(outputNames)
      let inputs = Array(inputs)
      let inputNames = inputs.map(\.key)
      let inputValues = inputs.map { Optional($0.value.withUnsafePointer { $0 }) }
      var outputValues = [OpaquePointer?](repeating: nil, count: outputNames.count)
      try self.runtime.withUnsafeAPIPointer { api in
        do {
          try withCopiedCStringPointerBuffer(inputNames) { inputNames in
            try withCopiedCStringPointerBuffer(outputNames) { outputNames in
              try inputValues.withUnsafeBufferPointer { inputValues in
                try outputValues.withUnsafeMutableBufferPointer { outputValues in
                  try check(
                    api: api,
                    status: api.pointee.Run(
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
          outputValues.compactMap { $0 }.forEach { api.pointee.ReleaseValue($0) }
          throw error
        }
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
        self.runtime.withUnsafeAPIPointer { api in
          handles.forEach { api.pointee.ReleaseValue($0) }
        }
        throw error
      }
    }

    private static func names(
      api: UnsafePointer<OrtApi>,
      allocator: UnsafeMutablePointer<OrtAllocator>,
      handle: OpaquePointer,
      count getCount: (OpaquePointer?, UnsafeMutablePointer<Int>?) -> OpaquePointer?,
      name: (
        OpaquePointer?, Int, UnsafeMutablePointer<OrtAllocator>?,
        UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
      ) -> OpaquePointer?
    ) throws -> [String] {
      var count = 0
      try check(api: api, status: getCount(handle, &count))
      return try (0..<count)
        .map { index in
          let name: UnsafeMutablePointer<CChar> = try output(api: api) {
            name(handle, index, allocator, $0)
          }
          defer { allocator.pointee.Free(allocator, name) }
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
    private let handle: OpaquePointer

    public let dtype: DType
    public let shape: [Int]

    fileprivate init(runtime: CONNXRuntime, handle: OpaquePointer, dtype: DType, shape: [Int]) {
      self.runtime = runtime
      self.handle = handle
      self.dtype = dtype
      self.shape = shape
    }

    fileprivate convenience init(runtime: CONNXRuntime, handle: OpaquePointer) throws {
      let metadata = try Self.metadata(runtime: runtime, handle: handle)
      self.init(runtime: runtime, handle: handle, dtype: metadata.dtype, shape: metadata.shape)
    }

    deinit {
      self.runtime.withUnsafeAPIPointer { $0.pointee.ReleaseValue(self.handle) }
    }

    public func withUnsafePointer<Result, Failure: Error>(
      _ body: (OpaquePointer) throws(Failure) -> Result
    ) throws(Failure) -> Result {
      try body(self.handle)
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
      try self.runtime.withUnsafeAPIPointer { api in
        try check(api: api, status: api.pointee.GetTensorMutableData(self.handle, &data))
      }
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
      try runtime.withUnsafeAPIPointer { api in
        let info: OpaquePointer = try output(api: api) {
          api.pointee.GetTensorTypeAndShape(handle, $0)
        }
        defer { api.pointee.ReleaseTensorTypeAndShapeInfo(info) }
        var elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        try check(api: api, status: api.pointee.GetTensorElementType(info, &elementType))
        var count = 0
        try check(api: api, status: api.pointee.GetDimensionsCount(info, &count))
        var dimensions = [Int64](repeating: 0, count: count)
        try dimensions.withUnsafeMutableBufferPointer {
          try check(
            api: api,
            status: api.pointee.GetDimensions(info, $0.baseAddress, $0.count)
          )
        }
        return (DType(rawValue: Int(elementType.rawValue)), dimensions.map(Int.init))
      }
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

    public func session(model: String, configuration: Configuration) throws -> CONNXRuntimeSession {
      try self.session(modelPath: model, configuration: configuration)
    }
  }

  private func output<Value>(
    api: UnsafePointer<OrtApi>,
    _ operation: (UnsafeMutablePointer<Value?>) -> OpaquePointer?
  ) throws -> Value {
    var value: Value?
    try check(api: api, status: operation(&value))
    return value!
  }
#endif
