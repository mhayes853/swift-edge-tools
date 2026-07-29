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

      public static let invalidGraphOptimizationLevel =
        Self(rawValue: "invalid-graph-optimization-level")
      public static let duplicateOutputName = Self(rawValue: "duplicate-output-name")
      public static let invalidTensorShape = Self(rawValue: "invalid-tensor-shape")
      public static let invalidTensorValueCount = Self(rawValue: "invalid-tensor-value-count")
      public static let onnxRuntime = Self(rawValue: "onnx-runtime")
      public static let tensorElementCountOverflow =
        Self(rawValue: "tensor-element-count-overflow")
      public static let unexpectedTensorElementType =
        Self(rawValue: "unexpected-tensor-element-type")
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
    // Safe because the API table and environment pointers are immutable after initialization, and
    // ONNX Runtime environments support concurrent session creation and inference.
    fileprivate static let apiVersion = UInt32(ORT_API_VERSION)

    fileprivate let api: UnsafePointer<OrtApi>
    fileprivate let allocator: UnsafeMutablePointer<OrtAllocator>
    private let environment: OpaquePointer
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
      self.allocator = allocator
      self.environment = environment
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

    public func session(modelPath: String) throws -> CONNXRuntimeSession {
      try modelPath.withCString {
        try self.session(modelPath: $0, configuration: self.configuration)
      }
    }

    #if Foundation
      public func session(modelURL: URL) throws -> CONNXRuntimeSession {
        try self.session(modelPath: modelURL.path())
      }
    #endif

    private func session(
      modelPath: UnsafePointer<CChar>,
      configuration: Configuration
    ) throws -> CONNXRuntimeSession {
      let options: OpaquePointer = try Self.output(api: self.api) {
        self.api.pointee.CreateSessionOptions($0)
      }
      defer { self.api.pointee.ReleaseSessionOptions(options) }

      try Self.check(
        api: self.api,
        status: self.api.pointee.SetSessionGraphOptimizationLevel(
          options,
          try Self.graphOptimizationLevel(configuration.graphOptimizationLevel)
        )
      )
      for provider in configuration.executionProviders where provider.name.lowercased() != "cpu" {
        try self.append(provider: provider, to: options)
      }

      let session: OpaquePointer = try Self.output(api: self.api) {
        self.api.pointee.CreateSession(self.environment, modelPath, options, $0)
      }
      return try CONNXRuntimeSession.make(runtime: self, session: session)
    }

    public func tensor(values: [Int64], shape: [Int]) throws -> CONNXRuntimeTensor {
      try self.tensor(
        values: values,
        shape: shape,
        elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
      )
    }

    public func tensor(values: [Int32], shape: [Int]) throws -> CONNXRuntimeTensor {
      try self.tensor(
        values: values,
        shape: shape,
        elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
      )
    }

    public func tensor(values: [Float], shape: [Int]) throws -> CONNXRuntimeTensor {
      try self.tensor(
        values: values,
        shape: shape,
        elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
      )
    }

    public func tensor(
      repeating value: Float,
      shape: [Int]
    ) throws -> CONNXRuntimeTensor {
      try self.tensor(
        values: [Float](repeating: value, count: try Self.elementCount(for: shape)),
        shape: shape
      )
    }

    private func tensor<Element>(
      values: [Element],
      shape: [Int],
      elementType: ONNXTensorElementDataType
    ) throws -> CONNXRuntimeTensor {
      let dimensions = shape.map(Int64.init)
      let expectedCount = try Self.elementCount(for: shape)
      guard values.count == expectedCount else {
        throw CONNXRuntimeError(
          code: .invalidTensorValueCount,
          message: "Tensor shape \(shape) requires \(expectedCount) values, got \(values.count)."
        )
      }

      let tensor: OpaquePointer = try dimensions.withUnsafeBufferPointer { dimensions in
        try Self.output(api: self.api) {
          self.api.pointee.CreateTensorAsOrtValue(
            self.allocator,
            dimensions.baseAddress,
            dimensions.count,
            elementType,
            $0
          )
        }
      }
      do {
        let bytes: UnsafeMutableRawPointer = try Self.output(api: self.api) {
          self.api.pointee.GetTensorMutableData(tensor, $0)
        }
        values.withUnsafeBufferPointer { source in
          guard let sourceAddress = source.baseAddress else { return }
          bytes.copyMemory(
            from: sourceAddress,
            byteCount: values.count * MemoryLayout<Element>.stride
          )
        }
        return CONNXRuntimeTensor(
          runtime: self,
          tensor: tensor,
          dtype: CONNXRuntimeTensor.DType(rawValue: Int(elementType.rawValue)),
          shape: shape
        )
      } catch {
        self.api.pointee.ReleaseValue(tensor)
        throw error
      }
    }

    fileprivate static func elementCount(for shape: [Int]) throws -> Int {
      try shape.reduce(1) { count, dimension in
        guard dimension >= 0 else {
          throw CONNXRuntimeError(
            code: .invalidTensorShape,
            message: "Tensor dimensions must not be negative: \(shape)."
          )
        }
        let result = count.multipliedReportingOverflow(by: dimension)
        guard !result.overflow else {
          throw CONNXRuntimeError(
            code: .tensorElementCountOverflow,
            message: "Tensor shape has too many elements: \(shape)."
          )
        }
        return result.partialValue
      }
    }

    private func append(
      provider: ExecutionProvider,
      to sessionOptions: OpaquePointer
    ) throws {
      let keys = provider.options.map(\.name)
      let values = provider.options.map(\.value)
      let name = Self.executionProviderName(provider.name)
      try withCopiedCStringPointerBuffer(keys) { keyPointers in
        try withCopiedCStringPointerBuffer(values) { valuePointers in
          try name.withCString { providerName in
            try Self.check(
              api: self.api,
              status: self.api.pointee.SessionOptionsAppendExecutionProvider(
                sessionOptions,
                providerName,
                keyPointers.baseAddress,
                valuePointers.baseAddress,
                keyPointers.count
              )
            )
          }
        }
      }
    }

    private static func executionProviderName(_ name: String) -> String {
      switch name.lowercased() {
      case "webgpu": "WebGPU"
      case "coreml": "CoreML"
      case "cuda": "CUDA"
      case "dml": "DML"
      default: name
      }
    }

    private static func graphOptimizationLevel(
      _ level: EdgeToolsONNXGraphOptimizationLevel
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

    fileprivate static func output<Value>(
      api: UnsafePointer<OrtApi>,
      _ operation: (UnsafeMutablePointer<Value?>) -> OpaquePointer?
    ) throws -> Value {
      var value: Value?
      try Self.check(api: api, status: operation(&value))
      return value!
    }

    fileprivate static func check(api: UnsafePointer<OrtApi>, status: OpaquePointer?) throws {
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
    // MARK: - Configuration

    public struct Configuration: Hashable, Sendable {
      public var logIdentifier: String
      public var graphOptimizationLevel: EdgeToolsONNXGraphOptimizationLevel
      public var executionProviders: [ExecutionProvider]

      public init(
        logIdentifier: String = "swift-edge-tools",
        graphOptimizationLevel: EdgeToolsONNXGraphOptimizationLevel = .all,
        executionProviders: [ExecutionProvider] = []
      ) {
        self.logIdentifier = logIdentifier
        self.graphOptimizationLevel = graphOptimizationLevel
        self.executionProviders = executionProviders
      }
    }

    public typealias ExecutionProvider = EdgeToolsONNXExecutionProvider
    public typealias ExecutionProviderOption = EdgeToolsONNXExecutionProvider.Option

    // MARK: - CoreML

    public struct CoreMLComputeUnits: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let all = Self(rawValue: "ALL")
      public static let cpuAndGPU = Self(rawValue: "CPUAndGPU")
      public static let cpuAndNeuralEngine = Self(rawValue: "CPUAndNeuralEngine")
      public static let cpuOnly = Self(rawValue: "CPUOnly")
    }

  }

  extension EdgeToolsONNXExecutionProvider {
    public static func coreML(
      computeUnits: CONNXRuntime.CoreMLComputeUnits,
      modelFormat: String = "MLProgram",
      requireStaticInputShapes: Bool = true,
      enableOnSubgraphs: Bool = true
    ) -> Self {
      Self(
        name: "coreml",
        options: [
          Option(name: "MLComputeUnits", value: computeUnits.rawValue),
          Option(name: "ModelFormat", value: modelFormat),
          Option(
            name: "RequireStaticInputShapes",
            value: requireStaticInputShapes ? "1" : "0"
          ),
          Option(
            name: "EnableOnSubgraphs",
            value: enableOnSubgraphs ? "1" : "0"
          )
        ]
      )
    }
  }

  // MARK: - CONNXRuntimeSession

  public final class CONNXRuntimeSession: @unchecked Sendable {
    // ONNX Runtime sessions support concurrent inference, and the pointer is immutable after init,
    // so this is safe.
    private let runtime: CONNXRuntime
    private let session: OpaquePointer

    public let inputNames: [String]
    public let outputNames: [String]

    private var api: UnsafePointer<OrtApi> { self.runtime.api }

    private init(
      runtime: CONNXRuntime,
      session: OpaquePointer,
      inputNames: [String],
      outputNames: [String]
    ) {
      self.runtime = runtime
      self.session = session
      self.inputNames = inputNames
      self.outputNames = outputNames
    }

    deinit {
      self.api.pointee.ReleaseSession(self.session)
    }

    fileprivate static func make(runtime: CONNXRuntime, session: OpaquePointer) throws
      -> CONNXRuntimeSession
    {
      do {
        let inputNames = try Self.names(
          runtime: runtime,
          session: session,
          count: runtime.api.pointee.SessionGetInputCount,
          name: runtime.api.pointee.SessionGetInputName
        )
        let outputNames = try Self.names(
          runtime: runtime,
          session: session,
          count: runtime.api.pointee.SessionGetOutputCount,
          name: runtime.api.pointee.SessionGetOutputName
        )
        return CONNXRuntimeSession(
          runtime: runtime,
          session: session,
          inputNames: inputNames,
          outputNames: outputNames
        )
      } catch {
        runtime.api.pointee.ReleaseSession(session)
        throw error
      }
    }

    public func run(
      inputs: [String: CONNXRuntimeTensor],
      outputNames: [String]
    ) throws -> [String: CONNXRuntimeTensor] {
      guard Set(outputNames).count == outputNames.count else {
        throw CONNXRuntimeError(
          code: .duplicateOutputName,
          message: "Output names must be unique."
        )
      }
      let inputPairs = Array(inputs)
      let inputNames = inputPairs.map(\.key)
      let inputValues = inputPairs.map { Optional($0.value.tensor) }
      var outputValues = [OpaquePointer?](repeating: nil, count: outputNames.count)
      do {
        try withCopiedCStringPointerBuffer(inputNames) { inputNamePointers in
          try withCopiedCStringPointerBuffer(outputNames) { outputNamePointers in
            try inputValues.withUnsafeBufferPointer { inputValues in
              try outputValues.withUnsafeMutableBufferPointer { outputValues in
                try CONNXRuntime.check(
                  api: self.api,
                  status: self.api.pointee.Run(
                    self.session,
                    nil,
                    inputNamePointers.baseAddress,
                    inputValues.baseAddress,
                    inputValues.count,
                    outputNamePointers.baseAddress,
                    outputNamePointers.count,
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

      let outputPointers = outputValues.map { $0! }
      let metadata: [(dtype: CONNXRuntimeTensor.DType, shape: [Int])]
      do {
        metadata = try outputPointers.map {
          try CONNXRuntimeTensor.metadata(runtime: self.runtime, tensor: $0)
        }
      } catch {
        outputPointers.forEach { self.api.pointee.ReleaseValue($0) }
        throw error
      }

      return Dictionary(
        uniqueKeysWithValues: zip(outputNames, zip(outputPointers, metadata))
          .map { pair in
            let (tensor, metadata) = pair.1
            return (
              pair.0,
              CONNXRuntimeTensor(
                runtime: self.runtime,
                tensor: tensor,
                dtype: metadata.dtype,
                shape: metadata.shape
              )
            )
          }
      )
    }

    private static func names(
      runtime: CONNXRuntime,
      session: OpaquePointer,
      count getCount: (OpaquePointer?, UnsafeMutablePointer<Int>?) -> OpaquePointer?,
      name getName: (
        OpaquePointer?,
        Int,
        UnsafeMutablePointer<OrtAllocator>?,
        UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
      ) -> OpaquePointer?
    ) throws -> [String] {
      var count = 0
      try CONNXRuntime.check(api: runtime.api, status: getCount(session, &count))
      return try (0..<count)
        .map { index in
          let name: UnsafeMutablePointer<CChar> = try CONNXRuntime.output(api: runtime.api) {
            getName(session, index, runtime.allocator, $0)
          }
          defer { runtime.allocator.pointee.Free(runtime.allocator, name) }
          return String(cString: name)
        }
    }
  }

  // MARK: - CONNXRuntimeTensor

  public final class CONNXRuntimeTensor: @unchecked Sendable {
    // Tensor storage is initialized before publication and only immutable reads are exposed, so
    // this is safe.
    public typealias DType = EdgeToolsONNXDType

    fileprivate let runtime: CONNXRuntime
    fileprivate let tensor: OpaquePointer

    public let dtype: DType
    public let shape: [Int]

    fileprivate init(
      runtime: CONNXRuntime,
      tensor: OpaquePointer,
      dtype: DType,
      shape: [Int]
    ) {
      self.runtime = runtime
      self.tensor = tensor
      self.dtype = dtype
      self.shape = shape
    }

    deinit {
      self.runtime.api.pointee.ReleaseValue(self.tensor)
    }

    fileprivate static func metadata(
      runtime: CONNXRuntime,
      tensor: OpaquePointer
    ) throws -> (dtype: DType, shape: [Int]) {
      try Self.withShapeInfo(runtime: runtime, tensor: tensor) { shapeInfo in
        var elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        try CONNXRuntime.check(
          api: runtime.api,
          status: runtime.api.pointee.GetTensorElementType(shapeInfo, &elementType)
        )

        var count = 0
        try CONNXRuntime.check(
          api: runtime.api,
          status: runtime.api.pointee.GetDimensionsCount(shapeInfo, &count)
        )
        var dimensions = [Int64](repeating: 0, count: count)
        try dimensions.withUnsafeMutableBufferPointer { dimensions in
          try CONNXRuntime.check(
            api: runtime.api,
            status: runtime.api.pointee.GetDimensions(
              shapeInfo,
              dimensions.baseAddress,
              dimensions.count
            )
          )
        }
        return (
          DType(rawValue: Int(elementType.rawValue)),
          dimensions.map(Int.init)
        )
      }
    }

    public func floatValues() throws -> [Float] {
      try self.values(elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, valueType: Float.self)
    }

    public func int32Values() throws -> [Int32] {
      try self.values(elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, valueType: Int32.self)
    }

    public func int64Values() throws -> [Int64] {
      try self.values(elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, valueType: Int64.self)
    }

    private func values<Element>(
      elementType expectedElementType: ONNXTensorElementDataType,
      valueType: Element.Type
    ) throws -> [Element] {
      let expectedDType = DType(rawValue: Int(expectedElementType.rawValue))
      guard self.dtype == expectedDType else {
        throw CONNXRuntimeError(
          code: .unexpectedTensorElementType,
          message:
            "Expected tensor element type \(expectedDType.rawValue), got \(self.dtype.rawValue)."
        )
      }

      let bytes: UnsafeMutableRawPointer = try CONNXRuntime.output(api: self.runtime.api) {
        self.runtime.api.pointee.GetTensorMutableData(self.tensor, $0)
      }
      return Array(
        UnsafeBufferPointer(
          start: bytes.assumingMemoryBound(to: Element.self),
          count: try CONNXRuntime.elementCount(for: self.shape)
        )
      )
    }

    private static func withShapeInfo<Result>(
      runtime: CONNXRuntime,
      tensor: OpaquePointer,
      _ body: (OpaquePointer) throws -> Result
    ) throws -> Result {
      let shapeInfo: OpaquePointer = try CONNXRuntime.output(api: runtime.api) {
        runtime.api.pointee.GetTensorTypeAndShape(tensor, $0)
      }
      defer { runtime.api.pointee.ReleaseTensorTypeAndShapeInfo(shapeInfo) }
      return try body(shapeInfo)
    }
  }

  // MARK: - Protocol Conformances

  extension CONNXRuntimeTensor: EdgeToolsONNXTensor {}

  extension CONNXRuntimeSession: EdgeToolsONNXSession {
    public typealias Tensor = CONNXRuntimeTensor
  }

  extension CONNXRuntime: EdgeToolsONNXRuntime {
    public typealias ModelSource = String
    public typealias Session = CONNXRuntimeSession
    public typealias Tensor = CONNXRuntimeTensor

    public func session(
      model: String,
      configuration: Configuration
    ) throws -> CONNXRuntimeSession {
      try self.session(modelPath: model, configuration: configuration)
    }
  }
#endif
