#if ONNXCore && canImport(COnnxRuntime)
  import COnnxRuntime

  #if Foundation
    import Foundation
  #endif

  #if System
    import SystemPackage
  #endif

  // MARK: - CONNXRuntimeError

  public struct CONNXRuntimeError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let integerConversionFailure = Self(rawValue: "integer-conversion-failure")
      public static let invalidModelSignature = Self(rawValue: "invalid-model-signature")
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
    public static let apiVersion = UInt32(ORT_API_VERSION)

    let api: UnsafePointer<OrtApi>
    let allocator: UnsafeMutablePointer<OrtAllocator>
    private let environment: OpaquePointer

    public init(api: OpaquePointer, configuration: Configuration = Configuration()) throws {
      let api = UnsafePointer<OrtApi>(api)
      self.api = api

      var environment: OpaquePointer?
      try configuration.logIdentifier.withCString { identifier in
        try Self.check(
          api: api,
          status: api.pointee.CreateEnv(
            ORT_LOGGING_LEVEL_WARNING,
            identifier,
            &environment
          )
        )
      }
      let environmentPointer = environment.unsafelyUnwrapped
      self.environment = environmentPointer

      var allocator: UnsafeMutablePointer<OrtAllocator>?
      do {
        try Self.check(
          api: api,
          status: api.pointee.GetAllocatorWithDefaultOptions(&allocator)
        )
        self.allocator = allocator.unsafelyUnwrapped
      } catch {
        api.pointee.ReleaseEnv(environmentPointer)
        throw error
      }
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

    public func session(
      modelPath: String,
      configuration: SessionConfiguration = SessionConfiguration()
    ) throws -> CONNXRuntimeSession {
      try modelPath.withCString {
        try self.session(modelPath: $0, configuration: configuration)
      }
    }

    #if Foundation
      public func session(
        modelURL: URL,
        configuration: SessionConfiguration = SessionConfiguration()
      ) throws -> CONNXRuntimeSession {
        try self.session(modelPath: modelURL.path(), configuration: configuration)
      }
    #endif

    #if System
      public func session(
        modelPath: FilePath,
        configuration: SessionConfiguration = SessionConfiguration()
      ) throws -> CONNXRuntimeSession {
        try self.session(modelPath: modelPath.string, configuration: configuration)
      }
    #endif

    private func session(
      modelPath: UnsafePointer<CChar>,
      configuration: SessionConfiguration
    ) throws -> CONNXRuntimeSession {
      var options: OpaquePointer?
      try Self.check(api: self.api, status: self.api.pointee.CreateSessionOptions(&options))
      let optionsPointer = options.unsafelyUnwrapped
      defer { self.api.pointee.ReleaseSessionOptions(optionsPointer) }

      try Self.check(
        api: self.api,
        status: self.api.pointee.SetSessionGraphOptimizationLevel(optionsPointer, ORT_ENABLE_ALL)
      )
      for provider in configuration.executionProviders {
        try self.append(provider: provider, to: optionsPointer)
      }

      var session: OpaquePointer?
      try Self.check(
        api: self.api,
        status: self.api.pointee.CreateSession(
          self.environment,
          modelPath,
          optionsPointer,
          &session
        )
      )
      return try CONNXRuntimeSession.make(runtime: self, session: session.unsafelyUnwrapped)
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
        values: [Float](repeating: value, count: try self.elementCount(for: shape)),
        shape: shape
      )
    }

    private func tensor<Element>(
      values: [Element],
      shape: [Int],
      elementType: ONNXTensorElementDataType
    ) throws -> CONNXRuntimeTensor {
      let dimensions = shape.map(Int64.init)
      let expectedCount = try self.elementCount(for: shape)
      guard values.count == expectedCount else {
        throw CONNXRuntimeError(
          code: .invalidTensorValueCount,
          message: "Tensor shape \(shape) requires \(expectedCount) values, got \(values.count)."
        )
      }

      var tensor: OpaquePointer?
      try dimensions.withUnsafeBufferPointer { dimensions in
        try Self.check(
          api: self.api,
          status: self.api.pointee.CreateTensorAsOrtValue(
            self.allocator,
            dimensions.baseAddress,
            dimensions.count,
            elementType,
            &tensor
          )
        )
      }
      let tensorPointer = tensor.unsafelyUnwrapped

      do {
        var bytes: UnsafeMutableRawPointer?
        try Self.check(
          api: self.api,
          status: self.api.pointee.GetTensorMutableData(tensorPointer, &bytes)
        )
        let bytesPointer = bytes.unsafelyUnwrapped
        values.withUnsafeBufferPointer { source in
          guard let sourceAddress = source.baseAddress else { return }
          bytesPointer.copyMemory(
            from: sourceAddress,
            byteCount: values.count * MemoryLayout<Element>.stride
          )
        }
        return CONNXRuntimeTensor(
          runtime: self,
          tensor: tensorPointer,
          dtype: CONNXRuntimeTensor.DType(rawValue: Int(elementType.rawValue)),
          shape: shape
        )
      } catch {
        self.api.pointee.ReleaseValue(tensorPointer)
        throw error
      }
    }

    private func elementCount(for shape: [Int]) throws -> Int {
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
      try withCopiedCStringPointerBuffer(keys) { keyPointers in
        try withCopiedCStringPointerBuffer(values) { valuePointers in
          try provider.name.withCString { providerName in
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
    // MARK: - Configuration

    public struct Configuration: Hashable, Sendable {
      public var logIdentifier: String

      public init(logIdentifier: String = "swift-edge-tools") {
        self.logIdentifier = logIdentifier
      }
    }

    // MARK: - SessionConfiguration

    public struct SessionConfiguration: Hashable, Sendable {
      public var executionProviders: [ExecutionProvider]

      public init(executionProviders: [ExecutionProvider] = []) {
        self.executionProviders = executionProviders
      }
    }

    // MARK: - ExecutionProviderOption

    public struct ExecutionProviderOption: Hashable, Sendable {
      public var name: String
      public var value: String

      public init(name: String, value: String) {
        self.name = name
        self.value = value
      }
    }

    // MARK: - ExecutionProvider

    public struct ExecutionProvider: Hashable, Sendable {
      public var name: String
      public var options: [ExecutionProviderOption]

      public init(name: String, options: [ExecutionProviderOption] = []) {
        self.name = name
        self.options = options
      }

      public static var webGPU: Self { Self(name: "WebGPU") }

      public static func coreML(
        computeUnits: String,
        modelFormat: String = "MLProgram",
        requireStaticInputShapes: Bool = true,
        enableOnSubgraphs: Bool = true
      ) -> Self {
        Self(
          name: "CoreML",
          options: [
            ExecutionProviderOption(name: "MLComputeUnits", value: computeUnits),
            ExecutionProviderOption(name: "ModelFormat", value: modelFormat),
            ExecutionProviderOption(
              name: "RequireStaticInputShapes",
              value: requireStaticInputShapes ? "1" : "0"
            ),
            ExecutionProviderOption(
              name: "EnableOnSubgraphs",
              value: enableOnSubgraphs ? "1" : "0"
            )
          ]
        )
      }
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

    static func make(runtime: CONNXRuntime, session: OpaquePointer) throws -> CONNXRuntimeSession {
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
      let inputNames = Array(inputs.keys)
      let inputValues = inputNames.map { inputs[$0].map(\.tensor) }
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

      let outputPointers = outputValues.map(\.unsafelyUnwrapped)
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
        uniqueKeysWithValues: zip(outputNames, zip(outputPointers, metadata)).map { pair in
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

    func validateSignature(inputNames: [String], outputNames: [String]) throws {
      guard self.inputNames == inputNames, self.outputNames == outputNames else {
        throw CONNXRuntimeError(
          code: .invalidModelSignature,
          message: "Invalid ONNX model signature. Expected inputs \(inputNames) and outputs "
            + "\(outputNames), got inputs \(self.inputNames) and outputs \(self.outputNames)."
        )
      }
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
      return try (0..<count).map { index in
        var name: UnsafeMutablePointer<CChar>?
        try CONNXRuntime.check(
          api: runtime.api,
          status: getName(session, index, runtime.allocator, &name)
        )
        let namePointer = name.unsafelyUnwrapped
        defer { runtime.allocator.pointee.Free(runtime.allocator, namePointer) }
        return String(cString: namePointer)
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

    init(
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

    static func metadata(
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

    func floatValues(count: Int) throws -> [Float] {
      let values = try self.floatValues()
      guard values.count == count else {
        throw CONNXRuntimeError(
          code: .invalidTensorValueCount,
          message: "Expected a Float32 tensor with \(count) values, got \(values.count)."
        )
      }
      return values
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
      let count = try self.withShapeInfo { shapeInfo in
        var elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        try CONNXRuntime.check(
          api: self.runtime.api,
          status: self.runtime.api.pointee.GetTensorElementType(shapeInfo, &elementType)
        )
        var elementCount = 0
        try CONNXRuntime.check(
          api: self.runtime.api,
          status: self.runtime.api.pointee.GetTensorShapeElementCount(shapeInfo, &elementCount)
        )
        guard elementType == expectedElementType else {
          throw CONNXRuntimeError(
            code: .unexpectedTensorElementType,
            message:
              "Expected tensor element type \(expectedElementType.rawValue), got \(elementType.rawValue)."
          )
        }
        return elementCount
      }

      var bytes: UnsafeMutableRawPointer?
      try CONNXRuntime.check(
        api: self.runtime.api,
        status: self.runtime.api.pointee.GetTensorMutableData(self.tensor, &bytes)
      )
      let bytesPointer = bytes.unsafelyUnwrapped
      return Array(
        UnsafeBufferPointer(
          start: bytesPointer.assumingMemoryBound(to: Element.self),
          count: count
        )
      )
    }

    private func withShapeInfo<Result>(
      _ body: (OpaquePointer) throws -> Result
    ) throws -> Result {
      try Self.withShapeInfo(runtime: self.runtime, tensor: self.tensor, body)
    }

    private static func withShapeInfo<Result>(
      runtime: CONNXRuntime,
      tensor: OpaquePointer,
      _ body: (OpaquePointer) throws -> Result
    ) throws -> Result {
      var shapeInfo: OpaquePointer?
      try CONNXRuntime.check(
        api: runtime.api,
        status: runtime.api.pointee.GetTensorTypeAndShape(tensor, &shapeInfo)
      )
      let shapeInfoPointer = shapeInfo.unsafelyUnwrapped
      defer { runtime.api.pointee.ReleaseTensorTypeAndShapeInfo(shapeInfoPointer) }
      return try body(shapeInfoPointer)
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
      configuration: SessionConfiguration
    ) throws -> CONNXRuntimeSession {
      try self.session(modelPath: model, configuration: configuration)
    }
  }
#endif
