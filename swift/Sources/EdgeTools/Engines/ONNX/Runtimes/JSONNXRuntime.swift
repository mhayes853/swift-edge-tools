#if ONNXCore && JS
  import JavaScriptBigIntSupport
  import JavaScriptEventLoop
  import JavaScriptKit

  // MARK: - JSONNXRuntimeError

  public struct JSONNXRuntimeError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let invalidGraphOptimizationLevel =
        Self(rawValue: "invalid-graph-optimization-level")
      public static let duplicateOutputName = Self(rawValue: "duplicate-output-name")
      public static let invalidJavaScriptValue = Self(rawValue: "invalid-javascript-value")
      public static let invalidRuntimeObject = Self(rawValue: "invalid-runtime-object")
      public static let invalidTensorShape = Self(rawValue: "invalid-tensor-shape")
      public static let invalidTensorValueCount = Self(rawValue: "invalid-tensor-value-count")
      public static let javaScriptException = Self(rawValue: "javascript-exception")
      public static let tensorElementCountOverflow =
        Self(rawValue: "tensor-element-count-overflow")
      public static let unexpectedTensorElementType =
        Self(rawValue: "unexpected-tensor-element-type")
    }

    public let code: Code
    public let message: String
    public let operation: String?
    public let javaScriptStack: String?

    public init(
      code: Code,
      message: String,
      operation: String? = nil,
      javaScriptStack: String? = nil
    ) {
      self.code = code
      self.message = message
      self.operation = operation
      self.javaScriptStack = javaScriptStack
    }
  }

  // MARK: - JSONNXRuntime

  public final class JSONNXRuntime {
    private let inferenceSession: JSObject
    private let tensorConstructor: JSObject

    public init(onnxRuntime: JSObject) throws {
      let namespace = Self.namespace(from: onnxRuntime)
      guard
        let inferenceSession = namespace["InferenceSession"].object,
        inferenceSession["create"].object != nil,
        let tensorConstructor = namespace["Tensor"].object
      else {
        throw JSONNXRuntimeError(
          code: .invalidRuntimeObject,
          message: "The JavaScript object must expose InferenceSession.create and Tensor."
        )
      }
      self.inferenceSession = inferenceSession
      self.tensorConstructor = tensorConstructor
    }

    public func tensor(values: [Float], shape: [Int]) throws -> JSONNXTensor {
      try self.tensor(
        values: values,
        shape: shape,
        dtype: .float,
        javaScriptType: "float32"
      )
    }

    public func tensor(values: [Int32], shape: [Int]) throws -> JSONNXTensor {
      try self.tensor(
        values: values,
        shape: shape,
        dtype: .int32,
        javaScriptType: "int32"
      )
    }

    public func tensor(values: [Int64], shape: [Int]) throws -> JSONNXTensor {
      try self.tensor(
        values: values,
        shape: shape,
        dtype: .int64,
        javaScriptType: "int64"
      )
    }

    nonisolated(nonsending) public func session(
      model: ModelSource,
      configuration: Configuration
    ) async throws -> JSONNXSession {
      let modelValue = Self.javaScriptValue(for: model)
      let options = Self.javaScriptOptions(for: configuration)
      do {
        guard let create = self.inferenceSession["create"].object else {
          throw JSONNXRuntimeError(
            code: .invalidRuntimeObject,
            message:
              "The JavaScript ONNX Runtime namespace no longer exposes InferenceSession.create."
          )
        }
        let promiseValue = try create.throws(
          this: self.inferenceSession,
          modelValue,
          options
        )
        let session = try await Self.promiseObject(
          promiseValue,
          operation: "InferenceSession.create"
        )
        return try JSONNXSession(session: session)
      } catch let error as JSONNXRuntimeError {
        throw error
      } catch let error as JSException {
        throw Self.runtimeError(error, operation: "InferenceSession.create")
      }
    }

    private func tensor<Element: TypedArrayElement>(
      values: [Element],
      shape: [Int],
      dtype: EdgeToolsONNXDType,
      javaScriptType: String
    ) throws -> JSONNXTensor where Element.Element == Element {
      let expectedCount = try Self.elementCount(for: shape)
      guard values.count == expectedCount else {
        throw JSONNXRuntimeError(
          code: .invalidTensorValueCount,
          message: "Tensor shape \(shape) requires \(expectedCount) values, got \(values.count)."
        )
      }

      do {
        let values = JSTypedArray<Element>(values)
        let tensor = try self.tensorConstructor.throws.new(
          javaScriptType,
          values.jsObject,
          shape
        )
        return JSONNXTensor(tensor: tensor, dtype: dtype, shape: shape)
      } catch let error as JSException {
        throw Self.runtimeError(error, operation: "Tensor")
      }
    }

    private static func namespace(from object: JSObject) -> JSObject {
      if object["InferenceSession"].object != nil, object["Tensor"].object != nil {
        object
      } else {
        object["default"].object ?? object
      }
    }

    private static func javaScriptValue(for source: ModelSource) -> JSValue {
      switch source {
      case .location(let location): .string(location)
      case .bytes(let bytes): .object(JSUint8Array(bytes).jsObject)
      case .javaScriptFile(let object): .object(object)
      }
    }

    private static func javaScriptOptions(for configuration: Configuration) -> JSObject {
      let options = JSObject()
      if let level = configuration.graphOptimizationLevel {
        options["graphOptimizationLevel"] = .string(level.rawValue)
      }
      if !configuration.executionProviders.isEmpty {
        options["executionProviders"] =
          configuration.executionProviders
          .map { provider in
            guard !provider.options.isEmpty else {
              return JSValue.string(Self.executionProviderName(provider.name))
            }
            let object = JSObject()
            object["name"] = .string(Self.executionProviderName(provider.name))
            for option in provider.options {
              object[option.name] = .string(option.value)
            }
            return .object(object)
          }
          .jsValue
      }
      if !configuration.externalData.isEmpty {
        options["externalData"] =
          configuration.externalData
          .map { file in
            let object = JSObject()
            object["path"] = .string(file.path)
            object["data"] = Self.javaScriptValue(for: file.data)
            return object.jsValue
          }
          .jsValue
      }
      return options
    }

    private static func executionProviderName(_ name: String) -> String {
      switch name.lowercased() {
      case "webgpu": "webgpu"
      case "coreml": "coreml"
      case "cuda": "cuda"
      case "dml": "dml"
      default: name
      }
    }

    fileprivate static func elementCount(for shape: [Int]) throws -> Int {
      try shape.reduce(1) { count, dimension in
        guard dimension >= 0 else {
          throw JSONNXRuntimeError(
            code: .invalidTensorShape,
            message: "Tensor dimensions must not be negative: \(shape)."
          )
        }
        let result = count.multipliedReportingOverflow(by: dimension)
        guard !result.overflow else {
          throw JSONNXRuntimeError(
            code: .tensorElementCountOverflow,
            message: "Tensor shape has too many elements: \(shape)."
          )
        }
        return result.partialValue
      }
    }

    fileprivate nonisolated(nonsending) static func promiseObject(
      _ value: JSValue,
      operation: String
    ) async throws -> JSObject {
      guard let promise = value.object else {
        throw JSONNXRuntimeError(
          code: .invalidJavaScriptValue,
          message: "\(operation) did not return a Promise.",
          operation: operation
        )
      }
      do {
        let result = try await JSPromise(unsafelyWrapping: promise).value(isolation: #isolation)
        guard let object = result.object else {
          throw JSONNXRuntimeError(
            code: .invalidJavaScriptValue,
            message: "\(operation) did not resolve to an object.",
            operation: operation
          )
        }
        return object
      } catch let error as JSONNXRuntimeError {
        throw error
      } catch let error as JSException {
        throw Self.runtimeError(error, operation: operation)
      }
    }

    fileprivate static func runtimeError(
      _ error: JSException,
      operation: String
    ) -> JSONNXRuntimeError {
      let object = error.thrownValue.object
      let message =
        object?["message"].string ?? error.thrownValue.string
        ?? "JavaScript operation failed."
      return JSONNXRuntimeError(
        code: .javaScriptException,
        message: message,
        operation: operation,
        javaScriptStack: object?["stack"].string
      )
    }
  }

  extension JSONNXRuntime {
    // MARK: - ModelSource

    public enum ModelSource {
      case location(String)
      case bytes([UInt8])
      case javaScriptFile(JSObject)
    }

    // MARK: - ExternalDataFile

    public struct ExternalDataFile {
      public var path: String
      public var data: ModelSource

      public init(path: String, data: ModelSource) {
        self.path = path
        self.data = data
      }
    }

    // MARK: - Configuration

    public struct Configuration {
      public var graphOptimizationLevel: EdgeToolsONNXGraphOptimizationLevel?
      public var executionProviders: [EdgeToolsONNXExecutionProvider]
      public var externalData: [ExternalDataFile]

      public init(
        graphOptimizationLevel: EdgeToolsONNXGraphOptimizationLevel? = nil,
        executionProviders: [EdgeToolsONNXExecutionProvider] = [],
        externalData: [ExternalDataFile] = []
      ) {
        self.graphOptimizationLevel = graphOptimizationLevel
        self.executionProviders = executionProviders
        self.externalData = externalData
      }
    }
  }

  // MARK: - JSONNXSession

  public final class JSONNXSession {
    private let session: JSObject

    public let inputNames: [String]
    public let outputNames: [String]

    fileprivate init(session: JSObject) throws {
      self.inputNames = try Self.names(session["inputNames"], property: "inputNames")
      self.outputNames = try Self.names(session["outputNames"], property: "outputNames")
      self.session = session
    }

    deinit {
      guard let release = self.session["release"].object else { return }
      guard let value = try? release.throws(this: self.session) else { return }
      guard let promise = value.object else { return }
      JSPromise(unsafelyWrapping: promise).catch { _ in .undefined }
    }

    nonisolated(nonsending) public func run(
      inputs: [String: JSONNXTensor],
      outputNames: [String]
    ) async throws -> [String: JSONNXTensor] {
      guard Set(outputNames).count == outputNames.count else {
        throw JSONNXRuntimeError(
          code: .duplicateOutputName,
          message: "Output names must be unique."
        )
      }
      let feeds = JSObject()
      for (name, tensor) in inputs {
        feeds[name] = tensor.javaScriptObject.jsValue
      }

      do {
        guard let run = self.session["run"].object else {
          throw JSONNXRuntimeError(
            code: .invalidJavaScriptValue,
            message: "The JavaScript session does not expose run()."
          )
        }
        let promiseValue = try run.throws(this: self.session, feeds, outputNames)
        let outputs = try await JSONNXRuntime.promiseObject(
          promiseValue,
          operation: "InferenceSession.run"
        )
        return try Dictionary(
          uniqueKeysWithValues: outputNames.map { name in
            guard let tensor = outputs[name].object else {
              throw JSONNXRuntimeError(
                code: .invalidJavaScriptValue,
                message: "InferenceSession.run did not return output \(name).",
                operation: "InferenceSession.run"
              )
            }
            return (name, try JSONNXTensor(tensor: tensor))
          }
        )
      } catch let error as JSONNXRuntimeError {
        throw error
      } catch let error as JSException {
        throw JSONNXRuntime.runtimeError(error, operation: "InferenceSession.run")
      }
    }

    private static func names(_ value: JSValue, property: String) throws -> [String] {
      guard let array = value.array else {
        throw JSONNXRuntimeError(
          code: .invalidJavaScriptValue,
          message: "The JavaScript session's \(property) property must be an array."
        )
      }
      return try array.map { value in
        guard let name = value.string else {
          throw JSONNXRuntimeError(
            code: .invalidJavaScriptValue,
            message: "The JavaScript session's \(property) property must contain strings."
          )
        }
        return name
      }
    }
  }

  // MARK: - JSONNXTensor

  public final class JSONNXTensor {
    public typealias DType = EdgeToolsONNXDType

    fileprivate let javaScriptObject: JSObject

    public let dtype: DType
    public let shape: [Int]

    fileprivate init(tensor: JSObject, dtype: DType, shape: [Int]) {
      self.javaScriptObject = tensor
      self.dtype = dtype
      self.shape = shape
    }

    fileprivate convenience init(tensor: JSObject) throws {
      guard let type = tensor["type"].string else {
        throw JSONNXRuntimeError(
          code: .invalidJavaScriptValue,
          message: "A JavaScript ONNX tensor must expose a string type."
        )
      }
      guard let dimensions = tensor["dims"].array else {
        throw JSONNXRuntimeError(
          code: .invalidJavaScriptValue,
          message: "A JavaScript ONNX tensor must expose a dims array."
        )
      }
      let shape = try dimensions.map { value in
        guard let dimension = value.number, let dimension = Int(exactly: dimension) else {
          throw JSONNXRuntimeError(
            code: .invalidJavaScriptValue,
            message: "A JavaScript ONNX tensor's dimensions must be integers."
          )
        }
        return dimension
      }
      let dtype: DType
      switch type {
      case "float32": dtype = .float
      case "int32": dtype = .int32
      case "int64": dtype = .int64
      default: dtype = .undefined
      }
      self.init(tensor: tensor, dtype: dtype, shape: shape)
    }

    deinit {
      guard let dispose = self.javaScriptObject["dispose"].object else { return }
      _ = try? dispose.throws(this: self.javaScriptObject)
    }

    nonisolated(nonsending) public func floatValues() async throws -> [Float] {
      try await self.values(
        dtype: .float,
        type: Float.self,
        operation: "Tensor.getData<float32>"
      )
    }

    nonisolated(nonsending) public func int32Values() async throws -> [Int32] {
      try await self.values(
        dtype: .int32,
        type: Int32.self,
        operation: "Tensor.getData<int32>"
      )
    }

    nonisolated(nonsending) public func int64Values() async throws -> [Int64] {
      try await self.values(
        dtype: .int64,
        type: Int64.self,
        operation: "Tensor.getData<int64>"
      )
    }

    private nonisolated(nonsending) func values<Element: TypedArrayElement>(
      dtype: DType,
      type: Element.Type,
      operation: String
    ) async throws -> [Element] where Element.Element == Element {
      guard self.dtype == dtype else {
        throw JSONNXRuntimeError(
          code: .unexpectedTensorElementType,
          message: "Expected tensor element type \(dtype.rawValue), got \(self.dtype.rawValue)."
        )
      }

      do {
        guard let getData = self.javaScriptObject["getData"].object else {
          throw JSONNXRuntimeError(
            code: .invalidJavaScriptValue,
            message: "A JavaScript ONNX tensor must expose getData()."
          )
        }
        let promiseValue = try getData.throws(this: self.javaScriptObject)
        let object = try await JSONNXRuntime.promiseObject(promiseValue, operation: operation)
        guard object.isInstanceOf(Element.typedArrayClass) else {
          throw JSONNXRuntimeError(
            code: .invalidJavaScriptValue,
            message: "\(operation) returned an incompatible typed array.",
            operation: operation
          )
        }
        let values = JSTypedArray<Element>(unsafelyWrapping: object)
        return values.withUnsafeBytes { Array($0) }
      } catch let error as JSONNXRuntimeError {
        throw error
      } catch let error as JSException {
        throw JSONNXRuntime.runtimeError(error, operation: operation)
      }
    }
  }

  // MARK: - Protocol Conformances

  extension JSONNXTensor: EdgeToolsONNXTensor {}

  extension JSONNXSession: EdgeToolsONNXSession {
    public typealias Tensor = JSONNXTensor
  }

  extension JSONNXRuntime: EdgeToolsONNXRuntime {
    public typealias Session = JSONNXSession
    public typealias Tensor = JSONNXTensor
  }
#endif
