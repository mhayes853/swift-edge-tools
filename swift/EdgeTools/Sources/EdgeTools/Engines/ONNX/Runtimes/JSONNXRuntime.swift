#if ONNXCore && JS && canImport(JavaScriptKit)
  import JavaScriptEventLoop
  import JavaScriptKit

  // MARK: - JSONNXRuntimeError

  public struct JSONNXRuntimeError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let invalidJSValue = Self(rawValue: "invalid-js-value")
      public static let invalidRuntimeObject = Self(rawValue: "invalid-runtime-object")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  // MARK: - JSONNXRuntime

  public final class JSONNXRuntime {
    public typealias Configuration = JSObject

    @nonexhaustive
    public enum ModelSource {
      case location(String)
      case bytes([UInt8])
      case object(JSObject)
    }

    public let object: JSObject

    private let inferenceSession: JSObject
    private let tensorConstructor: JSObject

    public init(object: JSObject) throws {
      let object = object["default"].object ?? object
      guard
        let inferenceSession = object["InferenceSession"].object,
        inferenceSession["create"].object != nil,
        let tensorConstructor = object["Tensor"].object
      else {
        throw JSONNXRuntimeError(
          code: .invalidRuntimeObject,
          message: "The JS object must expose InferenceSession.create and Tensor."
        )
      }
      self.object = object
      self.inferenceSession = inferenceSession
      self.tensorConstructor = tensorConstructor
    }

    public convenience init(onnxRuntime: JSObject) throws {
      try self.init(object: onnxRuntime)
    }

    public func tensor<Values: Sequence>(
      values: Values,
      shape: [Int]
    ) throws -> JSONNXTensor where Values.Element: ONNXElement {
      let values = ContiguousArray(values)
      try validateONNXTensorValueCount(values.count, shape: shape)
      let dtype = Values.Element.onnxDType
      let bytes = values.withUnsafeBytes {
        JSUint8Array(buffer: $0.bindMemory(to: UInt8.self)).jsObject
      }
      guard let buffer = bytes["buffer"].object else {
        throw JSONNXRuntimeError(
          code: .invalidJSValue,
          message: "A JS typed array must expose an ArrayBuffer."
        )
      }
      let object = try self.tensorConstructor.throws.new(
        try Self.jsType(for: dtype),
        try Self.jsTypedArrayConstructor(for: dtype).new(buffer),
        shape
      )
      return JSONNXTensor(object: object, dtype: dtype, shape: shape)
    }

    nonisolated(nonsending) public func session(
      model: ModelSource,
      configuration: JSObject
    ) async throws -> JSONNXSession {
      guard let create = self.inferenceSession["create"].object else {
        throw JSONNXRuntimeError(
          code: .invalidRuntimeObject,
          message: "The JS runtime no longer exposes InferenceSession.create."
        )
      }
      let value = try create.throws(
        this: self.inferenceSession,
        Self.jsValue(for: model),
        configuration
      )
      return try JSONNXSession(object: try await Self.promiseObject(value))
    }

    nonisolated(nonsending) static func promiseObject(
      _ value: JSValue
    ) async throws -> JSObject {
      guard let promise = value.object else {
        throw JSONNXRuntimeError(
          code: .invalidJSValue,
          message: "The JS operation did not return a Promise."
        )
      }
      let value = try await JSPromise(unsafelyWrapping: promise).value(isolation: #isolation)
      guard let object = value.object else {
        throw JSONNXRuntimeError(
          code: .invalidJSValue,
          message: "The JS Promise did not resolve to an object."
        )
      }
      return object
    }

    static func jsType(for dtype: ONNXDType) throws -> String {
      guard let name = dtype.name else {
        throw ONNXRuntimeError(
          code: .unexpectedTensorElementType,
          message: "ONNX tensor element type \(dtype.rawValue) has no JS representation."
        )
      }
      return name
    }

    static func jsTypedArrayConstructor(for dtype: ONNXDType) throws -> JSObject {
      let name: String
      switch dtype {
      case .bool: name = "Uint8Array"
      case .float16: name = "Uint16Array"
      case .int64: name = "BigInt64Array"
      case .uint64: name = "BigUint64Array"
      default:
        let type = try Self.jsType(for: dtype)
        name = type.prefix(1).uppercased() + type.dropFirst() + "Array"
      }
      guard let constructor = JSObject.global[name].object else {
        throw ONNXRuntimeError(
          code: .unexpectedTensorElementType,
          message: "JS does not provide \(name)."
        )
      }
      return constructor
    }

    private static func jsValue(for source: ModelSource) -> JSValue {
      switch source {
      case .location(let location): .string(location)
      case .bytes(let bytes): .object(JSUint8Array(bytes).jsObject)
      case .object(let object): .object(object)
      }
    }
  }

  // MARK: - JSONNXSession

  public final class JSONNXSession {
    public let object: JSObject
    public let inputNames: [String]
    public let outputNames: [String]

    init(object: JSObject) throws {
      self.object = object
      self.inputNames = try Self.names(object["inputNames"])
      self.outputNames = try Self.names(object["outputNames"])
    }

    deinit {
      guard let release = self.object["release"].object else { return }
      guard let value = try? release.throws(this: self.object).object else { return }
      JSPromise(unsafelyWrapping: value).catch { _ in .undefined }
    }

    nonisolated(nonsending) public func run(
      inputs: [String: JSONNXTensor],
      outputNames: [String]
    ) async throws -> [String: JSONNXTensor] {
      try validateONNXOutputNames(outputNames)
      let feeds = JSObject()
      for (name, tensor) in inputs {
        feeds[name] = tensor.object.jsValue
      }
      guard let run = self.object["run"].object else {
        throw JSONNXRuntimeError(
          code: .invalidJSValue,
          message: "The JS session does not expose run()."
        )
      }
      let outputs = try await JSONNXRuntime.promiseObject(
        try run.throws(this: self.object, feeds, outputNames)
      )
      return try Dictionary(
        uniqueKeysWithValues: outputNames.map { name in
          guard let object = outputs[name].object else {
            throw JSONNXRuntimeError(
              code: .invalidJSValue,
              message: "The JS session did not return output \(name)."
            )
          }
          return (name, try JSONNXTensor(object: object))
        }
      )
    }

    private static func names(_ value: JSValue) throws -> [String] {
      guard let values = value.array else {
        throw JSONNXRuntimeError(code: .invalidJSValue, message: "Expected an array of names.")
      }
      return try values.map {
        guard let name = $0.string else {
          throw JSONNXRuntimeError(code: .invalidJSValue, message: "Expected a string name.")
        }
        return name
      }
    }
  }

  // MARK: - JSONNXTensor

  public final class JSONNXTensor {
    public let object: JSObject
    public let dtype: ONNXDType
    public let shape: [Int]

    init(object: JSObject, dtype: ONNXDType, shape: [Int]) {
      self.object = object
      self.dtype = dtype
      self.shape = shape
    }

    convenience init(object: JSObject) throws {
      guard
        let name = object["type"].string,
        let dtype = ONNXDType(name: name),
        let dimensions = object["dims"].array
      else {
        throw JSONNXRuntimeError(code: .invalidJSValue, message: "Invalid JS tensor metadata.")
      }
      let shape = try dimensions.map {
        guard let value = $0.number, let dimension = Int(exactly: value) else {
          throw JSONNXRuntimeError(code: .invalidJSValue, message: "Invalid tensor dimension.")
        }
        return dimension
      }
      self.init(object: object, dtype: dtype, shape: shape)
    }

    deinit {
      guard let dispose = self.object["dispose"].object else { return }
      _ = try? dispose.throws(this: self.object)
    }

    nonisolated(nonsending) public func withUnsafeMutableBufferPointer<
      Element: ONNXElement,
      Result
    >(
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
      guard let getData = self.object["getData"].object else {
        throw JSONNXRuntimeError(code: .invalidJSValue, message: "The JS tensor has no getData().")
      }
      let object = try await JSONNXRuntime.promiseObject(try getData.throws(this: self.object))
      guard object.isInstanceOf(try JSONNXRuntime.jsTypedArrayConstructor(for: self.dtype)) else {
        throw JSONNXRuntimeError(code: .invalidJSValue, message: "Unexpected JS typed array.")
      }

      let count = try onnxTensorElementCount(for: self.shape)
      let byteCount = count * MemoryLayout<Element>.stride
      guard
        let buffer = object["buffer"].object,
        let byteOffset = object["byteOffset"].number,
        object["byteLength"].number.map(Int.init) == byteCount
      else {
        throw JSONNXRuntimeError(code: .invalidJSValue, message: "Invalid JS tensor buffer.")
      }
      let baseAddress = count == 0 ? nil : UnsafeMutablePointer<Element>.allocate(capacity: count)
      defer { baseAddress?.deallocate() }
      if let baseAddress {
        let bytes = UInt8.typedArrayClass.new(buffer, byteOffset, byteCount)
        JSTypedArray<UInt8>(unsafelyWrapping: bytes)
          .copyMemory(
            to: UnsafeMutableRawBufferPointer(start: baseAddress, count: byteCount)
              .bindMemory(to: UInt8.self)
          )
      }
      return try await body(UnsafeMutableBufferPointer(start: baseAddress, count: count))
    }
  }

  extension JSONNXTensor: ONNXTensor {}

  extension JSONNXSession: ONNXSession {
    public typealias Tensor = JSONNXTensor
  }

  extension JSONNXRuntime: ONNXRuntime {
    public typealias Session = JSONNXSession
    public typealias Tensor = JSONNXTensor
  }
#endif
