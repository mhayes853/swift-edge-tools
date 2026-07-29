#if ONNXCore
  // MARK: - EdgeToolsONNXDType

  public struct EdgeToolsONNXDType: RawRepresentable, Hashable, Sendable {
    private static let names = [
      1: "float32",
      2: "uint8",
      3: "int8",
      4: "uint16",
      5: "int16",
      6: "int32",
      7: "int64",
      8: "string",
      9: "bool",
      10: "float16",
      11: "float64",
      12: "uint32",
      13: "uint64",
      14: "complex64",
      15: "complex128",
      16: "bfloat16",
      17: "float8e4m3fn",
      18: "float8e4m3fnuz",
      19: "float8e5m2",
      20: "float8e5m2fnuz",
      21: "uint4",
      22: "int4",
      23: "float4e2m1",
      24: "uint2",
      25: "int2",
      26: "float8e8m0"
    ]

    public let rawValue: Int

    public var name: String? {
      Self.names[self.rawValue]
    }

    public init(rawValue: Int) {
      self.rawValue = rawValue
    }

    public init?(name: String) {
      guard let rawValue = Self.names.first(where: { $0.value == name })?.key else { return nil }
      self.init(rawValue: rawValue)
    }

    public static let undefined = Self(rawValue: 0)
    public static let float = Self(rawValue: 1)
    public static let uint8 = Self(rawValue: 2)
    public static let int8 = Self(rawValue: 3)
    public static let uint16 = Self(rawValue: 4)
    public static let int16 = Self(rawValue: 5)
    public static let int32 = Self(rawValue: 6)
    public static let int64 = Self(rawValue: 7)
    public static let string = Self(rawValue: 8)
    public static let bool = Self(rawValue: 9)
    public static let float16 = Self(rawValue: 10)
    public static let double = Self(rawValue: 11)
    public static let uint32 = Self(rawValue: 12)
    public static let uint64 = Self(rawValue: 13)
    public static let complex64 = Self(rawValue: 14)
    public static let complex128 = Self(rawValue: 15)
    public static let bfloat16 = Self(rawValue: 16)
    public static let float8E4M3FN = Self(rawValue: 17)
    public static let float8E4M3FNUZ = Self(rawValue: 18)
    public static let float8E5M2 = Self(rawValue: 19)
    public static let float8E5M2FNUZ = Self(rawValue: 20)
    public static let uint4 = Self(rawValue: 21)
    public static let int4 = Self(rawValue: 22)
    public static let float4E2M1 = Self(rawValue: 23)
    public static let uint2 = Self(rawValue: 24)
    public static let int2 = Self(rawValue: 25)
    public static let float8E8M0 = Self(rawValue: 26)
  }

  // MARK: - EdgeToolsONNXElement

  public protocol EdgeToolsONNXElement: BitwiseCopyable, Sendable {
    static var onnxDType: EdgeToolsONNXDType { get }
  }

  extension Float: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .float }
  }

  extension Double: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .double }
  }

  extension Float16: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .float16 }
  }

  extension Bool: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .bool }
  }

  extension UInt8: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .uint8 }
  }

  extension Int8: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .int8 }
  }

  extension UInt16: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .uint16 }
  }

  extension Int16: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .int16 }
  }

  extension UInt32: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .uint32 }
  }

  extension Int32: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .int32 }
  }

  extension UInt64: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .uint64 }
  }

  extension Int64: EdgeToolsONNXElement {
    public static var onnxDType: EdgeToolsONNXDType { .int64 }
  }

  // MARK: - EdgeToolsONNXRuntimeError

  public struct EdgeToolsONNXRuntimeError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let duplicateOutputName = Self(rawValue: "duplicate-output-name")
      public static let invalidTensorShape = Self(rawValue: "invalid-tensor-shape")
      public static let invalidTensorValueCount = Self(rawValue: "invalid-tensor-value-count")
      public static let tensorElementCountOverflow = Self(rawValue: "tensor-element-count-overflow")
      public static let unexpectedTensorElementType = Self(rawValue: "unexpected-tensor-element-type")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  // MARK: - Validation Helpers

  func edgeToolsONNXElementCount(for shape: [Int]) throws -> Int {
    try shape.reduce(1) { count, dimension in
      guard dimension >= 0 else {
        throw EdgeToolsONNXRuntimeError(
          code: .invalidTensorShape,
          message: "Tensor dimensions must not be negative: \(shape)."
        )
      }
      let result = count.multipliedReportingOverflow(by: dimension)
      guard !result.overflow else {
        throw EdgeToolsONNXRuntimeError(
          code: .tensorElementCountOverflow,
          message: "Tensor shape has too many elements: \(shape)."
        )
      }
      return result.partialValue
    }
  }

  func edgeToolsONNXValidateValueCount(_ count: Int, shape: [Int]) throws {
    let expectedCount = try edgeToolsONNXElementCount(for: shape)
    guard count == expectedCount else {
      throw EdgeToolsONNXRuntimeError(
        code: .invalidTensorValueCount,
        message: "Tensor shape \(shape) requires \(expectedCount) values, got \(count)."
      )
    }
  }

  func edgeToolsONNXValidateOutputNames(_ names: [String]) throws {
    guard Set(names).count == names.count else {
      throw EdgeToolsONNXRuntimeError(
        code: .duplicateOutputName,
        message: "Output names must be unique."
      )
    }
  }

  // MARK: - EdgeToolsONNXTensorView

  @_addressableForDependencies
  public struct EdgeToolsONNXTensorView<Element: EdgeToolsONNXElement>: ~Copyable {
    private let baseAddress: UnsafeMutablePointer<Element>?

    public let count: Int

    public init<Values: Collection>(copying values: Values)
    where Values.Element == Element {
      let values = ContiguousArray(values)
      let baseAddress =
        values.isEmpty ? nil : UnsafeMutablePointer<Element>.allocate(capacity: values.count)
      if let baseAddress {
        values.withUnsafeBufferPointer {
          baseAddress.initialize(from: $0.baseAddress!, count: $0.count)
        }
      }
      self.baseAddress = baseAddress
      self.count = values.count
    }

    public init(owning baseAddress: consuming UnsafeMutablePointer<Element>?, count: Int) {
      self.baseAddress = baseAddress
      self.count = count
    }

    deinit {
      guard let baseAddress = self.baseAddress else { return }
      baseAddress.deinitialize(count: self.count)
      baseAddress.deallocate()
    }

    public subscript(index: Int) -> Element {
      borrowing get {
        precondition(index >= 0 && index < self.count, "Tensor view index is out of bounds.")
        return self.baseAddress.unsafelyUnwrapped[index]
      }
      set {
        precondition(index >= 0 && index < self.count, "Tensor view index is out of bounds.")
        self.baseAddress.unsafelyUnwrapped[index] = newValue
      }
    }

    public func withUnsafeBufferPointer<Result>(
      _ body: (UnsafeBufferPointer<Element>) throws -> Result
    ) rethrows -> Result {
      try body(UnsafeBufferPointer(start: self.baseAddress, count: self.count))
    }

    public mutating func withUnsafeMutableBufferPointer<Result>(
      _ body: (UnsafeMutableBufferPointer<Element>) throws -> Result
    ) rethrows -> Result {
      try body(UnsafeMutableBufferPointer(start: self.baseAddress, count: self.count))
    }

    public var span: Span<Element> {
      @_lifetime(borrow self)
      borrowing get {
        guard self.baseAddress != nil else { return Span() }
        return Span(_unsafeStart: self.baseAddress!, count: self.count)
      }
    }

    public var mutableSpan: MutableSpan<Element> {
      @_lifetime(&self)
      mutating get {
        guard self.baseAddress != nil else { return MutableSpan() }
        return MutableSpan(_unsafeStart: self.baseAddress!, count: self.count)
      }
    }
  }

  // MARK: - EdgeToolsONNXLogitsProcessor

  public protocol EdgeToolsONNXLogitsProcessor {
    func prompt(_ prompt: [EdgeToolsToken.ID])

    nonisolated(nonsending) func process(
      logits: inout EdgeToolsONNXTensorView<Float>
    ) async throws

    func didSample(token: EdgeToolsToken)
  }

  // MARK: - EdgeToolsONNXGraphOptimizationLevel

  public struct EdgeToolsONNXGraphOptimizationLevel: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public static let disabled = Self(rawValue: "disabled")
    public static let basic = Self(rawValue: "basic")
    public static let extended = Self(rawValue: "extended")
    public static let layout = Self(rawValue: "layout")
    public static let all = Self(rawValue: "all")
  }

  // MARK: - Runtime Protocols

  public protocol EdgeToolsONNXTensor {
    var dtype: EdgeToolsONNXDType { get }
    var shape: [Int] { get }

    nonisolated(nonsending) func view<Element: EdgeToolsONNXElement>(
      as type: Element.Type
    ) async throws -> EdgeToolsONNXTensorView<Element>
  }

  extension EdgeToolsONNXTensor {
    public nonisolated(nonsending) func array<Element: EdgeToolsONNXElement>(
      as type: Element.Type
    ) async throws -> [Element] {
      let view = try await self.view(as: type)
      return (0..<view.count).map { view[$0] }
    }
  }

  public protocol EdgeToolsONNXSession {
    associatedtype Tensor: EdgeToolsONNXTensor

    var inputNames: [String] { get }
    var outputNames: [String] { get }

    nonisolated(nonsending) func run(
      inputs: [String: Tensor],
      outputNames: [String]
    ) async throws -> [String: Tensor]
  }

  public protocol EdgeToolsONNXRuntime: SendableMetatype {
    associatedtype ModelSource
    associatedtype SessionConfiguration
    associatedtype Session: EdgeToolsONNXSession where Session.Tensor == Tensor
    associatedtype Tensor: EdgeToolsONNXTensor

    nonisolated(nonsending) func session(
      model: ModelSource,
      configuration: SessionConfiguration
    ) async throws -> Session

    func tensor<Values: Sequence>(
      values: Values,
      shape: [Int]
    ) throws -> Tensor where Values.Element: EdgeToolsONNXElement
  }
#endif
