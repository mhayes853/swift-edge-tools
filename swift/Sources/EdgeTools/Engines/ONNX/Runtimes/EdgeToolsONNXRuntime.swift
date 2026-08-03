#if ONNXCore
  // MARK: - ONNXDType

  public struct ONNXDType: RawRepresentable, Hashable, Sendable {
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

  // MARK: - ONNXElement

  public protocol ONNXElement: BitwiseCopyable, Sendable {
    static var onnxDType: ONNXDType { get }
  }

  extension Float: ONNXElement {
    public static var onnxDType: ONNXDType { .float }
  }

  extension Double: ONNXElement {
    public static var onnxDType: ONNXDType { .double }
  }

  extension Float16: ONNXElement {
    public static var onnxDType: ONNXDType { .float16 }
  }

  extension Bool: ONNXElement {
    public static var onnxDType: ONNXDType { .bool }
  }

  extension UInt8: ONNXElement {
    public static var onnxDType: ONNXDType { .uint8 }
  }

  extension Int8: ONNXElement {
    public static var onnxDType: ONNXDType { .int8 }
  }

  extension UInt16: ONNXElement {
    public static var onnxDType: ONNXDType { .uint16 }
  }

  extension Int16: ONNXElement {
    public static var onnxDType: ONNXDType { .int16 }
  }

  extension UInt32: ONNXElement {
    public static var onnxDType: ONNXDType { .uint32 }
  }

  extension Int32: ONNXElement {
    public static var onnxDType: ONNXDType { .int32 }
  }

  extension UInt64: ONNXElement {
    public static var onnxDType: ONNXDType { .uint64 }
  }

  extension Int64: ONNXElement {
    public static var onnxDType: ONNXDType { .int64 }
  }

  // MARK: - ONNXRuntimeError

  public struct ONNXRuntimeError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let duplicateOutputName = Self(rawValue: "duplicate-output-name")
      public static let invalidTensorShape = Self(rawValue: "invalid-tensor-shape")
      public static let invalidTensorValueCount = Self(rawValue: "invalid-tensor-value-count")
      public static let materializedTensorShapeMismatch = Self(
        rawValue: "materialized-tensor-shape-mismatch"
      )
      public static let tensorElementCountOverflow = Self(rawValue: "tensor-element-count-overflow")
      public static let unexpectedTensorElementType = Self(
        rawValue: "unexpected-tensor-element-type"
      )
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
        throw ONNXRuntimeError(
          code: .invalidTensorShape,
          message: "Tensor dimensions must not be negative: \(shape)."
        )
      }
      let result = count.multipliedReportingOverflow(by: dimension)
      guard !result.overflow else {
        throw ONNXRuntimeError(
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
      throw ONNXRuntimeError(
        code: .invalidTensorValueCount,
        message: "Tensor shape \(shape) requires \(expectedCount) values, got \(count)."
      )
    }
  }

  func edgeToolsONNXValidateOutputNames(_ names: [String]) throws {
    guard Set(names).count == names.count else {
      throw ONNXRuntimeError(
        code: .duplicateOutputName,
        message: "Output names must be unique."
      )
    }
  }

  // MARK: - ONNXTensorView

  @_addressableForDependencies
  public struct ONNXTensorView<Element: BitwiseCopyable>: ~Copyable, ~Escapable {
    private let baseAddress: UnsafeMutablePointer<Element>?

    public let shape: [Int]

    @_unsafeNonescapableResult
    public init(unsafelyWrapping baseAddress: UnsafeMutablePointer<Element>?, shape: [Int]) {
      self.baseAddress = baseAddress
      self.shape = shape
    }

    public var count: Int { self.shape.reduce(1, *) }

    public var rank: Int { self.shape.count }

    public var strides: [Int] { edgeToolsONNXRowMajorStrides(for: self.shape) }

    public subscript(scalarAt indices: [Int]) -> Element {
      borrowing get {
        let index = edgeToolsONNXFlatIndex(
          shape: self.shape,
          strides: self.strides,
          indices: indices
        )
        return self.baseAddress![index]
      }
      mutating set {
        let index = edgeToolsONNXFlatIndex(
          shape: self.shape,
          strides: self.strides,
          indices: indices
        )
        self.baseAddress![index] = newValue
      }
    }

    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, visionOS 26, *)
    public subscript<let rank: Int>(scalarAt indices: InlineArray<rank, Int>) -> Element {
      borrowing get {
        self[scalarAt: indices.span.withUnsafeBufferPointer { Array($0) }]
      }
      mutating set {
        self[scalarAt: indices.span.withUnsafeBufferPointer { Array($0) }] = newValue
      }
    }

    @_lifetime(borrow self)
    public borrowing func slice(at leadingIndices: [Int]) -> ONNXTensorView<Element> {
      let (offset, slicedShape) = edgeToolsONNXAxisSliceOffset(
        shape: self.shape,
        strides: self.strides,
        leadingIndices: leadingIndices
      )
      let view = ONNXTensorView(
        unsafelyWrapping: self.baseAddress.map { $0 + offset },
        shape: slicedShape
      )
      return _overrideLifetime(view, borrowing: self)
    }

    public func withUnsafePointer<Result, Failure: Error>(
      _ body: (UnsafePointer<Element>?) throws(Failure) -> Result
    ) throws(Failure) -> Result {
      try body(self.baseAddress.map { UnsafePointer($0) })
    }

    public mutating func withUnsafeMutablePointer<Result, Failure: Error>(
      _ body: (UnsafeMutablePointer<Element>?) throws(Failure) -> Result
    ) throws(Failure) -> Result {
      try body(self.baseAddress)
    }

    public func withUnsafeBufferPointer<Result, Failure: Error>(
      _ body: (UnsafeBufferPointer<Element>) throws(Failure) -> Result
    ) throws(Failure) -> Result {
      try body(UnsafeBufferPointer(start: self.baseAddress, count: self.count))
    }

    public mutating func withUnsafeMutableBufferPointer<Result, Failure: Error>(
      _ body: (UnsafeMutableBufferPointer<Element>) throws(Failure) -> Result
    ) throws(Failure) -> Result {
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

  // MARK: - ONNXGraphOptimizationLevel

  public struct ONNXGraphOptimizationLevel: RawRepresentable, Hashable, Sendable {
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

  public protocol ONNXTensor {
    var dtype: ONNXDType { get }
    var shape: [Int] { get }

    nonisolated(nonsending) func withUnsafeMutableBufferPointer<Element: ONNXElement, Result>(
      as type: Element.Type,
      _ body: nonisolated(nonsending) (UnsafeMutableBufferPointer<Element>) async throws -> Result
    ) async throws -> Result
  }

  extension ONNXTensor {
    public nonisolated(nonsending) func withUnsafeBufferPointer<Element: ONNXElement, Result>(
      as type: Element.Type,
      _ body: nonisolated(nonsending) (UnsafeBufferPointer<Element>) async throws -> Result
    ) async throws -> Result {
      try await self.withUnsafeMutableBufferPointer(as: type) {
        try await body(UnsafeBufferPointer($0))
      }
    }

    public nonisolated(nonsending) func withMutableView<
      Source: ONNXElement, Materialized: BitwiseCopyable, Result
    >(
      as sourceType: Source.Type,
      materializing materializedType: Materialized.Type,
      shape: [Int],
      _ body: nonisolated(nonsending) (inout ONNXTensorView<Materialized>) async throws -> Result
    ) async throws -> Result {
      try await self.withUnsafeMutableBufferPointer(as: sourceType) { buffer in
        let materializedCount = try edgeToolsONNXElementCount(for: shape)
        let byteCount = buffer.count * MemoryLayout<Source>.stride
        let materializedByteCount = materializedCount * MemoryLayout<Materialized>.stride
        guard byteCount == materializedByteCount else {
          throw ONNXRuntimeError(
            code: .materializedTensorShapeMismatch,
            message:
              "Shape \(shape) of \(Materialized.self) (\(materializedByteCount) bytes) does not "
              + "match the tensor's \(byteCount) underlying bytes."
          )
        }
        let materializedBuffer = UnsafeMutableRawBufferPointer(buffer)
          .bindMemory(to: Materialized.self)
        var view = ONNXTensorView(
          unsafelyWrapping: materializedBuffer.baseAddress,
          shape: shape
        )
        return try await body(&view)
      }
    }

    public nonisolated(nonsending) func withView<
      Source: ONNXElement, Materialized: BitwiseCopyable, Result
    >(
      as sourceType: Source.Type,
      materializing materializedType: Materialized.Type,
      shape: [Int],
      _ body: nonisolated(nonsending) (borrowing ONNXTensorView<Materialized>) async throws -> Result
    ) async throws -> Result {
      try await self.withMutableView(
        as: sourceType,
        materializing: materializedType,
        shape: shape
      ) { try await body($0) }
    }

    public nonisolated(nonsending) func withView<Element: ONNXElement, Result>(
      as type: Element.Type,
      _ body: nonisolated(nonsending) (borrowing ONNXTensorView<Element>) async throws -> Result
    ) async throws -> Result {
      try await self.withView(as: type, materializing: type, shape: self.shape, body)
    }

    public nonisolated(nonsending) func withMutableView<Element: ONNXElement, Result>(
      as type: Element.Type,
      _ body: nonisolated(nonsending) (inout ONNXTensorView<Element>) async throws -> Result
    ) async throws -> Result {
      try await self.withMutableView(as: type, materializing: type, shape: self.shape, body)
    }
  }

  public protocol ONNXSession {
    associatedtype Tensor: ONNXTensor

    var inputNames: [String] { get }
    var outputNames: [String] { get }

    nonisolated(nonsending) func run(
      inputs: [String: Tensor],
      outputNames: [String]
    ) async throws -> [String: Tensor]
  }

  public protocol ONNXRuntime: SendableMetatype {
    associatedtype ModelSource
    associatedtype SessionConfiguration
    associatedtype Session: ONNXSession where Session.Tensor == Tensor
    associatedtype Tensor: ONNXTensor

    nonisolated(nonsending) func session(
      model: ModelSource,
      configuration: SessionConfiguration
    ) async throws -> Session

    func tensor<Values: Sequence>(
      values: Values,
      shape: [Int]
    ) throws -> Tensor where Values.Element: ONNXElement
  }

  extension ONNXRuntime {
    public func tensor<Values: Sequence>(
      values: Values
    ) throws -> Tensor where Values.Element: ONNXElement {
      let values = Array(values)
      return try self.tensor(values: values, shape: [values.count])
    }
  }

  // MARK: - Axis Helpers

  private func edgeToolsONNXRowMajorStrides(for shape: [Int]) -> [Int] {
    guard !shape.isEmpty else { return [] }
    return Array(
      shape.dropFirst().reversed()
        .reduce(into: [1]) { strides, dimension in
          strides.append(strides[strides.count - 1] * dimension)
        }
        .reversed()
    )
  }

  private func edgeToolsONNXFlatIndex(shape: [Int], strides: [Int], indices: [Int]) -> Int {
    precondition(
      indices.count == shape.count,
      "Expected \(shape.count) indices for tensor shape \(shape), got \(indices.count)."
    )
    return zip(indices.enumerated(), zip(shape, strides))
      .reduce(into: 0) { flatIndex, element in
        let ((axis, index), (dimension, stride)) = element
        precondition(
          index >= 0 && index < dimension,
          "Index \(index) at axis \(axis) is out of bounds for dimension \(dimension)."
        )
        flatIndex += index * stride
      }
  }

  private func edgeToolsONNXAxisSliceOffset(
    shape: [Int],
    strides: [Int],
    leadingIndices: [Int]
  ) -> (offset: Int, shape: [Int]) {
    precondition(
      leadingIndices.count <= shape.count,
      "Cannot index \(leadingIndices.count) axes on a rank-\(shape.count) tensor."
    )
    let offset = zip(leadingIndices.enumerated(), zip(shape, strides))
      .reduce(into: 0) { offset, element in
        let ((axis, index), (dimension, stride)) = element
        precondition(
          index >= 0 && index < dimension,
          "Index \(index) at axis \(axis) is out of bounds for dimension \(dimension)."
        )
        offset += index * stride
      }
    return (offset, Array(shape.dropFirst(leadingIndices.count)))
  }
#endif
