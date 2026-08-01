import EdgeTools
import JavaScriptKit
import Testing

@Suite
struct `JSONNXRuntime tests` {
  @Test
  func `Run Model Through JavaScript Runtime`() async throws {
    let runtime = try Self.runtime()
    let session = try await runtime.session(
      model: JSONNXRuntime.ModelSource.object(try Self.addModel()),
      configuration: JSONNXRuntime.Configuration()
    )
    let firstInput = try runtime.tensor(values: [Float(1), 2, 3], shape: [3])
    let secondInput = try runtime.tensor(values: [Float(4), 5, 6], shape: [3])
    let outputs = try await session.run(
      inputs: ["x": firstInput, "y": secondInput],
      outputNames: ["sum"]
    )
    let output = try #require(outputs["sum"])

    #expect(session.inputNames == ["x", "y"])
    #expect(session.outputNames == ["sum"])
    #expect(output.dtype == .float)
    #expect(output.shape == [3])
    try await output.withView(as: Float.self) { view in
      #expect(view.count == 3)
      #expect(view[scalarAt: [0]] == 5)
      #expect(view[scalarAt: [1]] == 7)
      #expect(view[scalarAt: [2]] == 9)
    }
    _ = runtime.object
    _ = session.object
    _ = output.object
  }

  @Test
  func `Reject Duplicate Output Names`() async throws {
    let runtime = try Self.runtime()
    let session = try await runtime.session(
      model: JSONNXRuntime.ModelSource.object(try Self.addModel()),
      configuration: JSONNXRuntime.Configuration()
    )

    let error = await #expect(throws: ONNXRuntimeError.self) {
      _ = try await session.run(inputs: [:], outputNames: ["sum", "sum"])
    }

    #expect(error?.code == .duplicateOutputName)
  }

  @Test
  func `Create And Read Integer Tensors`() async throws {
    let runtime = try Self.runtime()
    let int32Values = (1...3).lazy.map(Int32.init)
    let int32Tensor = try runtime.tensor(values: int32Values, shape: [3])
    let int64Tensor = try runtime.tensor(values: [Int64(4), 5, 6], shape: [3])

    let int32Scalars = try await int32Tensor.withView(as: Int32.self) {
      $0.withUnsafeBufferPointer { Array($0) }
    }
    let int64Scalars = try await int64Tensor.withView(as: Int64.self) {
      $0.withUnsafeBufferPointer { Array($0) }
    }
    #expect(int32Scalars == [1, 2, 3])
    #expect(int64Scalars == [4, 5, 6])
  }

  @Test
  func `Reject Invalid Runtime Namespace`() {
    let error = #expect(throws: JSONNXRuntimeError.self) {
      try JSONNXRuntime(onnxRuntime: JSObject())
    }

    #expect(error?.code == .invalidRuntimeObject)
  }

  @Test
  func `Propagate Rejected JS Promise`() async throws {
    let namespace = try #require(JSObject.global["edgeToolsRejectedONNXRuntime"].object)
    let runtime = try JSONNXRuntime(onnxRuntime: namespace)

    let error = await #expect(throws: JSException.self) {
      _ = try await runtime.session(
        model: JSONNXRuntime.ModelSource.bytes([0]),
        configuration: JSONNXRuntime.Configuration()
      )
    }

    #expect(error?.thrownValue.object?["message"].string == "The test session was rejected.")
    #expect(error?.thrownValue.object?["stack"].string != nil)
  }

  @Test
  func `Reject Tensor Values That Do Not Match Shape`() throws {
    let runtime = try Self.runtime()

    let error = #expect(throws: ONNXRuntimeError.self) {
      try runtime.tensor(values: [Float(1), 2], shape: [3])
    }
    #expect(error?.code == .invalidTensorValueCount)
  }

  @Test
  func `Reject Reading Tensor As Incorrect Element Type`() async throws {
    let runtime = try Self.runtime()
    let tensor = try runtime.tensor(values: [Int64(1), 2, 3], shape: [3])

    let error = await #expect(throws: ONNXRuntimeError.self) {
      try await tensor.withView(as: Float.self) { _ in }
    }
    #expect(error?.code == .unexpectedTensorElementType)
  }

  private static func runtime() throws -> JSONNXRuntime {
    let namespace = try #require(JSObject.global["edgeToolsONNXRuntime"].object)
    return try JSONNXRuntime(onnxRuntime: namespace)
  }

  private static func addModel() throws -> JSObject {
    try #require(JSObject.global["edgeToolsAddModel"].object)
  }
}
