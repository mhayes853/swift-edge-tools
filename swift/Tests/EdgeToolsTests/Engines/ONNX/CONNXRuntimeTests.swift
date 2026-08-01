#if ONNX && canImport(COnnxRuntime)
  import COnnxRuntime
  import CustomDump
  import EdgeTools
  import Foundation
  import Testing

  @Suite
  struct `CONNXRuntime tests` {
    @Test
    func `Run Model Through Execution Protocols`() async throws {
      let runtime = try CONNXRuntime()
      let values = try await Self.executeAdd(
        runtime: runtime,
        model: try Self.modelURL().path(),
        configuration: CONNXRuntime.Configuration()
      )

      expectNoDifference(values, [5, 7, 9])
    }

    @Test
    func `Use Runtime Configuration For Session Defaults`() throws {
      let runtime = try CONNXRuntime(configuration: CONNXRuntime.Configuration())
      let session = try runtime.session(modelURL: try Self.modelURL())

      expectNoDifference(session.inputNames, ["x", "y"])
    }

    @Test
    func `Run Model Using Vendored ONNX Runtime`() async throws {
      let runtime = try CONNXRuntime()
      let session = try runtime.session(modelURL: try Self.modelURL())
      let firstInput = try runtime.tensor(values: [Float(2), 4, 8], shape: [3])
      let secondInput = try runtime.tensor(values: [Float(1), 3, 5], shape: [3])
      let outputs = try session.run(
        inputs: ["x": firstInput, "y": secondInput],
        outputNames: ["sum"]
      )

      let sum = try #require(outputs["sum"])
      let values = try await sum.array(as: Float.self)
      expectNoDifference(values, [3, 7, 13])
    }

    @Test
    func `Reject Duplicate Output Names`() throws {
      let runtime = try CONNXRuntime()
      let session = try runtime.session(modelURL: try Self.modelURL())
      let firstInput = try runtime.tensor(values: [Float(2), 4, 8], shape: [3])
      let secondInput = try runtime.tensor(values: [Float(1), 3, 5], shape: [3])

      let error = #expect(throws: ONNXRuntimeError.self) {
        try session.run(
          inputs: ["x": firstInput, "y": secondInput],
          outputNames: ["sum", "sum"]
        )
      }

      expectNoDifference(error?.code, .duplicateOutputName)
    }

    @Test
    func `Expose Session And Tensor Metadata`() throws {
      let runtime = try CONNXRuntime()
      let session = try runtime.session(modelURL: try Self.modelURL())
      let tensor = try runtime.tensor(values: [Float(1), 2, 3], shape: [1, 3])

      expectNoDifference(session.inputNames, ["x", "y"])
      expectNoDifference(session.outputNames, ["sum"])
      expectNoDifference(tensor.dtype, .float)
      expectNoDifference(tensor.shape, [1, 3])
      _ = runtime.api
      _ = runtime.environment
      _ = session.handle
      _ = tensor.handle
    }

    @Test
    func `Create And Read Integer Tensors`() async throws {
      let runtime = try CONNXRuntime()
      let int32InputValues = (1...3).lazy.map(Int32.init)
      let int32Tensor = try runtime.tensor(values: int32InputValues, shape: [3])
      let int64Tensor = try runtime.tensor(values: [Int64(4), 5, 6], shape: [3])
      let mutableTensor = try runtime.tensor(values: [Int32(7), 8, 9], shape: [3])

      let int32Values = try await int32Tensor.array(as: Int32.self)
      let int64Values = try await int64Tensor.array(as: Int64.self)
      expectNoDifference(int32Values, [1, 2, 3])
      expectNoDifference(int64Values, [4, 5, 6])

      try await mutableTensor.withMutableView(as: Int32.self) { view in
        var span = view.mutableSpan
        span[1] = 10
      }
      let mutatedValues = try await mutableTensor.array(as: Int32.self)
      expectNoDifference(mutatedValues, [7, 10, 9])
    }

    @Test
    func `Axis-Aware View Supports Scalar Indexing And Slicing`() async throws {
      let runtime = try CONNXRuntime()
      let tensor = try runtime.tensor(values: (0..<6).map(Float.init), shape: [2, 3])

      let secondRow = try await tensor.withView(as: Float.self) { view -> [Float] in
        expectNoDifference(view.shape, [2, 3])
        expectNoDifference(view[scalarAt: [1, 2]], 5)
        return view.slice(at: [1]).span.withUnsafeBufferPointer { Array($0) }
      }
      expectNoDifference(secondRow, [3, 4, 5])

      try await tensor.withMutableView(as: Float.self) { view in
        view[scalarAt: [0, 1]] = 42
      }
      let updatedValues = try await tensor.array(as: Float.self)
      expectNoDifference(updatedValues, [0, 42, 2, 3, 4, 5])
    }

    @Test
    func `Reject Tensor Values That Do Not Match Shape`() throws {
      let runtime = try CONNXRuntime()

      let error = #expect(throws: ONNXRuntimeError.self) {
        try runtime.tensor(values: [Float(1), 2], shape: [3])
      }
      expectNoDifference(error?.code, .invalidTensorValueCount)
    }

    @Test
    func `Reject Reading Tensor As Incorrect Element Type`() async throws {
      let runtime = try CONNXRuntime()
      let tensor = try runtime.tensor(values: [Int64(1), 2, 3], shape: [3])

      let error = await #expect(throws: ONNXRuntimeError.self) {
        _ = try await tensor.array(as: Float.self)
      }
      expectNoDifference(error?.code, .unexpectedTensorElementType)
    }

    private static func executeAdd<Runtime: ONNXRuntime>(
      runtime: Runtime,
      model: Runtime.ModelSource,
      configuration: Runtime.SessionConfiguration
    ) async throws -> [Float] {
      let session = try await runtime.session(model: model, configuration: configuration)
      let firstInput = try runtime.tensor(values: [Float(1), 2, 3], shape: [3])
      let secondInput = try runtime.tensor(values: [Float(4), 5, 6], shape: [3])
      let outputs = try await session.run(
        inputs: ["x": firstInput, "y": secondInput],
        outputNames: ["sum"]
      )
      let output = try #require(outputs["sum"])
      return try await output.array(as: Float.self)
    }

    private static func modelURL() throws -> URL {
      try #require(Bundle.module.url(forResource: "add", withExtension: "onnx"))
    }
  }

  extension ONNXTensor {
    fileprivate nonisolated(nonsending) func array<Element: ONNXElement>(
      as type: Element.Type
    ) async throws -> [Element] {
      try await self.withUnsafeBufferPointer(as: type) { Array($0) }
    }
  }
#endif
