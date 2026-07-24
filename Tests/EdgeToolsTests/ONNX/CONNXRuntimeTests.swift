#if ONNX && canImport(COnnxRuntime)
  import COnnxRuntime
  import CustomDump
  import EdgeTools
  import Foundation
  import Testing

  #if System
    import SystemPackage
  #endif

  @Suite
  struct `CONNXRuntime tests` {
    @Test
    func `Run Model Using Vendored ONNX Runtime`() throws {
      let runtime = try CONNXRuntime()
      let session = try runtime.session(modelURL: try Self.modelURL())
      let firstInput = try runtime.tensor(values: [Float(2), 4, 8], shape: [3])
      let secondInput = try runtime.tensor(values: [Float(1), 3, 5], shape: [3])
      let outputs = try session.run(
        inputs: ["x": firstInput, "y": secondInput],
        outputNames: ["sum"]
      )

      let sum = try #require(outputs["sum"])
      expectNoDifference(try sum.floatValues(), [3, 7, 13])
    }

    #if System
      @Test
      func `Run Model Using Swift System File Path`() throws {
        let runtime = try CONNXRuntime()
        let session = try runtime.session(modelPath: FilePath(try Self.modelURL().path()))
        let firstInput = try runtime.tensor(values: [Float(3), 6, 9], shape: [3])
        let secondInput = try runtime.tensor(values: [Float(2), 4, 8], shape: [3])
        let outputs = try session.run(
          inputs: ["x": firstInput, "y": secondInput],
          outputNames: ["sum"]
        )

        let sum = try #require(outputs["sum"])
        expectNoDifference(try sum.floatValues(), [5, 10, 17])
      }
    #endif

    @Test
    func `Expose Session And Tensor Metadata`() throws {
      let runtime = try CONNXRuntime()
      let session = try runtime.session(modelURL: try Self.modelURL())
      let tensor = try runtime.tensor(values: [Float(1), 2, 3], shape: [1, 3])

      expectNoDifference(try session.inputNames, ["x", "y"])
      expectNoDifference(try session.outputNames, ["sum"])
      expectNoDifference(try tensor.elementType, .float)
      expectNoDifference(try tensor.shape, [1, 3])
    }

    @Test
    func `Create And Read Integer Tensors`() throws {
      let runtime = try CONNXRuntime()
      let int32Tensor = try runtime.tensor(values: [Int32(1), 2, 3], shape: [3])
      let int64Tensor = try runtime.tensor(values: [Int64(4), 5, 6], shape: [3])

      expectNoDifference(try int32Tensor.int32Values(), [1, 2, 3])
      expectNoDifference(try int64Tensor.int64Values(), [4, 5, 6])
    }

    @Test
    func `Reject Tensor Values That Do Not Match Shape`() throws {
      let runtime = try CONNXRuntime()

      let error = #expect(throws: CONNXRuntimeError.self) {
        try runtime.tensor(values: [Float(1), 2], shape: [3])
      }
      expectNoDifference(error?.code, .invalidTensorValueCount)
    }

    @Test
    func `Reject Reading Tensor As Incorrect Element Type`() throws {
      let runtime = try CONNXRuntime()
      let tensor = try runtime.tensor(values: [Int64(1), 2, 3], shape: [3])

      let error = #expect(throws: CONNXRuntimeError.self) {
        try tensor.floatValues()
      }
      expectNoDifference(error?.code, .unexpectedTensorElementType)
    }

    private static func modelURL() throws -> URL {
      try #require(Bundle.module.url(forResource: "add", withExtension: "onnx"))
    }
  }
#endif
