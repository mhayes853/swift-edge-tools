#if CoreAI && canImport(CoreAI)
  import CoreAI

  @available(anyAppleOS 27.0, *)
  extension InferenceFunction {
    nonisolated(nonsending) func invoke(
      inputs: [String: AsyncValue],
      outputNames: some Sequence<String>,
      on stream: ComputeStream?
    ) async throws -> [String: AsyncValue] {
      guard let stream else {
        return try await self.run(inputs: inputs, outputNames: outputNames)
      }
      return try self.encode(inputs: inputs, to: stream)
    }

    private nonisolated(nonsending) func run(
      inputs: [String: AsyncValue],
      outputNames: some Sequence<String>
    ) async throws -> [String: AsyncValue] {
      var arrays = [String: NDArray]()
      for (name, value) in inputs {
        guard let array = try await value.ndArray else {
          throw EdgeToolsError.missingModelOutputs
        }
        arrays[name] = array
      }
      var functionOutputs = try await self.run(inputs: arrays)
      var outputs = [String: AsyncValue]()
      for name in outputNames {
        guard let array = functionOutputs.remove(name)?.ndArray else {
          throw EdgeToolsError.missingModelOutputs
        }
        outputs[name] = AsyncValue(array)
      }
      return outputs
    }
  }
#endif
