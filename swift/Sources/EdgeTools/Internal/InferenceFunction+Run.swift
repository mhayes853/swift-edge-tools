#if CoreAI && canImport(CoreAI)
  import CoreAI

  @available(anyAppleOS 27.0, *)
  // @unchecked Sendable is safe because each state is generation-owned and the model actor
  // serializes every mutation of its backing arrays and asynchronous mutable values.
  final class InferenceFunctionState: @unchecked Sendable {
    final class Value {
      let name: String
      var array: NDArray
      var asyncMutableValue: InferenceFunction.AsyncMutableValue

      init(name: String, array: NDArray) {
        self.name = name
        self.array = array
        self.asyncMutableValue = InferenceFunction.AsyncMutableValue(array)
      }
    }

    let values: [Value]

    init(values: [(name: String, array: NDArray)]) {
      self.values = values.map { Value(name: $0.name, array: $0.array) }
    }
  }

  @available(anyAppleOS 27.0, *)
  extension InferenceFunction {
    nonisolated(nonsending) func invoke(
      inputs: [String: AsyncValue],
      outputNames: some Sequence<String>,
      state: InferenceFunctionState,
      on stream: ComputeStream?
    ) async throws -> [String: AsyncValue] {
      guard let stream else {
        return try await self.run(
          inputs: inputs,
          outputNames: Array(outputNames),
          state: state
        )
      }
      let outputs = try self.encode(
        inputs: inputs,
        stateValues: state.values[...],
        stateViews: AsyncMutableViews(),
        to: stream
      )
      return try Self.select(outputs: outputs, names: outputNames)
    }

    private func encode(
      inputs: [String: AsyncValue],
      stateValues: ArraySlice<InferenceFunctionState.Value>,
      stateViews: consuming AsyncMutableViews,
      to stream: ComputeStream
    ) throws -> [String: AsyncValue] {
      guard let value = stateValues.first else {
        return try self.encode(inputs: inputs, states: stateViews, to: stream)
      }
      return try Self.withMutableValue(
        &value.asyncMutableValue,
        stateViews: consume stateViews
      ) { mutableValue, stateViews in
        var stateViews = consume stateViews
        stateViews.insert(&mutableValue, for: value.name)
        return try self.encode(
          inputs: inputs,
          stateValues: stateValues.dropFirst(),
          stateViews: consume stateViews,
          to: stream
        )
      }
    }

    private static func withMutableValue<Result>(
      _ value: inout AsyncMutableValue,
      stateViews: consuming AsyncMutableViews,
      _ body: (inout AsyncMutableValue, consuming AsyncMutableViews) throws -> Result
    ) rethrows -> Result {
      try body(&value, stateViews)
    }

    nonisolated(nonsending) func invoke(
      inputs: [String: AsyncValue],
      outputNames: some Sequence<String>,
      on stream: ComputeStream?
    ) async throws -> [String: AsyncValue] {
      guard let stream else {
        return try await self.run(inputs: inputs, outputNames: outputNames)
      }
      let outputs = try self.encode(inputs: inputs, to: stream)
      return try Self.select(outputs: outputs, names: outputNames)
    }

    private nonisolated(nonsending) func run(
      inputs: [String: AsyncValue],
      outputNames: some Sequence<String>
    ) async throws -> [String: AsyncValue] {
      let arrays = try await Self.arrays(from: inputs)
      return try await self.run(
        inputs: arrays,
        outputNames: Array(outputNames),
        stateValues: [],
        stateViews: MutableViews()
      )
    }

    @concurrent private func run(
      inputs: [String: AsyncValue],
      outputNames: [String],
      state: InferenceFunctionState
    ) async throws -> [String: AsyncValue] {
      let arrays = try await Self.arrays(from: inputs)
      return try await self.run(
        inputs: arrays,
        outputNames: outputNames,
        stateValues: state.values[...],
        stateViews: MutableViews()
      )
    }

    @concurrent private func run(
      inputs: [String: NDArray],
      outputNames: [String],
      stateValues: ArraySlice<InferenceFunctionState.Value>,
      stateViews: consuming MutableViews
    ) async throws -> [String: AsyncValue] {
      guard let value = stateValues.first else {
        var functionOutputs = try await self.run(inputs: inputs, states: stateViews)
        var outputs = [String: AsyncValue]()
        for name in outputNames {
          guard let array = functionOutputs.remove(name)?.ndArray else {
            throw EdgeToolsError.missingModelOutputs
          }
          outputs[name] = AsyncValue(array)
        }
        return outputs
      }
      return try await Self.withMutableRawView(
        of: &value.array,
        stateViews: consume stateViews
      ) { rawView, stateViews in
        var stateViews = consume stateViews
        stateViews.insert(consume rawView, for: value.name)
        return try await self.run(
          inputs: inputs,
          outputNames: outputNames,
          stateValues: stateValues.dropFirst(),
          stateViews: consume stateViews
        )
      }
    }

    @concurrent private static func withMutableRawView<Result>(
      of array: inout NDArray,
      stateViews: consuming MutableViews,
      _ body: (
        consuming NDArray.MutableRawView,
        consuming MutableViews
      ) async throws -> Result
    ) async rethrows -> Result {
      let rawView = array.mutableRawView()
      return try await body(consume rawView, consume stateViews)
    }

    private static func select(
      outputs: [String: AsyncValue],
      names: some Sequence<String>
    ) throws -> [String: AsyncValue] {
      var outputs = outputs
      var selected = [String: AsyncValue]()
      for name in names {
        guard let value = outputs.removeValue(forKey: name) else {
          throw EdgeToolsError.missingModelOutputs
        }
        selected[name] = value
      }
      return selected
    }

    private nonisolated(nonsending) static func arrays(
      from values: [String: AsyncValue]
    ) async throws -> [String: NDArray] {
      var arrays = [String: NDArray]()
      for (name, value) in values {
        guard let array = try await value.ndArray else {
          throw EdgeToolsError.missingModelOutputs
        }
        arrays[name] = array
      }
      return arrays
    }
  }
#endif
