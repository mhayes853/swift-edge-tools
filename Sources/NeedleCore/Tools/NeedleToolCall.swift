// MARK: - NeedleToolCallOf

public typealias NeedleToolCallOf<Tool: NeedleTool> = NeedleToolCall<Tool.Input, Tool.Output>

// MARK: - NeedleRawToolCall

public typealias NeedleRawToolCall = NeedleToolCall<NeedleValue, NeedleValue>

// MARK: - NeedleToolCall

public struct NeedleToolCall<Input: ConvertibleFromNeedleValue, Output> {
  public let name: String
  public let input: Input
  public let output: Output

  public init(name: String, input: Input, output: Output) {
    self.name = name
    self.input = input
    self.output = output
  }
}

extension NeedleToolCall: Sendable where Input: Sendable, Output: Sendable {}
extension NeedleToolCall: Equatable where Input: Equatable, Output: Equatable {}
extension NeedleToolCall: Hashable where Input: Hashable, Output: Hashable {}
