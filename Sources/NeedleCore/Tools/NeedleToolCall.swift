// MARK: - NeedleToolCallOf

public typealias NeedleToolCallOf<Tool: NeedleTool> = NeedleToolCall<Tool.Input, Tool.Output>

// MARK: - NeedleToolCall

public struct NeedleToolCall<Input: ConvertibleFromNeedleValue, Output> {
  public let input: Input
  public let output: Output

  public init(input: Input, output: Output) {
    self.input = input
    self.output = output
  }
}
