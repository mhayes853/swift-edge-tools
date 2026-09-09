import EdgeToolsCore

// MARK: - EdgeToolCallOutcome

public enum EdgeToolCallOutcome: Sendable {
  case resolved(AnyEdgeToolCall)
  case unknownTool(EdgeRawToolCall)
  case invalidArguments(EdgeRawToolCall)

  public var call: AnyEdgeToolCall? {
    switch self {
    case .resolved(let call): call
    case .unknownTool, .invalidArguments: nil
    }
  }

  public var rawValue: EdgeRawToolCall {
    switch self {
    case .resolved(let call): call.rawValue
    case .unknownTool(let rawCall), .invalidArguments(let rawCall): rawCall
    }
  }

  public var name: String {
    self.rawValue.name
  }
}

// MARK: - EdgeToolsGeneration

public struct EdgeToolsGeneration: Sendable {
  public let engineGeneration: EdgeToolsEngineGeneration
  public let toolCalls: EdgeToolCallCollection

  public let toolCallOutcomes: [EdgeToolCallOutcome]

  public var response: String {
    self.engineGeneration.response
  }

  public var text: String {
    self.engineGeneration.text
  }

  public var parts: [EdgeToolsGenerationPart] {
    self.engineGeneration.parts
  }

  public var reasoning: [String] {
    self.engineGeneration.reasoning
  }

  public func decoded<Response: ConvertibleFromEdgeToolsValue>(
    as type: Response.Type
  ) throws -> Response {
    return try Response(edgeToolsValue: EdgeToolsValue(json: self.text))
  }
}
