#if Needle2
  import EdgeToolsCore
  import OrderedCollections

  // MARK: - Needle2Prompt

  public enum Needle2Prompt: Hashable, Sendable {
    case user(String)
    case toolResponses([EdgeToolsValue])

    public init(_ userMessage: String) {
      self = .user(userMessage)
    }

    var needle2Input: String {
      switch self {
      case .user(let message): message
      case .toolResponses(let responses): EdgeToolsValue.array(responses).orderedJSONString()
      }
    }
  }

  extension Needle2Prompt: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
      self = .user(value)
    }
  }

  public protocol Needle2SessionEngine: EdgeToolsEngine where Prompt == Needle2Prompt {
    func reset(_ context: Context) async throws
  }

  // MARK: - Needle2Response

  struct Needle2Response: Sendable {
    var type: String
    var success: Bool
    var error: String?
    var errorCode: String?
    var functionCalls: [EdgeRawToolCall]
    var reasoning: String?
    var confidence: Float?
    var prefillTokensPerSecond: Double?
    var decodeTokensPerSecond: Double?
    var peakRAMMegabytes: Double?
    var tokenCount: Int?

    var parts: [EdgeToolsGenerationPart] {
      let reasoning = self.reasoning.map { [EdgeToolsGenerationPart.reasoning($0)] } ?? []
      return reasoning + self.functionCalls.map(EdgeToolsGenerationPart.toolCall)
    }

    init(json: String) throws {
      let value: EdgeToolsValue
      do {
        value = try EdgeToolsValue(json: json)
      } catch {
        throw Needle2Error(
          code: .invalidResponse,
          message: "Needle 2 returned invalid JSON."
        )
      }
      guard case .object(let object) = value,
        case .string(let type) = object["type"],
        case .boolean(let success) = object["success"]
      else {
        throw Needle2Error(
          code: .invalidResponse,
          message: "Needle 2 returned a response without a valid type and success value."
        )
      }

      self.type = type
      self.success = success
      self.error = object["error"]?.string
      self.errorCode = object["error_code"]?.string
      self.reasoning = object["reasoning"]?.string
      self.confidence = object["confidence"]?.double.map(Float.init)
      self.prefillTokensPerSecond = object["prefill_tps"]?.double
      self.decodeTokensPerSecond = object["decode_tps"]?.double
      self.peakRAMMegabytes = object["peak_ram_mb"]?.double
      self.tokenCount = nil
      self.functionCalls = try object["function_calls"]?.needle2ToolCalls ?? []
    }
  }

  // MARK: - Tools

  extension Array where Element == EdgeToolDefinition {
    var needle2Value: EdgeToolsValue {
      .array(
        self.map { tool in
          [
            "name": .string(tool.name.snakeCased()),
            "description": .string(tool.description),
            "parameters": tool.arguments.edgeToolsValue
          ]
        }
      )
    }

    var needle2JSON: String {
      self.needle2Value.orderedJSONString()
    }
  }

  extension EdgeToolsValue {
    var needle2ToolCalls: [EdgeRawToolCall] {
      get throws {
        if self.isNull {
          return []
        }
        guard case .array(let values) = self else {
          throw Needle2Error(
            code: .invalidResponse,
            message: "Needle 2 returned an invalid function_calls value."
          )
        }
        var calls = [EdgeRawToolCall]()
        for value in values {
          guard let call = EdgeRawToolCall(jsonValue: value) else {
            throw Needle2Error(
              code: .invalidResponse,
              message: "Needle 2 returned an invalid function call."
            )
          }
          calls.append(call)
        }
        return calls
      }
    }
  }
#endif
