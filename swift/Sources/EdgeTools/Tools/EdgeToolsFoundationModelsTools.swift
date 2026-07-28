#if FoundationModels && canImport(FoundationModels)
  internal import _EdgeToolsFoundation
  import FoundationModels

  // MARK: - EdgeToolsFMTool

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  public struct EdgeToolsFMTool<Base: Tool>: EdgeTool where Base.Arguments: Sendable {
    public typealias Input = EdgeToolsFMToolInput<Base.Arguments>
    public typealias Output = Base.Output

    public let base: Base
    public let arguments: EdgeToolsGenerationSchema

    public var name: String { self.base.name }
    public var description: String { self.base.description }
    public var includesSchemaInInstructions: Bool { self.base.includesSchemaInInstructions }

    public init(_ base: Base) {
      self.base = base

      do {
        let data = try JSONEncoder().encode(self.base.parameters)
        self.arguments = try JSONDecoder().decode(EdgeToolsGenerationSchema.self, from: data)
      } catch {
        preconditionFailure("FoundationModels produced an invalid generation schema: \(error)")
      }
    }

    public func invoke(input: Input) async throws -> Base.Output {
      try await self.base.call(arguments: input.arguments)
    }
  }

  // MARK: - EdgeToolsFMToolInput

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  public struct EdgeToolsFMToolInput<
    Arguments: ConvertibleFromGeneratedContent & Sendable
  >: ConvertibleFromEdgeToolsValue, Sendable {
    public let arguments: Arguments

    public init(arguments: Arguments) {
      self.arguments = arguments
    }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      let jsonData = try JSONEncoder().encode(edgeToolsValue)
      let json = String(decoding: jsonData, as: UTF8.self)
      let generatedContent = try GeneratedContent(json: json)
      self.init(arguments: try Arguments(generatedContent))
    }
  }

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension EdgeToolsFMToolInput: Equatable where Arguments: Equatable {}

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension EdgeToolsFMToolInput: Hashable where Arguments: Hashable {}

  // MARK: - FMEdgeTool

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  public struct FMEdgeTool<Base: EdgeTool>: Tool where Base.Output: PromptRepresentable {
    public typealias Arguments = FMEdgeToolArguments<Base.Input>
    public typealias Output = Base.Output

    public let base: Base
    public let parameters: GenerationSchema

    public var name: String { self.base.name }
    public var description: String { self.base.description }
    public var includesSchemaInInstructions: Bool { self.base.includesSchemaInInstructions }

    public init(_ base: Base) throws {
      self.base = base
      self.parameters = try GenerationSchema(edgeToolsGenerationSchema: base.arguments)
    }

    public func call(arguments: Arguments) async throws -> Output {
      try await self.base.invoke(input: arguments.input)
    }
  }

  // MARK: - FMEdgeToolInput

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  public struct FMEdgeToolArguments<Input: ConvertibleFromEdgeToolsValue & Sendable>: Sendable {
    public let input: Input

    public init(input: Input) {
      self.input = input
    }
  }

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension FMEdgeToolArguments: ConvertibleFromGeneratedContent {
    public init(_ content: GeneratedContent) throws {
      let value = try EdgeToolsValue(generatedContent: content)
      self.init(input: try Input(edgeToolsValue: value))
    }
  }

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension FMEdgeToolArguments: Equatable where Input: Equatable {}

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension FMEdgeToolArguments: Hashable where Input: Hashable {}
#endif
