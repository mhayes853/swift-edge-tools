#if canImport(FoundationModels)
  import Foundation
  import FoundationModels

  // MARK: - NeedleFMTool

  @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
  public struct NeedleFMTool<Base: Tool>: NeedleTool where Base.Arguments: Sendable {
    public typealias Input = NeedleFMToolInput<Base.Arguments>
    public typealias Output = Base.Output

    public let base: Base
    public let arguments: NeedleGenerationSchema

    public var name: String { self.base.name }
    public var description: String { self.base.description }

    public init(_ base: Base) {
      self.base = base

      let data = Data(self.base.parameters.debugDescription.utf8)
      self.arguments = try! JSONDecoder().decode(NeedleGenerationSchema.self, from: data)
    }

    public func invoke(input: Input) async throws -> Base.Output {
      try await self.base.call(arguments: input.arguments)
    }
  }

  // MARK: - NeedleFMToolInput

  @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
  public struct NeedleFMToolInput<
    Arguments: ConvertibleFromGeneratedContent & Sendable
  >: ConvertibleFromNeedleValue, Sendable {
    public let arguments: Arguments
    public let needleValue: NeedleValue

    public init(arguments: Arguments, needleValue: NeedleValue) {
      self.arguments = arguments
      self.needleValue = needleValue
    }

    public init(needleValue: NeedleValue) throws {
      let generatedContent = try GeneratedContent(needleValue: needleValue)
      self.init(arguments: try Arguments(generatedContent), needleValue: needleValue)
    }
  }

  @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
  extension NeedleFMToolInput: Equatable where Arguments: Equatable {}

  @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
  extension NeedleFMToolInput: Hashable where Arguments: Hashable {}

  // MARK: - GeneratedContent Helper

  @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
  extension GeneratedContent: ConvertibleFromNeedleValue {
    public init(needleValue: NeedleValue) throws {
      let jsonData = try JSONEncoder().encode(needleValue)
      try self.init(json: String(decoding: jsonData, as: UTF8.self))
    }
  }
#endif
