#if ONNXCore
  // MARK: - EdgeToolsONNXDType

  public struct EdgeToolsONNXDType: RawRepresentable, Hashable, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) {
      self.rawValue = rawValue
    }

    public static let undefined = Self(rawValue: 0)
    public static let float = Self(rawValue: 1)
    public static let int32 = Self(rawValue: 6)
    public static let int64 = Self(rawValue: 7)
  }

  // MARK: - EdgeToolsONNXGraphOptimizationLevel

  public struct EdgeToolsONNXGraphOptimizationLevel:
    RawRepresentable,
    Hashable,
    Sendable
  {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public static let disabled = Self(rawValue: "disabled")
    public static let basic = Self(rawValue: "basic")
    public static let extended = Self(rawValue: "extended")
    public static let layout = Self(rawValue: "layout")
    public static let all = Self(rawValue: "all")
  }

  // MARK: - EdgeToolsONNXExecutionProvider

  public struct EdgeToolsONNXExecutionProvider: Hashable, Sendable {
    public struct Option: Hashable, Sendable {
      public var name: String
      public var value: String

      public init(name: String, value: String) {
        self.name = name
        self.value = value
      }
    }

    public var name: String
    public var options: [Option]

    public init(name: String, options: [Option] = []) {
      self.name = name
      self.options = options
    }

    public static var cpu: Self { Self(name: "cpu") }
    public static var wasm: Self { Self(name: "wasm") }
    public static var webGPU: Self { Self(name: "webgpu") }
    public static var webNN: Self { Self(name: "webnn") }
    public static var coreML: Self { Self(name: "coreml") }
    public static var cuda: Self { Self(name: "cuda") }
    public static var directML: Self { Self(name: "dml") }
  }

  // MARK: - EdgeToolsONNXTensor

  public protocol EdgeToolsONNXTensor {
    var dtype: EdgeToolsONNXDType { get }
    var shape: [Int] { get }

    nonisolated(nonsending) func floatValues() async throws -> [Float]
    nonisolated(nonsending) func int32Values() async throws -> [Int32]
    nonisolated(nonsending) func int64Values() async throws -> [Int64]
  }

  // MARK: - EdgeToolsONNXSession

  public protocol EdgeToolsONNXSession {
    associatedtype Tensor: EdgeToolsONNXTensor

    var inputNames: [String] { get }
    var outputNames: [String] { get }

    nonisolated(nonsending) func run(
      inputs: [String: Tensor],
      outputNames: [String]
    ) async throws -> [String: Tensor]
  }

  // MARK: - EdgeToolsONNXRuntime

  public protocol EdgeToolsONNXRuntime: SendableMetatype {
    associatedtype ModelSource
    associatedtype SessionConfiguration
    associatedtype Session: EdgeToolsONNXSession where Session.Tensor == Tensor
    associatedtype Tensor: EdgeToolsONNXTensor

    nonisolated(nonsending) func session(
      model: ModelSource,
      configuration: SessionConfiguration
    ) async throws -> Session

    func tensor(values: [Float], shape: [Int]) throws -> Tensor
    func tensor(values: [Int32], shape: [Int]) throws -> Tensor
    func tensor(values: [Int64], shape: [Int]) throws -> Tensor
  }
#endif
