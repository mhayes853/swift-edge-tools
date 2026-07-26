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

  // MARK: - EdgeToolsONNXTensor

  public protocol EdgeToolsONNXTensor: Sendable {
    var dtype: EdgeToolsONNXDType { get }
    var shape: [Int] { get }

    func floatValues() async throws -> [Float]
    func int32Values() async throws -> [Int32]
    func int64Values() async throws -> [Int64]
  }

  // MARK: - EdgeToolsONNXSession

  public protocol EdgeToolsONNXSession: Sendable {
    associatedtype Tensor: EdgeToolsONNXTensor

    var inputNames: [String] { get }
    var outputNames: [String] { get }

    func run(
      inputs: [String: Tensor],
      outputNames: [String]
    ) async throws -> [String: Tensor]
  }

  // MARK: - EdgeToolsONNXRuntime

  public protocol EdgeToolsONNXRuntime: Sendable {
    associatedtype ModelSource: Sendable
    associatedtype SessionConfiguration: Sendable
    associatedtype Session: EdgeToolsONNXSession where Session.Tensor == Tensor
    associatedtype Tensor: EdgeToolsONNXTensor

    func session(
      model: ModelSource,
      configuration: SessionConfiguration
    ) async throws -> Session

    func tensor(values: [Float], shape: [Int]) throws -> Tensor
    func tensor(values: [Int32], shape: [Int]) throws -> Tensor
    func tensor(values: [Int64], shape: [Int]) throws -> Tensor
  }
#endif
