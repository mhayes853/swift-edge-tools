#if SwiftNeedleXGrammar
  import CNeedleXGrammar
  import Foundation

  // MARK: - NeedleXGrammarEngine

  public final class NeedleXGrammarEngine: NeedleGrammarEngine {
    public let compiler: needle_xgrammar_compiler_t
    public var toolCallInvocationRange: ToolCallInvocationRange

    public init(
      encodedVocab: [String],
      eosTokenId: NeedleToken.ID,
      toolCallInvocationRange: ToolCallInvocationRange = .unbounded(minimum: 0),
      configuration: CompilerConfiguration = CompilerConfiguration()
    ) {
      self.compiler = withCStringPointerBuffer(encodedVocab) { buffer in
        needle_xgrammar_compiler_init(buffer.baseAddress, buffer.count, Int32(eosTokenId))
      }
      self.toolCallInvocationRange = toolCallInvocationRange
      self.apply(configuration: configuration)
    }

    public init(
      compiler: consuming needle_xgrammar_compiler_t,
      toolCallInvocationRange: ToolCallInvocationRange = .unbounded(minimum: 0)
    ) {
      self.compiler = compiler
      self.toolCallInvocationRange = toolCallInvocationRange
    }

    deinit { needle_xgrammar_compiler_destroy(self.compiler) }

    public func apply(configuration: CompilerConfiguration) {
      needle_xgrammar_compiler_set_max_threads(self.compiler, configuration.maxThreads.value)
      needle_xgrammar_compiler_set_memory_limit(self.compiler, configuration.memoryLimit.value)
      needle_xgrammar_compiler_set_cache_enabled(
        self.compiler,
        configuration.isCacheEnabled.intValue(as: Int32.self)
      )
    }

    public func compile(tools: [NeedleToolDefinition]) async throws -> Matcher {
      let toolsJSON = try String(
        decoding: JSONEncoder.needleTools.encode(tools.map { $0.normalized() }),
        as: UTF8.self
      )
      let (minCalls, maxCalls) = self.toolCallInvocationRange.cRange
      guard minCalls >= 0 else { throw NeedleXGrammarEngineError.invalidToolInvocationRange }
      let grammar = toolsJSON.withCString {
        needle_xgrammar_grammar_init($0, minCalls, maxCalls)
      }
      defer { needle_xgrammar_grammar_destroy(grammar) }
      return Matcher(matcher: needle_xgrammar_compile_matcher(self.compiler, grammar))
    }
  }

  // MARK: - Configuration

  extension NeedleXGrammarEngine {
    public struct CompilerConfiguration: Hashable, Sendable {
      public enum MemoryLimit: Hashable, Sendable {
        case noLimit
        case limit(UInt32)

        fileprivate var value: Int64 {
          switch self {
          case .limit(let limit): Int64(limit)
          case .noLimit: Int64(kNeedleXGrammarCompilerNoMemoryLimit)
          }
        }
      }

      public enum MaxThreads: Hashable, Sendable {
        case hardwareConcurrency
        case limit(UInt32)

        fileprivate var value: Int64 {
          switch self {
          case .limit(let limit): Int64(limit)
          case .hardwareConcurrency: Int64(kNeedleXGrammarCompilerHardwareConcurrency)
          }
        }
      }

      public var memoryLimit: MemoryLimit
      public var maxThreads: MaxThreads
      public var isCacheEnabled: Bool

      public init(
        memoryLimit: MemoryLimit = .noLimit,
        maxThreads: MaxThreads = .hardwareConcurrency,
        isCacheEnabled: Bool = true
      ) {
        self.memoryLimit = memoryLimit
        self.maxThreads = maxThreads
        self.isCacheEnabled = isCacheEnabled
      }
    }
  }

  // MARK: - NeedleXGrammarEngineError

  public enum NeedleXGrammarEngineError: Error, Hashable, Sendable {
    case invalidToolInvocationRange
  }

  // MARK: - ToolCallInvocationRange

  extension NeedleXGrammarEngine {
    public enum ToolCallInvocationRange: Hashable, Sendable {
      case unbounded(minimum: Int)
      case bounded(ClosedRange<Int>)
      case exact(Int)

      public static func unbounded(_ range: PartialRangeFrom<Int>) -> Self {
        .unbounded(minimum: range.lowerBound)
      }

      public static func bounded(_ range: PartialRangeThrough<Int>) -> Self {
        .bounded(0...range.upperBound)
      }

      public static func bounded(_ range: PartialRangeUpTo<Int>) -> Self {
        .bounded(0..<range.upperBound)
      }

      public static func bounded(_ range: Range<Int>) -> Self {
        .bounded(range.lowerBound...(range.upperBound - 1))
      }

      fileprivate var cRange: (Int32, Int32) {
        switch self {
        case .bounded(let range):
          (Int32(range.lowerBound), Int32(range.upperBound))
        case .unbounded(let minimum):
          (Int32(minimum), kNeedleXGrammarToolCallsUnbounded)
        case .exact(let count):
          (Int32(count), kNeedleXGrammarToolCallsOnlyLowerBound)
        }
      }
    }
  }

  // MARK: - Matcher

  extension NeedleXGrammarEngine {
    public final class Matcher {
      public let matcher: needle_xgrammar_matcher_t

      public init(matcher: consuming needle_xgrammar_matcher_t) {
        self.matcher = matcher
      }

      deinit { needle_xgrammar_matcher_destroy(self.matcher) }

      public var isCompleted: Bool {
        needle_xgrammar_matcher_is_completed(self.matcher).boolValue
      }

      public var isTerminated: Bool {
        needle_xgrammar_matcher_is_terminated(self.matcher).boolValue
      }

      public func rollback(_ numTokens: Int) {
        needle_xgrammar_matcher_rollback(self.matcher, Int32(numTokens))
      }

      public func reset() {
        needle_xgrammar_matcher_reset(self.matcher)
      }

      public func bitmask() -> NeedleGrammarBitmask {
        var bitmask = NeedleGrammarBitmask()
        _ = bitmask.storage.withUnsafeMutableBufferPointer { buffer in
          needle_xgrammar_matcher_bitmask(self.matcher, buffer.baseAddress)
        }
        return bitmask
      }

      @discardableResult
      public func accept(tokenId: NeedleToken.ID) -> Bool {
        needle_xgrammar_matcher_accept_token(self.matcher, Int32(tokenId)).boolValue
      }

      public func fork() -> Matcher {
        Matcher(matcher: needle_xgrammar_matcher_fork(self.matcher))
      }
    }
  }

  // MARK: - SP Tokenizer

  #if SwiftNeedleSentencepiece
    extension NeedleXGrammarEngine {
      public convenience init?(
        tokenizer: NeedleSPTokenizingModel,
        configuration: CompilerConfiguration = CompilerConfiguration()
      ) {
        guard let eosTokenId = tokenizer.eosTokenId else { return nil }
        self.init(
          encodedVocab: tokenizer.encodedVocab(),
          eosTokenId: eosTokenId,
          configuration: configuration
        )
      }
    }
  #endif
#endif
