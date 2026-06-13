#if SwiftNeedleXGrammar
  import CNeedleXGrammar

  // MARK: - NeedleXGrammarEngine

  public final class NeedleXGrammarEngine: NeedleGrammarEngine {
    public let value: needle_xgrammar_compiler_t

    public init(encodedVocab: [String], configuration: Configuration = Configuration()) {
      self.value = withCStringPointerBuffer(encodedVocab) { buffer in
        needle_xgrammar_compiler_init(buffer.baseAddress, buffer.count)
      }
      needle_xgrammar_compiler_set_max_threads(self.value, configuration.maxThreads.value)
      needle_xgrammar_compiler_set_memory_limit(self.value, configuration.memoryLimit.value)
      needle_xgrammar_compiler_set_cache_enabled(
        self.value,
        configuration.isCacheEnabled.intValue(as: Int32.self)
      )
    }

    public init(value: consuming needle_xgrammar_compiler_t) {
      self.value = value
    }

    deinit { needle_xgrammar_compiler_destroy(self.value) }

    public func compile(tools: [NeedleToolDefinition]) async throws -> Matcher {
      fatalError()
    }
  }

  // MARK: - Configuration

  extension NeedleXGrammarEngine {
    public struct Configuration: Hashable, Sendable {
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

  // MARK: - Matcher

  extension NeedleXGrammarEngine {
    public final class Matcher: NeedleGrammarMatcher {
      public let value: needle_xgrammar_matcher_t

      public init(value: consuming needle_xgrammar_matcher_t) {
        self.value = value
      }

      deinit { needle_xgrammar_matcher_destroy(self.value) }

      public func isCompleted() -> Bool {
        needle_xgrammar_matcher_is_completed(self.value).boolValue
      }

      public func isTerminated() -> Bool {
        needle_xgrammar_matcher_is_terminated(self.value).boolValue
      }

      public func rollback(_ numTokens: Int) {
        needle_xgrammar_matcher_rollback(self.value, Int32(numTokens))
      }

      public func reset() {
        needle_xgrammar_matcher_reset(self.value)
      }

      public func bitmask() -> NeedleGrammarBitmask {
        var bitmask = NeedleGrammarBitmask()
        _ = bitmask.storage.withUnsafeMutableBufferPointer { buffer in
          needle_xgrammar_matcher_next_bitmask(self.value, buffer.baseAddress)
        }
        return bitmask
      }

      public func accept(tokenId: NeedleToken.ID) {
        needle_xgrammar_matcher_accept_token(self.value, Int32(tokenId))
      }

      public func fork() -> Matcher {
        Matcher(value: needle_xgrammar_matcher_fork(self.value))
      }
    }
  }
#endif
