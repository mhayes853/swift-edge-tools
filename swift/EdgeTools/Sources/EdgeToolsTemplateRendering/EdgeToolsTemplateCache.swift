import EdgeToolsCore
import OrderedCollections

// MARK: - EdgeToolsTemplateCache

/// A thread-safe cache of parsed templates.
///
/// The cache keys templates by their source and options, and parses a template the first time
/// it's requested. When the cached templates exceed the limits of the cache's ``Policy``, the
/// cache evicts the least recently used templates until they fit.
///
/// ```swift
/// let policy = EdgeToolsTemplateCache.Policy(maximumMemorySize: 512 * 1024)
/// let cache = EdgeToolsTemplateCache(policy: policy)
/// let template = try cache.template(source: "Hello {{ name }}!")
/// ```
public final class EdgeToolsTemplateCache: Sendable {
  /// The limits a cache keeps its templates within.
  public struct Policy: Hashable, Sendable {
    /// The maximum number of cached templates, or `nil` for no limit.
    public var maximumCount: Int?

    /// The maximum combined ``EdgeToolsTemplate/estimatedMemorySize`` of the cached templates in
    /// bytes, or `nil` for no limit.
    ///
    /// The cache doesn't store a template whose estimated size exceeds this limit on its own.
    public var maximumMemorySize: Int?

    public init(maximumCount: Int? = nil, maximumMemorySize: Int? = nil) {
      self.maximumCount = maximumCount
      self.maximumMemorySize = maximumMemorySize
    }
  }

  /// A cache shared across the process, which holds up to 4 templates by default.
  public static let shared = EdgeToolsTemplateCache(policy: Policy(maximumCount: 4))

  private let state: Lock<State>

  /// The limits the cache keeps its templates within.
  ///
  /// Setting a policy with lower limits evicts templates immediately.
  public var policy: Policy {
    get { self.state.withLock { $0.policy } }
    set {
      self.state.withLock { state in
        state.policy = newValue
        state.evict()
      }
    }
  }

  /// The number of cached templates.
  public var count: Int {
    self.state.withLock { $0.templates.count }
  }

  /// The combined ``EdgeToolsTemplate/estimatedMemorySize`` of the cached templates in bytes.
  public var estimatedMemorySize: Int {
    self.state.withLock { $0.estimatedMemorySize }
  }

  /// Creates a cache.
  ///
  /// - Parameter policy: The limits the cache keeps its templates within.
  public init(policy: Policy) {
    self.state = Lock(State(policy: policy))
  }

  /// Returns the cached template for a source and options, parsing and caching it on a miss.
  ///
  /// Parse failures aren't cached.
  ///
  /// - Parameters:
  ///   - source: The template source.
  ///   - options: The whitespace handling to parse with.
  /// - Returns: The parsed template.
  /// - Throws: An ``EdgeToolsTemplateError`` when the source is not a valid template.
  public func template(
    source: String,
    options: EdgeToolsTemplate.Options = EdgeToolsTemplate.Options()
  ) throws(EdgeToolsTemplateError) -> EdgeToolsTemplate {
    let key = Key(source: source, options: options)
    if let cached = self.state.withLock({ $0.access(key) }) {
      return cached
    }
    let template = try EdgeToolsTemplate(source: source, options: options)
    return self.state.withLock { $0.insert(template, for: key) }
  }

  /// Removes the cached template for a source and options.
  ///
  /// - Parameters:
  ///   - source: The template source.
  ///   - options: The options the template was parsed with.
  /// - Returns: The removed template, or `nil` if the cache didn't contain it.
  @discardableResult
  public func removeTemplate(
    source: String,
    options: EdgeToolsTemplate.Options = EdgeToolsTemplate.Options()
  ) -> EdgeToolsTemplate? {
    self.state.withLock { $0.remove(Key(source: source, options: options)) }
  }

  /// Removes every cached template.
  public func removeAll() {
    self.state.withLock { state in
      state.templates.removeAll()
      state.estimatedMemorySize = 0
    }
  }
}

// MARK: - State

private struct Key: Hashable {
  let source: String
  let options: EdgeToolsTemplate.Options
}

private struct State {
  var policy: EdgeToolsTemplateCache.Policy
  var templates = OrderedDictionary<Key, EdgeToolsTemplate>()
  var estimatedMemorySize = 0

  mutating func access(_ key: Key) -> EdgeToolsTemplate? {
    guard let template = self.templates.removeValue(forKey: key) else {
      return nil
    }
    self.templates[key] = template
    return template
  }

  mutating func insert(_ template: EdgeToolsTemplate, for key: Key) -> EdgeToolsTemplate {
    if let cached = self.access(key) {
      return cached
    }
    if let maximum = self.policy.maximumMemorySize, template.estimatedMemorySize > maximum {
      return template
    }
    self.templates[key] = template
    self.estimatedMemorySize += template.estimatedMemorySize
    self.evict()
    return template
  }

  mutating func remove(_ key: Key) -> EdgeToolsTemplate? {
    let template = self.templates.removeValue(forKey: key)
    self.estimatedMemorySize -= template?.estimatedMemorySize ?? 0
    return template
  }

  mutating func evict() {
    while !self.templates.isEmpty && self.exceedsLimits {
      self.estimatedMemorySize -= self.templates.removeFirst().value.estimatedMemorySize
    }
  }

  private var exceedsLimits: Bool {
    let exceedsCount = self.policy.maximumCount.map { self.templates.count > $0 } ?? false
    let exceedsMemory =
      self.policy.maximumMemorySize.map { self.estimatedMemorySize > $0 } ?? false
    return exceedsCount || exceedsMemory
  }
}
