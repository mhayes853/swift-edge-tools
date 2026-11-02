import CustomDump
import EdgeToolsTemplateRendering
import Testing

@Suite
struct `EdgeToolsTemplateCache tests` {
  @Test
  func `Caches Templates By Source And Options`() throws {
    let cache = EdgeToolsTemplateCache(policy: EdgeToolsTemplateCache.Policy())
    let trimmed = EdgeToolsTemplate.Options(trimBlocks: true)
    let source = "{% if true %}\nyes{% endif %}"

    expectNoDifference(try cache.template(source: source).render(), "\nyes")
    expectNoDifference(try cache.template(source: source, options: trimmed).render(), "yes")
    expectNoDifference(try cache.template(source: source).render(), "\nyes")
    expectNoDifference(cache.count, 2)
  }

  @Test
  func `Evicts The Least Recently Used Template Beyond The Count Limit`() throws {
    let cache = EdgeToolsTemplateCache(policy: EdgeToolsTemplateCache.Policy(maximumCount: 2))
    _ = try cache.template(source: "a")
    _ = try cache.template(source: "b")
    _ = try cache.template(source: "a")
    _ = try cache.template(source: "c")

    expectNoDifference(removedSources(from: cache, ["a", "b", "c"]), ["a", "c"])
  }

  @Test
  func `Evicts The Least Recently Used Template Beyond The Memory Limit`() throws {
    let size = try EdgeToolsTemplate(source: "{{ a }}").estimatedMemorySize
    let cache = EdgeToolsTemplateCache(
      policy: EdgeToolsTemplateCache.Policy(maximumMemorySize: size * 2)
    )
    _ = try cache.template(source: "{{ a }}")
    _ = try cache.template(source: "{{ b }}")
    _ = try cache.template(source: "{{ a }}")
    _ = try cache.template(source: "{{ c }}")

    expectNoDifference(cache.estimatedMemorySize, size * 2)
    expectNoDifference(
      removedSources(from: cache, ["{{ a }}", "{{ b }}", "{{ c }}"]),
      ["{{ a }}", "{{ c }}"]
    )
  }

  @Test
  func `Returns Templates Larger Than The Memory Limit Without Caching Them`() throws {
    let source = "{{ greeting }}, {{ name }}!"
    let size = try EdgeToolsTemplate(source: source).estimatedMemorySize
    let cache = EdgeToolsTemplateCache(
      policy: EdgeToolsTemplateCache.Policy(maximumMemorySize: size - 1)
    )
    _ = try cache.template(source: "a")

    let template = try cache.template(source: source)

    expectNoDifference(try template.render(context: ["greeting": "Hi", "name": "Al"]), "Hi, Al!")
    expectNoDifference(removedSources(from: cache, ["a", source]), ["a"])
  }

  @Test
  func `Evicts Templates When The Policy Lowers Its Limits`() throws {
    let cache = EdgeToolsTemplateCache(policy: EdgeToolsTemplateCache.Policy())
    for source in ["a", "b", "c"] {
      _ = try cache.template(source: source)
    }

    cache.policy.maximumCount = 1

    expectNoDifference(removedSources(from: cache, ["a", "b", "c"]), ["c"])
  }

  @Test
  func `Removes A Single Template`() throws {
    let cache = EdgeToolsTemplateCache(policy: EdgeToolsTemplateCache.Policy())
    let kept = try cache.template(source: "{{ kept }}")
    _ = try cache.template(source: "{{ removed }}")

    let removed = cache.removeTemplate(source: "{{ removed }}")

    expectNoDifference(removed?.source, "{{ removed }}")
    expectNoDifference(cache.removeTemplate(source: "{{ removed }}")?.source, nil)
    expectNoDifference(cache.count, 1)
    expectNoDifference(cache.estimatedMemorySize, kept.estimatedMemorySize)
  }

  @Test
  func `Removes All Templates`() throws {
    let cache = EdgeToolsTemplateCache(policy: EdgeToolsTemplateCache.Policy())
    _ = try cache.template(source: "a")
    _ = try cache.template(source: "b")

    cache.removeAll()

    expectNoDifference(cache.count, 0)
    expectNoDifference(cache.estimatedMemorySize, 0)
  }

  @Test
  func `Does Not Cache Templates That Fail To Parse`() {
    let cache = EdgeToolsTemplateCache(policy: EdgeToolsTemplateCache.Policy())

    let error = #expect(throws: EdgeToolsTemplateError.self) {
      try cache.template(source: "{% if %}")
    }

    expectNoDifference(error?.code, .invalidSyntax)
    expectNoDifference(cache.count, 0)
  }

  @Test
  func `Caches Concurrently Requested Templates Once`() async throws {
    let cache = EdgeToolsTemplateCache(policy: EdgeToolsTemplateCache.Policy())
    try await withThrowingTaskGroup(of: Void.self) { group in
      for _ in 0..<16 {
        group.addTask {
          _ = try cache.template(source: "{{ shared }}")
        }
      }
      try await group.waitForAll()
    }

    expectNoDifference(cache.count, 1)
    expectNoDifference(
      cache.estimatedMemorySize,
      try EdgeToolsTemplate(source: "{{ shared }}").estimatedMemorySize
    )
  }
}

// MARK: - Helpers

private func removedSources(from cache: EdgeToolsTemplateCache, _ sources: [String]) -> [String] {
  sources.filter { cache.removeTemplate(source: $0) != nil }
}
