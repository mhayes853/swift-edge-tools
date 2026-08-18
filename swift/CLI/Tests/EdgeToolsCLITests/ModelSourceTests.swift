import CustomDump
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `ModelSource tests` {
  @Test
  func `Always Resolves A Cached Main Revision Through Hub Snapshot`() async throws {
    let calls = LockedBox([(String, String, URL)]())
    let cache = URL(fileURLWithPath: "/cache")
    let expected = URL(fileURLWithPath: "/cache/models/org/model")
    let source = ModelSource(
      location: .huggingFace(repo: "org/model", revision: "main"),
      cacheDirectory: cache
    )

    let result = try await source.resolve { repo, revision, cacheDirectory, _ in
      calls.withValue { $0.append((repo, revision, cacheDirectory)) }
      return expected
    }

    expectNoDifference(result, expected)
    expectNoDifference(calls.value.map(\.0), ["org/model"])
    expectNoDifference(calls.value.map(\.1), ["main"])
    expectNoDifference(calls.value.map(\.2), [cache])
  }

  @Test
  func `Downloads Every File Without A Quant`() async throws {
    let globs = LockedBox<[String]?>(["unset"])
    _ = try await ModelSource(location: .huggingFace(repo: "org/model", revision: "main"))
      .resolve { _, _, _, requested in
        globs.value = requested
        return URL(fileURLWithPath: "/cache")
      }

    expectNoDifference(globs.value, nil)
  }

  @Test
  func `Narrows The Download To The Requested Quant`() async throws {
    let globs = LockedBox<[String]?>(nil)
    _ = try await ModelSource(
      location: .huggingFace(repo: "org/model", revision: "main"),
      quant: "Q4_K_M"
    )
    .resolve { _, _, _, requested in
      globs.value = requested
      return URL(fileURLWithPath: "/cache")
    }

    expectNoDifference(globs.value, ["*Q4_K_M*.gguf", "*.json", "*.jinja"])
  }
}
