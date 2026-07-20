import CustomDump
import EdgeTools
import Foundation
import Testing

@Suite
struct `HuggingFace Backend JSON tests` {
  @Test
  func `Projects Only Metadata Fields`() throws {
    let source = #"""
      {
        "model": {"vocab": {"decoder": "not metadata"}},
        "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "first"},
        "added_tokens": [{"content": "[decoder]"}],
        "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}]},
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}]}
      }
      """#

    let projected = try projectedBackendJSON(from: source)
    let object = try JSONDecoder().decode(ProjectedBackend.self, from: Data(projected.utf8))

    expectNoDifference(object.decoder.type, "Sequence")
    expectNoDifference(object.normalizer.type, "Sequence")
    expectNoDifference(object.preTokenizer.type, "Metaspace")
  }

  @Test
  func `Projected Metadata Matches Complete Backend`() throws {
    let source = #"""
      {
        "decoder": {"type": "ByteLevel"},
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"vocab": {"token": 0}}
      }
      """#
    let projected = try projectedBackendJSON(from: source)

    expectNoDifference(
      try XGRTokenizerInfo.metadata(huggingFaceBackendJSON: projected),
      try XGRTokenizerInfo.metadata(huggingFaceBackendJSON: source)
    )
  }

  @Test
  func `Does Not Include Large Vocabulary`() throws {
    let vocabulary = String(repeating: #""token":0,"#, count: 100_000) + #""last":1"#
    let source =
      #"{"decoder":{"type":"ByteFallback"},"model":{"vocab":{"#
      + vocabulary
      + #"}}}"#

    let projected = try projectedBackendJSON(from: source)

    expectNoDifference(projected, #"{"decoder":{"type":"ByteFallback"}}"#)
    #expect(projected.utf8.count < 100)
  }

  private func projectedBackendJSON(from source: String) throws -> String {
    let bytes = Array(source.utf8)
    return try bytes.withUnsafeBufferPointer { try huggingFaceBackendJSON(from: $0) }
  }

  private struct ProjectedBackend: Decodable {
    struct Component: Decodable {
      let type: String
    }

    let decoder: Component
    let normalizer: Component
    let preTokenizer: Component

    enum CodingKeys: String, CodingKey {
      case decoder
      case normalizer
      case preTokenizer = "pre_tokenizer"
    }
  }
}
