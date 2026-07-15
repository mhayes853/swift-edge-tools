import Foundation

extension URL {
  static let testTokenizerModel = Bundle.module.url(
    forResource: "test_tokenizer",
    withExtension: "model"
  )!
}
