import Foundation

#if System
  import SystemPackage
#endif

extension URL {
  static let testTokenizerModel = Bundle.module.url(
    forResource: "test_tokenizer",
    withExtension: "model"
  )!
}

#if System
  extension FilePath {
    static let testTokenizerModel = FilePath(URL.testTokenizerModel.path())
  }
#endif
