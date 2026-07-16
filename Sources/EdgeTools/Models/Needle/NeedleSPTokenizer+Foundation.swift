import Foundation

extension NeedleSPTokenizer {
  public init(modelURL: URL) throws {
    guard modelURL.isFileURL, !modelURL.hasDirectoryPath else {
      throw NeedleSPTokenizerError.fileNotFound(atPath: modelURL.path())
    }

    do {
      try self.init(data: Data(contentsOf: modelURL))
    } catch let error as NeedleSPTokenizerError {
      throw error
    } catch {
      throw NeedleSPTokenizerError.fileNotFound(atPath: modelURL.path())
    }
  }
}
