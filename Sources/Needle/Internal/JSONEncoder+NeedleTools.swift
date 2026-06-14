import Foundation

extension JSONEncoder {
  static let needleTools = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    return encoder
  }()
}
