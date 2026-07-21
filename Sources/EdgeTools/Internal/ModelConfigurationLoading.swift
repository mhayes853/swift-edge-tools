#if Foundation
  import Foundation

  package func decodeModelConfiguration<Configuration: Decodable>(
    _ configuration: Configuration.Type,
    in directoryURL: URL,
    decoder: JSONDecoder = JSONDecoder()
  ) throws -> Configuration? {
    let configurationURLs = [
      directoryURL.appending(path: "configuration.json"),
      directoryURL.appending(path: "config.json")
    ]
    guard let configurationURL = configurationURLs.first(where: fileExists(at:)) else {
      return nil
    }
    return try decoder.decode(Configuration.self, from: Data(contentsOf: configurationURL))
  }
#endif
