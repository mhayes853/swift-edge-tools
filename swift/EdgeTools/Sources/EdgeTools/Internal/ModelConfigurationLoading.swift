#if Foundation
  import _EdgeToolsFoundation

  package func decodeModelConfiguration<Configuration: Decodable>(
    _ configuration: Configuration.Type,
    in directoryURL: URL,
    decoder: JSONDecoder = JSONDecoder()
  ) throws -> Configuration? {
    let configurationURLs = [
      directoryURL.appending(path: "configuration.json"),
      directoryURL.appending(path: "config.json")
    ]
    guard
      let configurationURL = configurationURLs.first(where: {
        FileManager.default.fileExists(atPath: $0.path())
      })
    else {
      return nil
    }
    return try decoder.decode(Configuration.self, from: Data(contentsOf: configurationURL))
  }

  package func decodeModelConfiguration<Configuration: Decodable>(
    _ configuration: Configuration.Type,
    named fileName: String,
    in directoryURL: URL,
    decoder: JSONDecoder = JSONDecoder()
  ) throws -> Configuration? {
    let configurationURL = directoryURL.appending(path: fileName)
    guard FileManager.default.fileExists(atPath: configurationURL.path()) else { return nil }
    return try decoder.decode(Configuration.self, from: Data(contentsOf: configurationURL))
  }
#endif
