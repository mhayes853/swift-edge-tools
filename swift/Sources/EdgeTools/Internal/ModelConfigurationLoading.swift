#if System
  import SystemPackage

  package func decodeModelConfiguration<Configuration: Decodable>(
    _ configuration: Configuration.Type,
    in directoryPath: FilePath,
    decoder: EdgeToolsJSONDecoder = EdgeToolsJSONDecoder()
  ) throws -> Configuration? {
    let configurationPaths = [
      directoryPath.appending("configuration.json"),
      directoryPath.appending("config.json")
    ]
    guard let configurationPath = configurationPaths.first(where: fileExists(at:)) else {
      return nil
    }
    return try decoder.decode(Configuration.self, from: readFile(at: configurationPath))
  }
#endif

#if Foundation
  package import _EdgeToolsFoundation

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
