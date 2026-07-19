#if Foundation
  import SystemPackage

  #if canImport(FoundationEssentials)
    import FoundationEssentials
  #else
    import Foundation
  #endif

  package func decodeModelConfiguration<Configuration: Decodable>(
    _ configuration: Configuration.Type,
    in directoryURL: URL,
    decoder: JSONDecoder = JSONDecoder()
  ) throws -> Configuration? {
    let directoryPath = FilePath(directoryURL.path())
    let configurationPaths = [
      directoryPath.appending("configuration.json"),
      directoryPath.appending("config.json")
    ]
    guard let configurationPath = configurationPaths.first(where: fileExists(at:)) else {
      return nil
    }
    return try decoder.decode(Configuration.self, from: Data(try readFile(at: configurationPath)))
  }
#endif
