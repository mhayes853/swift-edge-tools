#if $Embedded
  package protocol EdgeToolsEncodable {}
  package protocol EdgeToolsDecodable {}
#else
  package protocol EdgeToolsEncodable: Encodable {}
  package protocol EdgeToolsDecodable: Decodable {}
#endif

package typealias EdgeToolsCodable = EdgeToolsEncodable & EdgeToolsDecodable
