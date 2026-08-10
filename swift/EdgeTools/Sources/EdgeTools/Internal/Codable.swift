#if $Embedded
  protocol EdgeToolsEncodable {}
  protocol EdgeToolsDecodable {}
#else
  protocol EdgeToolsEncodable: Encodable {}
  protocol EdgeToolsDecodable: Decodable {}
#endif

typealias EdgeToolsCodable = EdgeToolsEncodable & EdgeToolsDecodable
