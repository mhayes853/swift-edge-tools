#if FullFoundation
  @_exported import Foundation
#elseif canImport(FoundationEssentials)
  @_exported import FoundationEssentials
#else
  @_exported import Foundation
#endif
