#if !$Embedded
  import Observation

  package typealias _ObservationRegistrar = ObservationRegistrar
#else
  package struct _ObservationRegistrar: Sendable {
    package init() {}
  }
#endif
