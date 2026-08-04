#if !$Embedded
  import Observation

  package typealias _ObservationRegistrar = ObservationRegistrar
#else
  // NB: Embedded Swift ships neither the Observation module nor the key paths its registrar
  // requires, so observation degrades to a no-op there.
  package struct _ObservationRegistrar: Sendable {
    package init() {}
  }
#endif
