#if !$Embedded
  import Observation

  typealias _ObservationRegistrar = ObservationRegistrar
#else
  struct _ObservationRegistrar: Sendable {
    init() {}
  }
#endif
