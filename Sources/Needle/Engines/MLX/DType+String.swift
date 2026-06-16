#if SwiftNeedleMLX
  import MLX

  // MARK: - Init

  extension DType {
    public init?(string: String) {
      switch string {
      case "float16": self = .float16
      case "float32": self = .float32
      case "float64": self = .float64
      case "bfloat16": self = .bfloat16
      case "complex64": self = .complex64
      case "int2", "bool": self = .bool
      case "int8": self = .int8
      case "int16": self = .int16
      case "int32": self = .int32
      case "int64": self = .int64
      case "uint8": self = .uint8
      case "uint16": self = .uint16
      case "uint32": self = .uint32
      case "uint64": self = .uint64
      default: return nil
      }
    }
  }

  // MARK: - Configuration

  extension NeedleModelConfiguration {
    public var mlxDType: DType {
      DType(string: self.dtype) ?? .bfloat16
    }
  }
#endif
