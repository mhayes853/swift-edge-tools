import OrderedCollections
import yyjson

package func decodeEdgeToolsJSON(_ bytes: [UInt8]) throws -> EdgeToolsValue {
  guard !bytes.isEmpty else { throw EdgeToolsError.emptyJSONInput }

  var bytes = bytes
  var error = yyjson_read_err()
  let document = bytes.withUnsafeMutableBytes { buffer in
    yyjson_read_opts(
      buffer.baseAddress?.assumingMemoryBound(to: CChar.self),
      buffer.count,
      YYJSON_READ_NOFLAG,
      nil,
      &error
    )
  }
  guard let document else {
    throw EdgeToolsError(
      code: .invalidJSON,
      message:
        "\(error.msg.map(String.init(cString:)) ?? "Invalid JSON.") (byte offset \(error.pos))."
    )
  }
  defer { yyjson_doc_free(document) }

  guard let root = yyjson_doc_get_root(document) else {
    throw EdgeToolsError.emptyJSONInput
  }
  return try edgeToolsValue(from: root)
}

private func edgeToolsValue(from value: UnsafeMutablePointer<yyjson_val>) throws -> EdgeToolsValue {
  if yyjson_is_null(value) {
    return .null
  } else if yyjson_is_bool(value) {
    return .boolean(yyjson_get_bool(value))
  } else if yyjson_is_sint(value) {
    guard let integer = Int(exactly: yyjson_get_sint(value)) else {
      throw EdgeToolsError(
        code: .jsonIntegerOutOfRange,
        message: "JSON integer is outside the supported range."
      )
    }
    return .integer(integer)
  } else if yyjson_is_uint(value) {
    let integer = yyjson_get_uint(value)
    return Int(exactly: integer).map(EdgeToolsValue.integer)
      ?? .number(Double(integer))
  } else if yyjson_is_real(value) {
    let number = yyjson_get_real(value)
    guard number.isFinite else {
      throw EdgeToolsError(code: .nonFiniteJSONNumber, message: "JSON number is not finite.")
    }
    return .number(number)
  } else if yyjson_is_str(value) {
    return .string(edgeToolsString(from: value))
  } else if yyjson_is_arr(value) {
    return try edgeToolsArray(from: value)
  } else if yyjson_is_obj(value) {
    return try edgeToolsObject(from: value)
  } else {
    throw EdgeToolsError.invalidJSONValue
  }
}

private func edgeToolsArray(
  from value: UnsafeMutablePointer<yyjson_val>
) throws -> EdgeToolsValue {
  var iterator = yyjson_arr_iter()
  guard yyjson_arr_iter_init(value, &iterator) else {
    throw EdgeToolsError.invalidJSONValue
  }
  var values = [EdgeToolsValue]()
  values.reserveCapacity(Int(yyjson_arr_size(value)))
  while let element = yyjson_arr_iter_next(&iterator) {
    values.append(try edgeToolsValue(from: element))
  }
  return .array(values)
}

private func edgeToolsObject(
  from value: UnsafeMutablePointer<yyjson_val>
) throws -> EdgeToolsValue {
  var iterator = yyjson_obj_iter()
  guard yyjson_obj_iter_init(value, &iterator) else {
    throw EdgeToolsError.invalidJSONValue
  }
  var object = OrderedDictionary<String, EdgeToolsValue>()
  object.reserveCapacity(Int(yyjson_obj_size(value)))
  while let key = yyjson_obj_iter_next(&iterator) {
    guard let child = yyjson_obj_iter_get_val(key) else {
      throw EdgeToolsError.invalidJSONValue
    }
    object[edgeToolsString(from: key)] = try edgeToolsValue(from: child)
  }
  return .object(object)
}

private func edgeToolsString(from value: UnsafeMutablePointer<yyjson_val>) -> String {
  let count = Int(yyjson_get_len(value))
  guard let characters = yyjson_get_str(value) else { return "" }
  let bytes = UnsafeRawPointer(characters).assumingMemoryBound(to: UInt8.self)
  return String(decoding: UnsafeBufferPointer(start: bytes, count: count), as: UTF8.self)
}
