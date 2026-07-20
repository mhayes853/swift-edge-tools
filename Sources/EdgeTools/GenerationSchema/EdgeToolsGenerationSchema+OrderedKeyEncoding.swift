import OrderedCollections
import yyjson

// MARK: - Ordered key encoding

extension EdgeToolsGenerationSchema {
  public func orderedKeyEncoded() -> String {
    OrderedKeyJSONWriter.encode(self.edgeToolsValue)
  }
}

package struct OrderedKeyJSONWriter: ~Copyable {
  let document: UnsafeMutablePointer<yyjson_mut_doc>

  deinit { yyjson_mut_doc_free(self.document) }

  package static func encode(_ value: EdgeToolsValue) -> String {
    let document = yyjson_mut_doc_new(nil)!
    let writer = Self(document: document)
    let root = writer.value(value)
    yyjson_mut_doc_set_root(writer.document, root)

    var length = 0
    let output = yyjson_mut_write(writer.document, YYJSON_WRITE_INF_AND_NAN_AS_NULL, &length)!
    defer { output.deallocate() }

    let bytes = UnsafeRawPointer(output).assumingMemoryBound(to: UInt8.self)
    let buffer = UnsafeBufferPointer<UInt8>(start: bytes, count: length)
    return String(bytes: buffer, encoding: .utf8)!
  }

  func value(_ value: EdgeToolsValue) -> UnsafeMutablePointer<yyjson_mut_val>? {
    switch value {
    case .null: yyjson_mut_null(self.document)
    case .boolean(let value): yyjson_mut_bool(self.document, value)
    case .integer(let value): yyjson_mut_sint(self.document, Int64(value))
    case .number(let value): yyjson_mut_real(self.document, value)
    case .string(let value): self.string(value)
    case .array(let values): self.array(values)
    case .object(let object): self.object(object)
    }
  }

  func string(_ value: String) -> UnsafeMutablePointer<yyjson_mut_val>? {
    value.withCString { characters in
      yyjson_mut_strncpy(self.document, characters, value.utf8.count)
    }
  }

  private func array(_ values: [EdgeToolsValue]) -> UnsafeMutablePointer<yyjson_mut_val>? {
    guard let array = yyjson_mut_arr(self.document) else { return nil }
    for value in values {
      guard let encodedValue = self.value(value), yyjson_mut_arr_add_val(array, encodedValue)
      else { return nil }
    }
    return array
  }

  private func object(
    _ object: OrderedDictionary<String, EdgeToolsValue>
  ) -> UnsafeMutablePointer<yyjson_mut_val>? {
    guard let encodedObject = yyjson_mut_obj(self.document) else { return nil }
    for (key, value) in object {
      guard let encodedKey = self.string(key),
        let encodedValue = self.value(value),
        yyjson_mut_obj_add(encodedObject, encodedKey, encodedValue)
      else { return nil }
    }
    return encodedObject
  }
}
