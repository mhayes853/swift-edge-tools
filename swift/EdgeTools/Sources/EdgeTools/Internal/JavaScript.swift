#if JS && canImport(JavaScriptKit)
  import EdgeToolsCore
  import JavaScriptEventLoop
  import JavaScriptKit
  import OrderedCollections

  // MARK: - JSRemote

  extension JSRemote where T == JSObject {
    package func promiseValue(
      failure: @Sendable @escaping (JSValue?) -> any Error,
      operation: @Sendable @escaping (JSObject) -> JSValue
    ) async throws {
      _ = try await self.promiseValue(
        transform: { _ in () },
        failure: failure,
        operation: operation
      )
    }

    package func promiseValue<Output: Sendable>(
      transform: @Sendable @escaping (JSValue) throws -> Output,
      failure: @Sendable @escaping (JSValue?) -> any Error,
      operation: @Sendable @escaping (JSObject) -> JSValue
    ) async throws -> Output {
      let result: Result<Output, any Error> = await withUnsafeContinuation { continuation in
        Task { @concurrent in
          await self.withJSObject { object in
            guard let promise = operation(object).object else {
              continuation.resume(returning: .failure(failure(nil)))
              return
            }
            JSPromise(unsafelyWrapping: promise)
              .then(
                success: { value in
                  do {
                    continuation.resume(returning: .success(try transform(value)))
                  } catch {
                    continuation.resume(returning: .failure(error))
                  }
                  return .undefined
                },
                failure: { value in
                  continuation.resume(returning: .failure(failure(value)))
                  return .undefined
                }
              )
          }
        }
      }
      return try result.get()
    }
  }

  // MARK: - JSObject

  extension JSObject {
    package func promisingCall(
      this receiver: JSObject? = nil,
      arguments: [JSValue] = []
    ) -> JSValue {
      let reflect = JSObject.global["Reflect"].object!
      let apply = reflect["apply"].object!
      let bind = apply["bind"].object!
      let boundApply = bind(
        this: apply,
        arguments: [
          .undefined,
          self.jsValue,
          receiver?.jsValue ?? .undefined,
          arguments.jsValue
        ]
      )
      .object!
      let promise = JSPromise.resolve(JSValue.undefined)
      return promise.jsObject["then"].object!(
        this: promise.jsObject,
        arguments: [boundApply.jsValue]
      )
    }
  }

  // MARK: - JSValue

  extension JSValue {
    var stringValue: String? {
      if let string = string {
        return string
      }
      if let boolean = boolean {
        return String(boolean)
      }
      if let number = number {
        return String(number)
      }
      return nil
    }
  }

  // MARK: - EdgeToolsValue

  extension EdgeToolsValue: ConvertibleToJSValue {
    public var jsValue: JSValue {
      switch self {
      case .string(let value): return value.jsValue
      case .boolean(let value): return value.jsValue
      case .array(let value): return value.jsValue
      case .object(let value):
        let object = JSObject()
        for (key, value) in value {
          object[key] = value.jsValue
        }
        return object.jsValue
      case .number(let value): return value.jsValue
      case .integer(let value): return Double(value).jsValue
      case .null: return .null
      }
    }
  }

  extension EdgeToolsValue: ConstructibleFromJSValue {
    public static func construct(from value: JSValue) -> Self? {
      if value.isNull {
        return .null
      }
      if let string = value.string {
        return .string(string)
      }
      if let boolean = value.boolean {
        return .boolean(boolean)
      }
      if let number = value.number {
        return Int(exactly: number).map(Self.integer) ?? .number(number)
      }
      if let array = value.array {
        var values = [Self]()
        for value in array {
          guard let value = Self.construct(from: value) else { return nil }
          values.append(value)
        }
        return .array(values)
      }
      guard let object = value.object,
        let keys = [String]
          .construct(from: JSObject.global["Object"].object!["keys"].object!(object))
      else { return nil }

      var values = OrderedDictionary<String, Self>()
      for key in keys {
        guard let value = Self.construct(from: object[key]) else { return nil }
        values[key] = value
      }
      return .object(values)
    }
  }
#endif
