import CustomDump
import EdgeToolsTemplateRendering
import OrderedCollections
import Testing

@Suite
struct `EdgeToolsTemplateJSON tests` {
  @Test
  func `Serializes With Python Separators In Insertion Order`() throws {
    expectNoDifference(
      try render(#"{{ {"b": [1, 2.5, true, none], "a": {"c": "d"}} | tojson }}"#),
      #"{"b": [1, 2.5, true, null], "a": {"c": "d"}}"#
    )
  }

  @Test
  func `Keeps Non ASCII Characters Unless Escaping Is Requested`() throws {
    expectNoDifference(try render(#"{{ "héllo 🌍" | tojson }}"#), #""héllo 🌍""#)
    expectNoDifference(
      try render(#"{{ "héllo 🌍" | tojson(ensure_ascii=true) }}"#),
      #""h\u00e9llo \ud83c\udf0d""#
    )
    expectNoDifference(
      try render(#"{{ {"é": "ü"} | tojson(ensure_ascii=true) }}"#),
      #"{"\u00e9": "\u00fc"}"#
    )
  }

  @Test
  func `Escapes Quotes Backslashes And Control Characters`() throws {
    expectNoDifference(
      try render(
        "{{ text | tojson }}",
        ["text": "quote\" backslash\\ newline\n tab\t bell\u{07} delete\u{7F}"]
      ),
      #""quote\" backslash\\ newline\n tab\t bell\u0007 delete\#u{7F}""#
    )
  }

  @Test
  func `Sorts Keys At Every Depth When Requested`() throws {
    expectNoDifference(
      try render(#"{{ {"b": 1, "a": {"d": 2, "c": 3}} | tojson(sort_keys=true) }}"#),
      #"{"a": {"c": 3, "d": 2}, "b": 1}"#
    )
  }

  @Test
  func `Indents Like Python`() throws {
    expectNoDifference(
      try render(#"{{ {"a": [1, {"b": 2}], "c": [], "d": {}} | tojson(indent=2) }}"#),
      "{\n  \"a\": [\n    1,\n    {\n      \"b\": 2\n    }\n  ],\n  \"c\": [],\n  \"d\": {}\n}"
    )
    expectNoDifference(try render("{{ [1, 2] | tojson(indent=0) }}"), "[\n1,\n2\n]")
    expectNoDifference(try render("{{ [1, 2] | tojson(1) }}"), "[\n 1,\n 2\n]")
    expectNoDifference(try render("{{ [1, 2] | tojson(indent=none) }}"), "[1, 2]")
  }

  @Test
  func `Formats Numbers Like JSON`() throws {
    expectNoDifference(
      try render("{{ numbers | tojson }}", ["numbers": [1, 1.0, 1.5, 1e20, 0.00001, -0.25]]),
      "[1, 1.0, 1.5, 1e+20, 1e-05, -0.25]"
    )
  }

  @Test
  func `Stringifies Non String Keys`() throws {
    expectNoDifference(
      try render(#"{{ {1: "a", 2.5: "b", true: "c", none: "d"} | tojson }}"#),
      #"{"1": "a", "2.5": "b", "true": "c", "null": "d"}"#
    )
  }

  @Test
  func `Rejects Unsupported Arguments`() {
    let error = #expect(throws: EdgeToolsTemplateError.self) {
      try render("{{ [1] | tojson(separators=[',', ':']) }}")
    }
    expectNoDifference(
      error?.message.hasPrefix("Unknown argument separators for function tojson"),
      true
    )
  }
}

private func render(
  _ source: String,
  _ context: OrderedDictionary<String, EdgeToolsValue> = [:]
) throws -> String {
  try EdgeToolsTemplate(source: source).render(context: context)
}
