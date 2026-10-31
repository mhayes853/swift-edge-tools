import CustomDump
import EdgeToolsTemplateRendering
import OrderedCollections
import Testing

@Suite
struct `EdgeToolsTemplate tests` {
  @Test
  func `Concatenates Adjacent String Literals`() throws {
    expectNoDifference(try render(#"{{ "Hello, " 'world' "!" }}"#), "Hello, world!")
    expectNoDifference(try render("{% set message = 'one '\n    'two' %}{{ message }}"), "one two")
    expectNoDifference(try render(#"{{ ["a" "b", "c"] }}"#), "['ab', 'c']")
    expectNoDifference(try render(#"{{ ("a" "b") | upper }}"#), "AB")
    expectNoDifference(try render(#"{{ "a" ~ "b" }}"#), "ab")
  }

  @Test
  func `Finds Minimums And Maximums`() throws {
    expectNoDifference(try render("{{ [3, 1.5, 2] | min }}"), "1.5")
    expectNoDifference(try render("{{ [3, 1.5, 2] | max }}"), "3")
    expectNoDifference(try render("{{ ['b', 'c', 'a'] | min }}"), "a")
    expectNoDifference(try render("{{ ['b', 'c', 'a'] | max }}"), "c")
    expectNoDifference(try render("{{ [] | max }}"), "")
  }

  @Test
  func `Calls Custom Functions And Filters`() throws {
    let greet = EdgeToolsTemplateFunction { arguments in
      let name = arguments.positional.first?.string ?? "nobody"
      let punctuation = arguments.named["punctuation"]?.string ?? "."
      return .string("Hello, \(name)\(punctuation)")
    }
    let template = try EdgeToolsTemplate(
      source: "{{ greet('Ada', punctuation='!') }} {{ name | greet }}"
    )
    expectNoDifference(
      try template.render(context: ["name": "Grace"], functions: ["greet": greet]),
      "Hello, Ada! Hello, Grace."
    )
  }

  @Test
  func `Passes Template Values To Custom Functions As JSON Values`() throws {
    let describe = EdgeToolsTemplateFunction { arguments in
      .string(arguments.positional.map { $0.orderedJSONString() }.joined(separator: " "))
    }
    expectNoDifference(
      try EdgeToolsTemplate(source: "{{ describe([1, 2.5], {1: none}, 'a', true) }}")
        .render(functions: ["describe": describe]),
      #"[1,2.5] {"1":null} "a" true"#
    )
  }

  @Test
  func `Custom Function Errors Fail Rendering`() {
    let failing = EdgeToolsTemplateFunction { _ in
      throw EdgeToolsTemplateError(code: .renderFailure, message: "Nope")
    }
    let error = #expect(throws: EdgeToolsTemplateError.self) {
      try EdgeToolsTemplate(source: "{{ fail() }}").render(functions: ["fail": failing])
    }
    expectNoDifference(error?.code, .renderFailure)
    expectNoDifference(error?.message.hasPrefix("Nope at row 1, column 8"), true)
  }

  @Test
  func `Reports Syntax Errors With Their Location`() {
    let error = #expect(throws: EdgeToolsTemplateError.self) {
      try EdgeToolsTemplate(source: "Hello\n{{ name")
    }
    expectNoDifference(
      error,
      EdgeToolsTemplateError(
        code: .invalidSyntax,
        message: "Expected closing expression tag at row 2, column 8:\nHello\n{{ name\n       ^\n"
      )
    )
  }

  @Test
  func `Reports Render Errors With Their Location`() {
    let error = #expect(throws: EdgeToolsTemplateError.self) {
      try render("First\n{{ raise_exception('Bad input') }}\nLast")
    }
    expectNoDifference(
      error,
      EdgeToolsTemplateError(
        code: .renderFailure,
        message: "Bad input at row 2, column 19:\nFirst\n{{ raise_exception('Bad input') }}\n"
          + "                  ^\nLast\n"
      )
    )
  }

  @Test
  func `Measures And Slices Strings By Character`() throws {
    expectNoDifference(try render("{{ 'café' | length }}"), "4")
    expectNoDifference(try render("{{ 'café'[1:] }}"), "afé")
    expectNoDifference(try render("{{ 'café'[::-1] }}"), "éfac")
    expectNoDifference(try render("{% for c in 'né' %}[{{ c }}]{% endfor %}"), "[n][é]")
  }

  @Test
  func `Clamps Out Of Range Slices`() throws {
    expectNoDifference(try render("{{ [1, 2, 3][1:100] }}"), "[2, 3]")
    expectNoDifference(try render("{{ [1, 2, 3][-100:1] }}"), "[1]")
    expectNoDifference(try render("{{ 'abc'[5:] }}"), "")
  }

  @Test
  func `Renders Recursive Loops`() throws {
    let items: EdgeToolsValue = [
      ["name": "a", "children": [["name": "b", "children": [["name": "c"]]]]],
      ["name": "d"]
    ]
    expectNoDifference(
      try render(
        "{% for item in items recursive %}{{ item.name }}"
          + "{% if item.children %}({{ loop(item.children) }}){% endif %}{% endfor %}",
        ["items": items]
      ),
      "a(b(c))d"
    )
  }

  @Test
  func `Rejects Division By Zero`() {
    for source in ["{{ 1 // 0 }}", "{{ 1 % 0 }}", "{{ 1 / 0 }}"] {
      #expect(throws: EdgeToolsTemplateError.self) {
        try render(source)
      }
    }
  }

  @Test
  func `Renders Concurrently From A Shared Template`() async throws {
    let template = try EdgeToolsTemplate(
      source: "{% for i in range(n) %}{{ i }}{% if not loop.last %},{% endif %}{% endfor %}"
    )
    let outputs = try await withThrowingTaskGroup(of: String.self) { group in
      for count in 1...8 {
        group.addTask {
          try template.render(context: ["n": .integer(count)])
        }
      }
      return try await group.reduce(into: Set<String>()) { $0.insert($1) }
    }
    expectNoDifference(
      outputs,
      Set((1...8).map { count in (0..<count).map(String.init).joined(separator: ",") })
    )
  }
}

private func render(
  _ source: String,
  _ context: OrderedDictionary<String, EdgeToolsValue> = [:]
) throws -> String {
  try EdgeToolsTemplate(source: source).render(context: context)
}
