#if ChatTemplates
  import CustomDump
  import EdgeToolsTokenizers
  import Testing

  @Suite
  struct `EdgeToolsChatTemplate tests` {
    @Test
    func `Formats The Current Time`() throws {
      let rendered = try EdgeToolsChatTemplate(source: "{{ strftime_now('%Y-%m-%d %H:%M:%S') }}")
        .render(context: [:])
      expectNoDifference(
        rendered.wholeMatch(of: /[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/) != nil,
        true
      )
    }

    @Test
    func `Formats A Pinned Time In UTC`() throws {
      let template = EdgeToolsChatTemplate(
        source: lines(
          "{{ strftime_now('%d %b %Y') }}",
          "{{ strftime_now('%A, %B %e %y %j') }}",
          "{{ strftime_now('%F %T %I:%M %p %u %w %%') }}",
          "{{ strftime_now('%-d/%-m %G-W%V %c') }}"
        )
      )
      expectNoDifference(
        try template.render(context: ["edge_tools_now": 1_577_923_200]),
        lines(
          "02 Jan 2020",
          "Thursday, January  2 20 002",
          "2020-01-02 00:00:00 12:00 AM 4 4 %",
          "2/1 2020-W01 Thu Jan  2 00:00:00 2020"
        )
      )
      expectNoDifference(
        try template.render(context: ["edge_tools_now": 1_609_459_199]),
        lines(
          "31 Dec 2020",
          "Thursday, December 31 20 366",
          "2020-12-31 23:59:59 11:59 PM 4 4 %",
          "31/12 2020-W53 Thu Dec 31 23:59:59 2020"
        )
      )
    }

    @Test
    func `Renders With Chat Template Whitespace Handling`() throws {
      let template = EdgeToolsChatTemplate(
        source: lines(
          "{% for message in messages %}",
          "    {% if message.role == 'user' %}",
          "User: {{ message.content }}",
          "    {% endif %}",
          "{% endfor %}"
        )
      )
      expectNoDifference(
        try template.render(context: ["messages": [["role": "user", "content": "Hi"]]]),
        "User: Hi\n"
      )
    }

    @Test
    func `Renders Correctly After Evicting Cached Templates`() throws {
      let templates = (0..<6).map { EdgeToolsChatTemplate(source: "{{ value }}-\($0)") }
      for _ in 0..<2 {
        for (index, template) in templates.enumerated() {
          expectNoDifference(try template.render(context: ["value": "v"]), "v-\(index)")
        }
      }
    }

    @Test
    func `Rejects Pinned Times That Are Not Whole Seconds`() {
      let error = #expect(throws: EdgeToolsChatTemplateError.self) {
        try EdgeToolsChatTemplate(source: "{{ strftime_now('%Y') }}")
          .render(context: ["edge_tools_now": 1.5])
      }
      expectNoDifference(
        error,
        EdgeToolsChatTemplateError(
          code: .renderFailure,
          message: "`edge_tools_now` must be whole seconds since the Unix epoch."
        )
      )
    }

    @Test
    func `Rejects Contexts That Are Not Objects`() {
      let error = #expect(throws: EdgeToolsChatTemplateError.self) {
        try EdgeToolsChatTemplate(source: "").render(context: [])
      }
      expectNoDifference(
        error,
        EdgeToolsChatTemplateError(
          code: .renderFailure,
          message: "The template context must be a JSON object."
        )
      )
    }

    @Test
    func `Wraps Template Errors`() {
      let error = #expect(throws: EdgeToolsChatTemplateError.self) {
        try EdgeToolsChatTemplate(source: "{{ raise_exception('Unsupported role') }}")
          .render(context: [:])
      }
      expectNoDifference(error?.code, .renderFailure)
      expectNoDifference(error?.message.hasPrefix("Unsupported role at row 1"), true)
    }
  }

  private func lines(_ lines: String...) -> String {
    lines.joined(separator: "\n")
  }
#endif
