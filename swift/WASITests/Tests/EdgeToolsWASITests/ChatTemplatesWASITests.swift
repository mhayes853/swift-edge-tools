import EdgeTools
import Testing

@Suite
struct `ChatTemplatesWASI tests` {
  @Test
  func `Renders Chat Templates`() throws {
    let template = EdgeToolsChatTemplate(
      source: """
        {%- for message in messages %}
        <|{{ message.role }}|>{{ message.content | trim }}
        {%- endfor %}
        {{ tools | tojson }} {{ strftime_now('%d %b %Y') }}
        """
    )
    let rendered = try template.render(
      context: [
        "messages": [["role": "user", "content": " Hello "]],
        "tools": [["name": "weather", "threshold": 0.5]],
        "edge_tools_now": 1_577_923_200
      ]
    )

    #expect(rendered == #"<|user|>Hello[{"name": "weather", "threshold": 0.5}] 02 Jan 2020"#)
  }

  @Test
  func `Invalid Template Throws`() {
    #expect(throws: EdgeToolsChatTemplateError.self) {
      _ = try EdgeToolsChatTemplate(source: "{% if true %}").render(context: [:])
    }
  }
}
