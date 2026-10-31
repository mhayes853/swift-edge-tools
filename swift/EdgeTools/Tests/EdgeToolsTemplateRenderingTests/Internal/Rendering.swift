import EdgeToolsTemplateRendering
import OrderedCollections

func render(
  _ source: String,
  _ context: OrderedDictionary<String, EdgeToolsValue> = [:],
  options: EdgeToolsTemplate.Options = EdgeToolsTemplate.Options()
) throws -> String {
  try EdgeToolsTemplate(source: source, options: options).render(context: context)
}

func lines(_ lines: String...) -> String {
  lines.joined(separator: "\n")
}
