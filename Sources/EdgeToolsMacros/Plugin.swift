import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct EdgeToolsMacrosPlugin: CompilerPlugin {
  let providingMacros: [any Macro.Type] = [
    EdgeToolsGenerableMacro.self,
    EdgeToolsGuideMacro.self,
    EdgeToolsIgnoredMacro.self
  ]
}
