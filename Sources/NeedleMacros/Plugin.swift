import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct NeedleMacrosPlugin: CompilerPlugin {
  let providingMacros: [any Macro.Type] = [
    NeedleGenerableMacro.self,
    NeedleGuideMacro.self,
    NeedleIgnoredMacro.self
  ]
}
