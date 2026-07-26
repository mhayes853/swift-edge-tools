import Foundation
import PackagePlugin

@main
struct PatchPlugin: BuildToolPlugin {
  func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
    let packageDirectory = context.package.directoryURL
    let sourceRoot = target.directoryURL.appendingPathComponent("xgrammar")
    let patches = [
      "patches/XGrammar/0001-support-single-threaded-wasi.patch",
      "patches/XGrammar/0002-support-wasi-without-cxx-exceptions.patch"
    ]
    .map { packageDirectory.appendingPathComponent($0) }
    let script = packageDirectory.appendingPathComponent("scripts/prepare-patched-sources.sh")
    let outputRoot = context.pluginWorkDirectoryURL.appendingPathComponent("patched")
    let sourcePaths = [
      "cpp/compiled_grammar.cc",
      "cpp/config.cc",
      "cpp/earley_parser.cc",
      "cpp/fsm.cc",
      "cpp/fsm_builder.cc",
      "cpp/grammar.cc",
      "cpp/grammar_builder.cc",
      "cpp/grammar_compiler.cc",
      "cpp/grammar_functor.cc",
      "cpp/grammar_matcher.cc",
      "cpp/grammar_parser.cc",
      "cpp/grammar_printer.cc",
      "cpp/json_schema_converter.cc",
      "cpp/json_schema_converter_ext.cc",
      "cpp/lark_converter.cc",
      "cpp/regex_converter.cc",
      "cpp/structural_tag.cc",
      "cpp/support/logging.cc",
      "cpp/support/recursion_guard.cc",
      "cpp/testing.cc",
      "cpp/tokenizer_info.cc",
      "cpp/support/thread_pool.h",
      "cpp/support/thread_safe_cache.h"
    ]
    let outputPaths =
      sourcePaths + [
        "cpp/support/no_exceptions.h",
        "cpp/support/threading.h"
      ]

    return [
      .buildCommand(
        displayName: "Preparing patched XGrammar sources",
        executable: URL(fileURLWithPath: "/bin/sh"),
        arguments: [script.path, sourceRoot.path, outputRoot.path]
          + patches.flatMap { ["--patch", $0.path] }
          + ["--prepend-include", "cpp/support/no_exceptions.h", "--"]
          + sourcePaths,
        inputFiles: [script] + patches
          + sourcePaths.map {
            sourceRoot.appendingPathComponent($0)
          },
        outputFiles: outputPaths.map { outputRoot.appendingPathComponent($0) }
      )
    ]
  }
}
