import Foundation
import PackagePlugin

@main
struct PatchPlugin: BuildToolPlugin {
  func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
    let packageDirectory = context.package.directoryURL
    let sourceRoot = target.directoryURL.appendingPathComponent("xgrammar")
    let patch = packageDirectory.appendingPathComponent(
      "Patches/XGrammar/0001-support-single-threaded-wasi.patch"
    )
    let script = packageDirectory.appendingPathComponent("Scripts/prepare-patched-sources.sh")
    let outputRoot = context.pluginWorkDirectoryURL.appendingPathComponent("patched")
    let sourcePaths = [
      "cpp/grammar_compiler.cc",
      "cpp/grammar_functor.cc",
      "cpp/grammar_matcher.cc",
      "cpp/support/thread_pool.h",
      "cpp/support/thread_safe_cache.h"
    ]
    let outputPaths = sourcePaths + ["cpp/support/threading.h"]

    return [
      .buildCommand(
        displayName: "Preparing patched XGrammar sources",
        executable: URL(fileURLWithPath: "/bin/sh"),
        arguments: [script.path, sourceRoot.path, patch.path, outputRoot.path] + sourcePaths,
        inputFiles: [script, patch] + sourcePaths.map {
          sourceRoot.appendingPathComponent($0)
        },
        outputFiles: outputPaths.map { outputRoot.appendingPathComponent($0) }
      )
    ]
  }
}
