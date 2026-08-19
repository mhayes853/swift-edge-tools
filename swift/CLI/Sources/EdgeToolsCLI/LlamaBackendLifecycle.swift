import Darwin
import EdgeTools

package func initializeLlamaBackendIfRequested(arguments: [String]) {
  guard argumentsRequestLlamaBackend(arguments) else { return }
  LlamaBackend.initialize()
  atexit(freeLlamaBackend)
}

private func argumentsRequestLlamaBackend(_ arguments: [String]) -> Bool {
  let options = arguments.prefix { $0 != "--" }
  return options.indices.contains { index in
    let argument = options[index]
    if argument == "--engine" {
      let valueIndex = options.index(after: index)
      return valueIndex < options.endIndex
        && EngineKind(argument: options[valueIndex]) == .llama
    }
    guard argument.hasPrefix("--engine=") else { return false }
    return EngineKind(argument: String(argument.dropFirst("--engine=".count))) == .llama
  }
}

private func freeLlamaBackend() {
  LlamaBackend.free()
}
