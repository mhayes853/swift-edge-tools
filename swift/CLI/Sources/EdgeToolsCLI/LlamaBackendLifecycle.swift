import EdgeTools

package func llamaBackendIfRequested(arguments: [String]) -> LlamaBackend? {
  guard argumentsRequestLlamaBackend(arguments) else { return nil }
  return LlamaBackend()
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
