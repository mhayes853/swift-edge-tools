import EdgeTools

protocol TranscriptContextTestable: AnyObject, EdgeToolsEngineContext {
  var transcript: EdgeToolsTranscript { get }
  var reasoningEffort: EdgeToolsReasoningEffort { get set }
  func fork() -> Self
}

func expectTranscriptContextSemantics(_ context: some TranscriptContextTestable) {
  expectNoDifference(context.transcript.messages, [.system("System")])
  expectNoDifference(context.reasoningEffort, .high)
  expectNoDifference(context.tools.map(\.name), ["echo"])

  let fork = context.fork()
  context.reasoningEffort = .none

  expectNoDifference(context.reasoningEffort, .none)
  expectNoDifference(fork.reasoningEffort, .high)
}
