import Foundation

// MARK: - Standard Output

/// CLI output goes here rather than through `print`.
///
/// Engine runtimes (CoreAI in particular) write diagnostics straight to file descriptor 1, which
/// would corrupt `--json`. `claimStandardOutput` takes a private duplicate of the real standard
/// output and points descriptor 1 at standard error, so those diagnostics stay visible without
/// mixing into CLI output.
public func claimStandardOutput() {
  _ = standardOutput
}

func output(_ string: String = "", terminator: String = "\n") {
  guard let data = (string + terminator).data(using: .utf8) else { return }
  try? standardOutput.write(contentsOf: data)
}

private let standardOutput: FileHandle = {
  fflush(stdout)
  let duplicate = dup(STDOUT_FILENO)
  guard duplicate >= 0, dup2(STDERR_FILENO, STDOUT_FILENO) >= 0 else {
    return FileHandle.standardOutput
  }
  setvbuf(stdout, nil, _IONBF, 0)
  return FileHandle(fileDescriptor: duplicate, closeOnDealloc: false)
}()
