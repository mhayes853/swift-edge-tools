#if canImport(Accelerate) && !$Embedded
  import Accelerate
#endif

import EdgeToolsCore
import HeapModule

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Musl)
  import Musl
#elseif canImport(WASILibc)
  import WASILibc
#elseif canImport(Android)
  import Android
#elseif canImport(ucrt)
  import ucrt
#endif

// MARK: - EdgeToolsCPUSample

public struct EdgeToolsCPUSample: Hashable, Sendable {
  public let tokenId: EdgeToolsToken.ID
  public let confidence: Float

  public init(tokenId: EdgeToolsToken.ID, confidence: Float) {
    self.tokenId = tokenId
    self.confidence = confidence
  }
}

// MARK: - EdgeToolsCPUTokenHistory

public final class EdgeToolsCPUTokenHistory {
  public let capacity: Int

  private var buffer = [EdgeToolsToken.ID]()
  private var writeIndex = 0

  public init(capacity: Int) {
    precondition(capacity > 0, "Token history capacity must be greater than zero.")
    self.capacity = capacity
  }

  public var tokenIds: [EdgeToolsToken.ID] {
    self.buffer
  }

  public func reset() {
    self.buffer.removeAll(keepingCapacity: true)
    self.writeIndex = 0
  }

  public func seed(_ tokens: some Sequence<EdgeToolsToken.ID>) {
    self.buffer = Array(tokens.suffix(self.capacity))
    self.writeIndex = self.buffer.count % self.capacity
  }

  public func append(_ tokenId: EdgeToolsToken.ID) {
    if self.buffer.count < self.capacity {
      self.buffer.append(tokenId)
    } else {
      self.buffer[self.writeIndex] = tokenId
    }
    self.writeIndex = (self.writeIndex + 1) % self.capacity
  }
}

// MARK: - EdgeToolsCPUFusedSampler

public final class EdgeToolsCPUFusedSampler {
  public let parameters: EdgeToolsFusedSamplingParameters
  public let history: EdgeToolsCPUTokenHistory

  private var rngState: UInt64

  public convenience init(parameters: EdgeToolsFusedSamplingParameters) {
    self.init(
      parameters: parameters,
      history: EdgeToolsCPUTokenHistory(capacity: parameters.repetitionContextSize ?? 20)
    )
  }

  public init(parameters: EdgeToolsFusedSamplingParameters, history: EdgeToolsCPUTokenHistory) {
    self.parameters = parameters
    self.history = history
    #if $Embedded
      self.rngState = parameters.seed ?? 0x9E37_79B9_7F4A_7C15
    #else
      self.rngState = parameters.seed ?? UInt64.random(in: 0...UInt64.max)
    #endif
  }

  public func sample(
    logits: inout [Float],
    bitmask: GrammarBitmask? = nil
  ) -> EdgeToolsCPUSample {
    logits.withUnsafeMutableBufferPointer { self.sample(logits: $0, bitmask: bitmask) }
  }

  public func sample(
    logits: UnsafeMutableBufferPointer<Float>,
    bitmask: GrammarBitmask? = nil
  ) -> EdgeToolsCPUSample {
    precondition(!logits.isEmpty, "Cannot sample from empty logits.")
    if let bitmask {
      applyBitmaskCPU(logits: logits, mask: bitmask)
    }
    let (top1, top2) = topTwoContiguous(UnsafeBufferPointer(logits))
    let confidence = tokenConfidence(top1: top1, top2: top2)
    if self.parameters.penalizesHistory {
      self.applyHistoryPenalties(logits: logits)
    }
    let tokenId = self.sampledTokenId(logits: UnsafeBufferPointer(logits))
    if self.parameters.penalizesHistory {
      self.history.append(tokenId)
    }
    return EdgeToolsCPUSample(tokenId: tokenId, confidence: confidence)
  }

  private func applyHistoryPenalties(logits: UnsafeMutableBufferPointer<Float>) {
    let repetitionPenalty = self.parameters.repetitionPenalty ?? 1
    let presencePenalty = self.parameters.presencePenalty ?? 0
    var penalized = Set<EdgeToolsToken.ID>()
    for tokenId in self.history.tokenIds where penalized.insert(tokenId).inserted {
      guard logits.indices.contains(tokenId) else { continue }
      let logit = logits[tokenId]
      let scaled = logit < 0 ? logit * repetitionPenalty : logit / repetitionPenalty
      logits[tokenId] = scaled - presencePenalty
    }
  }

  private func sampledTokenId(logits: UnsafeBufferPointer<Float>) -> EdgeToolsToken.ID {
    guard !self.parameters.isGreedy else { return argmaxContiguous(logits) }
    let invTemperature = 1 / (self.parameters.temperature ?? 0.6)
    let maxLogit = maxContiguous(logits)
    let topK = self.parameters.topK.flatMap { $0 > 0 && $0 < logits.count ? $0 : nil }
    let topP = self.parameters.topP.flatMap { $0 < 1 ? $0 : nil }
    let minP = self.parameters.minP.flatMap { $0 > 0 ? $0 : nil }

    guard topK != nil || topP != nil || minP != nil else {
      return self.categoricalTokenId(
        over: logits,
        maxLogit: maxLogit,
        invTemperature: invTemperature
      )
    }

    let logSumExp = topP.map { _ in maxLogit + logf(sumExpContiguous(logits, shiftedBy: maxLogit)) }
    var candidates: [LogitCandidate]
    if let topK {
      candidates = topCandidates(logits, count: topK)
    } else if let topP, let logSumExp {
      candidates = nucleusCandidates(logits, mass: topP, logSumExp: logSumExp)
    } else {
      let threshold = maxLogit + logf(minP!)
      candidates = logits.enumerated()
        .filter { $0.element >= threshold }
        .map { LogitCandidate(logit: $0.element, id: $0.offset) }
    }

    if let topP, let logSumExp {
      var cumulative = Float.zero
      var keepCount = 0
      for candidate in candidates {
        guard cumulative < topP else { break }
        keepCount += 1
        cumulative += expf(candidate.logit - logSumExp)
      }
      candidates.removeLast(candidates.count - keepCount)
    }
    if let minP, topK != nil || topP != nil {
      let threshold = maxLogit + logf(minP)
      candidates = candidates.filter { $0.logit >= threshold }
    }
    return self.categoricalTokenId(candidates, invTemperature: invTemperature)
  }

  private func categoricalTokenId(
    _ candidates: [LogitCandidate],
    invTemperature: Float
  ) -> EdgeToolsToken.ID {
    guard candidates.count > 1 else { return candidates[0].id }
    let reference = candidates.max { $0.logit < $1.logit }!.logit
    let weights = candidates.map { expf(($0.logit - reference) * invTemperature) }
    let total = weights.reduce(0, +)
    let uniform = self.nextUniform() * total
    var accumulated = Float.zero
    for (candidate, weight) in zip(candidates, weights) {
      accumulated += weight
      if uniform < accumulated {
        return candidate.id
      }
    }
    return candidates[candidates.count - 1].id
  }

  private func categoricalTokenId(
    over logits: UnsafeBufferPointer<Float>,
    maxLogit: Float,
    invTemperature: Float
  ) -> EdgeToolsToken.ID {
    let cutoff = maxLogit - (negligibleLogitOffset / Swift.max(invTemperature, 1e-6))
    var total = Float.zero
    for logit in logits where logit >= cutoff {
      total += expf((logit - maxLogit) * invTemperature)
    }
    let uniform = self.nextUniform() * total
    var accumulated = Float.zero
    var lastCandidate = 0
    for (index, logit) in logits.enumerated() where logit >= cutoff {
      accumulated += expf((logit - maxLogit) * invTemperature)
      lastCandidate = index
      if uniform < accumulated {
        return index
      }
    }
    return lastCandidate
  }

  private func nextUniform() -> Float {
    self.rngState &+= 0x9E37_79B9_7F4A_7C15
    var mixed = self.rngState
    mixed = (mixed ^ (mixed >> 30)) &* 0xBF58_476D_1CE4_E5B9
    mixed = (mixed ^ (mixed >> 27)) &* 0x94D0_49BB_1331_11EB
    mixed ^= mixed >> 31
    return Float(mixed >> 40) * (1.0 / Float(1 << 24))
  }
}

// MARK: - Bitmask

public func applyBitmaskCPU(logits: UnsafeMutableBufferPointer<Float>, mask: GrammarBitmask) {
  validateBitmaskCoverage(mask: mask, vocabularySize: logits.count)
  guard let base = logits.baseAddress else { return }
  var index = 0
  for byte in mask.storage {
    guard index < logits.count else { break }
    if byte == .max {
      index += 8
      continue
    }
    if index + 8 <= logits.count {
      let pointer = UnsafeMutableRawPointer(base.advanced(by: index))
      let block = pointer.loadUnaligned(as: SIMD8<Float>.self)
      pointer.storeBytes(of: block + bitmaskSIMDTable[Int(byte)], as: SIMD8<Float>.self)
    } else {
      for bit in 0..<8 where index + bit < logits.count {
        if byte & (1 &<< UInt8(bit)) == 0 {
          logits[index + bit] = -.infinity
        }
      }
    }
    index += 8
  }
}

// MARK: - Candidate Selection

private struct LogitCandidate: Comparable {
  let logit: Float
  let id: EdgeToolsToken.ID

  static func < (lhs: Self, rhs: Self) -> Bool {
    lhs.logit != rhs.logit ? lhs.logit < rhs.logit : lhs.id > rhs.id
  }
}

private func topCandidates(
  _ logits: UnsafeBufferPointer<Float>,
  count: Int
) -> [LogitCandidate] {
  var heap = Heap<LogitCandidate>()
  for (index, logit) in logits.enumerated() {
    let candidate = LogitCandidate(logit: logit, id: index)
    if heap.count < count {
      heap.insert(candidate)
    } else if let minimum = heap.min, candidate > minimum {
      heap.replaceMin(with: candidate)
    }
  }
  var descending = [LogitCandidate]()
  descending.reserveCapacity(heap.count)
  while let maximum = heap.popMax() {
    descending.append(maximum)
  }
  return descending
}

private func nucleusCandidates(
  _ logits: UnsafeBufferPointer<Float>,
  mass: Float,
  logSumExp: Float
) -> [LogitCandidate] {
  var count = Swift.min(256, logits.count)
  while true {
    let candidates = topCandidates(logits, count: count)
    let cumulative = candidates.reduce(Float.zero) { $0 + expf($1.logit - logSumExp) }
    if cumulative >= mass || count == logits.count {
      return candidates
    }
    count = Swift.min(count * 4, logits.count)
  }
}

// MARK: - SIMD Reductions

private let negligibleLogitOffset = Float(20)

private let bitmaskSIMDTable = (0..<256)
  .map { byte in
    SIMD8<Float>((0..<8).map { bit in ((byte >> bit) & 1) != 0 ? 0 : -Float.infinity })
  }

@inline(always)
func maxContiguous(_ values: UnsafeBufferPointer<Float>) -> Float {
  #if canImport(Accelerate) && !$Embedded
    guard !values.isEmpty else { return -.infinity }
    var maximum = Float.zero
    vDSP_maxv(values.baseAddress!, 1, &maximum, vDSP_Length(values.count))
    return maximum
  #else
    guard !values.isEmpty else { return -.infinity }
    let width = SIMD16<Float>.scalarCount
    var best = SIMD16<Float>(repeating: -.infinity)
    var index = 0
    while index + width <= values.count {
      let block = UnsafeRawPointer(values.baseAddress!.advanced(by: index))
        .loadUnaligned(as: SIMD16<Float>.self)
      best = best.replacing(with: block, where: block .> best)
      index += width
    }
    var maximum = best.max()
    while index < values.count {
      maximum = Swift.max(maximum, values[index])
      index += 1
    }
    return maximum
  #endif
}

@inline(always)
func argmaxContiguous(_ values: UnsafeBufferPointer<Float>) -> Int {
  #if canImport(Accelerate) && !$Embedded
    guard !values.isEmpty else { return 0 }
    var maximum = Float.zero
    var index = vDSP_Length.zero
    vDSP_maxvi(values.baseAddress!, 1, &maximum, &index, vDSP_Length(values.count))
    return Int(index)
  #else
    let (top1Index, _, _) = topTwoIndexed(values)
    return top1Index
  #endif
}

@inline(always)
func topTwoContiguous(_ values: UnsafeBufferPointer<Float>) -> (top1: Float, top2: Float) {
  let (_, top1, top2) = topTwoIndexed(values)
  return (top1, top2)
}

@inline(always)
func sumExpContiguous(
  _ values: UnsafeBufferPointer<Float>,
  shiftedBy maximum: Float
) -> Float {
  guard !values.isEmpty else { return 0 }
  let width = SIMD16<Float>.scalarCount
  let cutoff = maximum - negligibleLogitOffset
  let cutoffBlock = SIMD16<Float>(repeating: cutoff)
  var sum = Float.zero
  var index = 0
  while index + width <= values.count {
    let block = UnsafeRawPointer(values.baseAddress!.advanced(by: index))
      .loadUnaligned(as: SIMD16<Float>.self)
    if any(block .>= cutoffBlock) {
      for lane in 0..<width where block[lane] >= cutoff {
        sum += expf(block[lane] - maximum)
      }
    }
    index += width
  }
  while index < values.count {
    if values[index] >= cutoff {
      sum += expf(values[index] - maximum)
    }
    index += 1
  }
  return sum
}

private let topTwoLaneOffsets = SIMD16<Int32>(
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
)

@inline(always)
private func topTwoIndexed(
  _ values: UnsafeBufferPointer<Float>
) -> (top1Index: Int, top1: Float, top2: Float) {
  guard !values.isEmpty else { return (0, -.infinity, -.infinity) }
  let width = SIMD16<Float>.scalarCount
  var bestValues = SIMD16<Float>(repeating: -.infinity)
  var bestIndices = SIMD16<Int32>(repeating: 0)
  var secondValues = SIMD16<Float>(repeating: -.infinity)
  var index = 0
  while index + width <= values.count {
    let block = UnsafeRawPointer(values.baseAddress!.advanced(by: index))
      .loadUnaligned(as: SIMD16<Float>.self)
    let isBest = block .> bestValues
    secondValues.replace(with: bestValues, where: isBest)
    bestValues.replace(with: block, where: isBest)
    bestIndices.replace(with: topTwoLaneOffsets &+ Int32(index), where: isBest)
    let isSecond = (block .> secondValues) .& (.!isBest)
    secondValues.replace(with: block, where: isSecond)
    index += width
  }

  var top1 = -Float.infinity
  var top1Index = 0
  var top2 = -Float.infinity
  for lane in 0..<width {
    let value = bestValues[lane]
    let valueIndex = Int(bestIndices[lane])
    if value > top1 || (value == top1 && valueIndex < top1Index) {
      top2 = Swift.max(top2, top1)
      top1 = value
      top1Index = valueIndex
    } else if value > top2 {
      top2 = value
    }
    if secondValues[lane] > top2 {
      top2 = secondValues[lane]
    }
  }
  while index < values.count {
    let value = values[index]
    if value > top1 {
      top2 = top1
      top1 = value
      top1Index = index
    } else if value > top2 {
      top2 = value
    }
    index += 1
  }
  return (top1Index, top1, top2)
}
