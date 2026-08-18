#if canImport(simd) && !$Embedded
  import simd
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
    self.buffer.reserveCapacity(capacity)
  }

  public var tokenIds: [EdgeToolsToken.ID] {
    self.buffer
  }

  public func reset() {
    self.buffer.removeAll(keepingCapacity: true)
    self.writeIndex = 0
  }

  public func seed(_ tokens: some Sequence<EdgeToolsToken.ID>) {
    self.buffer.removeAll(keepingCapacity: true)
    self.buffer.append(contentsOf: tokens.suffix(self.capacity))
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
  private var weights = [Float]()
  private var nucleusBuckets = [Float](repeating: 0, count: nucleusBucketCount)
  private var gatheredLogits = [Float]()
  private var gatheredIds = [EdgeToolsToken.ID]()

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
    logits: inout MutableSpan<Float>,
    bitmask: GrammarBitmask? = nil
  ) -> EdgeToolsCPUSample {
    logits.withUnsafeMutableBufferPointer { self.sample(logits: $0, bitmask: bitmask) }
  }

  public func sample(
    logits: UnsafeMutableBufferPointer<Float>,
    bitmask: GrammarBitmask? = nil
  ) -> EdgeToolsCPUSample {
    precondition(!logits.isEmpty, "Cannot sample from empty logits.")
    let penalizesHistory = self.parameters.penalizesHistory
    if penalizesHistory {
      self.applyHistoryPenalties(logits: logits)
    }
    let sample = self.constrainedSample(logits: logits, bitmask: bitmask)
    if penalizesHistory {
      self.history.append(sample.tokenId)
    }
    return sample
  }

  private func constrainedSample(
    logits: UnsafeMutableBufferPointer<Float>,
    bitmask: GrammarBitmask?
  ) -> EdgeToolsCPUSample {
    guard let bitmask else {
      return self.sample(logits: UnsafeBufferPointer(logits), ids: nil)
    }
    guard let count = self.gatherPermitted(logits: UnsafeBufferPointer(logits), mask: bitmask)
    else {
      applyBitmaskCPU(logits: logits, mask: bitmask)
      return self.sample(logits: UnsafeBufferPointer(logits), ids: nil)
    }
    return self.gatheredLogits.withUnsafeBufferPointer { values in
      self.gatheredIds.withUnsafeBufferPointer { ids in
        self.sample(
          logits: UnsafeBufferPointer(rebasing: values[..<count]),
          ids: UnsafeBufferPointer(rebasing: ids[..<count])
        )
      }
    }
  }

  private func sample(
    logits: UnsafeBufferPointer<Float>,
    ids: UnsafeBufferPointer<EdgeToolsToken.ID>?
  ) -> EdgeToolsCPUSample {
    let extremes = logitExtremes(logits)
    let index = self.sampledTokenId(logits: logits, extremes: extremes)
    return EdgeToolsCPUSample(
      tokenId: ids.map { $0[index] } ?? index,
      confidence: tokenConfidence(top1: extremes.top1, top2: extremes.top2)
    )
  }

  private func gatherPermitted(
    logits: UnsafeBufferPointer<Float>,
    mask: GrammarBitmask
  ) -> Int? {
    validateBitmaskCoverage(mask: mask, vocabularySize: logits.count)
    guard permittedCount(mask: mask, limit: sparseGatherLimit) != nil else { return nil }
    if self.gatheredLogits.count < sparseGatherLimit {
      self.gatheredLogits = [Float](repeating: 0, count: sparseGatherLimit)
      self.gatheredIds = [EdgeToolsToken.ID](repeating: 0, count: sparseGatherLimit)
    }
    var written = 0
    self.gatheredLogits.withUnsafeMutableBufferPointer { values in
      self.gatheredIds.withUnsafeMutableBufferPointer { ids in
        mask.storage.withUnsafeBytes { raw in
          var byteOffset = 0
          while byteOffset + maskBlockSize <= raw.count {
            let block = raw.loadUnaligned(fromByteOffset: byteOffset, as: MaskBlock.self)
            if any(block .!= MaskBlock()) {
              for lane in 0..<MaskBlock.scalarCount {
                var word = UInt64(littleEndian: block[lane])
                let base = (byteOffset + lane * 8) * 8
                while word != 0 {
                  let tokenId = base + word.trailingZeroBitCount
                  guard tokenId < logits.count else { return }
                  values[written] = logits[tokenId]
                  ids[written] = tokenId
                  written += 1
                  word &= word &- 1
                }
              }
            }
            byteOffset += maskBlockSize
          }
          while byteOffset < raw.count {
            var word = maskWord(raw, from: byteOffset)
            while word != 0 {
              let tokenId = byteOffset * 8 + word.trailingZeroBitCount
              guard tokenId < logits.count else { return }
              values[written] = logits[tokenId]
              ids[written] = tokenId
              written += 1
              word &= word &- 1
            }
            byteOffset += 8
          }
        }
      }
    }
    return written > 0 ? written : nil
  }

  private func applyHistoryPenalties(logits: UnsafeMutableBufferPointer<Float>) {
    let repetitionPenalty = self.parameters.repetitionPenalty ?? 1
    let presencePenalty = self.parameters.presencePenalty ?? 0
    let tokenIds = self.history.tokenIds
    for (offset, tokenId) in tokenIds.enumerated()
    where logits.indices.contains(tokenId) && !tokenIds[..<offset].contains(tokenId) {
      let logit = logits[tokenId]
      let scaled = logit < 0 ? logit * repetitionPenalty : logit / repetitionPenalty
      logits[tokenId] = scaled - presencePenalty
    }
  }

  private func sampledTokenId(
    logits: UnsafeBufferPointer<Float>,
    extremes: LogitExtremes
  ) -> EdgeToolsToken.ID {
    guard !self.parameters.isGreedy else { return extremes.top1Index }
    let invTemperature = 1 / (self.parameters.temperature ?? 0.6)
    let maxLogit = extremes.top1
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

    let total = topP.map { _ in self.fillWeights(from: logits, shiftedBy: maxLogit) }
    let logSumExp = total.map { maxLogit + logf($0) }
    var candidates: [LogitCandidate]
    if let topK {
      candidates = topCandidates(logits, count: topK)
    } else if let topP, let total, let logSumExp {
      candidates = self.nucleusCandidates(
        logits: logits,
        maxLogit: maxLogit,
        mass: topP,
        total: total,
        logSumExp: logSumExp
      )
    } else {
      candidates = candidatesAtOrAbove(maxLogit + logf(minP!), in: logits)
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
    let total = self.fillWeights(from: logits, shiftedBy: maxLogit, scaledBy: invTemperature)
    let uniform = self.nextUniform() * total
    return self.weights.withUnsafeBufferPointer {
      firstIndexCrossing(UnsafeBufferPointer(rebasing: $0[..<logits.count]), threshold: uniform)
    }
  }

  private func fillWeights(
    from logits: UnsafeBufferPointer<Float>,
    shiftedBy maximum: Float,
    scaledBy scale: Float = 1
  ) -> Float {
    if self.weights.count < logits.count {
      self.weights = [Float](repeating: 0, count: logits.count)
    }
    return self.weights.withUnsafeMutableBufferPointer {
      expShifted(
        logits,
        shiftedBy: maximum,
        scaledBy: scale,
        into: UnsafeMutableBufferPointer(rebasing: $0[..<logits.count])
      )
    }
  }

  private func nucleusCandidates(
    logits: UnsafeBufferPointer<Float>,
    maxLogit: Float,
    mass: Float,
    total: Float,
    logSumExp: Float
  ) -> [LogitCandidate] {
    let probe = topCandidates(logits, count: Swift.min(nucleusProbeCount, logits.count))
    if probe.reduce(Float.zero, { $0 + expf($1.logit - logSumExp) }) >= mass {
      return probe
    }
    let cutoff = self.nucleusCutoff(logits: logits, maxLogit: maxLogit, mass: mass, total: total)
    var candidates = candidatesAtOrAbove(cutoff, in: logits)
    let boundary = candidates.partition {
      $0.logit < cutoff + nucleusLogitRange / Float(nucleusBucketCount)
    }
    candidates[boundary...].sort(by: >)
    return candidates
  }

  private func nucleusCutoff(
    logits: UnsafeBufferPointer<Float>,
    maxLogit: Float,
    mass: Float,
    total: Float
  ) -> Float {
    let scale = Float(nucleusBucketCount) / nucleusLogitRange
    for index in self.nucleusBuckets.indices {
      self.nucleusBuckets[index] = 0
    }
    let floor = maxLogit - nucleusLogitRange
    let floorBlock = SIMD16<Float>(repeating: floor)
    let width = SIMD16<Float>.scalarCount
    self.weights.withUnsafeBufferPointer { weights in
      var index = 0
      while index + width <= logits.count {
        let block = UnsafeRawPointer(logits.baseAddress!.advanced(by: index))
          .loadUnaligned(as: SIMD16<Float>.self)
        if any(block .>= floorBlock) {
          for lane in 0..<width where block[lane] >= floor {
            let bucket = bucketIndex(maxLogit - block[lane], scale: scale)
            self.nucleusBuckets[bucket] += weights[index + lane]
          }
        }
        index += width
      }
      while index < logits.count {
        if logits[index] >= floor {
          self.nucleusBuckets[bucketIndex(maxLogit - logits[index], scale: scale)] += weights[index]
        }
        index += 1
      }
    }
    let target = mass * total
    var cumulative = Float.zero
    for (bucket, bucketMass) in self.nucleusBuckets.enumerated() {
      cumulative += bucketMass
      if cumulative >= target {
        return maxLogit - Float(bucket + 1) / scale
      }
    }
    return maxLogit - nucleusLogitRange
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

// MARK: - Candidate Selection

private struct LogitCandidate: Comparable {
  let logit: Float
  let id: EdgeToolsToken.ID

  static func < (lhs: Self, rhs: Self) -> Bool {
    lhs.logit != rhs.logit ? lhs.logit < rhs.logit : lhs.id > rhs.id
  }
}

private let sparseGatherLimit = 4096

private let nucleusProbeCount = 64

private let nucleusBucketCount = 512

private let nucleusLogitRange = Float(88)

private func topCandidates(
  _ logits: UnsafeBufferPointer<Float>,
  count: Int
) -> [LogitCandidate] {
  guard let base = logits.baseAddress else { return [] }
  var heap = Heap<LogitCandidate>()
  heap.reserveCapacity(count)
  let width = SIMD16<Float>.scalarCount
  var threshold = SIMD16<Float>(repeating: -.infinity)
  var index = 0
  while index + width <= logits.count {
    let block = UnsafeRawPointer(base.advanced(by: index)).loadUnaligned(as: SIMD16<Float>.self)
    if heap.count < count || any(block .>= threshold) {
      for lane in 0..<width {
        insert(LogitCandidate(logit: block[lane], id: index + lane), into: &heap, limit: count)
      }
      if heap.count == count, let minimum = heap.min {
        threshold = SIMD16<Float>(repeating: minimum.logit)
      }
    }
    index += width
  }
  while index < logits.count {
    insert(LogitCandidate(logit: logits[index], id: index), into: &heap, limit: count)
    index += 1
  }
  var descending = [LogitCandidate]()
  descending.reserveCapacity(heap.count)
  while let maximum = heap.popMax() {
    descending.append(maximum)
  }
  return descending
}

private func candidatesAtOrAbove(
  _ threshold: Float,
  in logits: UnsafeBufferPointer<Float>
) -> [LogitCandidate] {
  guard let base = logits.baseAddress else { return [] }
  let width = SIMD16<Float>.scalarCount
  let thresholdBlock = SIMD16<Float>(repeating: threshold)
  var candidates = [LogitCandidate]()
  var index = 0
  while index + width <= logits.count {
    let block = UnsafeRawPointer(base.advanced(by: index)).loadUnaligned(as: SIMD16<Float>.self)
    if any(block .>= thresholdBlock) {
      for lane in 0..<width where block[lane] >= threshold {
        candidates.append(LogitCandidate(logit: block[lane], id: index + lane))
      }
    }
    index += width
  }
  while index < logits.count {
    if logits[index] >= threshold {
      candidates.append(LogitCandidate(logit: logits[index], id: index))
    }
    index += 1
  }
  return candidates
}

@inline(always)
private func bucketIndex(_ distanceBelowMaximum: Float, scale: Float) -> Int {
  Swift.min(Int(distanceBelowMaximum * scale), nucleusBucketCount - 1)
}

@inline(always)
private func maskWord(_ raw: UnsafeRawBufferPointer, from byteOffset: Int) -> UInt64 {
  guard raw.count - byteOffset < 8 else {
    return UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: byteOffset, as: UInt64.self))
  }
  var word = UInt64.zero
  for offset in byteOffset..<raw.count {
    word |= UInt64(raw[offset]) &<< UInt64((offset - byteOffset) * 8)
  }
  return word
}

private typealias MaskBlock = SIMD8<UInt64>

private let maskBlockSize = MemoryLayout<MaskBlock>.size

private func permittedCount(mask: GrammarBitmask, limit: Int) -> Int? {
  mask.storage.withUnsafeBytes { raw in
    var count = 0
    var byteOffset = 0
    while byteOffset + maskBlockSize <= raw.count {
      let block = raw.loadUnaligned(fromByteOffset: byteOffset, as: MaskBlock.self)
      if any(block .!= MaskBlock()) {
        for lane in 0..<MaskBlock.scalarCount {
          count += block[lane].nonzeroBitCount
        }
        if count > limit {
          return nil
        }
      }
      byteOffset += maskBlockSize
    }
    while byteOffset < raw.count {
      count += maskWord(raw, from: byteOffset).nonzeroBitCount
      if count > limit {
        return nil
      }
      byteOffset += 8
    }
    return count
  }
}

private func insert(
  _ candidate: LogitCandidate,
  into heap: inout Heap<LogitCandidate>,
  limit: Int
) {
  if heap.count < limit {
    heap.insert(candidate)
  } else if let minimum = heap.min, candidate > minimum {
    heap.replaceMin(with: candidate)
  }
}
// MARK: - LogitExtremes

struct LogitExtremes {
  let top1Index: EdgeToolsToken.ID
  let top1: Float

  let top2: Float
}

// MARK: - SIMD Reductions

private let laneOffsets = SIMD16<Int32>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

@inline(always)
func logitExtremes(_ values: UnsafeBufferPointer<Float>) -> LogitExtremes {
  guard let base = values.baseAddress, !values.isEmpty else {
    return LogitExtremes(top1Index: 0, top1: -.infinity, top2: -.infinity)
  }
  let width = SIMD16<Float>.scalarCount
  var laneTop1 = SIMD16<Float>(repeating: -.infinity)
  var laneTop2 = SIMD16<Float>(repeating: -.infinity)
  var laneIndices = SIMD16<Int32>(repeating: 0)
  var index = 0
  while index + width <= values.count {
    let block = UnsafeRawPointer(base.advanced(by: index)).loadUnaligned(as: SIMD16<Float>.self)
    let winners = simdMax(laneTop1, block)
    let losers = simdMin(laneTop1, block)
    laneIndices.replace(with: laneOffsets &+ Int32(index), where: block .> laneTop1)
    laneTop2 = simdMax(laneTop2, losers.replacing(with: -.infinity, where: losers .== winners))
    laneTop1 = winners
    index += width
  }

  var top1 = -Float.infinity
  var top1Index = 0
  for lane in 0..<width {
    let value = laneTop1[lane]
    let valueIndex = Int(laneIndices[lane])
    if value > top1 || (value == top1 && valueIndex < top1Index) {
      top1 = value
      top1Index = valueIndex
    }
  }

  var top2 = -Float.infinity
  for lane in 0..<width {
    top2 = Swift.max(top2, laneTop2[lane])
    if laneTop1[lane] < top1 {
      top2 = Swift.max(top2, laneTop1[lane])
    }
  }

  while index < values.count {
    let value = values[index]
    if value > top1 {
      top2 = Swift.max(top2, top1)
      top1 = value
      top1Index = index
    } else if value > top2 && value < top1 {
      top2 = value
    }
    index += 1
  }
  return LogitExtremes(top1Index: top1Index, top1: top1, top2: top2)
}

@inline(always)
func expShifted(
  _ values: UnsafeBufferPointer<Float>,
  shiftedBy maximum: Float,
  scaledBy scale: Float,
  into destination: UnsafeMutableBufferPointer<Float>
) -> Float {
  guard let source = values.baseAddress, let target = destination.baseAddress else { return 0 }
  let width = SIMD16<Float>.scalarCount
  let shift = SIMD16<Float>(repeating: maximum)
  let factor = SIMD16<Float>(repeating: scale)
  var totals = SIMD16<Float>(repeating: 0)
  var index = 0
  while index + width <= values.count {
    let block = UnsafeRawPointer(source.advanced(by: index)).loadUnaligned(as: SIMD16<Float>.self)
    let weights = simdExp((block - shift) * factor)
    UnsafeMutableRawPointer(target.advanced(by: index))
      .storeBytes(of: weights, as: SIMD16<Float>.self)
    totals += weights
    index += width
  }
  var total = totals.sum()
  while index < values.count {
    let weight = expf((values[index] - maximum) * scale)
    destination[index] = weight
    total += weight
    index += 1
  }
  return total
}

@inline(always)
func firstIndexCrossing(
  _ weights: UnsafeBufferPointer<Float>,
  threshold: Float
) -> Int {
  guard let base = weights.baseAddress, !weights.isEmpty else { return 0 }
  let width = SIMD16<Float>.scalarCount
  var accumulated = Float.zero
  var index = 0
  while index + width <= weights.count {
    let block = UnsafeRawPointer(base.advanced(by: index)).loadUnaligned(as: SIMD16<Float>.self)
    let blockTotal = block.sum()
    if accumulated + blockTotal > threshold {
      break
    }
    accumulated += blockTotal
    index += width
  }
  while index < weights.count {
    accumulated += weights[index]
    if threshold < accumulated {
      return index
    }
    index += 1
  }
  return weights.count - 1
}

private let smallestExpInput = Float(-87.3)

private let log2Inverse = Float(1.442_695_04)

private let expRoundingMagic = Float(1 << 23) + Float(1 << 22)

@inline(always)
private func simdExp(_ values: SIMD16<Float>) -> SIMD16<Float> {
  let magic = SIMD16<Float>(repeating: expRoundingMagic)
  let scaled = simdMax(values, SIMD16(repeating: smallestExpInput)) * log2Inverse
  let rounded = scaled + magic
  let fraction = scaled - (rounded - magic)
  var polynomial = SIMD16<Float>(repeating: 0.001_333_355_8)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.009_618_129)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.055_504_108_7)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.240_226_506_9)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.693_147_180_5)
  polynomial = polynomial * fraction + SIMD16(repeating: 1)
  let exponent = unsafeBitCast(rounded, to: SIMD16<UInt32>.self) &<< 23
  let bits = unsafeBitCast(polynomial, to: SIMD16<UInt32>.self) &+ exponent
  return unsafeBitCast(bits, to: SIMD16<Float>.self)
}

@inline(always)
private func simdMax(_ lhs: SIMD16<Float>, _ rhs: SIMD16<Float>) -> SIMD16<Float> {
  #if canImport(simd) && !$Embedded
    simd_max(lhs, rhs)
  #else
    lhs.replacing(with: rhs, where: rhs .> lhs)
  #endif
}

@inline(always)
private func simdMin(_ lhs: SIMD16<Float>, _ rhs: SIMD16<Float>) -> SIMD16<Float> {
  #if canImport(simd) && !$Embedded
    simd_min(lhs, rhs)
  #else
    lhs.replacing(with: rhs, where: rhs .< lhs)
  #endif
}
