#if ChatTemplates
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

  // MARK: - UTC Formatting

  func currentUnixTime() -> Int64 {
    Int64(time(nil))
  }

  func formattedUTCTime(_ secondsSinceEpoch: Int64, format: String) -> String {
    let time = UTCTime(secondsSinceEpoch: secondsSinceEpoch)
    let scalars = Array(format.unicodeScalars)
    var output = ""
    var index = 0
    while index < scalars.count {
      guard scalars[index] == "%", index + 1 < scalars.count else {
        output.unicodeScalars.append(scalars[index])
        index += 1
        continue
      }
      var specifierIndex = index + 1
      let padding = Padding(flag: scalars[specifierIndex])
      if padding != nil && specifierIndex + 1 < scalars.count {
        specifierIndex += 1
      }
      if let formatted = time.formatted(scalars[specifierIndex], padding: padding) {
        output += formatted
      } else {
        output.unicodeScalars.append(contentsOf: scalars[index...specifierIndex])
      }
      index = specifierIndex + 1
    }
    return output
  }

  // MARK: - UTCTime

  private struct UTCTime {
    let secondsSinceEpoch: Int64
    let year: Int64
    let month: Int64
    let day: Int64
    let hour: Int64
    let minute: Int64
    let second: Int64
    let weekday: Int64
    let dayOfYear: Int64

    init(secondsSinceEpoch: Int64) {
      self.secondsSinceEpoch = secondsSinceEpoch
      let days = floorDivide(secondsSinceEpoch, 86_400)
      let secondOfDay = secondsSinceEpoch - days * 86_400
      self.hour = secondOfDay / 3_600
      self.minute = secondOfDay % 3_600 / 60
      self.second = secondOfDay % 60
      self.weekday = floorModulo(days + 4, 7)
      let (year, month, day) = civilDate(daysSinceEpoch: days)
      self.year = year
      self.month = month
      self.day = day
      self.dayOfYear = days - daysSinceEpoch(year: year, month: 1, day: 1)
    }

    func formatted(_ specifier: Unicode.Scalar, padding: Padding?) -> String? {
      func number(_ value: Int64, width: Int, pad: Padding = .zero) -> String {
        let digits = "\(value.magnitude)"
        let sign = value < 0 ? "-" : ""
        let fill = padding ?? pad
        guard fill != .none, digits.utf8.count < width else {
          return sign + digits
        }
        let filler = String(repeating: fill == .zero ? "0" : " ", count: width - digits.utf8.count)
        return sign + filler + digits
      }
      let twelveHour = self.hour % 12 == 0 ? 12 : self.hour % 12
      switch specifier {
      case "a": return String(weekdayNames[Int(self.weekday)].prefix(3))
      case "A": return weekdayNames[Int(self.weekday)]
      case "b", "h": return String(monthNames[Int(self.month) - 1].prefix(3))
      case "B": return monthNames[Int(self.month) - 1]
      case "c": return self.expanding("%a %b %e %H:%M:%S %Y")
      case "C": return number(floorDivide(self.year, 100), width: 2)
      case "d": return number(self.day, width: 2)
      case "D", "x": return self.expanding("%m/%d/%y")
      case "e": return number(self.day, width: 2, pad: .space)
      case "F": return self.expanding("%Y-%m-%d")
      case "g": return number(floorModulo(self.isoWeek.year, 100), width: 2)
      case "G": return number(self.isoWeek.year, width: 1)
      case "H": return number(self.hour, width: 2)
      case "I": return number(twelveHour, width: 2)
      case "j": return number(self.dayOfYear + 1, width: 3)
      case "k": return number(self.hour, width: 2, pad: .space)
      case "l": return number(twelveHour, width: 2, pad: .space)
      case "m": return number(self.month, width: 2)
      case "M": return number(self.minute, width: 2)
      case "n": return "\n"
      case "p": return self.hour < 12 ? "AM" : "PM"
      case "P": return self.hour < 12 ? "am" : "pm"
      case "r": return self.expanding("%I:%M:%S %p")
      case "R": return self.expanding("%H:%M")
      case "s": return "\(self.secondsSinceEpoch)"
      case "S": return number(self.second, width: 2)
      case "t": return "\t"
      case "T", "X": return self.expanding("%H:%M:%S")
      case "u": return number(self.weekday == 0 ? 7 : self.weekday, width: 1)
      case "U": return number((self.dayOfYear + 7 - self.weekday) / 7, width: 2)
      case "V": return number(self.isoWeek.week, width: 2)
      case "w": return number(self.weekday, width: 1)
      case "W": return number((self.dayOfYear + 7 - (self.weekday + 6) % 7) / 7, width: 2)
      case "y": return number(floorModulo(self.year, 100), width: 2)
      case "Y": return number(self.year, width: 1)
      case "z": return "+0000"
      case "Z": return "UTC"
      case "%": return "%"
      default: return nil
      }
    }

    private func expanding(_ format: String) -> String {
      formattedUTCTime(self.secondsSinceEpoch, format: format)
    }

    private var isoWeek: (year: Int64, week: Int64) {
      let mondayBasedWeekday = (self.weekday + 6) % 7
      let week = (self.dayOfYear - mondayBasedWeekday + 10) / 7
      if week < 1 {
        return (self.year - 1, isoWeekCount(year: self.year - 1))
      }
      if week > isoWeekCount(year: self.year) {
        return (self.year + 1, 1)
      }
      return (self.year, week)
    }
  }

  private enum Padding {
    case zero
    case space
    case none

    init?(flag: Unicode.Scalar) {
      switch flag {
      case "0": self = .zero
      case "_": self = .space
      case "-": self = .none
      default: return nil
      }
    }
  }

  private let weekdayNames = [
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
  ]

  private let monthNames = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December"
  ]

  private func civilDate(daysSinceEpoch days: Int64) -> (year: Int64, month: Int64, day: Int64) {
    let shifted = days + 719_468
    let era = floorDivide(shifted, 146_097)
    let dayOfEra = shifted - era * 146_097
    let yearOfEra = (dayOfEra - dayOfEra / 1_460 + dayOfEra / 36_524 - dayOfEra / 146_096) / 365
    let dayOfYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100)
    let shiftedMonth = (5 * dayOfYear + 2) / 153
    let day = dayOfYear - (153 * shiftedMonth + 2) / 5 + 1
    let month = shiftedMonth < 10 ? shiftedMonth + 3 : shiftedMonth - 9
    return (yearOfEra + era * 400 + (month <= 2 ? 1 : 0), month, day)
  }

  private func daysSinceEpoch(year: Int64, month: Int64, day: Int64) -> Int64 {
    let shiftedYear = month <= 2 ? year - 1 : year
    let era = floorDivide(shiftedYear, 400)
    let yearOfEra = shiftedYear - era * 400
    let dayOfYear = (153 * (month > 2 ? month - 3 : month + 9) + 2) / 5 + day - 1
    let dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear
    return era * 146_097 + dayOfEra - 719_468
  }

  private func isoWeekCount(year: Int64) -> Int64 {
    func weekdayOfDecember31(_ year: Int64) -> Int64 {
      floorModulo(year + floorDivide(year, 4) - floorDivide(year, 100) + floorDivide(year, 400), 7)
    }
    return weekdayOfDecember31(year) == 4 || weekdayOfDecember31(year - 1) == 3 ? 53 : 52
  }

  private func floorDivide(_ value: Int64, _ divisor: Int64) -> Int64 {
    let quotient = value / divisor
    return value % divisor < 0 ? quotient - 1 : quotient
  }

  private func floorModulo(_ value: Int64, _ divisor: Int64) -> Int64 {
    value - floorDivide(value, divisor) * divisor
  }
#endif
