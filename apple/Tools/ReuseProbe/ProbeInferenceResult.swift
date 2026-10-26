import Foundation

struct ProbeInferenceResult {
    var output = [Float]()
    var seconds = [Double]()
    var thermalStates = [[Int]]()

    mutating func append(seconds: Double, startingThermalState: Int) {
        self.seconds.append(seconds)
        thermalStates.append([startingThermalState, ProcessInfo.processInfo.thermalState.rawValue])
    }
}
