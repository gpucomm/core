import Foundation
import Metal

public enum KernelLibraryError: Error, CustomStringConvertible {
    case missingResource(String)

    public var description: String {
        switch self {
        case .missingResource(let name):
            return "missing kernel resource '\(name)'"
        }
    }
}

public final class KernelLibrary {
    private let device: MTLDevice
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    public init(context: MetalContext) throws {
        self.device = context.device

        guard let url = Bundle.module.url(forResource: "Kernels", withExtension: "metal") else {
            throw KernelLibraryError.missingResource("Resources/Kernels/Kernels.metal (SwiftPM flattens this to Kernels.metal in the resource bundle)")
        }

        let source = try String(contentsOf: url, encoding: .utf8)
        self.library = try context.device.makeLibrary(source: source, options: nil)
    }

    public func pipeline(named functionName: String) throws -> MTLComputePipelineState {
        if let cached = pipelines[functionName] { return cached }
        guard let function = library.makeFunction(name: functionName) else {
            throw NSError(
                domain: "gpucomm.core",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "kernel function not found: \(functionName)"]
            )
        }
        let pso = try device.makeComputePipelineState(function: function)
        pipelines[functionName] = pso
        return pso
    }
}
