import Metal

public final class MetalContext {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    public init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
    }
}

