// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "gpucomm-core",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "GPUCommCore", targets: ["GPUCommCore"]),
        .executable(name: "gpucomm", targets: ["gpucomm"]),
    ],
    targets: [
        .target(
            name: "GPUCommCore",
            resources: [
                .process("Resources"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
            ]
        ),
        .executableTarget(
            name: "gpucomm",
            dependencies: ["GPUCommCore"],
            linkerSettings: [
                .linkedFramework("Metal"),
            ]
        ),
    ]
)
