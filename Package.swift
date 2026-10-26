// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ReuseApple",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "ReuseAudio", targets: ["ReuseAudio"]),
        .library(name: "ReuseStreaming", targets: ["ReuseStreaming"]),
        .executable(name: "ReuseProbe", targets: ["ReuseProbe"]),
        .executable(name: "ReuseStreamingCLI", targets: ["ReuseStreamingCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.29.1")
    ],
    targets: [
        .target(name: "ReuseAudio", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift")
        ], path: "Core"),
        .target(name: "ReuseStreaming", dependencies: [
            "ReuseAudio", .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift")
        ], path: "Sources/ReuseStreaming", exclude: ["LICENSE"]),
        // Frozen current implementation for regression tests and device A/B runs.
        // Kept out of the public products and never used by the shipping library.
        .target(name: "ReuseStreamingReference", dependencies: [
            "ReuseAudio", .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift")
        ], path: "Benchmarks/Reference/Sources/ReuseStreaming", exclude: ["LICENSE"]),
        .executableTarget(name: "ReuseProbe", dependencies: [
            "ReuseAudio", .product(name: "MLX", package: "mlx-swift")
        ], path: "Probe"),
        .executableTarget(name: "ReuseStreamingCLI", dependencies: ["ReuseAudio", "ReuseStreaming"],
                          path: "Examples/StreamingCLI"),
        .testTarget(name: "ReuseAudioTests", dependencies: ["ReuseAudio"]),
        .testTarget(name: "ReuseStreamingTests", dependencies: ["ReuseStreaming", "ReuseStreamingReference"])
    ],
    cxxLanguageStandard: .cxx17
)
