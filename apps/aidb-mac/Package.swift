// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AIDBMac",
    platforms: [ .macOS(.v13) ],
    products: [
        .executable(name: "AIDBMac", targets: ["AIDBMac"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "AIDBMac",
            path: "Sources",
            resources: [
                // Add assets later as needed
            ]
        )
    ]
)
