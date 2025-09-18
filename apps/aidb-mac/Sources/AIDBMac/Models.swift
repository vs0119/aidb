import Foundation
import SwiftUI

// MARK: - Models matching server API

struct CollectionInfo: Codable, Identifiable, Hashable {
    var id: String { name }
    let name: String
    let dim: Int
    let metric: String
    let len: Int
}

struct CreateCollectionReq: Codable {
    let name: String
    let dim: Int
    let metric: String
    let wal_dir: String?
    let index: String
    let hnsw: HnswParams?
}

struct HnswParams: Codable {
    let m: Int?
    let ef_construction: Int?
    let ef_search: Int?
}

struct UpsertPointReq: Codable {
    let id: String?
    let vector: [Float]
    let payload: [String: AnyCodable]?
}

struct UpsertPointsBatchReq: Codable {
    let points: [UpsertPointReq]
}

struct SearchReq: Codable { let vector: [Float]; let top_k: Int; let filter: [String: AnyCodable]? }

struct SearchResult: Codable, Identifiable { let id: UUID; let score: Float; let payload: [String: AnyCodable]? }
struct SearchResp: Codable { let results: [SearchResult] }

struct UpdateParamsReq: Codable { let ef_search: Int? }

// MARK: - AnyCodable for flexible payloads

struct AnyCodable: Codable, Hashable {
    let value: Any
    init(_ value: Any) { self.value = value }
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self.value = Optional<Any>.none as Any; return }
        if let b = try? c.decode(Bool.self) { self.value = b; return }
        if let i = try? c.decode(Int.self) { self.value = i; return }
        if let d = try? c.decode(Double.self) { self.value = d; return }
        if let s = try? c.decode(String.self) { self.value = s; return }
        if let arr = try? c.decode([AnyCodable].self) { self.value = arr.map { $0.value }; return }
        if let dict = try? c.decode([String: AnyCodable].self) { self.value = dict.mapValues { $0.value }; return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "Unsupported JSON")
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch value {
        case Optional<Any>.none:
            try c.encodeNil()
        case let b as Bool:
            try c.encode(b)
        case let i as Int:
            try c.encode(i)
        case let d as Double:
            try c.encode(d)
        case let f as Float:
            try c.encode(Double(f))
        case let s as String:
            try c.encode(s)
        case let arr as [Any]:
            try c.encode(arr.map { AnyCodable($0) })
        case let dict as [String: Any]:
            try c.encode(dict.mapValues { AnyCodable($0) })
        default:
            throw EncodingError.invalidValue(value, .init(codingPath: c.codingPath, debugDescription: "Unsupported JSON"))
        }
    }
}

// MARK: - App Model

final class AppModel: ObservableObject {
    @Published var baseURL: URL? = URL(string: "http://127.0.0.1:8080")
    @Published var healthOK = false
    @Published var collections: [CollectionInfo] = []
    @Published var selection: CollectionInfo?

    let client = AIDBClient()

    func refreshHealth() async {
        guard let url = baseURL else { return }
        client.baseURL = url
        healthOK = (try? await client.health()) ?? false
    }

    func loadCollections() async {
        guard let _ = baseURL else { return }
        if let list = try? await client.listCollections() {
            await MainActor.run {
                collections = list
                if selection == nil { selection = list.first }
            }
        }
    }
}

// MARK: - Helpers

func parseJSONFragment(_ text: String) -> Any? {
    let data = Data(text.utf8)
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

func toAnyCodable(_ any: Any) -> AnyCodable { AnyCodable(any) }
