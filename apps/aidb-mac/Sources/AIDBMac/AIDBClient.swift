import Foundation

final class AIDBClient {
    var baseURL = URL(string: "http://127.0.0.1:8080")!
    private let session = URLSession(configuration: .default)

    func health() async throws -> Bool {
        let url = baseURL.appending(path: "/health")
        let (d, _) = try await session.data(from: url)
        return String(data: d, encoding: .utf8) == "ok"
    }

    func listCollections() async throws -> [CollectionInfo] {
        let url = baseURL.appending(path: "/collections")
        let (d, _) = try await session.data(from: url)
        return try JSONDecoder().decode([CollectionInfo].self, from: d)
    }

    func createCollection(_ req: CreateCollectionReq) async throws {
        let url = baseURL.appending(path: "/collections")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONEncoder().encode(req)
        _ = try await session.data(for: r)
    }

    func getCollectionInfo(name: String) async throws -> CollectionInfo {
        let url = baseURL.appending(path: "/collections/\(name)")
        let (d, _) = try await session.data(from: url)
        return try JSONDecoder().decode(CollectionInfo.self, from: d)
    }

    func upsertPoint(name: String, point: UpsertPointReq) async throws -> UUID {
        let url = baseURL.appending(path: "/collections/\(name)/points")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONEncoder().encode(point)
        let (d, _) = try await session.data(for: r)
        return try JSONDecoder().decode(UUID.self, from: d)
    }

    func upsertBatch(name: String, points: [UpsertPointReq]) async throws -> [UUID] {
        let url = baseURL.appending(path: "/collections/\(name)/points:batch")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = UpsertPointsBatchReq(points: points)
        r.httpBody = try JSONEncoder().encode(body)
        let (d, _) = try await session.data(for: r)
        return try JSONDecoder().decode([UUID].self, from: d)
    }

    func search(name: String, vector: [Float], topK: Int, filter: [String: AnyCodable]?) async throws -> [SearchResult] {
        let url = baseURL.appending(path: "/collections/\(name)/search")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let sreq = SearchReq(vector: vector, top_k: topK, filter: filter)
        r.httpBody = try JSONEncoder().encode(sreq)
        let (d, _) = try await session.data(for: r)
        let resp = try JSONDecoder().decode(SearchResp.self, from: d)
        return resp.results
    }

    func snapshot(name: String) async throws {
        let url = baseURL.appending(path: "/collections/\(name)/snapshot")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = Data("{}".utf8)
        _ = try await session.data(for: r)
    }

    func compact(name: String) async throws -> Bool {
        let url = baseURL.appending(path: "/collections/\(name)/compact")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = Data("{}".utf8)
        let (d, _) = try await session.data(for: r)
        return try JSONDecoder().decode(Bool.self, from: d)
    }

    func updateEfSearch(name: String, ef: Int) async throws -> Bool {
        let url = baseURL.appending(path: "/collections/\(name)/params")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONEncoder().encode(UpdateParamsReq(ef_search: ef))
        let (d, _) = try await session.data(for: r)
        return try JSONDecoder().decode(Bool.self, from: d)
    }

    func metrics() async throws -> String {
        let url = baseURL.appending(path: "/metrics")
        let (d, _) = try await session.data(from: url)
        return String(decoding: d, as: UTF8.self)
    }
}
