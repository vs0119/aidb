import Foundation
import SwiftUI

// MARK: - Models matching server API

struct CollectionInfo: Codable, Identifiable, Hashable {
    var id: String { name }
    let name: String
    let dim: Int
    let metric: String
    let len: Int
    let index: String
    let hnsw: HnswMeta?
    let walSizeBytes: UInt64
    let walLastTruncate: String?
    let walBytesSinceTruncate: UInt64

    enum CodingKeys: String, CodingKey {
        case name
        case dim
        case metric
        case len
        case index
        case hnsw
        case walSizeBytes = "wal_size_bytes"
        case walLastTruncate = "wal_last_truncate"
        case walBytesSinceTruncate = "wal_bytes_since_truncate"
    }
}

struct HnswMeta: Codable, Hashable {
    let m: Int
    let efConstruction: Int
    let efSearch: Int

    enum CodingKeys: String, CodingKey {
        case m
        case efConstruction = "ef_construction"
        case efSearch = "ef_search"
    }
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
    @Published var baseURL: URL?
    @Published var baseURLText: String
    @Published var healthOK = false
    @Published var collections: [CollectionInfo] = []
    @Published var selection: CollectionInfo?
    @Published var isRefreshing = false
    @Published var errorMessage: String?
    @Published var recentVectors: [String] = []
    @Published var autoRefreshEnabled: Bool = true {
        didSet {
            defaults.set(autoRefreshEnabled, forKey: autoRefreshEnabledKey)
            guard !suppressAutoRefresh else { return }
            if autoRefreshEnabled {
                startAutoRefresh()
            } else {
                stopAutoRefresh()
            }
        }
    }
    @Published var autoRefreshInterval: Double = 20 {
        didSet {
            defaults.set(autoRefreshInterval, forKey: autoRefreshIntervalKey)
            guard !suppressAutoRefresh else { return }
            startAutoRefresh()
        }
    }

    let client = AIDBClient()

    private let defaults = UserDefaults.standard
    private let baseURLKey = "aidb.baseURL"
    private let recentVectorsKey = "aidb.recentVectors"
    private let autoRefreshEnabledKey = "aidb.autoRefreshEnabled"
    private let autoRefreshIntervalKey = "aidb.autoRefreshInterval"
    private var refreshTask: Task<Void, Never>?
    private var errorClearTask: Task<Void, Never>?
    private var suppressAutoRefresh = false

    init() {
        suppressAutoRefresh = true
        let savedURL = defaults.string(forKey: baseURLKey) ?? "http://127.0.0.1:8080"
        baseURLText = savedURL
        baseURL = URL(string: savedURL)
        if let url = baseURL {
            client.baseURL = url
        }
        recentVectors = defaults.stringArray(forKey: recentVectorsKey) ?? []
        let savedEnabled = defaults.object(forKey: autoRefreshEnabledKey) as? Bool ?? true
        autoRefreshEnabled = savedEnabled
        let savedInterval = defaults.object(forKey: autoRefreshIntervalKey) as? Double ?? 20
        autoRefreshInterval = savedInterval
        suppressAutoRefresh = false
        if autoRefreshEnabled {
            startAutoRefresh()
        }
    }

    deinit {
        refreshTask?.cancel()
        errorClearTask?.cancel()
    }

    func updateBaseURL(_ text: String) {
        baseURLText = text
        defaults.set(text, forKey: baseURLKey)
        baseURL = URL(string: text)
        if let url = baseURL {
            client.baseURL = url
            Task { @MainActor in
                self.errorMessage = nil
            }
        } else {
            Task { @MainActor in
                self.errorMessage = "Invalid server URL"
            }
        }
        if autoRefreshEnabled {
            startAutoRefresh()
        }
    }

    func refreshAll(showSpinner: Bool = true) async {
        guard baseURL != nil else { return }
        if showSpinner {
            await MainActor.run { self.isRefreshing = true }
        }
        await refreshHealth()
        await loadCollections()
        if showSpinner {
            await MainActor.run { self.isRefreshing = false }
        }
    }

    func refreshHealth() async {
        guard baseURL != nil else { return }
        do {
            let ok = try await client.health()
            await MainActor.run { self.healthOK = ok }
        } catch {
            await report(error)
        }
    }

    func loadCollections() async {
        guard baseURL != nil else { return }
        do {
            let list = try await client.listCollections()
            await MainActor.run {
                self.collections = list
                if let sel = self.selection {
                    self.selection = list.first(where: { $0.id == sel.id }) ?? list.first
                } else {
                    self.selection = list.first
                }
            }
        } catch {
            await report(error)
        }
    }

    func fetchCollectionDetail(name: String) async -> CollectionInfo? {
        guard baseURL != nil else { return nil }
        do {
            let detail = try await client.getCollectionInfo(name: name)
            await MainActor.run {
                if let idx = self.collections.firstIndex(where: { $0.id == detail.id }) {
                    self.collections[idx] = detail
                } else {
                    self.collections.append(detail)
                }
            }
            return detail
        } catch {
            await report(error)
            return nil
        }
    }

    func startAutoRefresh() {
        refreshTask?.cancel()
        guard autoRefreshEnabled, baseURL != nil else { return }
        let interval = autoRefreshInterval
        refreshTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(interval))
                await self.refreshAll(showSpinner: false)
            }
        }
    }

    func stopAutoRefresh() {
        refreshTask?.cancel()
        refreshTask = nil
    }

    func report(_ error: Error) async {
        await MainActor.run {
            self.errorMessage = error.localizedDescription
            self.errorClearTask?.cancel()
            self.errorClearTask = Task {
                try? await Task.sleep(for: .seconds(5))
                await MainActor.run {
                    if self.errorMessage == error.localizedDescription {
                        self.errorMessage = nil
                    }
                }
            }
        }
    }

    func clearError() {
        errorMessage = nil
    }

    func rememberVector(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        Task { @MainActor [trimmed] in
            if let idx = recentVectors.firstIndex(of: trimmed) {
                recentVectors.remove(at: idx)
            }
            recentVectors.insert(trimmed, at: 0)
            if recentVectors.count > 10 {
                recentVectors = Array(recentVectors.prefix(10))
            }
            self.defaults.set(recentVectors, forKey: recentVectorsKey)
        }
    }
}

// MARK: - Helpers

func parseJSONFragment(_ text: String) -> Any? {
    let data = Data(text.utf8)
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

func toAnyCodable(_ any: Any) -> AnyCodable { AnyCodable(any) }
