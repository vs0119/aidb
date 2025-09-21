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

// MARK: - SQL Models

struct SqlResponse: Codable {
    let type: String
    let ok: Bool?
    let id: String?
    let results: [SearchResult]?
    let data: [[String: AnyCodable]]?
    let columns: [String]?
    let tableColumns: [TableColumn]?
    let rows: [[AnyCodable]]?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case type, ok, id, results, data, columns, rows, error
        case tableColumns = "table_columns"
    }
}

struct TableInfo: Codable, Identifiable, Hashable {
    var id: String { name }
    let name: String
    let columns: [TableColumn]
    let rowCount: Int?
    let createdAt: String?
    let size: Int64?
}

struct TableColumn: Codable, Hashable, Identifiable {
    var id: String { name }
    let name: String
    let type: String
    let nullable: Bool
    let primaryKey: Bool
    let defaultValue: String?
}

struct CreateTableColumn {
    var name: String = ""
    var type: String = "TEXT"
    var nullable: Bool = true
    var primaryKey: Bool = false
    var defaultValue: String = ""
}

enum ColumnType: String, CaseIterable {
    case text = "TEXT"
    case integer = "INTEGER"
    case real = "REAL"
    case blob = "BLOB"
    case vector = "VECTOR"
    case json = "JSON"

    var displayName: String {
        switch self {
        case .text: return "Text"
        case .integer: return "Integer"
        case .real: return "Real/Float"
        case .blob: return "Binary Data"
        case .vector: return "Vector"
        case .json: return "JSON"
        }
    }

    var icon: String {
        switch self {
        case .text: return "textformat.abc"
        case .integer: return "number"
        case .real: return "function"
        case .blob: return "doc.circle"
        case .vector: return "brain"
        case .json: return "curlybraces"
        }
    }
}

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

    // MARK: - Equatable
    static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        if isNil(lhs.value) && isNil(rhs.value) { return true }
        if isNil(lhs.value) || isNil(rhs.value) { return false }

        switch (lhs.value, rhs.value) {
        case (let l as Bool, let r as Bool): return l == r
        case (let l as Int, let r as Int): return l == r
        case (let l as Double, let r as Double): return l == r
        case (let l as String, let r as String): return l == r
        case (let l as [Any], let r as [Any]): return arrayEquals(l, r)
        case (let l as [String: Any], let r as [String: Any]): return dictEquals(l, r)
        default: return false
        }
    }

    // MARK: - Hashable
    func hash(into hasher: inout Hasher) {
        if Self.isNil(value) {
            hasher.combine(0)
            return
        }

        switch value {
        case let b as Bool: hasher.combine(b)
        case let i as Int: hasher.combine(i)
        case let d as Double: hasher.combine(d)
        case let s as String: hasher.combine(s)
        case let arr as [Any]: hashArray(arr, into: &hasher)
        case let dict as [String: Any]: hashDict(dict, into: &hasher)
        default: hasher.combine(ObjectIdentifier(type(of: value)))
        }
    }

    private static func arrayEquals(_ lhs: [Any], _ rhs: [Any]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        for (l, r) in zip(lhs, rhs) {
            if !AnyCodable(l).equals(AnyCodable(r)) { return false }
        }
        return true
    }

    private static func dictEquals(_ lhs: [String: Any], _ rhs: [String: Any]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        for (key, lValue) in lhs {
            guard let rValue = rhs[key] else { return false }
            if !AnyCodable(lValue).equals(AnyCodable(rValue)) { return false }
        }
        return true
    }

    private func equals(_ other: AnyCodable) -> Bool {
        return self == other
    }

    private func hashArray(_ array: [Any], into hasher: inout Hasher) {
        hasher.combine(array.count)
        for item in array {
            AnyCodable(item).hash(into: &hasher)
        }
    }

    private func hashDict(_ dict: [String: Any], into hasher: inout Hasher) {
        hasher.combine(dict.count)
        for (key, value) in dict.sorted(by: { $0.key < $1.key }) {
            hasher.combine(key)
            AnyCodable(value).hash(into: &hasher)
        }
    }

    private static func isNil(_ value: Any) -> Bool {
        if case Optional<Any>.none = value { return true }
        return false
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
    @Published var tables: [TableInfo] = []
    @Published var selectedTable: TableInfo?

    // Statistics
    @Published var tableStatistics: [TableStatistics] = []
    @Published var columnStatistics: [ColumnStatistics] = []
    @Published var statisticsSummary: StatisticsSummary?
    @Published var selectedTableStats: TableStatistics?
    @Published var isAnalyzing = false

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

    // MARK: - Statistics Methods

    func fetchStatistics() async {
        do {
            let summary = try await client.getStatisticsSummary()
            let tableStats = try await client.getTableStatistics()
            let columnStats = try await client.getColumnStatistics()

            await MainActor.run {
                self.statisticsSummary = summary
                self.tableStatistics = tableStats
                self.columnStatistics = columnStats
            }
        } catch {
            await report(error)
        }
    }

    func analyzeTable(_ tableName: String?) async {
        await MainActor.run { isAnalyzing = true }
        defer { Task { @MainActor in isAnalyzing = false } }

        do {
            try await client.analyzeTable(tableName)
            await fetchStatistics()
        } catch {
            await report(error)
        }
    }

    func selectTableStats(_ tableName: String) {
        selectedTableStats = tableStatistics.first { $0.table_name == tableName }
    }

    func getColumnStatistics(for tableName: String) -> [ColumnStatistics] {
        return columnStatistics.filter { $0.table_name == tableName }
    }
}

// MARK: - Helpers

func parseJSONFragment(_ text: String) -> Any? {
    let data = Data(text.utf8)
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

func toAnyCodable(_ any: Any) -> AnyCodable { AnyCodable(any) }

// MARK: - Statistics Models

struct StatisticsValue: Codable, Hashable {
    enum ValueType: String, Codable {
        case int = "Int"
        case float = "Float"
        case text = "Text"
        case bool = "Bool"
    }

    let type: ValueType
    let value: AnyCodable

    var displayString: String {
        switch type {
        case .int:
            if let intVal = value.value as? Int {
                return "\(intVal)"
            }
        case .float:
            if let floatVal = value.value as? Double {
                return String(format: "%.3f", floatVal)
            }
        case .text:
            if let textVal = value.value as? String {
                return textVal
            }
        case .bool:
            if let boolVal = value.value as? Bool {
                return boolVal ? "true" : "false"
            }
        }
        return "N/A"
    }
}

struct HistogramBucket: Codable, Hashable, Identifiable {
    var id: String { "\(lower.displayString)-\(upper.displayString)" }
    let lower: StatisticsValue
    let upper: StatisticsValue
    let count: UInt64
    let cumulative_count: UInt64

    enum CodingKeys: String, CodingKey {
        case lower, upper, count, cumulative_count
    }
}

enum HistogramType: String, Codable, CaseIterable, Hashable {
    case equiWidth = "equi_width"
    case equiDepth = "equi_depth"
    case topK = "top_k"

    var displayName: String {
        switch self {
        case .equiWidth: return "Equal Width"
        case .equiDepth: return "Equal Depth"
        case .topK: return "Top K"
        }
    }
}

struct Histogram: Codable, Hashable, Identifiable {
    let histogram_id: String
    let histogram_type: HistogramType
    let buckets: [HistogramBucket]
    let last_refreshed: String?

    var id: String { histogram_id }

    enum CodingKeys: String, CodingKey {
        case histogram_id, histogram_type, buckets, last_refreshed
    }
}

struct HistogramRef: Codable, Hashable {
    let histogram_id: String
    let histogram_type: HistogramType

    enum CodingKeys: String, CodingKey {
        case histogram_id, histogram_type
    }
}

struct TableStatistics: Codable, Hashable, Identifiable {
    let table_name: String
    let row_count: UInt64
    let analyzed_at: String
    let stats_version: UInt32

    var id: String { table_name }

    enum CodingKeys: String, CodingKey {
        case table_name, row_count, analyzed_at, stats_version
    }
}

struct ColumnStatistics: Codable, Hashable, Identifiable {
    let table_name: String
    let column_name: String
    let null_count: UInt64?
    let distinct_count: UInt64?
    let min: StatisticsValue?
    let max: StatisticsValue?
    let histogram: HistogramRef?
    let analyzed_at: String
    let stats_version: UInt32

    var id: String { "\(table_name).\(column_name)" }

    enum CodingKeys: String, CodingKey {
        case table_name, column_name, null_count, distinct_count
        case min, max, histogram, analyzed_at, stats_version
    }
}

struct AnalyzeRequest: Codable {
    let query: String
}

struct StatsRefreshConfig: Codable {
    let table: String
    let column: String
    let histogram_type: HistogramType
    let bucket_count: Int
    let refresh_interval_seconds: Int
}

struct StatisticsSummary: Codable, Hashable {
    let total_tables: Int
    let analyzed_tables: Int
    let total_columns: Int
    let analyzed_columns: Int
    let active_refresh_jobs: Int
}
