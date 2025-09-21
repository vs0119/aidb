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

    // MARK: - SQL Interface

    func executeSQL(_ query: String) async throws -> SqlResponse {
        let url = baseURL.appending(path: "/sql")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = ["query": query]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (d, response) = try await session.data(for: r)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode(SqlResponse.self, from: d)
    }

    func listTables() async throws -> [TableInfo] {
        let response = try await executeSQL("SHOW TABLES")
        return response.data?.compactMap { row in
            guard let tableName = row["name"]?.value as? String else { return nil }
            return TableInfo(
                name: tableName,
                columns: [],
                rowCount: row["row_count"]?.value as? Int,
                createdAt: row["created_at"]?.value as? String,
                size: row["size"]?.value as? Int64
            )
        } ?? []
    }

    func getTableInfo(_ tableName: String) async throws -> TableInfo? {
        let response = try await executeSQL("DESCRIBE \(tableName)")
        let columns = response.data?.compactMap { row -> TableColumn? in
            guard let name = row["name"]?.value as? String,
                  let type = row["type"]?.value as? String else { return nil }
            return TableColumn(
                name: name,
                type: type,
                nullable: (row["nullable"]?.value as? Bool) ?? true,
                primaryKey: (row["primary_key"]?.value as? Bool) ?? false,
                defaultValue: row["default_value"]?.value as? String
            )
        } ?? []

        return TableInfo(
            name: tableName,
            columns: columns,
            rowCount: nil,
            createdAt: nil,
            size: nil
        )
    }

    func getTableData(_ tableName: String, limit: Int = 100, offset: Int = 0) async throws -> [[String: AnyCodable]] {
        let query = "SELECT * FROM \(tableName) LIMIT \(limit) OFFSET \(offset)"
        let response = try await executeSQLTable(query)
        return response.data ?? []
    }

    // MARK: - SQL Table Interface

    func executeSQLTable(_ query: String) async throws -> SqlResponse {
        let url = baseURL.appending(path: "/sql/tables")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = ["query": query]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (d, response) = try await session.data(for: r)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode(SqlResponse.self, from: d)
    }

    func listTablesSQL() async throws -> [TableInfo] {
        let response = try await executeSQLTable("SHOW TABLES")
        return response.data?.compactMap { row in
            guard let tableName = row["name"]?.value as? String else { return nil }
            return TableInfo(
                name: tableName,
                columns: [],
                rowCount: row["row_count"]?.value as? Int,
                createdAt: row["created_at"]?.value as? String,
                size: row["size"]?.value as? Int64
            )
        } ?? []
    }

    func getTableInfoSQL(_ tableName: String) async throws -> TableInfo? {
        let response = try await executeSQLTable("DESCRIBE \(tableName)")
        let columns = response.data?.compactMap { row -> TableColumn? in
            guard let name = row["name"]?.value as? String,
                  let type = row["type"]?.value as? String else { return nil }
            return TableColumn(
                name: name,
                type: type,
                nullable: (row["nullable"]?.value as? Bool) ?? true,
                primaryKey: (row["primary_key"]?.value as? Bool) ?? false,
                defaultValue: row["default_value"]?.value as? String
            )
        } ?? []

        return TableInfo(
            name: tableName,
            columns: columns,
            rowCount: nil,
            createdAt: nil,
            size: nil
        )
    }

    func getTableDataSQL(_ tableName: String, limit: Int = 100, offset: Int = 0) async throws -> [[String: AnyCodable]] {
        let query = "SELECT * FROM \(tableName) LIMIT \(limit) OFFSET \(offset)"
        let response = try await executeSQLTable(query)
        return response.data ?? []
    }

    // MARK: - Statistics Interface

    func analyzeTable(_ tableName: String?) async throws {
        let query = tableName.map { "ANALYZE \($0)" } ?? "ANALYZE"
        _ = try await executeSQL(query)
    }

    func getTableStatistics() async throws -> [TableStatistics] {
        let response = try await executeSQL("SELECT * FROM __table_statistics")
        return response.data?.compactMap { row in
            guard let tableName = row["table_name"]?.value as? String,
                  let rowCount = row["row_count"]?.value as? Int,
                  let analyzedAt = row["analyzed_at"]?.value as? String,
                  let statsVersion = row["stats_version"]?.value as? Int else { return nil }
            return TableStatistics(
                table_name: tableName,
                row_count: UInt64(rowCount),
                analyzed_at: analyzedAt,
                stats_version: UInt32(statsVersion)
            )
        } ?? []
    }

    func getColumnStatistics(_ tableName: String? = nil) async throws -> [ColumnStatistics] {
        let query = tableName.map { "SELECT * FROM __column_statistics WHERE table_name = '\($0)'" } ?? "SELECT * FROM __column_statistics"
        let response = try await executeSQL(query)
        return response.data?.compactMap { row in
            guard let tableName = row["table_name"]?.value as? String,
                  let columnName = row["column_name"]?.value as? String,
                  let analyzedAt = row["analyzed_at"]?.value as? String,
                  let statsVersion = row["stats_version"]?.value as? Int else { return nil }

            let nullCount = row["null_count"]?.value as? Int
            let distinctCount = row["distinct_count"]?.value as? Int

            // Parse min/max values if present
            let min: StatisticsValue? = if let minStr = row["min_value"]?.value as? String, !minStr.isEmpty {
                parseStatisticsValue(minStr)
            } else { nil }

            let max: StatisticsValue? = if let maxStr = row["max_value"]?.value as? String, !maxStr.isEmpty {
                parseStatisticsValue(maxStr)
            } else { nil }

            return ColumnStatistics(
                table_name: tableName,
                column_name: columnName,
                null_count: nullCount.map(UInt64.init),
                distinct_count: distinctCount.map(UInt64.init),
                min: min,
                max: max,
                histogram: nil, // Histogram refs not implemented in simple form yet
                analyzed_at: analyzedAt,
                stats_version: UInt32(statsVersion)
            )
        } ?? []
    }

    func getStatisticsSummary() async throws -> StatisticsSummary {
        let tables = try await listTablesSQL()
        let tableStats = try await getTableStatistics()
        let columnStats = try await getColumnStatistics()

        // Count unique columns across all tables
        let totalColumns = tables.flatMap { $0.columns }.count
        let analyzedColumns = Set(columnStats.map { "\($0.table_name).\($0.column_name)" }).count

        return StatisticsSummary(
            total_tables: tables.count,
            analyzed_tables: tableStats.count,
            total_columns: totalColumns,
            analyzed_columns: analyzedColumns,
            active_refresh_jobs: 0 // Would come from server stats endpoint
        )
    }

    private func parseStatisticsValue(_ str: String) -> StatisticsValue? {
        // Simple parsing - in real implementation would need more robust parsing
        if str.lowercased() == "true" || str.lowercased() == "false" {
            return StatisticsValue(type: .bool, value: AnyCodable(str.lowercased() == "true"))
        } else if let intVal = Int(str) {
            return StatisticsValue(type: .int, value: AnyCodable(intVal))
        } else if let floatVal = Double(str) {
            return StatisticsValue(type: .float, value: AnyCodable(floatVal))
        } else {
            return StatisticsValue(type: .text, value: AnyCodable(str))
        }
    }

    // MARK: - Cost Estimation Interface

    func estimateCardinality(tableName: String, predicate: String?) async throws -> CardinalityEstimate {
        let url = baseURL.appending(path: "/cost/cardinality")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "table_name": tableName,
            "predicate": predicate as Any
        ]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (d, response) = try await session.data(for: r)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode(CardinalityEstimate.self, from: d)
    }

    func estimateQueryCost(tableName: String, queryType: String, predicate: String?) async throws -> CostEstimate {
        let url = baseURL.appending(path: "/cost/query")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "table_name": tableName,
            "query_type": queryType,
            "predicate": predicate as Any
        ]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (d, response) = try await session.data(for: r)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode(CostEstimate.self, from: d)
    }

    func estimateJoinCost(leftTable: String, rightTable: String, joinPredicate: JoinPredicate) async throws -> CostEstimate {
        let url = baseURL.appending(path: "/cost/join")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "left_table": leftTable,
            "right_table": rightTable,
            "join_predicate": try JSONEncoder().encode(joinPredicate).base64EncodedString()
        ]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (d, response) = try await session.data(for: r)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode(CostEstimate.self, from: d)
    }

    func analyzeQueryPlan(sql: String) async throws -> QueryPlan {
        let url = baseURL.appending(path: "/cost/analyze")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = ["sql": sql]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (d, response) = try await session.data(for: r)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode(QueryPlan.self, from: d)
    }

    func updateCostModel(config: CostModelConfig) async throws -> Bool {
        let url = baseURL.appending(path: "/cost/config")
        var r = URLRequest(url: url)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONEncoder().encode(config)

        let (d, response) = try await session.data(for: r)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode(Bool.self, from: d)
    }

    func getQueryPlans(tableName: String? = nil) async throws -> [QueryPlan] {
        var url = baseURL.appending(path: "/cost/plans")
        if let tableName = tableName {
            url.append(queryItems: [URLQueryItem(name: "table", value: tableName)])
        }

        let (d, response) = try await session.data(from: url)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode >= 400 {
            let errorText = String(data: d, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "AIDBClient", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorText])
        }

        return try JSONDecoder().decode([QueryPlan].self, from: d)
    }
}
