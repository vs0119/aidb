import SwiftUI
import Foundation

struct SidebarView: View {
    @EnvironmentObject var model: AppModel
    @State private var showingCreate = false
    var body: some View {
        List(selection: $model.selection) {
            Section("Connection") {
                ConnectionView()
            }
            Section("Collections") {
                if model.collections.isEmpty && !model.isRefreshing {
                    Text("No collections yet").foregroundStyle(.secondary)
                }
                ForEach(model.collections) { c in
                    VStack(alignment: .leading) {
                        Text(c.name)
                        Text("Vectors: \(c.len)").font(.caption).foregroundStyle(.secondary)
                    }
                    .tag(Optional(c))
                    .onTapGesture { model.selection = c }
                }
            }
        }
        .toolbar {
            Button {
                Task { await model.refreshAll() }
            } label: { Label("Refresh", systemImage: "arrow.clockwise") }
            Button { showingCreate = true } label: { Label("New", systemImage: "plus") }
        }
        .sheet(isPresented: $showingCreate) { CreateCollectionView(isPresented: $showingCreate) }
    }
}

struct ConnectionView: View {
    @EnvironmentObject var model: AppModel
    @State private var base = "http://127.0.0.1:8080"
    @State private var checking = false
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                TextField("Server URL", text: $base)
                    .textFieldStyle(.roundedBorder)
                    .textInputAutocapitalization(.never)
                    .disableAutocorrection(true)
                Button(checking ? "Updating…" : "Save & Refresh") {
                    checking = true
                    Task {
                        model.updateBaseURL(base)
                        await model.refreshAll()
                        await MainActor.run {
                            checking = false
                            model.clearError()
                        }
                    }
                }
                .disabled(checking || base.isEmpty)
            }
            HStack {
                Toggle("Auto refresh", isOn: $model.autoRefreshEnabled)
                Spacer()
                Stepper(value: $model.autoRefreshInterval, in: 5...120, step: 5) {
                    Text("Every \(Int(model.autoRefreshInterval))s")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(width: 140)
            }
        }
        .onAppear {
            base = model.baseURLText
        }
    }
}

struct CreateCollectionView: View {
    @EnvironmentObject var model: AppModel
    @Binding var isPresented: Bool
    @State private var name = "docs"
    @State private var dim = 768
    @State private var metric = "cosine"
    @State private var index = "hnsw"
    @State private var m = 16
    @State private var efc = 200
    @State private var efs = 50
    @State private var creating = false
    @State private var error: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Create Collection").font(.title2)
            HStack { Text("Name"); TextField("name", text: $name).frame(width: 200) }
            HStack { Text("Dim"); TextField("dim", value: $dim, formatter: NumberFormatter()).frame(width: 100) }
            HStack { Text("Metric"); Picker("metric", selection: $metric) { Text("cosine").tag("cosine"); Text("euclidean").tag("euclidean") }.pickerStyle(.segmented).frame(width: 220) }
            HStack { Text("Index"); Picker("index", selection: $index) { Text("HNSW").tag("hnsw"); Text("BruteForce").tag("bruteforce") }.pickerStyle(.segmented).frame(width: 220) }
            if index == "hnsw" {
                HStack { Text("M"); TextField("M", value: $m, formatter: NumberFormatter()).frame(width: 80) }
                HStack { Text("efc"); TextField("efc", value: $efc, formatter: NumberFormatter()).frame(width: 80) }
                HStack { Text("efs"); TextField("efs", value: $efs, formatter: NumberFormatter()).frame(width: 80) }
            }
            if let e = error { Text(e).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel") { isPresented = false }
                Button(creating ? "Creating…" : "Create") {
                    creating = true
                    Task {
                        do {
                            let req = CreateCollectionReq(name: name, dim: dim, metric: metric, wal_dir: "./data", index: index, hnsw: index == "hnsw" ? HnswParams(m: m, ef_construction: efc, ef_search: efs) : nil)
                            try await model.client.createCollection(req)
                            await model.loadCollections()
                            await MainActor.run { isPresented = false }
                        } catch {
                            await MainActor.run { self.error = error.localizedDescription }
                        }
                        creating = false
                    }
                }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 420)
    }
}

struct CollectionDetailView: View {
    @EnvironmentObject var model: AppModel
    @State private var detail: CollectionInfo
    @State private var efSearch: Int
    @State private var vectorText = "[0.1, 0.2, 0.3]"
    @State private var topK = 5
    @State private var results: [SearchResult] = []
    @State private var busy = false
    @State private var notice: String?
    @State private var showBatchIngest = false
    @State private var filters: [FilterEntry] = []
    @State private var selectedResult: SearchResult?
    @State private var metricsText: String = ""
    @State private var showingMetrics = false

    init(collection: CollectionInfo) {
        _detail = State(initialValue: collection)
        _efSearch = State(initialValue: collection.hnsw?.efSearch ?? 50)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("Schema") {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow { Text("Name").bold(); Text(detail.name) }
                        GridRow { Text("Metric").bold(); Text(detail.metric) }
                        GridRow { Text("Dimension").bold(); Text("\(detail.dim)") }
                        GridRow { Text("Index").bold(); Text(detail.index.capitalized) }
                        if let hnsw = detail.hnsw {
                            GridRow { Text("HNSW M").bold(); Text("\(hnsw.m)") }
                            GridRow { Text("ef_construction").bold(); Text("\(hnsw.efConstruction)") }
                            GridRow { Text("ef_search").bold(); Text("\(hnsw.efSearch)") }
                        }
                        GridRow { Text("Vectors").bold(); Text("\(detail.len)") }
                    }
                    .font(.subheadline)
                }

                GroupBox("WAL & Metrics") {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow { Text("Size Bytes").bold(); Text(ByteCountFormatter.string(fromByteCount: Int64(detail.walSizeBytes), countStyle: .decimal)) }
                        GridRow { Text("Bytes Since Truncate").bold(); Text("\(detail.walBytesSinceTruncate)") }
                        GridRow { Text("Last Truncate").bold(); Text(detail.walLastTruncate ?? "n/a") }
                    }
                    .font(.subheadline)
                }

                HStack(spacing: 12) {
                    Button(action: actionSnapshot) {
                        Label("Snapshot", systemImage: "tray.and.arrow.down")
                    }
                    .disabled(busy)

                    Button(action: actionCompact) {
                        Label("Compact", systemImage: "hammer")
                    }
                    .disabled(busy)

                    if detail.index == "hnsw" {
                        HStack(spacing: 8) {
                            Text("ef_search")
                            TextField("", value: $efSearch, formatter: NumberFormatter())
                                .frame(width: 80)
                            Button("Update", action: actionUpdateParams)
                                .disabled(busy)
                        }
                    }

                    Spacer(minLength: 12)

                    Button("Metrics", action: actionMetrics)
                    Button("Batch Ingest") { showBatchIngest = true }
                }

                Divider()

                VStack(alignment: .leading, spacing: 10) {
                    Text("Search").font(.headline)
                    HStack(spacing: 8) {
                        TextField("[floats]", text: $vectorText)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                        if !model.recentVectors.isEmpty {
                            Menu {
                                ForEach(model.recentVectors, id: \.self) { v in
                                    Button(v) { vectorText = v }
                                }
                            } label: {
                                Image(systemName: "clock.arrow.circlepath")
                            }
                        }
                    }
                    HStack(spacing: 12) {
                        Stepper(value: $topK, in: 1...100) {
                            Text("top_k: \(topK)")
                        }
                        Spacer()
                    }
                    FilterBuilderView(entries: $filters)
                    HStack {
                        Button(busy ? "Searching…" : "Search", action: actionSearch)
                            .buttonStyle(.borderedProminent)
                            .disabled(busy)
                        Spacer()
                    }
                    if !results.isEmpty {
                        Table(results) {
                            TableColumn("ID") { result in
                                Text(result.id.uuidString.prefix(8) + "…")
                                    .onTapGesture { selectedResult = result }
                            }
                            TableColumn("Score") { result in
                                Text(String(format: "%.4f", result.score))
                            }
                        }
                        .frame(minHeight: 160)
                    }
                }

                if let n = notice { Text(n).foregroundStyle(.secondary) }
            }
            .padding()
        }
        .navigationTitle(detail.name)
        .sheet(isPresented: $showBatchIngest) {
            BatchIngestView(isPresented: $showBatchIngest, collection: detail)
                .environmentObject(model)
        }
        .sheet(item: $selectedResult) { PayloadDetailView(result: $0) }
        .sheet(isPresented: $showingMetrics) {
            MetricsSheet(text: metricsText, title: detail.name)
        }
        .overlay {
            if busy {
                ProgressOverlay()
            }
        }
        .task(id: detail.id) {
            if let updated = await model.fetchCollectionDetail(name: detail.name) {
                await MainActor.run {
                    detail = updated
                    if let h = updated.hnsw {
                        efSearch = h.efSearch
                    }
                }
            }
        }
    }

    private func actionSnapshot() {
        busy = true
        Task {
            defer { busy = false }
            do {
                try await model.client.snapshot(name: detail.name)
                await MainActor.run { notice = "Snapshot written." }
                await reloadDetail()
            } catch {
                await MainActor.run { notice = error.localizedDescription }
            }
        }
    }

    private func actionCompact() {
        busy = true
        Task {
            defer { busy = false }
            do {
                let ok = try await model.client.compact(name: detail.name)
                await MainActor.run { notice = ok ? "Compacted HNSW." : "No-op (BruteForce)." }
                await reloadDetail()
            } catch {
                await MainActor.run { notice = error.localizedDescription }
            }
        }
    }

    private func actionUpdateParams() {
        busy = true
        Task {
            defer { busy = false }
            do {
                let ok = try await model.client.updateEfSearch(name: detail.name, ef: efSearch)
                await MainActor.run { notice = ok ? "ef_search updated." : "No-op (BruteForce)." }
                await reloadDetail()
            } catch {
                await MainActor.run { notice = error.localizedDescription }
            }
        }
    }

    private func actionSearch() {
        busy = true
        Task {
            defer { busy = false }
            do {
                guard let vec = try? JSONSerialization.jsonObject(with: Data(vectorText.utf8)) as? [Double] else {
                    throw NSError(domain: "AIDB", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid vector JSON"])
                }
                let floats = vec.map { Float($0) }
                var filt: [String: AnyCodable]? = nil
                if !filters.isEmpty {
                    var map: [String: AnyCodable] = [:]
                    for entry in filters where !entry.key.isEmpty {
                        if let fragment = parseJSONFragment(entry.value) {
                            map[entry.key] = toAnyCodable(fragment)
                        } else {
                            map[entry.key] = AnyCodable(entry.value)
                        }
                    }
                    filt = map
                }
                let res = try await model.client.search(name: detail.name, vector: floats, topK: topK, filter: filt)
                await MainActor.run {
                    self.results = res
                    self.notice = "Found \(res.count) results."
                }
                model.rememberVector(vectorText)
            } catch {
                await MainActor.run { notice = error.localizedDescription }
            }
        }
    }

    private func actionMetrics() {
        metricsText = ""
        Task {
            do {
                let metrics = try await model.client.metrics()
                await MainActor.run {
                    metricsText = metrics
                    showingMetrics = true
                }
            } catch {
                await MainActor.run { notice = error.localizedDescription }
            }
        }
    }

    private func reloadDetail() async {
        if let updated = await model.fetchCollectionDetail(name: detail.name) {
            await MainActor.run {
                detail = updated
                if let h = updated.hnsw {
                    efSearch = h.efSearch
                }
            }
        }
    }
}

// MARK: - Batch Ingest

struct BatchIngestView: View {
    @EnvironmentObject var model: AppModel
    @Binding var isPresented: Bool
    let collection: CollectionInfo
    @State private var jsonText: String = "[\n  {\n    \"vector\": [0.1, 0.2, 0.3],\n    \"payload\": {\"source\": \"demo\"}\n  }\n]"
    @State private var busy = false
    @State private var message: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Batch Ingest into \(collection.name)").font(.title3)
            Text("Paste a JSON array of points: {vector:[...], payload:{...}}. Optional id field.").font(.footnote)
            TextEditor(text: $jsonText)
                .font(.system(.body, design: .monospaced))
                .frame(minHeight: 220)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.gray.opacity(0.2)))
            if let m = message { Text(m).foregroundStyle(.secondary) }
            HStack {
                Spacer()
                Button("Cancel") { isPresented = false }
                Button(busy ? "Ingesting…" : "Ingest") { ingest() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 560)
    }

    func ingest() {
        busy = true
        Task {
            do {
                let data = Data(jsonText.utf8)
                guard let arr = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { throw NSError(domain: "AIDB", code: 2, userInfo: [NSLocalizedDescriptionKey: "Expect JSON array of objects"]) }
                var points: [UpsertPointReq] = []
                for obj in arr {
                    let idStr = obj["id"] as? String
                    guard let vecAny = obj["vector"] as? [Any] else { continue }
                    let floats: [Float] = vecAny.compactMap { v in
                        if let d = v as? Double { return Float(d) }
                        if let i = v as? Int { return Float(i) }
                        return nil
                    }
                    var payloadAC: [String: AnyCodable]? = nil
                    if let pay = obj["payload"] as? [String: Any] { payloadAC = pay.mapValues { AnyCodable($0) } }
                    points.append(UpsertPointReq(id: idStr, vector: floats, payload: payloadAC))
                }
                let ids = try await model.client.upsertBatch(name: collection.name, points: points)
                await MainActor.run { self.message = "Ingested \(ids.count) points." }
                await model.fetchCollectionDetail(name: collection.name)
            } catch { await MainActor.run { self.message = error.localizedDescription } }
            busy = false
        }
    }
}

// MARK: - Filter Builder

struct FilterEntry: Identifiable, Hashable { let id = UUID(); var key: String; var value: String }

struct FilterBuilderView: View {
    @Binding var entries: [FilterEntry]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { Text("Filters (key:value JSON fragment)").font(.subheadline); Spacer(); Button("Add") { entries.append(FilterEntry(key: "", value: "\"tag\"")) } }
            ForEach(entries) { e in
                HStack {
                    TextField("key", text: binding(for: e).key).frame(width: 160)
                    Text(":")
                    TextField("value (e.g. \"news\", 42, true)", text: binding(for: e).value)
                    Button(role: .destructive) { entries.removeAll { $0.id == e.id } } label: { Image(systemName: "trash") }
                }
            }
        }
    }
    private func binding(for entry: FilterEntry) -> (key: Binding<String>, value: Binding<String>) {
        let i = entries.firstIndex(of: entry)!
        return (Binding(get: { entries[i].key }, set: { entries[i].key = $0 }), Binding(get: { entries[i].value }, set: { entries[i].value = $0 }))
    }
}

struct ErrorBanner: View {
    let message: String
    var body: some View {
        Text(message)
            .padding(.vertical, 8)
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity)
            .background(Color.red.opacity(0.9))
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .padding(.horizontal)
    }
}

struct ProgressOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.15).ignoresSafeArea()
            ProgressView().progressViewStyle(.circular)
        }
    }
}

struct PayloadDetailView: View {
    let result: SearchResult
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Result Detail").font(.title2)
            Text("ID: \(result.id.uuidString)").font(.callout)
            Text(String(format: "Score: %.4f", result.score)).font(.callout)
            Divider()
            ScrollView {
                Text(prettyPayload())
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 200)
            Spacer()
        }
        .padding(24)
        .frame(minWidth: 420, minHeight: 320)
    }

    private func prettyPayload() -> String {
        guard let payload = result.payload,
              let data = try? JSONEncoder().encode(payload),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]),
              let string = String(data: pretty, encoding: .utf8) else {
            return "<empty payload>"
        }
        return string
    }
}

struct MetricsSheet: View {
    let text: String
    let title: String
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Metrics for \(title)").font(.title3)
            ScrollView {
                Text(text.isEmpty ? "No metrics available." : text)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 240)
        }
        .padding(24)
        .frame(minWidth: 520, minHeight: 360)
    }
}
