import SwiftUI

struct SidebarView: View {
    @EnvironmentObject var model: AppModel
    @State private var showingCreate = false
    var body: some View {
        List(selection: $model.selection) {
            Section("Connection") {
                ConnectionView()
            }
            Section("Collections") {
                ForEach(model.collections) { c in
                    Text("\(c.name) • \(c.len)")
                        .tag(Optional(c))
                        .onTapGesture { model.selection = c }
                }
            }
        }
        .toolbar {
            Button { showingCreate = true } label: { Label("New", systemImage: "plus") }
        }
        .sheet(isPresented: $showingCreate) { CreateCollectionView(isPresented: $showingCreate) }
        .task { await model.refreshHealth(); await model.loadCollections() }
    }
}

struct ConnectionView: View {
    @EnvironmentObject var model: AppModel
    @State private var base = "http://127.0.0.1:8080"
    @State private var checking = false
    var body: some View {
        HStack(spacing: 8) {
            TextField("Server URL", text: $base).textFieldStyle(.roundedBorder)
            Button(checking ? "Checking…" : "Check") {
                checking = true
                Task { @MainActor in
                    model.baseURL = URL(string: base)
                    await model.refreshHealth()
                    await model.loadCollections()
                    checking = false
                }
            }
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
    let collection: CollectionInfo
    @EnvironmentObject var model: AppModel
    @State private var efSearch = 50
    @State private var vectorText = "[0.1, 0.2, 0.3]"
    @State private var topK = 5
    @State private var results: [SearchResult] = []
    @State private var busy = false
    @State private var notice: String?
    @State private var showBatchIngest = false
    @State private var filters: [FilterEntry] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack { Text("Collection:").bold(); Text(collection.name) }
                HStack { Text("Dim:").bold(); Text("\(collection.dim)") }
                HStack { Text("Metric:").bold(); Text(collection.metric) }
                HStack { Text("Size:").bold(); Text("\(collection.len)") }
                Divider()
                HStack(spacing: 12) {
                    Button(busy ? "Snapshot…" : "Snapshot") { actionSnapshot() }
                    Button(busy ? "Compact…" : "Compact") { actionCompact() }
                    HStack { Text("ef_search"); TextField("", value: $efSearch, formatter: NumberFormatter()).frame(width: 80); Button("Update") { actionUpdateParams() } }
                    Spacer()
                    Button("Batch Ingest") { showBatchIngest = true }
                }
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    Text("Search").font(.headline)
                    TextField("[floats]", text: $vectorText).textFieldStyle(.roundedBorder)
                    HStack { Text("top_k"); TextField("", value: $topK, formatter: NumberFormatter()).frame(width: 80) }
                    FilterBuilderView(entries: $filters)
                    Button(busy ? "Searching…" : "Search") { actionSearch() }
                    if !results.isEmpty {
                        Table(results) {
                            TableColumn("ID") { Text($0.id.uuidString.prefix(8) + "…") }
                            TableColumn("Score") { Text(String(format: "%.4f", $0.score)) }
                        }
                        .frame(minHeight: 160)
                    }
                }
                if let n = notice { Text(n).foregroundStyle(.secondary) }
            }
            .padding()
        }
        .navigationTitle(collection.name)
        .sheet(isPresented: $showBatchIngest) {
            BatchIngestView(isPresented: $showBatchIngest, collection: collection)
                .environmentObject(model)
        }
    }

    func actionSnapshot() {
        busy = true
        Task {
            do { try await model.client.snapshot(name: collection.name); await MainActor.run { notice = "Snapshot written." } }
            catch { await MainActor.run { notice = error.localizedDescription } }
            busy = false
        }
    }

    func actionCompact() {
        busy = true
        Task {
            do { let ok = try await model.client.compact(name: collection.name); await MainActor.run { notice = ok ? "Compacted HNSW." : "No-op (BruteForce)." } }
            catch { await MainActor.run { notice = error.localizedDescription } }
            busy = false
        }
    }

    func actionUpdateParams() {
        Task {
            do { let ok = try await model.client.updateEfSearch(name: collection.name, ef: efSearch); await MainActor.run { notice = ok ? "ef_search updated." : "No-op (BruteForce)." } }
            catch { await MainActor.run { notice = error.localizedDescription } }
        }
    }

    func actionSearch() {
        busy = true
        Task {
            do {
                guard let vec = try? JSONSerialization.jsonObject(with: Data(vectorText.utf8)) as? [Double] else { throw NSError(domain: "AIDB", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid vector JSON"]) }
                let f = vec.map { Float($0) }
                var filt: [String: AnyCodable]? = nil
                if !filters.isEmpty {
                    var m: [String: AnyCodable] = [:]
                    for e in filters where !e.key.isEmpty {
                        if let v = parseJSONFragment(e.value) { m[e.key] = toAnyCodable(v) }
                        else { m[e.key] = AnyCodable(e.value) }
                    }
                    filt = m
                }
                let res = try await model.client.search(name: collection.name, vector: f, topK: topK, filter: filt)
                await MainActor.run { self.results = res }
            } catch { await MainActor.run { notice = error.localizedDescription } }
            busy = false
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
