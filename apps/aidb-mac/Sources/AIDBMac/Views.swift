import SwiftUI
import Foundation
import AppKit

struct SidebarView: View {
    @EnvironmentObject var model: AppModel
    @State private var showingCreate = false
    @State private var showingSQLQuery = false

    var body: some View {
        List(selection: $model.selection) {
            Section {
                ConnectionView()
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
            } header: {
                Label("Connection", systemImage: "network")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)
            }

            Section {
                NavigationLink(destination: TablesOverviewView().environmentObject(model)) {
                    Label("SQL Tables", systemImage: "tablecells.badge.ellipsis")
                        .font(.subheadline.weight(.medium))
                }
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))

                Button {
                    showingSQLQuery = true
                } label: {
                    Label("SQL Console", systemImage: "terminal")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
            } header: {
                Label("Database", systemImage: "cylinder.fill")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)
            }

            Section {
                NavigationLink(destination: StatisticsOverviewView().environmentObject(model)) {
                    Label("Statistics", systemImage: "chart.bar.doc.horizontal")
                        .font(.subheadline.weight(.medium))
                }
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
            } header: {
                Label("Analytics", systemImage: "chart.line.uptrend.xyaxis")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)
            }

            Section {
                if model.collections.isEmpty && !model.isRefreshing {
                    VStack(spacing: 12) {
                        Image(systemName: "cylinder.split.1x2")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                        Text("No collections yet")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Button("Create First Collection") {
                            showingCreate = true
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
                    .listRowInsets(EdgeInsets())
                    .listRowBackground(Color.clear)
                }

                ForEach(model.collections) { c in
                    CollectionRowView(collection: c)
                        .tag(Optional(c))
                        .onTapGesture {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                model.selection = c
                            }
                        }
                        .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
                }
            } header: {
                Label("Vector Collections (\(model.collections.count))", systemImage: "brain.head.profile")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)
            }
        }
        .listStyle(.sidebar)
        .background(Color(NSColor.controlBackgroundColor))
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    Task {
                        await model.refreshAll()
                    }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .keyboardShortcut("r", modifiers: .command)

                Button {
                    showingCreate = true
                } label: {
                    Label("New Collection", systemImage: "plus.circle.fill")
                }
                .keyboardShortcut("n", modifiers: .command)
            }
        }
        .sheet(isPresented: $showingCreate) {
            CreateCollectionView(isPresented: $showingCreate)
        }
        .sheet(isPresented: $showingSQLQuery) {
            SQLQueryInterface()
                .environmentObject(model)
        }
    }
}

struct CollectionRowView: View {
    let collection: CollectionInfo

    var body: some View {
        HStack(spacing: 12) {
            VStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(gradientForCollection(collection.name))
                    .frame(width: 40, height: 40)
                    .overlay {
                        Image(systemName: iconForIndex(collection.index))
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(.white)
                    }
                    .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(collection.name)
                    .font(.system(.body, weight: .medium))
                    .foregroundStyle(.primary)

                HStack(spacing: 12) {
                    Label("\(collection.len)", systemImage: "number.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Label(collection.metric.capitalized, systemImage: "ruler")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Label("\(collection.dim)D", systemImage: "cube")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text(collection.index.uppercased())
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.1))
                    .foregroundStyle(.blue)
                    .clipShape(Capsule())

                if collection.walSizeBytes > 0 {
                    Text(ByteCountFormatter.string(fromByteCount: Int64(collection.walSizeBytes), countStyle: .file))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }

    private func gradientForCollection(_ name: String) -> LinearGradient {
        let colors: [(Color, Color)] = [
            (.blue, .purple),
            (.green, .teal),
            (.orange, .red),
            (.purple, .pink),
            (.teal, .blue),
            (.indigo, .purple)
        ]
        let index = abs(name.hashValue) % colors.count
        let colorPair = colors[index]
        return LinearGradient(colors: [colorPair.0, colorPair.1], startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    private func iconForIndex(_ index: String) -> String {
        switch index.lowercased() {
        case "hnsw": return "brain"
        case "bruteforce": return "magnifyingglass"
        default: return "cylinder"
        }
    }
}

struct ConnectionView: View {
    @EnvironmentObject var model: AppModel
    @State private var base = "http://127.0.0.1:8080"
    @State private var checking = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: model.healthOK ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(model.healthOK ? .green : .orange)
                        .font(.caption)

                    TextField("Server URL", text: $base)
                        .textFieldStyle(.plain)
                        .disableAutocorrection(true)
                        .font(.system(.body, design: .monospaced))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color(NSColor.textBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(model.healthOK ? Color.green.opacity(0.3) : Color.gray.opacity(0.3), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 6))

                Button {
                    checking = true
                    Task {
                        model.updateBaseURL(base)
                        await model.refreshAll()
                        await MainActor.run {
                            checking = false
                            model.clearError()
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        if checking {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .scaleEffect(0.7)
                        } else {
                            Image(systemName: "arrow.clockwise")
                                .font(.caption.weight(.medium))
                        }
                        Text(checking ? "Updating…" : "Connect")
                            .font(.caption.weight(.medium))
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(checking || base.isEmpty)
            }

            GroupBox {
                HStack {
                    Toggle("Auto refresh", isOn: $model.autoRefreshEnabled)
                        .toggleStyle(.switch)

                    Spacer()

                    VStack(alignment: .trailing, spacing: 2) {
                        Text("Every \(Int(model.autoRefreshInterval))s")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.primary)

                        Stepper(value: $model.autoRefreshInterval, in: 5...120, step: 5) {
                            EmptyView()
                        }
                        .labelsHidden()
                    }
                }
            } label: {
                Label("Auto Refresh", systemImage: "timer")
                    .font(.caption.weight(.medium))
            }
            .disabled(!model.healthOK)
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
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.blue)
                    Text("Create New Collection")
                        .font(.title2.weight(.semibold))
                    Spacer()
                }

                VStack(alignment: .leading, spacing: 16) {
                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Text("Collection Name")
                                    .font(.subheadline.weight(.medium))
                                    .frame(width: 120, alignment: .leading)
                                TextField("e.g., documents, embeddings", text: $name)
                                    .textFieldStyle(.roundedBorder)
                            }

                            HStack {
                                Text("Dimensions")
                                    .font(.subheadline.weight(.medium))
                                    .frame(width: 120, alignment: .leading)
                                TextField("768", value: $dim, formatter: NumberFormatter())
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 100)
                                Spacer()
                                Text("Vector size (e.g., 768 for OpenAI)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    } label: {
                        Label("Basic Configuration", systemImage: "gear")
                            .font(.subheadline.weight(.medium))
                    }

                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Text("Distance Metric")
                                    .font(.subheadline.weight(.medium))
                                    .frame(width: 120, alignment: .leading)
                                Picker("metric", selection: $metric) {
                                    Label("Cosine", systemImage: "function")
                                        .tag("cosine")
                                    Label("Euclidean", systemImage: "ruler")
                                        .tag("euclidean")
                                }
                                .pickerStyle(.segmented)
                                .frame(width: 250)
                            }

                            HStack {
                                Text("Index Type")
                                    .font(.subheadline.weight(.medium))
                                    .frame(width: 120, alignment: .leading)
                                Picker("index", selection: $index) {
                                    Label("HNSW", systemImage: "brain")
                                        .tag("hnsw")
                                    Label("BruteForce", systemImage: "magnifyingglass")
                                        .tag("bruteforce")
                                }
                                .pickerStyle(.segmented)
                                .frame(width: 250)
                            }
                        }
                    } label: {
                        Label("Search Configuration", systemImage: "magnifyingglass.circle")
                            .font(.subheadline.weight(.medium))
                    }

                    if index == "hnsw" {
                        GroupBox {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Text("M Parameter")
                                        .font(.subheadline.weight(.medium))
                                        .frame(width: 120, alignment: .leading)
                                    TextField("16", value: $m, formatter: NumberFormatter())
                                        .textFieldStyle(.roundedBorder)
                                        .frame(width: 80)
                                    Spacer()
                                    Text("Graph connectivity (higher = better recall)")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }

                                HStack {
                                    Text("Construction")
                                        .font(.subheadline.weight(.medium))
                                        .frame(width: 120, alignment: .leading)
                                    TextField("200", value: $efc, formatter: NumberFormatter())
                                        .textFieldStyle(.roundedBorder)
                                        .frame(width: 80)
                                    Spacer()
                                    Text("ef_construction (higher = better quality)")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }

                                HStack {
                                    Text("Search")
                                        .font(.subheadline.weight(.medium))
                                        .frame(width: 120, alignment: .leading)
                                    TextField("50", value: $efs, formatter: NumberFormatter())
                                        .textFieldStyle(.roundedBorder)
                                        .frame(width: 80)
                                    Spacer()
                                    Text("ef_search (higher = better accuracy)")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        } label: {
                            Label("HNSW Parameters", systemImage: "brain.head.profile")
                                .font(.subheadline.weight(.medium))
                        }
                        .transition(.opacity.combined(with: .move(edge: .top)))
                    }

                    if let e = error {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                            Text(e)
                                .font(.callout)
                                .foregroundStyle(.red)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }

                Spacer()
            }
            .padding(24)

            Divider()

            HStack {
                Spacer()
                Button("Cancel") {
                    isPresented = false
                }
                .buttonStyle(.bordered)

                Button {
                    creating = true
                    Task {
                        do {
                            let req = CreateCollectionReq(
                                name: name,
                                dim: dim,
                                metric: metric,
                                wal_dir: "./data",
                                index: index,
                                hnsw: index == "hnsw" ? HnswParams(m: m, ef_construction: efc, ef_search: efs) : nil
                            )
                            try await model.client.createCollection(req)
                            await model.loadCollections()
                            await MainActor.run { isPresented = false }
                        } catch {
                            await MainActor.run { self.error = error.localizedDescription }
                        }
                        creating = false
                    }
                } label: {
                    HStack(spacing: 6) {
                        if creating {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "plus.circle.fill")
                        }
                        Text(creating ? "Creating Collection…" : "Create Collection")
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(creating || name.isEmpty)
            }
            .padding(20)
        }
        .frame(minWidth: 550, minHeight: 500)
        .background(Color(NSColor.controlBackgroundColor))
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
            LazyVStack(alignment: .leading, spacing: 20) {
                statsCardsView

                if detail.walSizeBytes > 0 {
                    walGroupBox
                }

                configurationGroupBox

                actionsView

                searchView

                if let n = notice {
                    Text(n).foregroundStyle(.secondary)
                }
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

    private var actionsView: some View {
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
    }

    private var searchView: some View {
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
                    SwiftUI.TableColumn("ID") { result in
                        Text(result.id.uuidString.prefix(8) + "…")
                            .onTapGesture { selectedResult = result }
                    }
                    SwiftUI.TableColumn("Score") { result in
                        Text(String(format: "%.4f", result.score))
                    }
                }
                .frame(minHeight: 160)
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

    private var statsCardsView: some View {
        HStack(spacing: 16) {
            CollectionStatsCardView(
                title: "Vectors",
                value: "\(detail.len)",
                icon: "number.circle.fill",
                color: .blue
            )

            CollectionStatsCardView(
                title: "Dimensions",
                value: "\(detail.dim)",
                icon: "cube.fill",
                color: .green
            )

            CollectionStatsCardView(
                title: "Index Type",
                value: detail.index.uppercased(),
                icon: iconForIndex(detail.index),
                color: .purple
            )

            CollectionStatsCardView(
                title: "Metric",
                value: detail.metric.capitalized,
                icon: "ruler.fill",
                color: .orange
            )
        }
    }

    private var walGroupBox: some View {
        GroupBox {
            WALVisualizationView(detail: detail)
        } label: {
            Label("Write-Ahead Log", systemImage: "doc.text.fill")
                .font(.subheadline.weight(.medium))
        }
    }

    private var configurationGroupBox: some View {
        GroupBox {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    Text("Collection Name")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)
                    Text(detail.name)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.primary)
                }

                if let hnsw = detail.hnsw {
                    GridRow {
                        Text("HNSW Parameters")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("M: \(hnsw.m), ef_construction: \(hnsw.efConstruction)")
                                .font(.subheadline)
                            Text("ef_search: \(hnsw.efSearch)")
                                .font(.subheadline)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .padding(.vertical, 4)
        } label: {
            Label("Configuration", systemImage: "gearshape.fill")
                .font(.subheadline.weight(.medium))
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
                _ = await model.fetchCollectionDetail(name: collection.name)
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
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    HStack(spacing: 8) {
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.title2)
                            .foregroundStyle(.blue)
                        Text("Search Result Detail")
                            .font(.title2.weight(.semibold))
                    }
                    Spacer()
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.cancelAction)
                }

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("ID", systemImage: "number.circle")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(result.id.uuidString)
                            .font(.system(.subheadline, design: .monospaced))
                            .textSelection(.enabled)
                    }

                    HStack {
                        Label("Score", systemImage: "target")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.6f", result.score))
                            .font(.system(.subheadline, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(NSColor.controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                VStack(alignment: .leading, spacing: 8) {
                    Text("Payload")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)

                    ScrollView {
                        Text(prettyPayload())
                            .font(.system(.body, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(minHeight: 250)
                    .background(Color(NSColor.textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )
                }

                Spacer()
            }
            .padding(24)

            Divider()

            HStack {
                Button("Copy Payload") {
                    let pasteboard = NSPasteboard.general
                    pasteboard.clearContents()
                    pasteboard.setString(prettyPayload(), forType: .string)
                }

                Button("Copy ID") {
                    let pasteboard = NSPasteboard.general
                    pasteboard.clearContents()
                    pasteboard.setString(result.id.uuidString, forType: .string)
                }

                Spacer()

                Button("Close") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(20)
        }
        .frame(minWidth: 500, minHeight: 400)
        .background(Color(NSColor.controlBackgroundColor))
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
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    HStack(spacing: 8) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.title2)
                            .foregroundStyle(.blue)
                        Text("Metrics for \(title)")
                            .font(.title2.weight(.semibold))
                    }
                    Spacer()
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.cancelAction)
                }

                ScrollView {
                    Text(text.isEmpty ? "No metrics available." : text)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 300)
                .background(Color(NSColor.textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                )

                Spacer()
            }
            .padding(24)

            Divider()

            HStack {
                Button("Copy to Clipboard") {
                    let pasteboard = NSPasteboard.general
                    pasteboard.clearContents()
                    pasteboard.setString(text, forType: .string)
                }
                .disabled(text.isEmpty)

                Spacer()

                Button("Close") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(20)
        }
        .frame(minWidth: 600, minHeight: 450)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

// MARK: - Modern UI Components

struct CollectionStatsCardView: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundStyle(color)
                    .frame(width: 24, height: 24)
                Spacer()
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.title2.weight(.bold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(color.opacity(0.2), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.05), radius: 2, x: 0, y: 1)
    }
}

struct WALVisualizationView: View {
    let detail: CollectionInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Total Size")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                    Text(ByteCountFormatter.string(fromByteCount: Int64(detail.walSizeBytes), countStyle: .file))
                        .font(.subheadline.weight(.semibold))
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    Text("Since Truncate")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                    Text(ByteCountFormatter.string(fromByteCount: Int64(detail.walBytesSinceTruncate), countStyle: .file))
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(detail.walBytesSinceTruncate > detail.walSizeBytes / 2 ? .orange : .primary)
                }
            }

            if detail.walBytesSinceTruncate > 0 {
                ProgressView(value: Double(detail.walBytesSinceTruncate), total: Double(max(detail.walSizeBytes, detail.walBytesSinceTruncate))) {
                    Text("WAL Usage")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .progressViewStyle(.linear)
                .tint(detail.walBytesSinceTruncate > detail.walSizeBytes / 2 ? .orange : .blue)
            }

            if let lastTruncate = detail.walLastTruncate, lastTruncate != "n/a" {
                Text("Last truncate: \(lastTruncate)")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 4)
    }
}

extension CollectionDetailView {
    private func iconForIndex(_ index: String) -> String {
        switch index.lowercased() {
        case "hnsw": return "brain.fill"
        case "bruteforce": return "magnifyingglass.circle.fill"
        default: return "cylinder.fill"
        }
    }
}

// MARK: - SQL Table Management UI

struct CreateTableView: View {
    @EnvironmentObject var model: AppModel
    @Binding var isPresented: Bool
    @State private var tableName = ""
    @State private var dimension = 768
    @State private var metric = "cosine"
    @State private var indexType = "hnsw"
    @State private var columns: [CreateTableColumn] = [CreateTableColumn()]
    @State private var creating = false
    @State private var error: String?
    @State private var creationType: CreationType = .collection

    enum CreationType: String, CaseIterable {
        case collection = "collection"
        case table = "table"

        var displayName: String {
            switch self {
            case .collection: return "Vector Collection"
            case .table: return "SQL Table"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            createTableContent
            Divider()
            createTableFooter
            sqlPreviewSection
        }
        .frame(minWidth: 700, minHeight: 600)
        .background(Color(NSColor.controlBackgroundColor))
    }

    private var createTableContent: some View {
        VStack(alignment: .leading, spacing: 20) {
            createTableHeader
            createTableForm
            Spacer()
        }
        .padding(24)
    }

    private var createTableHeader: some View {
        HStack {
            Image(systemName: creationType == .collection ? "brain.head.profile" : "tablecells.badge.ellipsis")
                .font(.title2)
                .foregroundStyle(.blue)
            Text(creationType == .collection ? "Create New Collection" : "Create New Table")
                .font(.title2.weight(.semibold))
            Spacer()
        }
    }

    private var createTableForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            typeSelectionSection
            basicInfoSection
            if creationType == .table {
                schemaDesignSection
            }
            errorSection
        }
    }

    private var typeSelectionSection: some View {
        GroupBox {
            Picker("Type", selection: $creationType) {
                ForEach(CreationType.allCases, id: \.self) { type in
                    Text(type.displayName).tag(type)
                }
            }
            .pickerStyle(.segmented)
        } label: {
            Label("Creation Type", systemImage: "rectangle.3.group")
                .font(.subheadline.weight(.medium))
        }
    }

    private var basicInfoSection: some View {
        GroupBox {
            VStack(spacing: 12) {
                HStack {
                    Text(creationType == .collection ? "Collection Name" : "Table Name")
                        .font(.subheadline.weight(.medium))
                        .frame(width: 120, alignment: .leading)
                    TextField(creationType == .collection ? "e.g., documents, embeddings" : "e.g., users, products", text: $tableName)
                        .textFieldStyle(.roundedBorder)
                }

                if creationType == .collection {
                    HStack {
                        Text("Vector Dimension")
                            .font(.subheadline.weight(.medium))
                            .frame(width: 120, alignment: .leading)
                        TextField("768", value: $dimension, formatter: NumberFormatter())
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 100)
                        Spacer()
                        Text("Size of each vector (e.g., 768 for OpenAI embeddings)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    HStack {
                        Text("Distance Metric")
                            .font(.subheadline.weight(.medium))
                            .frame(width: 120, alignment: .leading)
                        Picker("Metric", selection: $metric) {
                            Text("Cosine").tag("cosine")
                            Text("Euclidean").tag("euclidean")
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 200)
                        Spacer()
                    }

                    HStack {
                        Text("Index Type")
                            .font(.subheadline.weight(.medium))
                            .frame(width: 120, alignment: .leading)
                        Picker("Index", selection: $indexType) {
                            Text("HNSW").tag("hnsw")
                            Text("Brute Force").tag("bruteforce")
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 200)
                        Spacer()
                    }
                }
            }
        } label: {
            Label(creationType == .collection ? "Collection Configuration" : "Table Configuration",
                  systemImage: creationType == .collection ? "brain.head.profile" : "tablecells")
                .font(.subheadline.weight(.medium))
        }
    }

    private var schemaDesignHeader: some View {
        HStack {
            Text("Columns")
                .font(.subheadline.weight(.medium))
            Spacer()
            Button("Add Column") {
                withAnimation(.easeInOut(duration: 0.2)) {
                    columns.append(CreateTableColumn())
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private var schemaDesignContent: some View {
        VStack(spacing: 8) {
            ForEach(0..<columns.count, id: \.self) { index in
                ColumnDesignerRow(
                    column: $columns[index],
                    onDelete: columns.count > 1 ? {
                        columns.remove(at: index)
                    } : nil
                )
            }
        }
    }

    private var schemaDesignSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                schemaDesignHeader
                schemaDesignContent
            }
        } label: {
            Label("Schema Design", systemImage: "square.grid.3x3")
                .font(.subheadline.weight(.medium))
        }
    }

    @ViewBuilder
    private var errorSection: some View {
        if let e = error {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(e)
                    .font(.callout)
                    .foregroundStyle(.red)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.red.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var createTableFooter: some View {
        HStack {
            Text("Preview SQL:")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            Spacer()
            Button("Cancel") {
                isPresented = false
            }
            .buttonStyle(.bordered)

            Button {
                createTable()
            } label: {
                HStack(spacing: 6) {
                    if creating {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: creationType == .collection ? "brain.head.profile" : "tablecells.badge.ellipsis")
                    }
                    Text(creating ?
                         (creationType == .collection ? "Creating Collection…" : "Creating Table…") :
                         (creationType == .collection ? "Create Collection" : "Create Table"))
                }
            }
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.defaultAction)
            .disabled(creating || tableName.isEmpty || (creationType == .table && columns.isEmpty))
        }
        .padding(20)
    }

    @ViewBuilder
    private var sqlPreviewSection: some View {
        if !tableName.isEmpty && !columns.isEmpty {
            ScrollView(.horizontal) {
                Text(generateSQL())
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 12)
            }
            .frame(maxHeight: 60)
            .background(Color(NSColor.textBackgroundColor))
        }
    }

    private func createTable() {
        creating = true
        Task {
            do {
                let sql = generateSQL()
                if creationType == .collection {
                    _ = try await model.client.executeSQL(sql)
                } else {
                    _ = try await model.client.executeSQLTable(sql)
                }
                await MainActor.run { isPresented = false }
            } catch {
                await MainActor.run { self.error = error.localizedDescription }
            }
            creating = false
        }
    }

    private func generateSQL() -> String {
        if creationType == .collection {
            // Generate CREATE COLLECTION for AIDB vector database
            var params: [String] = []
            params.append("DIM = \(dimension)")
            params.append("METRIC = '\(metric)'")
            params.append("INDEX = '\(indexType)'")
            return "CREATE COLLECTION \(tableName) (\n  \(params.joined(separator: ",\n  "))\n);"
        } else {
            // Generate CREATE TABLE for SQL tables
            let columnDefs = columns.compactMap { col -> String? in
                guard !col.name.isEmpty else { return nil }
                var def = "\(col.name) \(col.type)"
                if col.primaryKey { def += " PRIMARY KEY" }
                if !col.nullable && !col.primaryKey { def += " NOT NULL" }
                if !col.defaultValue.isEmpty { def += " DEFAULT '\(col.defaultValue)'" }
                return def
            }
            return "CREATE TABLE \(tableName) (\n  \(columnDefs.joined(separator: ",\n  "))\n);"
        }
    }
}

struct ColumnDesignerRow: View {
    @Binding var column: CreateTableColumn
    let onDelete: (() -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: ColumnType(rawValue: column.type)?.icon ?? "textformat")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 16)

                TextField("Column name", text: $column.name)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
            }

            Picker("Type", selection: $column.type) {
                ForEach(ColumnType.allCases, id: \.rawValue) { type in
                    Label(type.displayName, systemImage: type.icon)
                        .tag(type.rawValue)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 120)

            Toggle("PK", isOn: $column.primaryKey)
                .toggleStyle(.checkbox)
                .help("Primary Key")

            Toggle("Null", isOn: $column.nullable)
                .toggleStyle(.checkbox)
                .help("Allow NULL values")
                .disabled(column.primaryKey)

            TextField("Default", text: $column.defaultValue)
                .textFieldStyle(.roundedBorder)
                .frame(width: 80)
                .disabled(column.primaryKey)

            if let onDelete = onDelete {
                Button {
                    onDelete()
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(.red)
                }
                .buttonStyle(.borderless)
                .help("Delete column")
            } else {
                Spacer().frame(width: 20)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: column.primaryKey)
    }
}

struct TablesOverviewView: View {
    @EnvironmentObject var model: AppModel
    @State private var showingCreateTable = false
    @State private var searchText = ""
    @State private var isRefreshing = false

    var filteredTables: [TableInfo] {
        if searchText.isEmpty {
            return model.tables
        }
        return model.tables.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Database Tables")
                            .font(.title2.weight(.bold))
                        Text("\(model.tables.count) tables • Hybrid Vector + SQL Database")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()

                    HStack(spacing: 8) {
                        Button {
                            Task {
                                isRefreshing = true
                                await loadTables()
                                isRefreshing = false
                            }
                        } label: {
                            Image(systemName: "arrow.clockwise")
                                .rotationEffect(.degrees(isRefreshing ? 360 : 0))
                        }
                        .buttonStyle(.bordered)
                        .animation(.linear(duration: 1).repeatForever(autoreverses: false), value: isRefreshing)

                        Button {
                            showingCreateTable = true
                        } label: {
                            Label("New Table", systemImage: "plus.circle.fill")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }

                HStack {
                    SearchField(text: $searchText)
                    Spacer()
                    if !model.tables.isEmpty {
                        Text("\(filteredTables.count) of \(model.tables.count)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(20)

            Divider()

            if model.tables.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "tablecells.badge.ellipsis")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("No tables yet")
                        .font(.title3.weight(.medium))
                        .foregroundStyle(.secondary)
                    Text("Create your first table to start storing structured data")
                        .font(.subheadline)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                    Button("Create First Table") {
                        showingCreateTable = true
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(NSColor.controlBackgroundColor))
            } else {
                ScrollView {
                    LazyVGrid(columns: [
                        GridItem(.adaptive(minimum: 320, maximum: 400), spacing: 16)
                    ], spacing: 16) {
                        ForEach(filteredTables) { table in
                            TableCard(table: table) {
                                model.selectedTable = table
                            }
                        }
                    }
                    .padding(20)
                }
            }
        }
        .sheet(isPresented: $showingCreateTable) {
            CreateTableView(isPresented: $showingCreateTable)
                .environmentObject(model)
        }
        .task {
            await loadTables()
        }
    }

    private func loadTables() async {
        do {
            let tables = try await model.client.listTables()
            await MainActor.run {
                model.tables = tables
            }
        } catch {
            await model.report(error)
        }
    }
}

struct TableCard: View {
    let table: TableInfo
    let onSelect: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(table.name)
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.primary)

                    if let rowCount = table.rowCount {
                        Text("\(rowCount) rows")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    Text("\(table.columns.count)")
                        .font(.title2.weight(.bold))
                        .foregroundStyle(.blue)
                    Text("columns")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 8) {
                ForEach(table.columns.prefix(4)) { column in
                    HStack(spacing: 6) {
                        Image(systemName: ColumnType(rawValue: column.type)?.icon ?? "textformat")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text(column.name)
                            .font(.caption)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        if column.primaryKey {
                            Image(systemName: "key.fill")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                        Spacer()
                    }
                }

                if table.columns.count > 4 {
                    Text("+\(table.columns.count - 4) more")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            HStack {
                if let size = table.size {
                    Label(ByteCountFormatter.string(fromByteCount: size, countStyle: .file), systemImage: "doc.circle")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button("View Data") {
                    onSelect()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(16)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.05), radius: 2, x: 0, y: 1)
        .onTapGesture {
            onSelect()
        }
    }
}

struct SearchField: View {
    @Binding var text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search tables...", text: $text)
                .textFieldStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(NSColor.textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
        )
    }
}

// MARK: - Data Panel and Query Interface

struct TableDataView: View {
    @EnvironmentObject var model: AppModel
    let table: TableInfo
    @State private var tableData: [[String: AnyCodable]] = []
    @State private var isLoading = false
    @State private var currentPage = 0
    @State private var rowsPerPage = 50
    @State private var showingSQLQuery = false
    @State private var sortColumn: String?
    @State private var sortAscending = true

    private let pageSize = 50

    var body: some View {
        VStack(spacing: 0) {
            TableDataHeader(
                table: table,
                rowCount: tableData.count,
                onRefresh: { await loadData() },
                onQuery: { showingSQLQuery = true }
            )

            Divider()

            if isLoading {
                VStack(spacing: 16) {
                    ProgressView()
                        .progressViewStyle(.circular)
                    Text("Loading table data...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if tableData.isEmpty {
                EmptyTableView {
                    showingSQLQuery = true
                }
            } else {
                ScrollView([.horizontal, .vertical]) {
                    DataTable(
                        columns: table.columns,
                        data: tableData,
                        sortColumn: $sortColumn,
                        sortAscending: $sortAscending
                    )
                    .padding(16)
                }
                .background(Color(NSColor.textBackgroundColor))

                TableDataFooter(
                    currentPage: currentPage,
                    totalRows: tableData.count,
                    rowsPerPage: rowsPerPage,
                    onPageChange: { page in
                        currentPage = page
                        Task { await loadData(page: page) }
                    }
                )
            }
        }
        .sheet(isPresented: $showingSQLQuery) {
            SQLQueryInterface(table: table)
                .environmentObject(model)
        }
        .task {
            await loadData()
        }
    }

    private func loadData(page: Int = 0) async {
        isLoading = true
        do {
            let data = try await model.client.getTableData(
                table.name,
                limit: pageSize,
                offset: page * pageSize
            )
            await MainActor.run {
                self.tableData = data
                self.currentPage = page
            }
        } catch {
            await model.report(error)
        }
        isLoading = false
    }
}

struct TableDataHeader: View {
    let table: TableInfo
    let rowCount: Int
    let onRefresh: () async -> Void
    let onQuery: () -> Void
    @State private var isRefreshing = false

    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(table.name)
                    .font(.title2.weight(.bold))
                Text("\(rowCount) rows • \(table.columns.count) columns")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            HStack(spacing: 8) {
                Button {
                    Task {
                        isRefreshing = true
                        await onRefresh()
                        isRefreshing = false
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .rotationEffect(.degrees(isRefreshing ? 360 : 0))
                }
                .buttonStyle(.bordered)
                .animation(.linear(duration: 1).repeatForever(autoreverses: false), value: isRefreshing)

                Button {
                    onQuery()
                } label: {
                    Label("Query", systemImage: "terminal")
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
    }
}

struct EmptyTableView: View {
    let onAddData: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "tablecells")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("No data yet")
                .font(.title3.weight(.medium))
                .foregroundStyle(.secondary)
            Text("This table is empty. Use SQL queries to insert data.")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            Button("Add Data with SQL") {
                onAddData()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct RowData: Identifiable {
    let id = UUID()
    let data: [String: AnyCodable]
}

struct DataTable: View {
    let columns: [TableColumn]
    let data: [[String: AnyCodable]]
    @Binding var sortColumn: String?
    @Binding var sortAscending: Bool

    private var rowData: [RowData] {
        data.map { RowData(data: $0) }
    }

    var body: some View {
        ScrollView(.horizontal) {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: columns.count), spacing: 1) {
                // Header row
                ForEach(columns) { column in
                    Text(column.name)
                        .font(.headline)
                        .padding(8)
                        .frame(maxWidth: .infinity)
                        .background(Color.secondary.opacity(0.1))
                }

                // Data rows
                ForEach(data.indices, id: \.self) { rowIndex in
                    ForEach(columns) { column in
                        DataCell(
                            value: data[rowIndex][column.name],
                            columnType: column.type
                        )
                        .padding(4)
                        .frame(maxWidth: .infinity)
                        .background(Color.primary.opacity(0.05))
                    }
                }
            }
        }
    }

    private func idealColumnWidth(for column: TableColumn) -> CGFloat {
        switch ColumnType(rawValue: column.type) {
        case .text, .json: return 200
        case .integer: return 100
        case .real: return 120
        case .vector: return 250
        case .blob: return 150
        default: return 150
        }
    }
}

struct DataCell: View {
    let value: AnyCodable?
    let columnType: String

    var body: some View {
        Group {
            if let value = value {
                switch ColumnType(rawValue: columnType) {
                case .vector:
                    VectorCell(value: value)
                case .json:
                    JSONCell(value: value)
                case .blob:
                    BlobCell(value: value)
                default:
                    Text(formatValue(value))
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
            } else {
                Text("NULL")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .italic()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }

    private func formatValue(_ value: AnyCodable) -> String {
        switch value.value {
        case let str as String: return str
        case let num as NSNumber: return num.stringValue
        case let bool as Bool: return bool ? "true" : "false"
        default: return String(describing: value.value)
        }
    }
}

struct VectorCell: View {
    let value: AnyCodable

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "brain")
                .font(.caption)
                .foregroundStyle(.blue)
            if let array = value.value as? [Any] {
                Text("[\(array.count) dims]")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            } else {
                Text("Vector")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct JSONCell: View {
    let value: AnyCodable

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "curlybraces")
                .font(.caption)
                .foregroundStyle(.purple)
            Text("JSON")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }
}

struct BlobCell: View {
    let value: AnyCodable

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "doc.circle")
                .font(.caption)
                .foregroundStyle(.orange)
            Text("Binary")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }
}

struct TableDataFooter: View {
    let currentPage: Int
    let totalRows: Int
    let rowsPerPage: Int
    let onPageChange: (Int) -> Void

    private var totalPages: Int {
        max(1, (totalRows + rowsPerPage - 1) / rowsPerPage)
    }

    var body: some View {
        HStack {
            Text("Page \(currentPage + 1) of \(totalPages)")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            HStack(spacing: 8) {
                Button {
                    onPageChange(max(0, currentPage - 1))
                } label: {
                    Image(systemName: "chevron.left")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(currentPage == 0)

                Button {
                    onPageChange(min(totalPages - 1, currentPage + 1))
                } label: {
                    Image(systemName: "chevron.right")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(currentPage >= totalPages - 1)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct SQLQueryInterface: View {
    @EnvironmentObject var model: AppModel
    let table: TableInfo?
    @State private var queryText = ""
    @State private var results: [[String: AnyCodable]] = []
    @State private var resultColumns: [TableColumn] = []
    @State private var isExecuting = false
    @State private var error: String?
    @State private var successMessage: String?
    @Environment(\.dismiss) private var dismiss

    init(table: TableInfo? = nil) {
        self.table = table
        if let table = table {
            _queryText = State(initialValue: "SELECT * FROM \(table.name) LIMIT 10;")
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    HStack(spacing: 8) {
                        Image(systemName: "terminal.fill")
                            .font(.title2)
                            .foregroundStyle(.green)
                        Text("SQL Query Interface")
                            .font(.title2.weight(.semibold))
                    }
                    Spacer()
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.cancelAction)
                }

                GroupBox {
                    VStack(spacing: 8) {
                        HStack {
                            Text("Query")
                                .font(.subheadline.weight(.medium))
                            Spacer()
                            if let table = table {
                                Text("Table: \(table.name)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }

                        TextEditor(text: $queryText)
                            .font(.system(.body, design: .monospaced))
                            .frame(minHeight: 120)
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                            )
                    }
                } label: {
                    Label("SQL Editor", systemImage: "pencil.and.outline")
                        .font(.subheadline.weight(.medium))
                }

                if let error = error {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                        Text(error)
                            .font(.callout)
                            .foregroundStyle(.red)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.red.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                if let message = successMessage {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text(message)
                            .font(.callout)
                            .foregroundStyle(.green)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                if !results.isEmpty {
                    GroupBox {
                        ScrollView([.horizontal, .vertical]) {
                            DataTable(
                                columns: resultColumns,
                                data: results,
                                sortColumn: .constant(nil),
                                sortAscending: .constant(true)
                            )
                        }
                        .frame(maxHeight: 300)
                    } label: {
                        Label("Query Results (\(results.count) rows)", systemImage: "tablecells")
                            .font(.subheadline.weight(.medium))
                    }
                }

                Spacer()
            }
            .padding(24)

            Divider()

            HStack {
                SQLTemplatesMenu(table: table) { template in
                    queryText = template
                }

                Spacer()

                Button("Clear") {
                    queryText = ""
                    results = []
                    resultColumns = []
                    error = nil
                    successMessage = nil
                }
                .buttonStyle(.bordered)

                Button {
                    executeQuery()
                } label: {
                    HStack(spacing: 6) {
                        if isExecuting {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "play.fill")
                        }
                        Text(isExecuting ? "Executing…" : "Execute")
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(isExecuting || queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(20)
        }
        .frame(minWidth: 800, minHeight: 600)
        .background(Color(NSColor.controlBackgroundColor))
    }

    private func executeQuery() {
        isExecuting = true
        error = nil
        successMessage = nil

        Task {
            do {
                let response = try await model.client.executeSQLTable(queryText)

                await MainActor.run {
                    if let data = response.data {
                        self.results = data
                        self.resultColumns = inferColumns(from: data)
                        self.successMessage = "Query executed successfully. \(data.count) rows returned."
                    } else if let rows = response.rows {
                        // Handle rows format from /sql/tables endpoint
                        // Use column names from response if available, otherwise fall back to generic names
                        let columnNames = response.columns ?? (0..<(rows.first?.count ?? 0)).map { "col_\($0)" }
                        let convertedData = rows.map { row in
                            var dict: [String: AnyCodable] = [:]
                            for (index, value) in row.enumerated() {
                                let columnName = index < columnNames.count ? columnNames[index] : "col_\(index)"
                                dict[columnName] = value
                            }
                            return dict
                        }
                        self.results = convertedData
                        self.resultColumns = columnNames.map {
                            TableColumn(name: $0, type: "TEXT", nullable: true, primaryKey: false, defaultValue: nil)
                        }
                        self.successMessage = "Query executed successfully. \(convertedData.count) rows returned."
                    } else if response.ok == true {
                        self.results = []
                        self.resultColumns = []
                        self.successMessage = "Query executed successfully."
                    }
                }
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    self.results = []
                    self.resultColumns = []
                }
            }
            isExecuting = false
        }
    }

    private func inferColumns(from data: [[String: AnyCodable]]) -> [TableColumn] {
        guard let firstRow = data.first else { return [] }
        return firstRow.keys.map { key in
            let value = firstRow[key]
            let type = inferType(from: value)
            return TableColumn(
                name: key,
                type: type,
                nullable: true,
                primaryKey: false,
                defaultValue: nil
            )
        }.sorted { $0.name < $1.name }
    }

    private func inferType(from value: AnyCodable?) -> String {
        guard let value = value else { return "TEXT" }
        switch value.value {
        case is String: return "TEXT"
        case is Int: return "INTEGER"
        case is Double, is Float: return "REAL"
        case is [Any]: return "VECTOR"
        case is [String: Any]: return "JSON"
        default: return "TEXT"
        }
    }
}

struct SQLTemplatesMenu: View {
    let table: TableInfo?
    let onSelect: (String) -> Void

    var body: some View {
        Menu {
            Section("Basic Queries") {
                Button("Select All") {
                    if let table = table {
                        onSelect("SELECT * FROM \(table.name);")
                    } else {
                        onSelect("SELECT * FROM table_name;")
                    }
                }

                Button("Count Rows") {
                    if let table = table {
                        onSelect("SELECT COUNT(*) FROM \(table.name);")
                    } else {
                        onSelect("SELECT COUNT(*) FROM table_name;")
                    }
                }

                Button("Insert Row") {
                    if let table = table {
                        let columns = table.columns.map { $0.name }.joined(separator: ", ")
                        let values = table.columns.map { _ in "?" }.joined(separator: ", ")
                        onSelect("INSERT INTO \(table.name) (\(columns)) VALUES (\(values));")
                    } else {
                        onSelect("INSERT INTO table_name (column1, column2) VALUES (value1, value2);")
                    }
                }
            }

            Section("Table Management") {
                Button("Create Collection") {
                    onSelect("CREATE COLLECTION documents (\n  DIM = 768,\n  METRIC = 'cosine',\n  INDEX = 'hnsw'\n);")
                }

                Button("Show Tables") {
                    onSelect("SHOW TABLES;")
                }

                if let table = table {
                    Button("Describe Table") {
                        onSelect("DESCRIBE \(table.name);")
                    }
                }
            }

            Section("Vector Operations") {
                Button("Vector Search") {
                    onSelect("SEARCH collection_name (\n  VECTOR = [0.1, 0.2, 0.3],\n  TOPK = 5,\n  FILTER = {\"category\": \"example\"}\n);")
                }

                Button("Insert Vector") {
                    onSelect("INSERT INTO docs VALUES (\n  ID = 'unique-id',\n  VECTOR = [0.1, 0.2, 0.3, 0.4],\n  PAYLOAD = {\"source\": \"document.txt\"}\n);")
                }
            }
        } label: {
            Label("Templates", systemImage: "doc.text.below.ecg")
        }
        .buttonStyle(.bordered)
    }
}

// MARK: - Dashboard

struct DashboardView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                VStack(spacing: 16) {
                    Image(systemName: "cylinder.split.1x2.fill")
                        .font(.system(size: 64))
                        .foregroundStyle(.blue)

                    VStack(spacing: 8) {
                        Text("AIDB Hybrid Database")
                            .font(.largeTitle.weight(.bold))

                        Text("Vector Search + SQL • The Ultimate Database Experience")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }

                    ConnectionView()
                        .frame(maxWidth: 400)

                    if model.healthOK {
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Text("Server connection healthy")
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.green)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.green.opacity(0.1))
                        .clipShape(Capsule())
                    }
                }
                .padding(.top, 40)

                HStack(spacing: 24) {
                    DashboardCard(
                        title: "Vector Collections",
                        count: model.collections.count,
                        icon: "brain.head.profile",
                        color: .purple,
                        description: "High-dimensional vector storage with HNSW indexing"
                    )

                    DashboardCard(
                        title: "SQL Tables",
                        count: model.tables.count,
                        icon: "tablecells.badge.ellipsis",
                        color: .blue,
                        description: "Structured data with full SQL compatibility"
                    )
                }
                .frame(maxWidth: 600)

                Spacer(minLength: 40)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 40)
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct DashboardCard: View {
    let title: String
    let count: Int
    let icon: String
    let color: Color
    let description: String

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundStyle(color)
                Spacer()
                Text("\(count)")
                    .font(.title.weight(.bold))
                    .foregroundStyle(color)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline.weight(.semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)

                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(20)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(color.opacity(0.2), lineWidth: 2)
        )
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
    }
}

// MARK: - Statistics Views

struct StatisticsOverviewView: View {
    @EnvironmentObject var model: AppModel
    @State private var selectedTab = 0

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Database Statistics")
                        .font(.largeTitle.weight(.bold))
                    Text("Analyze table structure and data patterns")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    Task { await model.fetchStatistics() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(model.isAnalyzing)
            }
            .padding()

            TabView(selection: $selectedTab) {
                StatisticsSummaryView()
                    .environmentObject(model)
                    .tabItem {
                        Label("Overview", systemImage: "chart.bar")
                    }
                    .tag(0)

                TableStatisticsView()
                    .environmentObject(model)
                    .tabItem {
                        Label("Tables", systemImage: "tablecells")
                    }
                    .tag(1)

                ColumnStatisticsView()
                    .environmentObject(model)
                    .tabItem {
                        Label("Columns", systemImage: "list.bullet.rectangle")
                    }
                    .tag(2)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            Task { await model.fetchStatistics() }
        }
    }
}

struct StatisticsSummaryView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ScrollView {
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 20) {
                if let summary = model.statisticsSummary {
                    StatsCardView(
                        title: "Total Tables",
                        value: "\(summary.total_tables)",
                        icon: "tablecells",
                        color: .blue
                    )

                    StatsCardView(
                        title: "Analyzed Tables",
                        value: "\(summary.analyzed_tables)",
                        icon: "chart.bar.doc.horizontal",
                        color: .green
                    )

                    StatsCardView(
                        title: "Total Columns",
                        value: "\(summary.total_columns)",
                        icon: "list.bullet",
                        color: .orange
                    )

                    StatsCardView(
                        title: "Analyzed Columns",
                        value: "\(summary.analyzed_columns)",
                        icon: "checkmark.circle",
                        color: .purple
                    )
                } else {
                    ForEach(0..<4, id: \.self) { _ in
                        StatsCardView(
                            title: "Loading...",
                            value: "-",
                            icon: "ellipsis",
                            color: .gray
                        )
                    }
                }
            }
            .padding()

            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Quick Actions")
                        .font(.headline.weight(.semibold))
                    Spacer()
                }

                HStack(spacing: 12) {
                    Button {
                        Task { await model.analyzeTable(nil) }
                    } label: {
                        Label("Analyze All Tables", systemImage: "chart.bar.doc.horizontal.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isAnalyzing)

                    if model.isAnalyzing {
                        ProgressView()
                            .scaleEffect(0.8)
                    }

                    Spacer()
                }
            }
            .padding()
        }
    }
}

struct TableStatisticsView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            if model.tableStatistics.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "chart.bar.xaxis")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)

                    VStack(spacing: 8) {
                        Text("No Statistics Available")
                            .font(.headline)
                        Text("Run ANALYZE on your tables to generate statistics")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }

                    Button("Analyze All Tables") {
                        Task { await model.analyzeTable(nil) }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isAnalyzing)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding()
            } else {
                List(model.tableStatistics) { stats in
                    TableStatsRowView(stats: stats)
                        .environmentObject(model)
                }
            }
        }
    }
}

struct TableStatsRowView: View {
    @EnvironmentObject var model: AppModel
    let stats: TableStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(stats.table_name)
                    .font(.headline.weight(.medium))
                Spacer()
                Menu {
                    Button("Analyze Table") {
                        Task { await model.analyzeTable(stats.table_name) }
                    }
                    Button("View Columns") {
                        model.selectTableStats(stats.table_name)
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .foregroundStyle(.secondary)
                }
                .disabled(model.isAnalyzing)
            }

            HStack(spacing: 20) {
                StatItemView(label: "Rows", value: "\(stats.row_count)")
                StatItemView(label: "Version", value: "\(stats.stats_version)")
                StatItemView(label: "Analyzed", value: formatDate(stats.analyzed_at))
            }
        }
        .padding(.vertical, 4)
    }

    private func formatDate(_ dateString: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"
        if let date = formatter.date(from: dateString) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateStyle = .short
            displayFormatter.timeStyle = .short
            return displayFormatter.string(from: date)
        }
        return dateString
    }
}

struct ColumnStatisticsView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            if model.columnStatistics.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "list.bullet.circle")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)

                    VStack(spacing: 8) {
                        Text("No Column Statistics")
                            .font(.headline)
                        Text("Analyze your tables to generate column statistics")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding()
            } else {
                List(model.columnStatistics) { stats in
                    ColumnStatsRowView(stats: stats)
                }
            }
        }
    }
}

struct ColumnStatsRowView: View {
    let stats: ColumnStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(stats.table_name).\(stats.column_name)")
                    .font(.headline.weight(.medium))
                Spacer()
            }

            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 8) {
                if let nullCount = stats.null_count {
                    StatItemView(label: "Nulls", value: "\(nullCount)")
                }
                if let distinctCount = stats.distinct_count {
                    StatItemView(label: "Distinct", value: "\(distinctCount)")
                }
                if let min = stats.min {
                    StatItemView(label: "Min", value: min.displayString)
                }
                if let max = stats.max {
                    StatItemView(label: "Max", value: max.displayString)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

struct StatItemView: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.medium))
        }
    }
}

struct StatsCardView: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundStyle(color)
                Spacer()
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.title.weight(.bold))
                    .foregroundStyle(.primary)

                Text(title)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(20)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(color.opacity(0.2), lineWidth: 1)
        )
    }
}
