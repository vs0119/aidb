import SwiftUI

@main
struct AIDBMacApp: App {
    @StateObject private var model = AppModel()
    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(model)
        }
        .windowStyle(.automatic)
    }
}

struct RootView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        ZStack(alignment: .top) {
            NavigationSplitView {
                SidebarView()
                    .navigationSplitViewColumnWidth(min: 220, ideal: 280, max: 400)
            } detail: {
                if let selectedTable = model.selectedTable {
                    TableDataView(table: selectedTable)
                        .environmentObject(model)
                } else if let sel = model.selection {
                    CollectionDetailView(collection: sel)
                } else {
                    DashboardView()
                        .environmentObject(model)
                }
            }
            if let message = model.errorMessage {
                ErrorBanner(message: message, onDismiss: {
                    model.clearError()
                })
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .padding(.top, 8)
            }
            if model.isRefreshing {
                ProgressOverlay()
            }
        }
        .task(id: model.baseURL) {
            await model.refreshAll()
            model.startAutoRefresh()
        }
        .onDisappear {
            model.stopAutoRefresh()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            // Force window focus when app becomes active
            DispatchQueue.main.async {
                if let window = NSApp.keyWindow ?? NSApp.windows.first {
                    window.makeKeyAndOrderFront(nil)
                    window.orderFrontRegardless()
                }
            }
        }
    }
}
