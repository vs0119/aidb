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
            } detail: {
                if let sel = model.selection {
                    CollectionDetailView(collection: sel)
                } else {
                    VStack(spacing: 12) {
                        Text("AIDB Mac UI").font(.largeTitle)
                        ConnectionView()
                        if model.healthOK { Text("Server: healthy").foregroundStyle(.green) }
                    }
                    .padding()
                }
            }
            if let message = model.errorMessage {
                ErrorBanner(message: message)
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
    }
}
