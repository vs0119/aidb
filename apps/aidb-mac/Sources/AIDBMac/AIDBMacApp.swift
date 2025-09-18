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
    }
}
