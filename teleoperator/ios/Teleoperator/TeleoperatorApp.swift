import SwiftUI

@main
struct TeleoperatorApp: App {
    @StateObject private var connection = ConnectionModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(connection)
                .preferredColorScheme(.dark)
                .onChange(of: scenePhase) { phase in
                    if phase != .active { connection.suspendControl() }
                    if phase == .background { connection.enterBackground() }
                }
        }
    }
}
