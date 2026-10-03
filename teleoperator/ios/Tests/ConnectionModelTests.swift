import Foundation

@main
struct ConnectionModelTests {
    @MainActor
    static func main() async throws {
        guard CommandLine.arguments.count == 2, let port = Int(CommandLine.arguments[1]) else {
            fatalError("Pass the local test-server port")
        }
        let model = ConnectionModel()
        func link(_ prefix: String) -> String {
            "http://127.0.0.1:\(port)/?token=\(prefix)\(String(repeating: "a", count: 64))"
        }
        func settle() async throws {
            for _ in 0..<200 {
                if !model.isConnecting { return }
                try await Task.sleep(nanoseconds: 10_000_000)
            }
            fatalError("Connection test timed out")
        }

        model.urlText = link("good")
        model.connect()
        try await settle()
        precondition(model.endpoint != nil && model.message == nil, "Good health response was rejected")
        model.suspendControl()
        precondition(model.endpoint == nil && !model.urlText.isEmpty, "Suspension must require an explicit reconnect")

        model.urlText = link("bad")
        model.connect()
        try await settle()
        precondition(model.endpoint == nil && model.message?.contains("Код сопряжения") == true, "Unauthorized pairing was accepted")

        model.urlText = link("redirect")
        model.connect()
        try await settle()
        precondition(model.endpoint == nil && model.message?.contains("сервера Teleoperator") == true, "Health redirect was accepted")

        model.urlText = link("slow")
        model.connect()
        try await Task.sleep(nanoseconds: 20_000_000)
        model.suspendControl()
        precondition(!model.isConnecting && model.endpoint == nil && model.urlText == link("slow"), "Pending suspension did not retain an explicit retry")
        try await Task.sleep(nanoseconds: 350_000_000)
        precondition(model.endpoint == nil && !model.isConnecting, "A delayed health response connected while inactive")
        model.connect()
        try await settle()
        precondition(model.endpoint != nil, "Explicit retry after suspension failed")
        model.disconnect()

        model.urlText = link("slow")
        model.connect()
        try await Task.sleep(nanoseconds: 20_000_000)
        model.disconnect()
        model.urlText = link("good")
        model.connect()
        try await settle()
        precondition(model.endpoint?.url.absoluteString == link("good"), "Cancelled attempt replaced the newer connection")
        try await Task.sleep(nanoseconds: 300_000_000)
        precondition(model.endpoint?.url.absoluteString == link("good") && !model.isConnecting, "Old cleanup overwrote the new connection")
        model.disconnect()
        precondition(model.endpoint == nil && model.urlText.isEmpty && !model.isConnecting, "Disconnect retained a pairing token")
        print("Connection model checks passed: success, unauthorized, redirect, active/pending suspension, cancellation, disconnect")
    }
}
