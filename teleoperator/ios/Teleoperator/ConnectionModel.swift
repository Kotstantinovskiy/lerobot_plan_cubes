import Foundation
import SwiftUI

@MainActor
final class ConnectionModel: ObservableObject {
    @Published var urlText = ""
    @Published var endpoint: PairingEndpoint?
    @Published var message: String?
    @Published var isConnecting = false
    @Published var scannerPresented = false
    private var connectionTask: Task<Void, Never>?
    private var connectionID = UUID()

    func connect() {
        guard !isConnecting else { return }
        let candidate: PairingEndpoint
        do {
            candidate = try PairingEndpoint(urlText)
        } catch {
            message = error.localizedDescription
            return
        }
        urlText = candidate.url.absoluteString
        message = nil
        isConnecting = true
        let attempt = UUID()
        connectionID = attempt
        connectionTask = Task { [weak self] in
            let configuration = URLSessionConfiguration.ephemeral
            configuration.waitsForConnectivity = true
            configuration.timeoutIntervalForRequest = 10
            configuration.timeoutIntervalForResource = 15
            let session = URLSession(configuration: configuration, delegate: RejectRedirects(), delegateQueue: nil)
            defer { session.invalidateAndCancel() }
            do {
                let (data, response) = try await session.data(from: candidate.healthURL)
                try Task.checkCancellation()
                guard let http = response as? HTTPURLResponse else { throw PairingError.wrongServer }
                if http.statusCode == 401 || http.statusCode == 403 { throw PairingError.unauthorized }
                guard http.statusCode == 200 else { throw PairingError.wrongServer }
                let health = try JSONDecoder().decode(Health.self, from: data)
                guard health.name == "Teleoperator", health.version == 1 else { throw PairingError.wrongServer }
                if self?.connectionID == attempt {
                    self?.endpoint = candidate
                    self?.message = nil
                }
            } catch is CancellationError {
                // An explicit disconnect owns the resulting screen state.
            } catch {
                if !Task.isCancelled, self?.connectionID == attempt {
                    self?.message = (error as? PairingError)?.localizedDescription
                        ?? "Не удалось подключиться. Проверьте Wi-Fi, адрес Mac и разрешение локальной сети."
                }
            }
            if self?.connectionID == attempt {
                self?.isConnecting = false
                self?.connectionTask = nil
            }
        }
    }

    func acceptQRCode(_ text: String) {
        scannerPresented = false
        urlText = text
        connect()
    }

    func disconnect() {
        connectionID = UUID()
        connectionTask?.cancel()
        connectionTask = nil
        isConnecting = false
        endpoint = nil
        urlText = ""
        message = nil
    }

    func suspendControl() {
        guard endpoint != nil || isConnecting else { return }
        connectionID = UUID()
        connectionTask?.cancel()
        connectionTask = nil
        isConnecting = false
        endpoint = nil
        message = "Управление приостановлено. Подключитесь снова, чтобы продолжить."
    }

    func enterBackground() {
        scannerPresented = false
        suspendControl()
    }

    func controlFailed(_ message: String) {
        endpoint = nil
        self.message = message
    }

    private struct Health: Decodable {
        let name: String
        let version: Int
    }
}

private final class RejectRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        // A health check must never follow a redirect to another machine.
        completionHandler(nil)
    }
}
