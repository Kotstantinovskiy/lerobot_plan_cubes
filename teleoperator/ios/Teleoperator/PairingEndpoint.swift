import Foundation

struct PairingEndpoint: Equatable, Identifiable {
    let url: URL
    let host: String
    let port: Int

    var id: String { url.absoluteString }

    var healthURL: URL {
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.path = "/health"
        return components.url!
    }

    init(_ input: String) throws {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.utf8.count <= 4096,
              var components = URLComponents(string: text),
              components.scheme?.lowercased() == "http",
              components.user == nil, components.password == nil,
              components.fragment == nil,
              let host = components.host?.lowercased(),
              ["", "/", "/index.html"].contains(components.percentEncodedPath),
              (1...65535).contains(components.port ?? 80) else {
            throw PairingError.invalidURL
        }
        guard Self.isLocalHost(host) else { throw PairingError.notLocal }
        let tokens = (components.queryItems ?? []).filter { $0.name == "token" }
        guard tokens.count == 1, let token = tokens[0].value,
              (32...256).contains(token.utf8.count),
              token.unicodeScalars.allSatisfy({
                  (65...90).contains($0.value) || (97...122).contains($0.value)
                      || (48...57).contains($0.value) || $0 == "-" || $0 == "_"
              }) else { throw PairingError.invalidToken }
        components.scheme = "http"
        components.host = host
        components.path = "/"
        components.queryItems = [URLQueryItem(name: "token", value: token)]
        guard let url = components.url else { throw PairingError.invalidURL }
        self.url = url
        self.host = host
        self.port = components.port ?? 80
    }

    func allowsNavigation(to url: URL) -> Bool {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return false }
        return components.scheme?.lowercased() == "http"
            && components.host?.lowercased() == host
            && (components.port ?? 80) == port
            && components.user == nil && components.password == nil
            && ["/", "/index.html"].contains(components.percentEncodedPath)
    }

    private static func isLocalHost(_ host: String) -> Bool {
        if host == "localhost" || host == "::1" || host == "[::1]" { return true }
        if host.hasSuffix(".local") {
            return host.split(separator: ".", omittingEmptySubsequences: false).allSatisfy { label in
                !label.isEmpty && label.utf8.count <= 63
                    && label.first != "-" && label.last != "-"
                    && label.utf8.allSatisfy { (97...122).contains($0) || (48...57).contains($0) || $0 == 45 }
            }
        }
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return false }
        var bytes: [Int] = []
        for part in parts {
            guard let value = Int(part), (0...255).contains(value), String(value) == part else { return false }
            bytes.append(value)
        }
        return bytes[0] == 10 || bytes[0] == 127
            || (bytes[0] == 172 && (16...31).contains(bytes[1]))
            || (bytes[0] == 192 && bytes[1] == 168)
            || (bytes[0] == 169 && bytes[1] == 254)
    }
}

enum PairingError: LocalizedError {
    case invalidURL, notLocal, invalidToken, unauthorized, wrongServer

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Вставьте полную ссылку сопряжения с Mac."
        case .notLocal: return "Нужен локальный адрес Mac из той же Wi-Fi сети."
        case .invalidToken: return "В ссылке нет корректного кода сопряжения. Отсканируйте новый QR-код с Mac."
        case .unauthorized: return "Код сопряжения не принят. Возьмите новую ссылку с Mac."
        case .wrongServer: return "По этому адресу нет совместимого сервера Teleoperator."
        }
    }
}
