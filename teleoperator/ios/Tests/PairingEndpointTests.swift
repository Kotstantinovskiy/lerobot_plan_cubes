import Foundation

@main
struct PairingEndpointTests {
    static func main() throws {
        var checks = 0
        let token = String(repeating: "a", count: 64)
        func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
            checks += 1
            precondition(condition(), message)
        }
        func reject(_ text: String) {
            checks += 1
            do {
                _ = try PairingEndpoint(text)
                preconditionFailure("Accepted invalid pairing URL")
            } catch { }
        }

        for host in ["10.0.0.4", "172.16.0.1", "172.31.255.254", "192.168.1.6", "169.254.4.5", "127.0.0.1", "localhost", "robot-mac.local"] {
            let endpoint = try PairingEndpoint("http://\(host):8080/?token=\(token)")
            expect(endpoint.host == host, "Allowed local host changed")
            expect(endpoint.port == 8080, "Pairing port changed")
            expect(endpoint.healthURL.path == "/health", "Health path is wrong")
            expect(endpoint.healthURL.query == "token=\(token)", "Health token is missing")
        }
        let clean = try PairingEndpoint(" http://ROBOT.local:8083/index.html?websocket=ws://evil.example&token=\(token)&other=value \n")
        let ipv6 = try PairingEndpoint("http://[::1]:8080/?token=\(token)")
        expect(ipv6.url.absoluteString == "http://[::1]:8080/?token=\(token)", "IPv6 loopback changed")
        expect(clean.url.absoluteString == "http://robot.local:8083/?token=\(token)", "Pairing must strip unrelated query parameters")
        expect(clean.allowsNavigation(to: URL(string: "http://robot.local:8083/?token=\(token)&darkMode")!), "Same-origin redirect blocked")
        expect(!clean.allowsNavigation(to: URL(string: "http://other.local:8083/?token=\(token)")!), "Cross-host navigation allowed")
        expect(!clean.allowsNavigation(to: URL(string: "http://robot.local:8084/?token=\(token)")!), "Cross-port navigation allowed")
        expect(!clean.allowsNavigation(to: URL(string: "https://robot.local:8083/?token=\(token)")!), "Cross-scheme navigation allowed")
        expect(!clean.allowsNavigation(to: URL(string: "http://robot.local:8083/files")!), "Unexpected path allowed")
        expect(!clean.allowsNavigation(to: URL(string: "http://user@robot.local:8083/")!), "Credential URL allowed")

        for host in ["example.com", "8.8.8.8", "172.15.0.1", "172.32.0.1", "192.169.1.1", "0.0.0.0", "127.1", "2130706433", "192.168.001.1", "-mac.local", "mac..local", "mac.local.evil.example"] {
            reject("http://\(host):8080/?token=\(token)")
        }
        for value in [
            "https://robot.local/?token=\(token)",
            "file:///tmp/controller?token=\(token)",
            "http://user:secret@robot.local/?token=\(token)",
            "http://robot.local:0/?token=\(token)",
            "http://robot.local:65536/?token=\(token)",
            "http://robot.local/../?token=\(token)",
            "http://robot.local/%69ndex.html?token=\(token)",
            "http://robot.local/?token=\(token)#fragment",
            "http://robot.local/",
            "http://robot.local/?token=short",
            "http://robot.local/?token=\(token)&token=\(token)",
            "http://robot.local/?token=\(token)%2F",
        ] { reject(value) }
        print("Pairing endpoint checks passed: \(checks)")
    }
}
