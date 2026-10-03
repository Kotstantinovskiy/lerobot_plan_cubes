import SwiftUI
import WebKit

struct RobotWebView: UIViewRepresentable {
    let endpoint: PairingEndpoint
    let onError: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(endpoint: endpoint, onError: onError) }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        // Keep the standard Mobile token: Viser's browser check recognizes a WebView by it.
        configuration.applicationNameForUserAgent = "Mobile/15E148 Teleoperator/1.0"
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.isOpaque = false
        view.backgroundColor = UIColor(red: 17 / 255, green: 19 / 255, blue: 21 / 255, alpha: 1)
        view.scrollView.backgroundColor = view.backgroundColor
        view.scrollView.bounces = false
        view.navigationDelegate = context.coordinator
        view.load(URLRequest(url: endpoint.url))
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {}

    static func dismantleUIView(_ view: WKWebView, coordinator: Coordinator) {
        // Replacing the document closes its WebSocket; stopLoading alone does not.
        view.navigationDelegate = nil
        view.stopLoading()
        view.loadHTMLString("<!doctype html><html></html>", baseURL: nil)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        let endpoint: PairingEndpoint
        let onError: (String) -> Void

        init(endpoint: PairingEndpoint, onError: @escaping (String) -> Void) {
            self.endpoint = endpoint
            self.onError = onError
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let url = navigationAction.request.url,
                  navigationAction.targetFrame != nil,
                  endpoint.allowsNavigation(to: url) else {
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            report(error)
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            report(error)
        }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            onError("Экран управления остановлен. Подключитесь снова.")
        }

        private func report(_ error: Error) {
            guard (error as NSError).code != NSURLErrorCancelled else { return }
            onError("Соединение с Mac прервано. Проверьте Wi-Fi и подключитесь снова.")
        }
    }
}
