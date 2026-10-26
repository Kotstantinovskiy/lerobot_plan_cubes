import SwiftUI

private let controlBackground = Color(red: 17 / 255, green: 19 / 255, blue: 21 / 255)
private let controlAccent = Color(red: 214 / 255, green: 243 / 255, blue: 154 / 255)

struct ContentView: View {
    @EnvironmentObject private var connection: ConnectionModel

    var body: some View {
        Group {
            if let endpoint = connection.endpoint {
                GeometryReader { geometry in
                    let landscape = geometry.size.width > geometry.size.height
                    VStack(spacing: 0) {
                        HStack {
                            if !landscape {
                                Text(endpoint.host).foregroundStyle(.secondary).lineLimit(1)
                            }
                            Spacer()
                            Button("Отключиться", action: connection.disconnect)
                                .tint(controlAccent)
                                .frame(minWidth: 112, maxHeight: .infinity)
                        }
                        .font(.system(size: landscape ? 13 : 14))
                        .padding(.horizontal, 16)
                        .frame(height: landscape ? 32 : 44)
                        RobotWebView(endpoint: endpoint, onError: connection.controlFailed)
                            .id(endpoint.id)
                    }
                }
            } else {
                connectionScreen
            }
        }
        .background(controlBackground.ignoresSafeArea())
        .sheet(isPresented: $connection.scannerPresented) {
            NavigationStack {
                QRScanner(onScan: connection.acceptQRCode)
                    .ignoresSafeArea(edges: .bottom)
                    .navigationTitle("QR-код с Mac")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Закрыть") { connection.scannerPresented = false }
                        }
                    }
            }
            .preferredColorScheme(.dark)
        }
    }

    private var connectionScreen: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("SO—101")
                    .font(.system(size: 38, weight: .medium))
                    .padding(.top, 44)
                Text("Подключение к Mac")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                Text("Mac и iPhone должны быть в одной Wi-Fi сети. Рука подключена к Mac по USB.")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Button {
                    connection.scannerPresented = true
                } label: {
                    Label("Сканировать QR", systemImage: "qrcode.viewfinder")
                        .frame(maxWidth: .infinity, minHeight: 52)
                }
                .buttonStyle(.borderedProminent)
                .tint(controlAccent)
                .foregroundStyle(controlBackground)
                .disabled(connection.isConnecting)

                VStack(alignment: .leading, spacing: 12) {
                    Text("Или вставьте ссылку с Mac")
                        .font(.system(size: 15))
                        .foregroundStyle(.secondary)
                    TextField("http://mac.local:8080/?token=…", text: $connection.urlText, axis: .vertical)
                        .font(.system(size: 15))
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .lineLimit(2...4)
                        .padding(16)
                        .background(Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
                        .disabled(connection.isConnecting)
                    HStack {
                        PasteButton(payloadType: String.self) { values in
                            if let text = values.first { connection.urlText = text }
                        }
                        .labelStyle(.titleOnly)
                        .tint(controlAccent)
                        .disabled(connection.isConnecting)
                        Spacer()
                        Button(action: connection.connect) {
                            if connection.isConnecting {
                                ProgressView().tint(controlAccent)
                            } else {
                                Text("Подключиться")
                            }
                        }
                        .buttonStyle(.bordered)
                        .tint(controlAccent)
                        .disabled(connection.isConnecting || connection.urlText.isEmpty)
                    }
                }

                if let message = connection.message {
                    Text(message)
                        .font(.system(size: 15))
                        .foregroundStyle(Color(red: 1, green: 0.62, blue: 0.55))
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityAddTraits(.isStaticText)
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 32)
        }
        .scrollDismissesKeyboard(.interactively)
    }
}
