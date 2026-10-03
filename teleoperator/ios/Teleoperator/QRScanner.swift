import AVFoundation
import SwiftUI
import UIKit

struct QRScanner: UIViewControllerRepresentable {
    let onScan: (String) -> Void

    func makeUIViewController(context: Context) -> ScannerViewController {
        ScannerViewController(onScan: onScan)
    }

    func updateUIViewController(_ controller: ScannerViewController, context: Context) {}

    static func dismantleUIViewController(_ controller: ScannerViewController, coordinator: ()) {
        controller.stop()
    }
}

final class ScannerViewController: UIViewController {
    private let onScan: (String) -> Void
    private var preview: AVCaptureVideoPreviewLayer?
    private let instruction = UILabel()
    private var capture: QRCodeCapture?

    init(onScan: @escaping (String) -> Void) {
        self.onScan = onScan
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError("Use init(onScan:)") }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        let capture = QRCodeCapture(onScan: { [weak self] text in
            self?.onScan(text)
        }, onUnavailable: { [weak self] in
            self?.instruction.text = "Камера недоступна. Разрешите доступ в настройках или вставьте ссылку с Mac."
        })
        self.capture = capture
        let preview = AVCaptureVideoPreviewLayer(session: capture.session)
        preview.videoGravity = .resizeAspectFill
        view.layer.addSublayer(preview)
        self.preview = preview
        instruction.text = "Наведите камеру на QR-код с Mac"
        instruction.textColor = .white
        instruction.font = .systemFont(ofSize: 17, weight: .medium)
        instruction.textAlignment = .center
        instruction.numberOfLines = 0
        instruction.backgroundColor = UIColor.black.withAlphaComponent(0.65)
        instruction.layer.cornerRadius = 12
        instruction.clipsToBounds = true
        view.addSubview(instruction)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        preview?.frame = view.bounds
        instruction.frame = CGRect(x: 24, y: view.bounds.height - view.safeAreaInsets.bottom - 96,
                                   width: view.bounds.width - 48, height: 68)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        capture?.start()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stop()
    }

    func stop() {
        capture?.stop()
    }
}

// Capture-session state is confined to sessionQueue; scan delivery is on main.
private final class QRCodeCapture: NSObject, AVCaptureMetadataOutputObjectsDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "Teleoperator.QRScanner")
    private var wantsRunning = false
    private var configured = false
    private var didScan = false
    private let onScan: @MainActor @Sendable (String) -> Void
    private let onUnavailable: @MainActor @Sendable () -> Void

    init(onScan: @escaping @MainActor @Sendable (String) -> Void,
         onUnavailable: @escaping @MainActor @Sendable () -> Void) {
        self.onScan = onScan
        self.onUnavailable = onUnavailable
    }

    func start() {
        sessionQueue.async { self.wantsRunning = true }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in
                if allowed { self?.configureAndStart() }
                else { self?.showUnavailable() }
            }
        default:
            showUnavailable()
        }
    }

    func stop() {
        sessionQueue.async {
            self.wantsRunning = false
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configureAndStart() {
        sessionQueue.async {
            guard self.wantsRunning else { return }
            if !self.configured {
                guard let camera = AVCaptureDevice.default(for: .video),
                      let input = try? AVCaptureDeviceInput(device: camera),
                      self.session.canAddInput(input) else {
                    self.showUnavailable()
                    return
                }
                self.session.addInput(input)
                let output = AVCaptureMetadataOutput()
                guard self.session.canAddOutput(output) else {
                    self.showUnavailable()
                    return
                }
                self.session.addOutput(output)
                output.setMetadataObjectsDelegate(self, queue: .main)
                output.metadataObjectTypes = [.qr]
                self.configured = true
            }
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    private func showUnavailable() {
        Task { @MainActor [onUnavailable] in onUnavailable() }
    }

    func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject],
                        from connection: AVCaptureConnection) {
        guard !didScan,
              let code = metadataObjects.first as? AVMetadataMachineReadableCodeObject,
              code.type == .qr, let text = code.stringValue else { return }
        didScan = true
        stop()
        Task { @MainActor [onScan] in onScan(text) }
    }
}
