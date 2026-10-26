#!/bin/sh
# Check portable logic and source syntax; full app builds require Xcode.
set -eu

ios_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
check_dir=$(mktemp -d "${TMPDIR:-/tmp}/so101-ios-check.XXXXXX")
trap 'rm -rf "$check_dir"' EXIT HUP INT TERM

plutil -lint "$ios_dir/SO101Control/Info.plist" "$ios_dir/SO101Control.xcodeproj/project.pbxproj"
swiftc -module-cache-path "$check_dir/modules" \
    "$ios_dir/SO101Control/PairingEndpoint.swift" \
    "$ios_dir/Tests/PairingEndpointTests.swift" -o "$check_dir/pairing-tests"
"$check_dir/pairing-tests"
swiftc -swift-version 5 -module-cache-path "$check_dir/modules" \
    "$ios_dir/SO101Control/PairingEndpoint.swift" \
    "$ios_dir/SO101Control/ConnectionModel.swift" \
    "$ios_dir/Tests/ConnectionModelTests.swift" -o "$check_dir/connection-tests"
python3 "$ios_dir/Tests/check_connection_model.py" "$check_dir/connection-tests"
swiftc -frontend -parse "$ios_dir"/SO101Control/*.swift

if xcodebuild -version >/dev/null 2>&1; then
    xcodebuild -project "$ios_dir/SO101Control.xcodeproj" -target SO101Control \
        -sdk iphoneos -configuration Debug \
        SYMROOT="$check_dir/Products" OBJROOT="$check_dir/Intermediates" \
        CODE_SIGNING_ALLOWED=NO build
else
    printf '%s\n' "Source checks passed. Full iOS build skipped: select a full Xcode installation with the iOS SDK."
fi
