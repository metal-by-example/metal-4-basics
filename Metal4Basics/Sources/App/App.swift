//
// Copyright 2025 Warren Moore
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

import SwiftUI

struct ContentView: View {
    @State private var renderer: MetalViewDelegate = {
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("This sample requires a device that supports Metal")
        }

        if #available(macOS 26.0, iOS 26.0, visionOS 26.0, *), metalDevice.supportsFamily(.metal3) {
            print("Selected Metal 4 renderer")
            return Metal4Renderer(device: metalDevice)
        } else {
            print("Runtime environment not new enough or Metal 4 not supported; falling back to Metal 2 renderer...")
            return Metal2Renderer(device: metalDevice)
        }
    }()

    var body: some View {
        MetalView(delegate: renderer)
            .ignoresSafeArea()
    }
}

@main
struct Metal4BasicsApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup {
            ContentView()
        }
    }
}

#Preview {
    ContentView()
}
