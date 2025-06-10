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
import Metal
import MetalKit

protocol MetalViewDelegate : MTKViewDelegate {
    @MainActor func configure(_ view: MTKView)
}

#if !os(macOS)
struct MetalView : UIViewRepresentable {
    public typealias UIViewType = MTKView
    public var delegate: MetalViewDelegate?

    public init(delegate: MetalViewDelegate) {
        self.delegate = delegate
    }

    public func makeUIView(context: Context) -> MTKView {
        return MTKView()
    }

    public func updateUIView(_ view: MTKView, context: Context) {
        delegate?.configure(view)
    }
}
#else
struct MetalView : NSViewRepresentable {
    public typealias NSViewType = MTKView
    public var delegate: MetalViewDelegate?

    public init(delegate: MetalViewDelegate) {
        self.delegate = delegate
    }

    public func makeNSView(context: Context) -> MTKView {
        return MTKView()
    }

    public func updateNSView(_ view: MTKView, context: Context) {
        delegate?.configure(view)
    }
}
#endif
