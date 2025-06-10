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

import Metal

func alignUp(_ n: Int, _ alignment: Int) -> Int {
    return ((n + alignment - 1) / alignment) * alignment
}

extension MTLDevice {
    var minimumConstantBufferAlignment: Int {
#if targetEnvironment(simulator)
        let isSimulator = true
#else
        let isSimulator = false
#endif
        if supportsFamily(.apple2) && !isSimulator {
            return 4
        } else if supportsFamily(.mac2) {
            return 32
        } else {
            return 256
        }
    }
}

struct BufferView {
    let buffer: MTLBuffer
    let offset: Int
    let length: Int

    init(buffer: MTLBuffer, offset: Int = 0, length: Int? = nil) {
        self.buffer = buffer
        self.offset = offset
        if let length {
            self.length = length
        } else {
            self.length = max(0, buffer.length - offset)
        }
    }

    var gpuAddress: UInt64 {
        return buffer.gpuAddress + UInt64(offset)
    }
}

class RingBuffer {
    let buffer: MTLBuffer
    private let minimumAlignment: Int
    private var nextOffset: Int = 0

    init(length: Int, device: MTLDevice) {
        buffer = device.makeBuffer(length: length, options: [.storageModeShared])!
        minimumAlignment = device.minimumConstantBufferAlignment
    }

    func reset() {
        nextOffset = 0
    }

    func allocate(length: Int, alignment: Int) -> BufferView {
        assert(length <= buffer.length)
        let effectiveAlignment = lcm(alignment, minimumAlignment)
        var offset = alignUp(nextOffset, effectiveAlignment)
        if offset + length >= buffer.length {
            offset = 0
        }
        nextOffset = offset + length
        return BufferView(buffer: buffer, offset: offset, length: length)
    }

    func copy<T>(_ value: T) -> BufferView {
        assert(_isPOD(T.self))
        let view = allocate(length: MemoryLayout<T>.size, alignment: MemoryLayout<T>.alignment)
        let store = view.buffer.contents().advanced(by: view.offset).bindMemory(to: T.self, capacity: 1)
        store.pointee = value
        return view
    }


    func copy<T>(_ elements: [T]) -> BufferView {
        assert(_isPOD(T.self))
        let view = allocate(length: MemoryLayout<T>.stride * elements.count, alignment: MemoryLayout<T>.alignment)
        let store = view.buffer.contents().advanced(by: view.offset).bindMemory(to: T.self, capacity: elements.count)
        elements.withUnsafeBufferPointer { ptr in
            store.update(from: ptr.baseAddress!, count: elements.count)
        }
        return view
    }
}
