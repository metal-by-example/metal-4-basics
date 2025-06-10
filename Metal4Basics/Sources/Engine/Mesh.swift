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
import ModelIO

class Submesh {
    let indexBuffer: BufferView
    let indexType: MTLIndexType
    let indexCount: Int
    let primitiveType: MTLPrimitiveType
    var materialIndex: Int

    init(primitiveType: MTLPrimitiveType,
         indexBuffer: BufferView,
         indexType: MTLIndexType,
         indexCount: Int,
         materialIndex: Int)
    {
        self.primitiveType = primitiveType
        self.indexBuffer = indexBuffer
        self.indexType = indexType
        self.indexCount = indexCount
        self.materialIndex = materialIndex
    }
}

class Mesh {
    var name = "UnnamedMesh"
    let vertexBuffers: [BufferView]
    let vertexCount: Int
    let vertexDescriptor: MDLVertexDescriptor
    let submeshes: [Submesh]
    var materials: [Material]

    init(vertexBuffers: [BufferView],
         vertexCount: Int,
         vertexDescriptor: MDLVertexDescriptor,
         submeshes: [Submesh],
         materials: [Material])
    {
        self.vertexBuffers = vertexBuffers
        self.vertexCount = vertexCount
        self.vertexDescriptor = vertexDescriptor
        self.submeshes = submeshes
        self.materials = materials
    }
}
