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

import ModelIO

// Extensions to work around Model I/O -> Swift rough edges

extension MDLVertexDescriptor {
    var vertexAttributes: [MDLVertexAttribute] {
        return attributes as! [MDLVertexAttribute]
    }

    var bufferLayouts: [MDLVertexBufferLayout] {
        return layouts as! [MDLVertexBufferLayout]
    }
}

extension MDLMesh {
    var submeshArray: [MDLSubmesh] {
        return submeshes as! [MDLSubmesh]
    }
}

// Slightly Swiftier child object access
extension MDLAsset {
    struct ChildObjectProxy {
        let asset: MDLAsset
        subscript<T>(componentType: T.Type) -> [T] where T : MDLObject  {
            return asset.childObjects(of: T.self) as? [T] ?? []
        }
    }

    var childObjects: ChildObjectProxy {
        return ChildObjectProxy(asset: self)
    }
}
