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
import Metal
import MetalKit
import Spatial

class Entity {
    var name = "UnnamedEntity"

    var transform: AffineTransform3D = .init()
    var children: [Entity] = []
    weak var parent: Entity?

    var worldTransform: AffineTransform3D {
        guard let parent else {
            return transform
        }
        return parent.worldTransform * transform
    }

    private func removeChild(_ child: Entity) {
        children.removeAll { $0 === child }
        child.parent = nil
    }

    func removeFromParent() {
        parent?.removeChild(self)
    }

    func addChild(_ child: Entity) {
        child.removeFromParent()
        children.append(child)
        child.parent = self
    }

    func visitHierarchy(_ visitor: (Entity) -> Void) {
        visitor(self)
        for child in children {
            child.visitHierarchy(visitor)
        }
    }

    func childNamed(_ name: String, recursively: Bool = true) -> Entity? {
        for child in children {
            if child.name == name {
                return child
            } else if recursively, let foundChild = child.childNamed(name, recursively: recursively) {
                return foundChild
            }
        }
        return nil
    }
}

class ModelEntity : Entity {
    var mesh: Mesh?
}
