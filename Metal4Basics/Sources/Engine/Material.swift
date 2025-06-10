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

enum AlphaMode: Int {
    case opaque
    case mask
    case blend
};

class Material {
    var name: String = "UnnamedMaterial"

    struct BaseColor {
        var texture: MTLTexture?
        var baseColorFactor: SIMD3<Float> = [0, 0, 0]
        var mappingChannel: Int = 0
    }

    struct Opacity {
        var texture: MTLTexture?
        var opacityFactor: Float = 1.0
        var mappingChannel: Int = 0
    }

    struct Metalness {
        var texture: MTLTexture?
        var metalnessFactor: Float = 0.0
        var mappingChannel: Int = 0
    }

    struct Roughness {
        var texture: MTLTexture?
        var roughnessFactor: Float = 1.0
        var mappingChannel: Int = 0
    }

    struct Emissive {
        var texture: MTLTexture?
        var emissiveFactor: SIMD3<Float> = [0, 0, 0]
        var mappingChannel: Int = 0
    }

    struct Normal {
        var texture: MTLTexture?
        var normalScale: Float = 1.0
        var mappingChannel: Int = 0
    }

    struct Occlusion {
        var texture: MTLTexture?
        var occlusionStrength: Float = 1.0
        var mappingChannel: Int = 0
    }

    var baseColor = BaseColor()
    var opacity = Opacity()
    var metalness = Metalness()
    var roughness = Roughness()
    var emissive = Emissive()
    var normal = Normal()
    var occlusion = Occlusion()

    var alphaMode = AlphaMode.opaque
    var alphaThreshold: Float = 0.5 // Only used if alphaMode != .opaque

    var relativeSortOrder: Int {
        switch alphaMode {
        case .opaque:
            -1
        case .mask:
            0
        case .blend:
            1
        }
    }

    var renderPipelineState: MTLRenderPipelineState?
    // A view onto a material argument buffer that binds this material's constants and textures
    var bufferView: BufferView?
}
