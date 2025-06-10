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
import MetalKit
import ModelIO
import Spatial

class Scene {
    let rootEntity = Entity()
    var lights: [Light] = []
    var lightingEnvironment: ImageBasedLight?

    var resources = [any MTLResource]()
}

fileprivate class AssetResourceContext {
    enum TextureSemantic {
        case raw
        case color
    }

    let device: MTLDevice
    let textureLoader: MTKTextureLoader
    let dataTextureOptions: [MTKTextureLoader.Option : Any]
    let colorTextureOptions: [MTKTextureLoader.Option : Any]
    var textureCache = [ObjectIdentifier : MTLTexture]()
    var materialCache = [ObjectIdentifier : Material]()
    var resources: [any MTLResource] = []

    init(device: MTLDevice) {
        self.device = device
        self.textureLoader = MTKTextureLoader(device: device)
        dataTextureOptions = [
            .SRGB : NSNumber(value: false),
            .generateMipmaps : NSNumber(value: true),
            .textureStorageMode : NSNumber(value: MTLStorageMode.private.rawValue)
        ]
        colorTextureOptions = [
            .SRGB : NSNumber(value: true),
            .generateMipmaps : NSNumber(value: true),
            .textureStorageMode : NSNumber(value: MTLStorageMode.private.rawValue)
        ]
    }

    func convert(texture mdlTexture: MDLTexture, semantic: TextureSemantic) throws -> MTLTexture? {
        if let existingTexture = textureCache[ObjectIdentifier(mdlTexture)] {
            return existingTexture
        }
        let options = semantic == .raw ? dataTextureOptions : colorTextureOptions
        var texture = try textureLoader.newTexture(texture: mdlTexture, options: options)
        if semantic == .color && texture.pixelFormat != .rgba8Unorm_srgb {
            // MTKTextureLoader has been broken since its introduction, sometimes ignoring the
            // SRGB option, so we make our own view to reinterpret the image contents correctly.
            texture = texture.makeTextureView(pixelFormat: .rgba8Unorm_srgb)!
        }
        resources.append(texture)
        textureCache[ObjectIdentifier(mdlTexture)] = texture
        return texture
    }

    func convert(material mdlMaterial: MDLMaterial) throws -> Material {
        let material = Material()
        material.name = mdlMaterial.name
        if let baseColor = mdlMaterial.property(with: MDLMaterialSemantic.baseColor) {
            if baseColor.type == .texture, let sampler = baseColor.textureSamplerValue {
                material.baseColor.texture = try convert(texture: sampler.texture!, semantic: .color)
                material.baseColor.texture?.label = material.name + " Base Color"
                material.baseColor.baseColorFactor = [1, 1, 1]
            } else if baseColor.type == .float3 {
                material.baseColor.baseColorFactor = baseColor.float3Value
            }
        }
        if let roughness = mdlMaterial.property(with: MDLMaterialSemantic.roughness) {
            if roughness.type == .texture, let sampler = roughness.textureSamplerValue {
                material.roughness.texture = try convert(texture: sampler.texture!, semantic: .raw)
                material.roughness.texture?.label = material.name + " Metalness/Roughness"
            } else if roughness.type == .float {
                material.roughness.roughnessFactor = roughness.floatValue
            }
        }
        if let metalness = mdlMaterial.property(with: MDLMaterialSemantic.metallic) {
            if metalness.type == .texture, let sampler = metalness.textureSamplerValue {
                material.metalness.texture = try convert(texture: sampler.texture!, semantic: .raw)
                material.metalness.texture?.label = material.name + " Metalness/Roughness"
                material.metalness.metalnessFactor = 1.0
            } else if metalness.type == .float {
                material.metalness.metalnessFactor = metalness.floatValue
            }
        }
        if let normal = mdlMaterial.property(with: MDLMaterialSemantic.tangentSpaceNormal) {
            if normal.type == .texture, let sampler = normal.textureSamplerValue {
                material.normal.texture = try convert(texture: sampler.texture!, semantic: .raw)
                material.normal.texture?.label = material.name + " Normal"
            }
        }
        if let emissive = mdlMaterial.property(with: MDLMaterialSemantic.emission) {
            if emissive.type == .texture, let sampler = emissive.textureSamplerValue {
                material.emissive.texture = try convert(texture: sampler.texture!, semantic: .color)
                material.emissive.texture?.label = material.name + " Emissive"
                material.emissive.emissiveFactor = [1, 1, 1]
            } else if emissive.type == .float3 {
                material.emissive.emissiveFactor = emissive.float3Value
            }
        }
        if let occlusion = mdlMaterial.property(with: MDLMaterialSemantic.ambientOcclusion) {
            if occlusion.type == .texture, let sampler = occlusion.textureSamplerValue {
                material.occlusion.texture = try convert(texture: sampler.texture!, semantic: .raw)
                material.occlusion.texture?.label = material.name + " Occlusion"
            }
        }
        if let occlusionStrength = mdlMaterial.property(with: MDLMaterialSemantic.ambientOcclusionScale) {
            if occlusionStrength.type == .float {
                material.occlusion.occlusionStrength = occlusionStrength.floatValue
            }
        }
        if let opacity = mdlMaterial.property(with: MDLMaterialSemantic.opacity) {
            if opacity.type == .texture, let sampler = opacity.textureSamplerValue {
                material.opacity.texture = try convert(texture: sampler.texture!, semantic: .raw)
                material.opacity.texture?.label = material.name + " Opacity"
                material.alphaMode = .blend
            }
        }
        return material
    }

    func convert(mesh mdlMesh: MDLMesh) throws -> Mesh {
        let mtkMesh = try MTKMesh(mesh: mdlMesh, device: device)
        let vertexBuffers: [BufferView] = mtkMesh.vertexBuffers.map {
            $0.buffer.label = mdlMesh.name + " Vertex Attributes"
            resources.append($0.buffer)
            return BufferView(buffer: $0.buffer, offset: $0.offset, length: $0.length)
        }
        let submeshes = mtkMesh.submeshes.enumerated().map { submeshIndex, submesh in
            let indexBuffer = BufferView(buffer: submesh.indexBuffer.buffer,
                                         offset: submesh.indexBuffer.offset,
                                         length: submesh.indexBuffer.length)
            submesh.indexBuffer.buffer.label = mdlMesh.name + " Indices"
            resources.append(submesh.indexBuffer.buffer)
            return Submesh(primitiveType: submesh.primitiveType,
                           indexBuffer: indexBuffer,
                           indexType: submesh.indexType,
                           indexCount: submesh.indexCount,
                           materialIndex: submeshIndex)
        }
        let defaultMaterial = Material()
        defaultMaterial.baseColor.baseColorFactor = [0.18, 0.18, 0.18]
        defaultMaterial.metalness.metalnessFactor = 0.0
        defaultMaterial.roughness.roughnessFactor = 0.8
        let materials = try mdlMesh.submeshArray.map {
            if let mdlMaterial = $0.material {
                return try convert(material: mdlMaterial)
            } else {
                return defaultMaterial
            }
        }
        let mesh = Mesh(vertexBuffers: vertexBuffers,
                        vertexCount: mtkMesh.vertexCount,
                        vertexDescriptor: mdlMesh.vertexDescriptor,
                        submeshes: submeshes,
                        materials: materials)
        mesh.name = mdlMesh.name
        return mesh
    }
}

extension Scene {
    static func load(url: URL, device: MTLDevice) throws -> Scene {
        let scene = Scene()

        let bufferAllocator = MTKMeshBufferAllocator(device: device)
        let asset = MDLAsset(url: url, vertexDescriptor: nil, bufferAllocator: bufferAllocator)
        asset.loadTextures()

        let resourceContext = AssetResourceContext(device: device)

        var entityMap = [ObjectIdentifier : Entity]()
        for object in asset.childObjects[MDLObject.self] {
            var entity: Entity!
            if let mdlMesh = object as? MDLMesh {
                let modelEntity = ModelEntity()

                // We need our attributes to be a certain order, but we also want to be flexible
                // with respect to multiple texture coordinate sets, so we interleave and reorganize
                // attribute data _after_ loading, in response to what the mesh actually contains.
                let hasMultipleUVs = mdlMesh.vertexDescriptor.vertexAttributes.count(where: {
                    $0.name == MDLVertexAttributeTextureCoordinate
                }) > 1
                let vertexDescriptor = MDLVertexDescriptor()
                vertexDescriptor.vertexAttributes[0].name = MDLVertexAttributePosition
                vertexDescriptor.vertexAttributes[0].format = .float4
                vertexDescriptor.vertexAttributes[0].offset = 0
                vertexDescriptor.vertexAttributes[0].bufferIndex = 0
                vertexDescriptor.vertexAttributes[1].name = MDLVertexAttributeNormal
                vertexDescriptor.vertexAttributes[1].format = .float3
                vertexDescriptor.vertexAttributes[1].offset = 16
                vertexDescriptor.vertexAttributes[1].bufferIndex = 0
                vertexDescriptor.vertexAttributes[2].name = MDLVertexAttributeTangent
                vertexDescriptor.vertexAttributes[2].format = .float4
                vertexDescriptor.vertexAttributes[2].offset = 32
                vertexDescriptor.vertexAttributes[2].bufferIndex = 0
                vertexDescriptor.vertexAttributes[3].name = MDLVertexAttributeTextureCoordinate
                vertexDescriptor.vertexAttributes[3].format = .float2
                vertexDescriptor.vertexAttributes[3].offset = 48
                vertexDescriptor.vertexAttributes[3].bufferIndex = 0
                if hasMultipleUVs {
                    vertexDescriptor.vertexAttributes[4].name = MDLVertexAttributeTextureCoordinate
                    vertexDescriptor.vertexAttributes[4].format = .float2
                    vertexDescriptor.vertexAttributes[4].offset = 56
                    vertexDescriptor.vertexAttributes[4].bufferIndex = 0
                }
                vertexDescriptor.bufferLayouts[0].stride = hasMultipleUVs ? 64 : 56
                mdlMesh.vertexDescriptor = vertexDescriptor

                mdlMesh.addOrthTanBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                        normalAttributeNamed: MDLVertexAttributeNormal,
                                        tangentAttributeNamed: MDLVertexAttributeTangent)
                modelEntity.mesh = try resourceContext.convert(mesh: mdlMesh)
                entity = modelEntity
            } else /* TODO: Lights and cameras */ {
                entity = Entity()
            }
            entity.name = object.name
            if let objectTransform = object.transform,
                let modelMatrix = objectTransform.localTransform?(atTime: 0.0)
            {
                entity.transform = AffineTransform3D(truncating: modelMatrix)
            }
            entityMap[ObjectIdentifier(object)] = entity

            if let parentObject = object.parent {
                if let parentEntity = entityMap[ObjectIdentifier(parentObject)] {
                    parentEntity.addChild(entity)
                } else {
                    print("Logic error: Could not look up parent entity for MDLObject even though it had a parent")
                }
            } else {
                scene.rootEntity.addChild(entity)
            }
        }
        scene.resources = resourceContext.resources
        return scene
    }
}
