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
import Spatial

@available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
extension MTL4LibraryFunctionDescriptor {
    convenience init(name: String, library: MTLLibrary) {
        self.init()
        self.name = name
        self.library = library
    }
}

@available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
extension MTL4SpecializedFunctionDescriptor {
    convenience init(name: String, library: MTLLibrary, functionConstants: MTLFunctionConstantValues? = nil) {
        self.init()
        self.functionDescriptor = MTL4LibraryFunctionDescriptor(name: name, library: library)
        self.constantValues = functionConstants
    }
}

@available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
class Metal4Renderer : NSObject, MetalViewDelegate {
    var scene = Scene()
    let camera = PerspectiveCamera()
    
    let device: MTLDevice
    let maxFramesInFlight: UInt64 = 3
    
    let colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
    let depthStencilPixelFormat = MTLPixelFormat.depth32Float
    let rasterSampleCount: Int
    
    private let commandQueue: MTL4CommandQueue
    private let commandBuffer: MTL4CommandBuffer
    private let commandAllocators: [MTL4CommandAllocator]
    private let library: MTLLibrary
    private let compiler: MTL4Compiler
    private let depthStencilStates: [AlphaMode : MTLDepthStencilState]
    private let constantsBuffers: [RingBuffer]
    private let materialsBuffer: RingBuffer
    private let residencySet: MTLResidencySet
    private let frameCompletionEvent: MTLSharedEvent
    private let vertexArgumentTable: MTL4ArgumentTable
    private let fragmentArgumentTable: MTL4ArgumentTable
    
    private var frameIndex: UInt64 = 0
    private var hasPreparedResources = false
    private var lastRenderTime: TimeInterval = 0
    private var time: TimeInterval = 0

    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeMTL4CommandQueue()!
        self.commandBuffer = device.makeCommandBuffer()!
        self.commandAllocators = (0..<maxFramesInFlight).map { _ in
            device.makeCommandAllocator()!
        }
        
        let supportedSampleCounts = [8, 5, 4, 2, 1].filter { device.supportsTextureSampleCount($0) }
        rasterSampleCount = supportedSampleCounts[0]
        
        self.frameCompletionEvent = device.makeSharedEvent()!
        
        self.library = device.makeDefaultLibrary()!
        
        let compilerDescriptor = MTL4CompilerDescriptor()
        self.compiler = try! device.makeCompiler(descriptor: compilerDescriptor)
        
        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.initialCapacity = 64
        residencySet = try! device.makeResidencySet(descriptor: residencyDescriptor)
        
        let argumentDescriptor = MTL4ArgumentTableDescriptor()
        argumentDescriptor.maxBufferBindCount = max(VertexBufferCount, FragmentBufferCount)
        argumentDescriptor.maxTextureBindCount = FragmentTextureCount
        vertexArgumentTable = try! device.makeArgumentTable(descriptor: argumentDescriptor)
        fragmentArgumentTable = try! device.makeArgumentTable(descriptor: argumentDescriptor)
        
        constantsBuffers = (0..<maxFramesInFlight).map { _ in
            RingBuffer(length: 256 * 1_024, device: device)
        }

        materialsBuffer = RingBuffer(length: 64 * 1024, device: device)

        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.isDepthWriteEnabled = true
        depthStencilDescriptor.depthCompareFunction = .less
        let opaqueDepthState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
        depthStencilDescriptor.isDepthWriteEnabled = false
        let blendDepthState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
        depthStencilStates = [
            .opaque: opaqueDepthState,
            .mask: opaqueDepthState,
            .blend: blendDepthState
        ]
        
        super.init()
        
        frameCompletionEvent.signaledValue = frameIndex
        
        Task {
            do {
                try await self.makeResources()
            } catch {
                fatalError("Failed to initialize resources or render pipeline: \(error)")
            }
        }
    }
    
    private func makeResources() async throws {
        let assetURL = Bundle.main.url(forResource: "Island", withExtension: "usdz")!
        let scene = try Scene.load(url: assetURL, device: device)
        
        let environmentURL = Bundle.main.url(forResource: "Beach", withExtension: "hdr")!
        scene.lightingEnvironment = try await ImageBasedLight.generateImageBasedLight(url: environmentURL)
        scene.lightingEnvironment?.intensity = 3.0

        scene.lights = [
            Light() // Dummy light. Assume we're using IBL.
        ]
        
        self.scene = scene

        var modelEntities: [ModelEntity] = []
        scene.rootEntity.visitHierarchy { entity in
            if let modelEntity = entity as? ModelEntity {
                modelEntities.append(modelEntity)
            }
        }
        for renderableEntity in modelEntities {
            for material in renderableEntity.mesh!.materials {
                if material.alphaMode == .blend {
                    // The alpha-blended materials in our scene are mostly foliage, so switch to
                    // alpha testing to avoid depth artifacts while still allowing transparency.
                    material.alphaMode = .mask
                }

                let renderPipelineState = try! makePipelineState(mesh: renderableEntity.mesh!,
                                                                 material: material)
                material.renderPipelineState = renderPipelineState

                let nullResource = MTLResourceID()
                var materialConstants = MaterialConstants()
                materialConstants.baseColorFactor = SIMD4<Float>(material.baseColor.baseColorFactor, 1.0)
                materialConstants.opacityFactor = material.opacity.opacityFactor
                materialConstants.normalScale = material.normal.normalScale
                materialConstants.occlusionStrength = material.occlusion.occlusionStrength
                materialConstants.metallicFactor = material.metalness.metalnessFactor
                materialConstants.roughnessFactor = material.roughness.roughnessFactor
                materialConstants.emissiveFactor = material.emissive.emissiveFactor
                materialConstants.alphaCutoff = material.alphaThreshold
                var materialArguments = MaterialArguments()
                materialArguments.constants = materialConstants
                materialArguments.baseColorTexture = material.baseColor.texture?.gpuResourceID ?? nullResource
                materialArguments.normalTexture = material.normal.texture?.gpuResourceID ?? nullResource
                materialArguments.emissiveTexture = material.emissive.texture?.gpuResourceID ?? nullResource
                materialArguments.occlusionTexture = material.occlusion.texture?.gpuResourceID ?? nullResource
                materialArguments.metalnessTexture = material.metalness.texture?.gpuResourceID ?? nullResource
                materialArguments.roughnessTexture = material.roughness.texture?.gpuResourceID ?? nullResource
                materialArguments.opacityTexture = material.opacity.texture?.gpuResourceID ?? nullResource
                material.bufferView = materialsBuffer.copy(materialArguments)
            }
        }

        if let cameraEntity = scene.rootEntity.childNamed("Camera") {
            camera.position = cameraEntity.worldTransform.translation.vector
            camera.orientation = cameraEntity.worldTransform.rotation!.quaternion
            camera.nearZ = 0.01
            camera.farZ = 500.0
        }

        try makeSceneResourcesResident(scene)
    }

    private func makeSceneResourcesResident(_ scene: Scene) throws {
        residencySet.addAllocations(scene.resources)

        residencySet.addAllocations(constantsBuffers.map(\.buffer))
        residencySet.addAllocation(materialsBuffer.buffer)

        if let light = scene.lightingEnvironment {
            residencySet.addAllocations([light.diffuseCubeTexture,
                                         light.specularCubeTexture,
                                         light.scaleAndBiasLookupTexture])
        }
        
        commandQueue.addResidencySet(residencySet)
        residencySet.commit()

        hasPreparedResources = true
    }
    
    private func makePipelineState(mesh: Mesh, material: Material) throws -> MTLRenderPipelineState {
        var hasNormals = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeNormal) != nil
        var hasTangents = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeTangent) != nil
        var hasTexCoords0 = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeTextureCoordinate) != nil
        var hasTexCoords1 = mesh.vertexDescriptor.vertexAttributes.count { $0.name == MDLVertexAttributeTextureCoordinate } > 1
        var hasVertexColors = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeColor) != nil
        var hasBaseColorTexture = material.baseColor.texture != nil
        var hasEmissiveTexture = material.emissive.texture != nil
        var hasNormalTexture = material.normal.texture != nil
        var hasMetalnessTexture = material.metalness.texture != nil
        var hasRoughnessTexture = material.roughness.texture != nil
        var hasOcclusionTexture = material.occlusion.texture != nil
        var hasOpacityTexture = material.opacity.texture != nil
        var baseColorUVSet = Int32(material.baseColor.mappingChannel)
        var emissiveUVSet = Int32(material.emissive.mappingChannel)
        var normalUVSet = Int32(material.normal.mappingChannel)
        var metalnessUVSet = Int32(material.metalness.mappingChannel)
        var roughnessUVSet = Int32(material.roughness.mappingChannel)
        var occlusionUVSet = Int32(material.occlusion.mappingChannel)
        var opacityUVSet = Int32(material.opacity.mappingChannel)
        var useIBL = scene.lightingEnvironment != nil
        var alphaMode = UInt32(material.alphaMode.rawValue)
        let functionConstants = MTLFunctionConstantValues()
        functionConstants.setConstantValue(&hasNormals,          type: .bool, withName: "hasNormals")
        functionConstants.setConstantValue(&hasTangents,         type: .bool, withName: "hasTangents")
        functionConstants.setConstantValue(&hasTexCoords0,       type: .bool, withName: "hasTexCoords0")
        functionConstants.setConstantValue(&hasTexCoords1,       type: .bool, withName: "hasTexCoords1")
        functionConstants.setConstantValue(&hasVertexColors,     type: .bool, withName: "hasColors")
        functionConstants.setConstantValue(&hasBaseColorTexture, type: .bool, withName: "hasBaseColorMap")
        functionConstants.setConstantValue(&baseColorUVSet,      type: .int,  withName: "baseColorUVSet")
        functionConstants.setConstantValue(&hasEmissiveTexture,  type: .bool, withName: "hasEmissiveMap")
        functionConstants.setConstantValue(&emissiveUVSet,       type: .int,  withName: "emissiveUVSet")
        functionConstants.setConstantValue(&hasNormalTexture,    type: .bool, withName: "hasNormalMap")
        functionConstants.setConstantValue(&normalUVSet,         type: .int,  withName: "normalUVSet")
        functionConstants.setConstantValue(&hasMetalnessTexture, type: .bool, withName: "hasMetalnessMap")
        functionConstants.setConstantValue(&metalnessUVSet,      type: .int,  withName: "metalnessUVSet")
        functionConstants.setConstantValue(&hasRoughnessTexture, type: .bool, withName: "hasRoughnessMap")
        functionConstants.setConstantValue(&roughnessUVSet,      type: .int,  withName: "roughnessUVSet")
        functionConstants.setConstantValue(&hasOcclusionTexture, type: .bool, withName: "hasOcclusionMap")
        functionConstants.setConstantValue(&occlusionUVSet,      type: .int,  withName: "occlusionUVSet")
        functionConstants.setConstantValue(&hasOpacityTexture,   type: .bool, withName: "hasOpacityMap")
        functionConstants.setConstantValue(&opacityUVSet,        type: .int,  withName: "opacityUVSet")
        functionConstants.setConstantValue(&useIBL,              type: .bool, withName: "useIBL")
        functionConstants.setConstantValue(&alphaMode,           type: .uint, withName: "alphaMode")
        
        let vertexFunction = MTL4SpecializedFunctionDescriptor(name: "pbr_vertex", library: library, functionConstants: functionConstants)
        let fragmentFunction = MTL4SpecializedFunctionDescriptor(name: "pbr_fragment", library: library, functionConstants: functionConstants)
        
        let vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mesh.vertexDescriptor)!
        
        let renderPipelineDescriptor = MTL4RenderPipelineDescriptor()
        renderPipelineDescriptor.vertexFunctionDescriptor = vertexFunction
        renderPipelineDescriptor.fragmentFunctionDescriptor = fragmentFunction
        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
        renderPipelineDescriptor.rasterSampleCount = rasterSampleCount
        
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        
        if material.alphaMode == .blend {
            renderPipelineDescriptor.colorAttachments[0].blendingState = .enabled
            renderPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
            renderPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            renderPipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            renderPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
            renderPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            renderPipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        }

        return try compiler.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
    }
    
    func configure(_ view: MTKView) {
        view.device = device
        view.colorPixelFormat = colorPixelFormat
        view.clearColor = .init(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        view.depthStencilPixelFormat = depthStencilPixelFormat
        view.sampleCount = rasterSampleCount
        view.delegate = self
        
        if let metalLayer = view.layer as? CAMetalLayer {
            commandQueue.addResidencySet(metalLayer.residencySet)
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }
    
    private func updateCamera() {
        guard let orbitEntity = scene.rootEntity.childNamed("TreasureChest", recursively: true) else { return }
        let orbitRadius = 12.0
        let orbitCenter = Point3D(orbitEntity.worldTransform.translation)
        let cameraPosition = Point3D(x: orbitCenter.x + orbitRadius * sin(time * 0.5),
                                     y: 5.0,
                                     z: orbitCenter.z + orbitRadius * cos(time * 0.5))
        let cameraTransform = AffineTransform3D(lookAt: orbitCenter, from: cameraPosition, up: .up)
        camera.setTransform(cameraTransform)
    }
    
    func draw(in view: MTKView) {
        if !hasPreparedResources { return }

        guard let renderPassDescriptor = view.currentMTL4RenderPassDescriptor else {
            return
        }
        
        if frameIndex >= maxFramesInFlight {
            let valueToWait = frameIndex - maxFramesInFlight
            frameCompletionEvent.wait(untilSignaledValue: valueToWait, timeoutMS: 8)
        }

        let currentTime = CACurrentMediaTime()
        if lastRenderTime == 0 { lastRenderTime = currentTime }
        let deltaTime = min(currentTime - lastRenderTime, 0.05)
        lastRenderTime = currentTime
        time += deltaTime

        updateCamera()

        frameIndex += 1
        let allocator = commandAllocators[Int(frameIndex % maxFramesInFlight)]
        allocator.reset()

        let constantsBuffer = constantsBuffers[Int(frameIndex % maxFramesInFlight)]
        constantsBuffer.reset()

        commandBuffer.beginCommandBuffer(allocator: allocator)
        
        let commandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        
        commandEncoder.setFrontFacing(.counterClockwise)

        commandEncoder.setArgumentTable(vertexArgumentTable, stages: .vertex)
        commandEncoder.setArgumentTable(fragmentArgumentTable, stages: .fragment)
        
        let lightView = constantsBuffer.copy(scene.lights)
        fragmentArgumentTable.setAddress(lightView.gpuAddress, index: fragmentBufferLights)
        
        let viewMatrix = camera.viewMatrix()
        let aspectRatio = view.drawableSize.width / view.drawableSize.height
        let projectionMatrix = camera.projectionMatrix(aspectRatio: aspectRatio)
        let frameConstants = FrameConstants(viewMatrix: viewMatrix,
                                            projectionMatrix: projectionMatrix,
                                            viewProjectionMatrix: projectionMatrix * viewMatrix,
                                            environmentRotation: matrix_identity_float3x3,
                                            environmentIntensity: scene.lightingEnvironment?.intensity ?? 1.0,
                                            specularEnvironmentMipCount: UInt32(scene.lightingEnvironment?.specularMipLevelCount ?? 1),
                                            cameraPosition: SIMD3<Float>(camera.position),
                                            activeLightCount: UInt32(scene.lights.count))
        
        let frameView = constantsBuffer.copy(frameConstants)
        vertexArgumentTable.setAddress(frameView.gpuAddress, index: vertexBufferFrameConstants)
        fragmentArgumentTable.setAddress(frameView.gpuAddress, index: fragmentBufferFrameConstants)

        struct DrawCall {
            var mesh: Mesh
            var submesh: Submesh
            var material: Material
            var modelTransform: AffineTransform3D
            var modelViewPosition: SIMD3<Float>
        }

        var drawCalls: [DrawCall] = []
        scene.rootEntity.visitHierarchy { entity in
            if let modelEntity = entity as? ModelEntity, let mesh = modelEntity.mesh {
                for submesh in mesh.submeshes {
                    let material = mesh.materials[submesh.materialIndex]
                    let modelViewPosition = viewMatrix * SIMD4<Float>(SIMD4<Double>(modelEntity.worldTransform.translation.vector, 1.0))
                    drawCalls.append(DrawCall(mesh: mesh,
                                              submesh: submesh,
                                              material: material,
                                              modelTransform: modelEntity.worldTransform,
                                              modelViewPosition: modelViewPosition.xyz))
                }
            }
        }
        drawCalls.sort {
            let leftSort = $0.material.relativeSortOrder
            let rightSort = $1.material.relativeSortOrder
            if leftSort > 0 && rightSort > 0 {
                let leftZ = $0.modelViewPosition.z
                let rightZ = $1.modelViewPosition.z
                return leftZ < rightZ
            }
            return leftSort < rightSort
        }

        if let ibl = scene.lightingEnvironment {
            // Wait for IBL prefiltering to complete before rendering so our lighting doesn't pop in gradually
            commandQueue.waitForEvent(ibl.readyEvent, value: 1)

            fragmentArgumentTable.setTexture(ibl.diffuseCubeTexture.gpuResourceID, index: fragmentTextureDiffuseEnvironment)
            fragmentArgumentTable.setTexture(ibl.specularCubeTexture.gpuResourceID, index: fragmentTextureSpecularEnvironment)
            fragmentArgumentTable.setTexture(ibl.scaleAndBiasLookupTexture.gpuResourceID, index: fragmentTextureGGXLookup)
        }
        
        for drawCall in drawCalls {
            let mesh = drawCall.mesh
            let submesh = drawCall.submesh
            let material = drawCall.material
            
            commandEncoder.pushDebugGroup("Draw \(mesh.name)")
            
            commandEncoder.setDepthStencilState(depthStencilStates[material.alphaMode])
            
            for (bufferIndex, vertexBuffer) in mesh.vertexBuffers.enumerated() {
                vertexArgumentTable.setAddress(vertexBuffer.gpuAddress, index: vertexBuffer0 + bufferIndex)
            }
            
            let modelMatrix = simd_float4x4(drawCall.modelTransform)
            let instanceConstants = InstanceConstants(modelMatrix: modelMatrix,
                                                      normalMatrix: modelMatrix.upperLeft3x3.adjugate.transpose)
            
            let instanceView = constantsBuffer.copy(instanceConstants)
            vertexArgumentTable.setAddress(instanceView.gpuAddress, index: vertexBufferInstanceConstants)
            fragmentArgumentTable.setAddress(instanceView.gpuAddress, index: fragmentBufferInstanceConstants)
            
            guard let renderPipelineState = material.renderPipelineState else {
                fatalError("Material \(material.name) had no corresponding render pipeline state")
            }
            
            commandEncoder.setRenderPipelineState(renderPipelineState)

            if let materialView = material.bufferView {
                fragmentArgumentTable.setAddress(materialView.gpuAddress, index: fragmentBufferMaterial)
            }

            commandEncoder.drawIndexedPrimitives(primitiveType: submesh.primitiveType,
                                                 indexCount: submesh.indexCount,
                                                 indexType: submesh.indexType,
                                                 indexBuffer: submesh.indexBuffer.gpuAddress,
                                                 indexBufferLength: submesh.indexBuffer.length)
            commandEncoder.popDebugGroup()
        }
        
        commandEncoder.endEncoding()
        
        commandBuffer.endCommandBuffer()
        
        let drawable = view.currentDrawable!
        
        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        drawable.present()
        
        let valueToSignal = frameIndex
        commandQueue.signalEvent(frameCompletionEvent, value: valueToSignal)
    }
}
