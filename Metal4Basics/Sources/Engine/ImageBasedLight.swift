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

import ImageIO
import Metal
import MetalKit

class ImageBasedLight {
    let diffuseCubeTexture: MTLTexture
    let specularCubeTexture: MTLTexture
    let specularMipLevelCount: Int
    let scaleAndBiasLookupTexture: MTLTexture
    let readyEvent: (any MTLEvent)

    var rotation: simd_float3x3 = matrix_identity_float3x3
    var intensity: Float = 1.0

    class func generateImageBasedLight(url fileURL: URL) async throws -> ImageBasedLight {
        let loadTask = Task.detached {
            return try await ImageBasedLightGenerator.defaultGenerator.makeLight(url: fileURL)
        }
        switch await loadTask.result {
        case .success(let light):
            return light
        case .failure(let error):
            throw error
        }
    }

    init(diffuseCubeTexture: MTLTexture,
         specularCubeTexture: MTLTexture,
         specularMipLevelCount: Int,
         scaleAndBiasLookupTexture: MTLTexture,
         readyEvent: any MTLEvent)
    {
        self.diffuseCubeTexture = diffuseCubeTexture
        self.specularCubeTexture = specularCubeTexture
        self.specularMipLevelCount = specularMipLevelCount
        self.scaleAndBiasLookupTexture = scaleAndBiasLookupTexture
        self.readyEvent = readyEvent
    }
}

class ImageBasedLightGenerator {
    enum Error: Swift.Error {
        case libraryNotFound
        case functionNotFound
        case allocationFailed
    }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let equirectToCubePipelineState: MTLComputePipelineState!
    let lookupTablePipelineState: MTLComputePipelineState!
    let prefilteringPipelineState: MTLComputePipelineState!

    static var defaultGenerator: ImageBasedLightGenerator = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        let commandQueue = device.makeCommandQueue()!
        do {
            return try ImageBasedLightGenerator(device: device, commandQueue: commandQueue)
        } catch {
            fatalError("Failed to create image-based light generator: \(error)")
        }
    }()

    init(device: MTLDevice, commandQueue: MTLCommandQueue) throws {
        self.device = device
        self.commandQueue = commandQueue

        guard let defaultLibrary = device.makeDefaultLibrary() else { throw Error.libraryNotFound }

        guard let equirectToCubeFunction = defaultLibrary.makeFunction(name: "CubeFromEquirectangular") else { throw Error.functionNotFound }
        equirectToCubePipelineState = try device.makeComputePipelineState(function: equirectToCubeFunction)

        guard let lutIntegration = defaultLibrary.makeFunction(name: "F0OffsetAndBiasLUT") else { throw Error.functionNotFound }
        lookupTablePipelineState = try device.makeComputePipelineState(function: lutIntegration)

        guard let prefilterFunction = defaultLibrary.makeFunction(name: "PrefilterEnvironmentMap") else { throw Error.functionNotFound }
        prefilteringPipelineState = try device.makeComputePipelineState(function: prefilterFunction)
    }

    func makeLight(url: URL) throws -> ImageBasedLight {
        let textureLoader = MTKTextureLoader(device: device)
        let hdrOptions: [MTKTextureLoader.Option : Any] = [
            .SRGB : NSNumber(value: false),
            .allocateMipmaps : NSNumber(value: false)
        ]
        let equirectTexture = try textureLoader.newTexture(URL: url, options: hdrOptions)

        let workingPixelFormat = MTLPixelFormat.rgba16Float
        let sourceHeight = equirectTexture.height
        let sourceCubeSize = min(512, sourceHeight / 2)
        let specularCubeSize = sourceCubeSize
        let diffuseCubeSize = 32
        let lookupTableSize = 512
        let specularSampleCount = 1024
        let diffuseSampleCount = 2048
        let lutSampleCount = 512

        let sourceCubeDescriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: workingPixelFormat,
                                                                              size:specularCubeSize,
                                                                              mipmapped:true)
        sourceCubeDescriptor.usage = [.shaderRead, .shaderWrite]

        guard let sourceCubeTexture = device.makeTexture(descriptor: sourceCubeDescriptor) else { throw Error.allocationFailed }
        sourceCubeTexture.label = "Environment Map (Cube)"

        let cubePixelFormat = MTLPixelFormat.rgba16Float

        let specularCubeDescriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat:cubePixelFormat,
                                                                                size:specularCubeSize,
                                                                                mipmapped:true)
        specularCubeDescriptor.usage = [.shaderRead, .shaderWrite]
        specularCubeDescriptor.storageMode = .private

        guard let specularCubeTexture = device.makeTexture(descriptor:specularCubeDescriptor) else { throw Error.allocationFailed }
        specularCubeTexture.label = "Prefiltered Environment (GGX)"

        let diffuseCubeDescriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat:cubePixelFormat,
                                                                               size:diffuseCubeSize,
                                                                               mipmapped:false)
        diffuseCubeDescriptor.usage = [.shaderRead, .shaderWrite]
        diffuseCubeDescriptor.storageMode = .private

        guard let diffuseCubeTexture = device.makeTexture(descriptor:diffuseCubeDescriptor) else { throw Error.allocationFailed }
        diffuseCubeTexture.label = "Prefiltered Environment (Lambertian)"

        let lookupTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                               width:lookupTableSize,
                                                                               height:lookupTableSize,
                                                                               mipmapped:false)
        lookupTextureDescriptor.usage = [.shaderRead, .shaderWrite]
        lookupTextureDescriptor.storageMode = .private

        guard let lookupTexture = device.makeTexture(descriptor:lookupTextureDescriptor) else { throw Error.allocationFailed }
        lookupTexture.label = "DFG Lookup Table (GGX)"

        var commandBuffer = commandQueue.makeCommandBuffer()!

        var computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        computeCommandEncoder.setComputePipelineState(self.equirectToCubePipelineState)
        computeCommandEncoder.setTexture(equirectTexture, index:1)
        computeCommandEncoder.setTexture(sourceCubeTexture, index:0)
        computeCommandEncoder.dispatchThreads(MTLSizeMake(sourceCubeSize, sourceCubeSize, 6),
                                              threadsPerThreadgroup:MTLSizeMake(32, 32, 1))

        computeCommandEncoder.endEncoding()

        commandBuffer.commit()

        commandBuffer = commandQueue.makeCommandBuffer()!

        let mipmapCommandEncoder = commandBuffer.makeBlitCommandEncoder()!
        mipmapCommandEncoder.generateMipmaps(for: sourceCubeTexture)
        mipmapCommandEncoder.endEncoding()

        computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        struct PrefilteringParams {
            var distribution: UInt32
            var sampleCount: UInt32
            var roughness: Float
            var lodBias: Float
            var cubemapSize: Float
        }

        let mipLevelCount = 5
        var levelSize = specularCubeSize
        for mipLevel in 0..<mipLevelCount {
            let levelView = specularCubeTexture.__newTextureView(with: specularCubeTexture.pixelFormat,
                                                                textureType: .typeCube,
                                                                levels: NSMakeRange(mipLevel, 1),
                                                                slices: NSMakeRange(0, 6))
            var params = PrefilteringParams(distribution: 1,
                                            sampleCount: UInt32(specularSampleCount),
                                            roughness: Float(mipLevel) / Float(mipLevelCount - 1),
                                            lodBias: 0.0,
                                            cubemapSize: Float(sourceCubeSize))
            computeCommandEncoder.setComputePipelineState(prefilteringPipelineState)
            computeCommandEncoder.setBytes(&params, length:MemoryLayout.size(ofValue:params), index: 0)
            computeCommandEncoder.setTexture(sourceCubeTexture, index: 0)
            computeCommandEncoder.setTexture(levelView, index: 1)
            computeCommandEncoder.dispatchThreads(MTLSizeMake(levelSize, levelSize, 6),
                                                  threadsPerThreadgroup:MTLSizeMake(32, 32, 1))
            levelSize >>= 1
        }

        var params = PrefilteringParams(distribution: 0,
                                        sampleCount: UInt32(diffuseSampleCount),
                                        roughness: 0.0,
                                        lodBias: 0.0,
                                        cubemapSize: Float(sourceCubeSize))
        computeCommandEncoder.setComputePipelineState(prefilteringPipelineState)
        computeCommandEncoder.setBytes(&params, length:MemoryLayout.size(ofValue: params), index: 0)
        computeCommandEncoder.setTexture(sourceCubeTexture, index: 0)
        computeCommandEncoder.setTexture(diffuseCubeTexture, index: 1)
        computeCommandEncoder.dispatchThreads(MTLSizeMake(diffuseCubeSize, diffuseCubeSize, 6),
                                              threadsPerThreadgroup:MTLSizeMake(32, 32, 1))

        var sampleCount = UInt32(lutSampleCount)
        computeCommandEncoder.setComputePipelineState(lookupTablePipelineState)
        computeCommandEncoder.setBytes(&sampleCount, length:MemoryLayout.size(ofValue: sampleCount), index: 0)
        computeCommandEncoder.setTexture(lookupTexture, index: 0)
        computeCommandEncoder.dispatchThreads(MTLSizeMake(lookupTableSize, lookupTableSize, 1),
                                              threadsPerThreadgroup:MTLSizeMake(32, 32, 1))

        computeCommandEncoder.endEncoding()

        let readyEvent = device.makeEvent()!
        commandBuffer.encodeSignalEvent(readyEvent, value: 1)

        commandBuffer.commit()

        let light = ImageBasedLight(diffuseCubeTexture:diffuseCubeTexture,
                                    specularCubeTexture:specularCubeTexture,
                                    specularMipLevelCount: mipLevelCount,
                                    scaleAndBiasLookupTexture:lookupTexture,
                                    readyEvent: readyEvent)
        return light
    }
}
