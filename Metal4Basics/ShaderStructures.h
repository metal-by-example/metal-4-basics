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

#include <simd/simd.h>

#ifndef __METAL__
#import <Metal/MTLTypes.h> // For MTLResourceID
#endif

typedef struct {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewProjectionMatrix;
    simd_float3x3 environmentRotation;
    float environmentIntensity;
    unsigned int specularEnvironmentMipCount;
    simd_float3 cameraPosition; // world space
    unsigned int activeLightCount;
} FrameConstants;

typedef struct {
    simd_float4x4 modelMatrix;
    simd_float3x3 normalMatrix;
} InstanceConstants;

typedef struct {
    simd_float3 position;
    simd_float3 direction;
    simd_float3 color;
    float intensity;
    float range;
    float innerConeCos;
    float outerConeCos;
    unsigned int type;
} Light;

typedef struct {
    simd_float4 baseColorFactor;
    float opacityFactor;
    float normalScale;
    float occlusionStrength;
    float metallicFactor;
    float roughnessFactor;
    simd_float3 emissiveFactor;
    float alphaCutoff;
} MaterialConstants;

typedef struct {
    MaterialConstants constants;
    MTLResourceID baseColorTexture;
    MTLResourceID normalTexture;
    MTLResourceID emissiveTexture;
    MTLResourceID occlusionTexture;
    MTLResourceID metalnessTexture;
    MTLResourceID roughnessTexture;
    MTLResourceID opacityTexture;
} MaterialArguments;

enum {
    vertexBuffer0,
    vertexBuffer1,
    vertexBuffer2,
    vertexBuffer3,
    vertexBufferFrameConstants,
    vertexBufferInstanceConstants,
    
    VertexBufferCount // Keep last
};

enum {
    fragmentBufferFrameConstants,
    fragmentBufferLights,
    fragmentBufferMaterial,
    fragmentBufferInstanceConstants,
    
    FragmentBufferCount // Keep last
};

enum {
    fragmentTextureDiffuseEnvironment,
    fragmentTextureSpecularEnvironment,
    fragmentTextureGGXLookup,

    FragmentTextureCount // Keep last
};
