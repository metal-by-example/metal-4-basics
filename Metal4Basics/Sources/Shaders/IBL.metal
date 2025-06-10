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
// Portions of this source are originally copyright 2012 Holger Dammertz.
// Licensed under the terms of the Creative Commons Attribution 3.0 Unported license.
//

#include <metal_stdlib>
using namespace metal;

enum class BRDF : uint {
    Lambert,
    TrowbridgeReitz,
};

typedef struct {
    uint distribution;
    uint sampleCount;
    float roughness;
    float lodBias;
    float cubemapSize;
} PrefilteringParameters;

typedef struct {
    uint sampleCount;
} LUTParameters;

typedef struct {
    float pdf;
    float cosTheta;
    float sinTheta;
    float phi;
} MicrofacetDistributionSample;

typedef struct {
    float3 direction;
    float pdf;
} ImportanceSample;

static float3 CubeDirectionFromFaceAndUV(int face, float2 uv) {
    float3 N;
    switch (face) {
        case 0: N = {  1.0f,  uv.y, -uv.x }; break;
        case 1: N = { -1.0f,  uv.y,  uv.x }; break;
        case 2: N = {  uv.x, -1.0f,  uv.y }; break;
        case 3: N = {  uv.x,  1.0f, -uv.y }; break;
        case 4: N = {  uv.x,  uv.y,  1.0f }; break;
        case 5: N = { -uv.x,  uv.y, -1.0f }; break;
    }
    return normalize(N);
}

static float2 EquirectUVFromCubeDirection(float3 v) {
    const float2 scales { 0.1591549f, 0.3183099f };
    const float2 biases { 0.5f, 0.5f };
    // Assumes +Z is forward. For -X forward, use atan2(v.z, v.x) below instead.
    float2 uv = float2(atan2(-v.x, v.z), asin(v.y)) * scales + biases;
    return uv;
}

// CC BY 3.0 (Holger Dammertz)
// Ref: http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// Ref: https://math.stackexchange.com/q/849848
static float VanDerCorputRadixInverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Generates a sequence of low-discrepancy, randomly distributed points on [0, 1)^2
static float2 Hammersley(int i, int N) {
    return { float(i) / float(N), VanDerCorputRadixInverse(uint(i)) };
}

// Generate an orthonormal tangent space basis given a normalized normal.
static float3x3 TangentSpaceFromNormal(float3 normal) {
    float3 bitangent { 0.0f, 1.0f, 0.0f };

    float Ny = normal.y;
    float epsilon = 0.0000001f;
    if (1.0 - abs(Ny) <= epsilon) {
        // We're very nearly vertical, so use a different axis
        // as the bitangent for numerical stability.
        if (Ny > 0.0f) {
            bitangent = { 0.0f, 0.0f, 1.0f };
        } else {
            bitangent = { 0.0f, 0.0f, -1.0f };
        }
    }

    float3 tangent = normalize(cross(bitangent, normal));
    bitangent = cross(normal, tangent);

    return float3x3(tangent, bitangent, normal);
}

// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html
static float D_TrowbridgeReitz(float NdotH, float roughness) {
    float a = NdotH * roughness;
    float k = roughness / (1.0f - NdotH * NdotH + a * a);
    return k * k * (1.0f / M_PI_F);
}

// Ref. https://bruop.github.io/ibl/
// Ref. https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html
// Ref. https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
static MicrofacetDistributionSample TrowbridgeReitz(float2 xi, float roughness)
{
    float alpha = roughness * roughness;
    MicrofacetDistributionSample ggx;
    ggx.cosTheta = saturate(sqrt((1.0f - xi.y) / (1.0f + (alpha * alpha - 1.0f) * xi.y)));
    ggx.sinTheta = sqrt(1.0f - ggx.cosTheta * ggx.cosTheta);
    ggx.phi = xi.x * 2.0 * M_PI_F;
    // PDF, normalized by the Jacobian. PDF = (D * NdotH) / (4 * VdotH),
    // so because of the V = N assumption, the factor is simply 1/4.
    ggx.pdf = D_TrowbridgeReitz(ggx.cosTheta, alpha) / 4.0f;
    return ggx;
}

// Ref. http://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations.html#Cosine-WeightedHemisphereSampling
static MicrofacetDistributionSample Lambertian(float2 xi, float /*roughness*/) {
    MicrofacetDistributionSample sample;
    sample.cosTheta = sqrt(1.0f - xi.y);
    sample.sinTheta = sqrt(xi.y);
    sample.phi = xi.x * 2.0f * M_PI_F;
    sample.pdf = sample.cosTheta / M_PI_F;
    return sample;
}

static ImportanceSample GetImportanceSample(BRDF distribution, float3 N, float roughness,
                                            int sampleIndex, uint sampleCount)
{
    float2 xi = Hammersley(sampleIndex, sampleCount);

    MicrofacetDistributionSample sample;
    switch (distribution) {
        case BRDF::Lambert:
            sample = Lambertian(xi, roughness);
            break;
        case BRDF::TrowbridgeReitz:
            sample = TrowbridgeReitz(xi, roughness);
            break;
    }

    float3 localSpaceDirection = normalize(float3(sample.sinTheta * cos(sample.phi),
                                                  sample.sinTheta * sin(sample.phi),
                                                  sample.cosTheta));
    float3x3 TBN = TangentSpaceFromNormal(N);
    float3 direction = TBN * localSpaceDirection;

    return { direction, sample.pdf };
}

// Ref: https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
// Ref: https://cgg.mff.cuni.cz/~jaroslav/papers/2007-sketch-fis/Final_sap_0073.pdf
static float LodForSample(uint width, uint sampleCount, float pdf)
{
    // glTF Sample Viewer formulation:
    //float lod = 0.5f * log2(6.0f * float(width) * float(width) / (float(sampleCount) * pdf));

    // Colbert and Křivánek formulation:
    // float lod = max(log4(((width * height / sampleCount) * 1 / pdf), 0);

    // Assuming we started from an equirectangular environment map with w = 2 * h, and
    // assuming that the texel density was preserved in the conversion, the texel count
    // is width^2 * 8. If it was resized, that would introduce a different constant factor.
    float texelCount = float(width * width) * 8.0f;
    float lod = max(0.5f * log2(texelCount / (float(sampleCount) * pdf)), 0.0f);
    return lod;
}

static float3 FilteredColor(BRDF distribution, float3 N, float roughness, float lodBias, uint width, uint sampleCount,
                            texturecube<float, access::sample> cubeTexture)
{
    constexpr sampler cubeSampler(coord::normalized, filter::linear, mip_filter::linear, address::repeat);

    float3 color { 0.0f };
    float weight = 0.0f;
    for(uint i = 0; i < sampleCount; ++i) {
        ImportanceSample importanceSample = GetImportanceSample(distribution, N, roughness, i, sampleCount);
        float3 H = importanceSample.direction;
        float pdf = importanceSample.pdf;

        float lod = LodForSample(width, sampleCount, pdf) + lodBias;

        switch (distribution) {
            case BRDF::Lambert: {
                float3 sampleColor = cubeTexture.sample(cubeSampler, H, level(lod)).rgb;
                // Summing over factors of NdotH and dividing by pi cancel, so we omit both.
                // Ref. https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
                color += sampleColor;
                break;
            }
            case BRDF::TrowbridgeReitz:
                float3 V = N;
                float3 L = normalize(reflect(-V, H));
                float NdotL = dot(N, L);
                if (NdotL > 0.0f) {
                    if (roughness == 0.0f) {
                        lod = lodBias;
                    }
                    float3 sampleColor = cubeTexture.sample(cubeSampler, L, level(lod)).rgb;
                    color += sampleColor * NdotL;
                    weight += NdotL;
                }
                break;
        }
    }

    if (weight == 0.0f) {
        color /= float(sampleCount);
    } else {
        color /= weight;
    }

    return color;
}

// https://google.github.io/filament/Filament.html#toc4.4.2
static float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
    float a2 = powr(roughness, 4.0f);
    float GGXV = NoL * sqrt(NoV * NoV * (1.0f - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0f - a2) + a2);
    return 0.5f / (GGXV + GGXL);
}

// Ref. https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
static float3 CombinedF0OffsetAndBias(uint width, uint sampleCount, float NdotV, float roughness)
{
    float3 V = float3(sqrt(1.0f - NdotV * NdotV), 0.0f, NdotV);
    float3 N = float3(0.0f, 0.0f, 1.0f);

    float A = 0.0;
    float B = 0.0;
    for(uint i = 0; i < sampleCount; ++i) {
        ImportanceSample importanceSample = GetImportanceSample(BRDF::TrowbridgeReitz, N, roughness, i, sampleCount);
        float3 H = importanceSample.direction;
        float3 L = normalize(reflect(-V, H));

        float NdotL = saturate(L.z);
        float NdotH = saturate(H.z);
        float VdotH = saturate(dot(V, H));
        if (NdotL > 0.0f) {
            float V_pdf = V_SmithGGXCorrelated(NdotV, NdotL, roughness) * VdotH * NdotL / NdotH;
            float Fc = powr(1.0f - VdotH, 5.0);
            A += (1.0f - Fc) * V_pdf;
            B += Fc * V_pdf;
        }
    }

    return 4.0f * float3(A, B, 0.0f) / float(sampleCount);
}

// Utility kernel for converting from an equirectangular environment map (LDR or HDR) to a cube map
kernel void CubeFromEquirectangular(texturecube<float, access::write> cubeMap [[texture(0)]],
                                    texture2d<float, access::sample> equirectangularMap [[texture(1)]],
                                    uint3 tpig [[thread_position_in_grid]])
{
    constexpr sampler equirectSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    uint2 coords = tpig.xy;
    int face = tpig.z;
    float cubeSize = cubeMap.get_width();
    float2 cubeUV = ((float2(tpig.xy) / (cubeSize - 1)) * 2 - 1);
    float3 dir = CubeDirectionFromFaceAndUV(face, cubeUV);
    float2 rectUV = EquirectUVFromCubeDirection(dir);
    float4 color = equirectangularMap.sample(equirectSampler, rectUV);
    cubeMap.write(color, coords, face);
}

// Generates the lookup texture for F0 scale and bias from BRDF in Karis split-sum approximation.
// GGX scale and bias are stored in r and g channels, respectively.
kernel void F0OffsetAndBiasLUT(texture2d<float, access::write> lutTexture [[texture(0)]],
                               constant LUTParameters &params [[buffer(0)]],
                               uint2 tpig [[thread_position_in_grid]])
{
    uint size = lutTexture.get_width();
    float NdotV = (tpig.x + 1) / float(size);
    float roughness = (tpig.y + 1) / float(lutTexture.get_height());
    float3 abc = CombinedF0OffsetAndBias(size, params.sampleCount, NdotV, roughness);
    lutTexture.write(float4(abc, 1.0f), tpig);
}

// Generates a radiance map for a particular roughness from an environment map.
// Can be run multiple times to store maps corresponding to different roughness to different mip levels.
kernel void PrefilterEnvironmentMap(texturecube<float, access::sample> environmentMap,
                                    texturecube<float, access::write> filteredMap,
                                    constant PrefilteringParameters &params [[buffer(0)]],
                                    uint3 threadIndex [[thread_position_in_grid]])
{
    BRDF distribution = static_cast<BRDF>(params.distribution);
    uint2 coords = threadIndex.xy;
    int face = threadIndex.z;
    float cubeSize = float(filteredMap.get_width());
    float2 cubeUV = ((float2(coords) / cubeSize) * 2.0f - 1.0f);
    float3 N = CubeDirectionFromFaceAndUV(face, cubeUV);
    N *= float3(1.0f, -1.0f, 1.0f);
    float3 color = FilteredColor(distribution, N, params.roughness, params.lodBias,
                                 params.cubemapSize, params.sampleCount, environmentMap);
    filteredMap.write(float4(color, 1.0f), coords, face);
}
