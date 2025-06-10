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

#include <metal_stdlib>
using namespace metal;

#include "ShaderStructures.h"

enum class AlphaMode : unsigned int {
    Opaque,
    Mask,
    Blend,
};

enum class LightType : unsigned int {
    Invalid,
    Directional,
    Point,
    Spot,
};

#pragma mark - Function constants

constexpr constant bool hasNormals        [[function_constant(0)]];
constexpr constant bool hasTangents       [[function_constant(1)]];
constexpr constant bool hasTexCoords0     [[function_constant(2)]];
constexpr constant bool hasTexCoords1     [[function_constant(3)]];
constexpr constant bool hasColors         [[function_constant(4)]];
constexpr constant bool hasBaseColorMap   [[function_constant(5)]];
constexpr constant int baseColorUVSet     [[function_constant(6)]];
constexpr constant bool hasEmissiveMap    [[function_constant(7)]];
constexpr constant int emissiveUVSet      [[function_constant(8)]];
constexpr constant bool hasNormalMap      [[function_constant(9)]];
constexpr constant int normalUVSet        [[function_constant(10)]];
constexpr constant bool hasMetalnessMap   [[function_constant(11)]];
constexpr constant int metalnessUVSet     [[function_constant(12)]];
constexpr constant bool hasRoughnessMap   [[function_constant(13)]];
constexpr constant int roughnessUVSet     [[function_constant(14)]];
constexpr constant bool hasOcclusionMap   [[function_constant(15)]];
constexpr constant int occlusionUVSet     [[function_constant(16)]];
constexpr constant bool hasOpacityMap     [[function_constant(17)]];
constexpr constant int opacityUVSet       [[function_constant(18)]];
constexpr constant bool useIBL            [[function_constant(19)]];
constexpr constant unsigned int alphaMode [[function_constant(20)]];

#pragma mark - Constexpr samplers

constexpr sampler trilinearSampler(coord::normalized, filter::linear, mip_filter::linear, address::repeat);
constexpr sampler bilinearClampSampler(coord::normalized, filter::linear, mip_filter::none, address::clamp_to_edge);

#pragma mark - Input structures

struct Material {
    MaterialConstants constants;
    texture2d<float, access::sample> baseColorTexture;
    texture2d<float, access::sample> normalTexture;
    texture2d<float, access::sample> emissiveTexture;
    texture2d<float, access::sample> occlusionTexture;
    texture2d<float, access::sample> metalnessTexture;
    texture2d<float, access::sample> roughnessTexture;
    texture2d<float, access::sample> opacityTexture;
};

typedef struct {
    float4 position   [[attribute(0)]];
    float3 normal     [[attribute(1) function_constant(hasNormals)]];
    float4 tangent    [[attribute(2) function_constant(hasTangents)]];
    float2 texCoords0 [[attribute(3) function_constant(hasTexCoords0)]];
    float2 texCoords1 [[attribute(4) function_constant(hasTexCoords1)]];
    float4 color      [[attribute(5) function_constant(hasColors)]];
} VertexIn;

#pragma mark - Fragment stage input structures

typedef struct {
    float4 clipPosition [[position]];
    float3 position; // world-space position
    float3 normal;
    float4 tangent;
    float2 texCoords0   [[function_constant(hasTexCoords0)]];
    float2 texCoords1   [[function_constant(hasTexCoords1)]];
    float4 color        [[function_constant(hasColors)]];
    float pointSize     [[point_size]];
} VertexOut;

typedef struct {
    float4 viewportPosition [[position]];
    float3 position;
    float3 normal;
    float4 tangent;
    float2 texCoords0     [[function_constant(hasTexCoords0)]];
    float2 texCoords1     [[function_constant(hasTexCoords1)]];
    float4 color          [[function_constant(hasColors)]];
    float pointSize       [[point_size]];
    bool frontFacing      [[front_facing]];
} FragmentIn;

#pragma mark - Intermediate fragment stage structures

// Per-fragment material properties
typedef struct FragmentMaterial {
    float4 baseColor { 1.0f, 1.0f, 1.0f, 1.0f };
    float3 c_diff { 1.0f, 1.0f, 1.0f };
    float3 F0 { 0.04f, 0.04f, 0.04f };
    float3 F90;
    float metalness = 1.0f;
    float perceptualRoughness = 1.0f;
    float alphaRoughness = 1.0f;
    float ior = 1.5f;
    float specularWeight = 1.0f;
} FragmentMaterial;

typedef struct TangentSpace {
    float3 T;  // Geometry tangent
    float3 B;  // Geometry bitangent
    float3 Ng; // Geometry normal
    float3 N;  // Shading normal (TBN() * Nt)
    float3 Nt; // Normal from texture, scaled to [0, 1] (only for debugging)

    float3x3 TBN() const {
        return { T, B, Ng };
    }
} TangentSpace;

#pragma mark - Math utilities

static float clampedDot(float3 x, float3 y) {
    return saturate(dot(x, y));
}

#pragma mark - Vertex attribute retrieval utilities

static float4 getVertexColor(FragmentIn v) {
    float4 color = float4(1.0);
    if (hasColors) {
        color = v.color;
    }
    return color;
}

static float2 getNormalUV(FragmentIn v) {
    float2 uv = normalUVSet == 0 ? v.texCoords0.xy : v.texCoords1.xy;
    uv.y = 1.0f - uv.y;
    return uv.xy;
}

static float2 getEmissiveUV(FragmentIn v) {
    float2 uv = emissiveUVSet == 0 ? v.texCoords0.xy : v.texCoords1.xy;
    uv.y = 1.0f - uv.y;
    return uv.xy;
}

static float2 getOcclusionUV(FragmentIn v) {
    float2 uv = occlusionUVSet == 0 ? v.texCoords0.xy : v.texCoords1.xy;
    uv.y = 1.0f - uv.y;
    return uv.xy;
}

static float2 getBaseColorUV(FragmentIn v) {
    float2 uv = baseColorUVSet == 0 ? v.texCoords0.xy : v.texCoords1.xy;
    uv.y = 1.0f - uv.y;
    return uv;
}

static float2 getOpacityUV(FragmentIn v) {
    float2 uv = opacityUVSet == 0 ? v.texCoords0.xy : v.texCoords1.xy;
    uv.y = 1.0f - uv.y;
    return uv;
}

static float2 getMetalnessUV(FragmentIn v) {
    float2 uv = metalnessUVSet == 0 ? v.texCoords0.xy : v.texCoords1.xy;
    uv.y = 1.0f - uv.y;
    return uv;
}

static float2 getRoughnessUV(FragmentIn v) {
    float2 uv = roughnessUVSet == 0 ? v.texCoords0.xy : v.texCoords1.xy;
    uv.y = 1.0f - uv.y;
    return uv;
}

#pragma mark - Vertex attribute accessors

static float4 getPosition(VertexIn v)
{
    float4 pos = float4(v.position.xyz, 1.0f);
    return pos;
}

static float3 getNormal(VertexIn v)
{
    float3 normal { 0.0f, 0.0f, 1.0f };
    if (hasNormals) {
        normal = v.normal;
    }
    return normalize(normal);
}

static float4 getTangent(VertexIn v)
{
    float3 tangent { 1.0f, 0.0f, 0.0f };
    float tangentW = 1.0f;
    if (hasTangents) {
        tangent = v.tangent.xyz;
        tangentW = v.tangent.w;
    }
    return float4(normalize(tangent), tangentW);
}

#pragma mark - Normal utilities

static float2x3 cotangents(float3 N, float3 p, float2 uv) {
    // get edge vectors of the pixel triangle
    float3 dp1 = dfdx(p);
    float3 dp2 = dfdy(p);
    float2 duv1 = dfdx(uv);
    float2 duv2 = dfdy(uv);
    // solve the linear system
    float3 dp2perp = cross(dp2, N);
    float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * -duv1.y + dp1perp * -duv2.y;
    // construct a scale-invariant frame
    float invmax = rsqrt(max(dot(T, T), dot(B, B)));
    return float2x3(T * invmax, B * invmax);
}

static TangentSpace getTangentSpace(FragmentIn v, constant MaterialConstants &material, texture2d<float, access::sample> normalMap)
{
    float2 uv = getNormalUV(v);
    float3 t, b, ng;

    // Compute geometrical TBN:
    if (hasNormals) {
        if (hasTangents) {
            t = normalize(v.tangent.xyz);
            ng = normalize(v.normal.xyz);
            b = cross(ng, t) * v.tangent.w;
        } else {
            // Normals are either present as vertex attributes or approximated.
            ng = normalize(v.normal.xyz);
            float2x3 TB = cotangents(ng, -v.position.xyz, uv);
            t = TB[0];
            b = TB[1];
        }
    } else {
        ng = normalize(cross(dfdx(v.position.xyz), dfdy(v.position.xyz)));
        float2x3 TB = cotangents(ng, -v.position.xyz, uv);
        t = TB[0];
        b = TB[1];
    }

    // For a back-facing surface, the tangent basis vectors are negated.
    if (!v.frontFacing) {
        t *= -1.0f;
        b *= -1.0f;
        ng *= -1.0f;
    }

    // Apply normal map if available:
    TangentSpace basis;
    basis.Ng = ng;
    if (hasNormalMap) {
        basis.Nt = normalMap.sample(trilinearSampler, uv).rgb * 2.0f - 1.0f;
        basis.Nt *= float3(material.normalScale, material.normalScale, 1.0);
        basis.Nt = normalize(basis.Nt);
        basis.N = normalize(float3x3(t, b, ng) * basis.Nt);
    } else {
        basis.N = ng;
    }

    basis.T = t;
    basis.B = b;
    return basis;
}

#pragma mark - Material property utilities

static float4 getBaseColor(FragmentIn v, constant MaterialConstants &material,
                           texture2d<float, access::sample> baseColorMap,
                           texture2d<float, access::sample> opacityMap)
{
    float4 baseColor = material.baseColorFactor;
    if (hasBaseColorMap) {
        baseColor *= baseColorMap.sample(trilinearSampler, getBaseColorUV(v));
    }
    if (hasOpacityMap) {
        baseColor.a = opacityMap.sample(trilinearSampler, getOpacityUV(v)).a;
    }
    baseColor.a *= material.opacityFactor;
    return baseColor * getVertexColor(v);
}

static void populateMaterialParameters(FragmentIn v,
                                       TangentSpace tangentSpace,
                                       thread FragmentMaterial &outMaterial,
                                       constant Material &material)
{
    outMaterial.metalness = material.constants.metallicFactor;
    if (hasMetalnessMap) {
        float sampledMetalness = material.metalnessTexture.sample(trilinearSampler, getMetalnessUV(v)).b;
        outMaterial.metalness *= sampledMetalness;
    }

    outMaterial.perceptualRoughness = material.constants.roughnessFactor;
    if (hasRoughnessMap) {
        float sampledRoughness = material.roughnessTexture.sample(trilinearSampler, getRoughnessUV(v)).g;
        outMaterial.perceptualRoughness *= sampledRoughness;
    }

    outMaterial.F0 = mix(float3(0.04f), outMaterial.baseColor.rgb, outMaterial.metalness);

    outMaterial.c_diff = mix(outMaterial.baseColor.rgb, float3(0.0f), outMaterial.metalness);
}

#pragma mark - BRDF components

static float V_GGX(float NdotL, float NdotV, float alphaRoughness)
{
    float alphaRoughnessSq = alphaRoughness * alphaRoughness;
    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0f - alphaRoughnessSq) + alphaRoughnessSq);
    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0f - alphaRoughnessSq) + alphaRoughnessSq);
    float GGX = GGXV + GGXL;
    if (GGX > 0.0f) {
        return 0.5f / GGX;
    }
    return 0.0f;
}

static float D_GGX(float NdotH, float alphaRoughness) {
    float alphaRoughnessSq = alphaRoughness * alphaRoughness;
    float f = (NdotH * NdotH) * (alphaRoughnessSq - 1.0f) + 1.0f;
    return alphaRoughnessSq / (M_PI_F * f * f);
}

static float3 F_Schlick(float3 f0, float3 f90, float VdotH)
{
    return f0 + (f90 - f0) * powr(clamp(1.0f - VdotH, 0.0f, 1.0f), 5.0f);
}

static float3 BRDF_lambertian(float3 f0, float3 f90, float3 diffuseColor, float specularWeight, float VdotH) {
    // see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
    return (1.0f - specularWeight * F_Schlick(f0, f90, VdotH)) * (diffuseColor / M_PI_F);
}

static float3 BRDF_specularGGX(float3 f0, float3 f90, float alphaRoughness, float specularWeight,
                               float VdotH, float NdotL, float NdotV, float NdotH)
{
    float3 F = F_Schlick(f0, f90, VdotH);
    float V = V_GGX(NdotL, NdotV, alphaRoughness);
    float D = D_GGX(NdotH, alphaRoughness);

    return specularWeight * F * V * D;
}

#pragma mark - Tonemapping functions

// Input color is non-negative and resides in the Linear Rec. 709 color space.
// Output color is also Linear Rec. 709, but in the [0, 1] range.
float3 toneMapKHR_pbrNeutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) {
        return color;
    }

    const float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return mix(color, float3(newPeak), g);
}

float3 toneMap(float3 color, float exposure)
{
    color *= exposure;
    color = toneMapKHR_pbrNeutral(color);
    return color;
}

#pragma mark - Analytic light functions

static float getDistanceAttenuation(float range, float distance) {
    if (range <= 0.0f) {
        // negative range means unlimited
        return 1.0f / powr(distance, 2.0f);
    }
    return max(min(1.0f - powr(distance / range, 4.0f), 1.0f), 0.0f) / powr(distance, 2.0f);
}

static float getSpotAttenuation(float3 pointToLight, float3 spotDirection, float outerConeCos, float innerConeCos)
{
    float actualCos = dot(normalize(spotDirection), normalize(-pointToLight));
    if (actualCos > outerConeCos) {
        if (actualCos < innerConeCos) {
            return smoothstep(outerConeCos, innerConeCos, actualCos);
        }
        return 1.0f;
    }
    return 0.0f;
}

static float3 getLightIntensity(Light light, float3 pointToLight)
{
    float rangeAttenuation = 1.0;
    float spotAttenuation = 1.0;

    if (light.type != (int)LightType::Directional) {
        rangeAttenuation = getDistanceAttenuation(light.range, length(pointToLight));
    }
    if (light.type == (int)LightType::Spot) {
        spotAttenuation = getSpotAttenuation(pointToLight, light.direction, light.outerConeCos, light.innerConeCos);
    }

    return rangeAttenuation * spotAttenuation * light.intensity * light.color;
}

#pragma mark - IBL environment map utilities

typedef struct {
    simd_float3x3 rotation;
    float intensity;
} EnvironmentParameters;

static float3 getDiffuseLight(texturecube<float, access::sample> lambertianEnvMap,
                              EnvironmentParameters env, float3 N)
{
    return lambertianEnvMap.sample(trilinearSampler, env.rotation * N).rgb * env.intensity;
}

static float4 getSpecularSample(texturecube<float, access::sample> GGXEnvMap,
                                EnvironmentParameters env, float3 R, float lod)
{
    return GGXEnvMap.sample(trilinearSampler, env.rotation * R, level(lod)) * env.intensity;
}

#pragma mark - IBL BRDF functions

static float3 getIBLRadianceGGX(texturecube<float, access::sample> envMap, EnvironmentParameters env,
                                texture2d<float, access::sample> GGXLUT, float3 N, float3 V,
                                float roughness, float3 F0, float specularWeight, int mipCount)
{
    float NdotV = clampedDot(N, V);
    float lod = roughness * float(mipCount - 1);
    float3 reflection = normalize(reflect(-V, N));

    float2 brdfSamplePoint = float2(NdotV, roughness);
    float2 f_ab = GGXLUT.sample(bilinearClampSampler, brdfSamplePoint).rg;
    float3 specularLight = getSpecularSample(envMap, env, reflection, lod).rgb;

    // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
    // Roughness-dependent Fresnel, from Fdez-Aguera
    float3 Fr = max(float3(1.0f - roughness), F0) - F0;
    float3 k_S = F0 + Fr * powr(1.0f - NdotV, 5.0f);
    float3 FssEss = k_S * f_ab.x + f_ab.y;

    float3 radiance = specularWeight * specularLight * FssEss;
    return radiance;
}

static float3 getIBLRadianceLambertian(texturecube<float, access::sample> lambertianEnvMap, EnvironmentParameters env,
                                       texture2d<float, access::sample> u_GGXLUT, float3 N, float3 V,
                                       float roughness, float3 diffuseColor, float3 F0, float specularWeight)
{
    float NdotV = clampedDot(N, V);
    float2 brdfSamplePoint = float2(NdotV, roughness);
    float2 f_ab = u_GGXLUT.sample(bilinearClampSampler, brdfSamplePoint).rg;

    float3 irradiance = getDiffuseLight(lambertianEnvMap, env, N);

    float3 Fr = max(float3(1.0 - roughness), F0) - F0;
    float3 k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);
    float3 FssEss = specularWeight * k_S * f_ab.x + f_ab.y;

    // Multiple scattering, from Fdez-Aguera
    float Ems = (1.0 - (f_ab.x + f_ab.y));
    float3 F_avg = specularWeight * (F0 + (1.0 - F0) / 21.0);
    float3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    float3 k_D = diffuseColor * (1.0 - FssEss + FmsEms); // Is this right?

    return (FmsEms + k_D) * irradiance;
}

#pragma mark - Primitive vertex function

vertex VertexOut pbr_vertex(VertexIn in [[stage_in]],
                            constant FrameConstants &frame       [[buffer(vertexBufferFrameConstants)]],
                            constant InstanceConstants &instance [[buffer(vertexBufferInstanceConstants)]])
{
    VertexOut out{};

    out.pointSize = 1.0f;
    float4 worldPosition = instance.modelMatrix * getPosition(in);
    out.position = worldPosition.xyz / worldPosition.w;

    if (hasNormals) {
        if (hasTangents) {
            float4 modelTangent = getTangent(in);
            float3 normal = normalize(instance.normalMatrix * getNormal(in));
            float3 tangent = normalize(instance.normalMatrix * modelTangent.xyz);
            out.tangent = float4(tangent, modelTangent.w);
            out.normal = normal;
        } else {
            out.normal = normalize(instance.normalMatrix * getNormal(in));
        }
    }

    out.texCoords0 = float2(0.0f, 0.0f);
    if (hasTexCoords0) {
        out.texCoords0 = in.texCoords0;
    }

    out.texCoords1 = float2(0.0f, 0.0f);
    if (hasTexCoords1) {
        out.texCoords1 = in.texCoords1;
    }

    if (hasColors) {
        out.color = in.color;
    }

    out.clipPosition = frame.viewProjectionMatrix * worldPosition;
    return out;
}

#pragma mark - Primitive fragment function

typedef float4 FragmentOut;

fragment FragmentOut pbr_fragment(FragmentIn in                                             [[stage_in]],
                                  constant FrameConstants &frame                            [[buffer(fragmentBufferFrameConstants)]],
                                  constant InstanceConstants &instance                      [[buffer(fragmentBufferInstanceConstants)]],
                                  constant Material &material                               [[buffer(fragmentBufferMaterial)]],
                                  constant Light *lights                                    [[buffer(fragmentBufferLights)]],
                                  texturecube<float, access::sample> lambertEnvironmentMap  [[texture(fragmentTextureDiffuseEnvironment)]],
                                  texturecube<float, access::sample> specularEnvironmentMap [[texture(fragmentTextureSpecularEnvironment)]],
                                  texture2d<float, access::sample> GGXLUT                   [[texture(fragmentTextureGGXLookup)]])
{
    FragmentOut out;

    float4 baseColor = getBaseColor(in, material.constants, material.baseColorTexture, material.opacityTexture);
    if (alphaMode == (uint)AlphaMode::Opaque) {
        baseColor.a = 1.0f;
    }

    float3 V = normalize(frame.cameraPosition - in.position);
    TangentSpace tangentSpace = getTangentSpace(in, material.constants, material.normalTexture);
    float3 N = tangentSpace.N;

    FragmentMaterial fragmentMaterial = {
        .baseColor = baseColor,
    };
    populateMaterialParameters(in, tangentSpace, fragmentMaterial, material);

    fragmentMaterial.perceptualRoughness = clamp(fragmentMaterial.perceptualRoughness, 0.004f, 1.0f);
    fragmentMaterial.metalness = clamp(fragmentMaterial.metalness, 0.0f, 1.0f);
    fragmentMaterial.alphaRoughness = fragmentMaterial.perceptualRoughness * fragmentMaterial.perceptualRoughness;

    fragmentMaterial.F90 = float3(1.0f);

    float3 f_specular = float3(0.0);
    float3 f_diffuse = float3(0.0);
    float3 f_emissive = float3(0.0);

    EnvironmentParameters environment = {
        frame.environmentRotation,
        frame.environmentIntensity,
    };

    if (useIBL) {
        f_specular += getIBLRadianceGGX(specularEnvironmentMap, environment, GGXLUT,
                                        N, V, fragmentMaterial.perceptualRoughness, fragmentMaterial.F0,
                                        fragmentMaterial.specularWeight, frame.specularEnvironmentMipCount);
        f_diffuse += getIBLRadianceLambertian(lambertEnvironmentMap, environment, GGXLUT,
                                              N, V, fragmentMaterial.perceptualRoughness,
                                              fragmentMaterial.c_diff, fragmentMaterial.F0, fragmentMaterial.specularWeight);
    }
    float ao = 1.0;
    if (hasOcclusionMap) {
        ao = material.occlusionTexture.sample(trilinearSampler, getOcclusionUV(in)).r;
        f_diffuse = mix(f_diffuse, f_diffuse * ao, material.constants.occlusionStrength);
        f_specular = mix(f_specular, f_specular * ao, material.constants.occlusionStrength);
    }

    for (uint i = 0; i < frame.activeLightCount; ++i) {
        constant Light &light = lights[i];

        float3 pointToLight;
        if (light.type != (int)LightType::Directional) {
            pointToLight = light.position - in.position;
        } else {
            pointToLight = -light.direction;
        }

        float3 L = normalize(pointToLight);   // Direction from surface point to light
        float3 H = normalize(L + V);          // Direction of the vector between l and v, i.e., the halfway vector
        float NdotL = clampedDot(N, L);
        float NdotV = clampedDot(N, V);
        float NdotH = clampedDot(N, H);
        float VdotH = clampedDot(V, H);
        if (NdotL > 0.0f || NdotV > 0.0f)
        {
            float3 intensity = getLightIntensity(light, pointToLight);

            f_diffuse += intensity * NdotL *  BRDF_lambertian(fragmentMaterial.F0, fragmentMaterial.F90, fragmentMaterial.c_diff,
                                                              fragmentMaterial.specularWeight, VdotH);
            f_specular += intensity * NdotL * BRDF_specularGGX(fragmentMaterial.F0, fragmentMaterial.F90, fragmentMaterial.alphaRoughness,
                                                               fragmentMaterial.specularWeight, VdotH, NdotL, NdotV, NdotH);
        }
    }

    f_emissive = material.constants.emissiveFactor;

    if (hasEmissiveMap) {
        f_emissive *= material.emissiveTexture.sample(trilinearSampler, getEmissiveUV(in)).rgb;
    }

    float3 color = f_emissive;
    // In lieu of "unlit" materials, check whether this material
    // should accumulate diffuse or specular light, or just emissive
    if (any(baseColor.rgb > 0.f)) {
        color += f_diffuse + f_specular;
    }

    if (alphaMode == (uint)AlphaMode::Mask) {
        if (baseColor.a < material.constants.alphaCutoff) {
            discard_fragment();
        }
        baseColor.a = 1.0f;
    }

    out = float4(toneMap(color, 1.0f/*frame.exposure*/) * baseColor.a, baseColor.a);

    return out;
}
