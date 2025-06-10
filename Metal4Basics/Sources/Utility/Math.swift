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

import simd
import Spatial

func gcd(_ x: Int, _ y: Int) -> Int {
    var a = 0
    var b = max(x, y)
    var r = min(x, y)

    while r != 0 {
        a = b
        b = r
        r = a % b
    }
    return b
}

func lcm(_ x: Int, _ y: Int) -> Int {
    return x / gcd(x, y) * y
}

extension SIMD4<Float> {
    var xyz: SIMD3<Float> {
        return SIMD3<Float>(x, y, z)
    }
}

extension simd_float3x3 {
    var adjugate: simd_float3x3 {
        return simd_float3x3(rows: [
            cross(columns.1, columns.2),
            cross(columns.2, columns.0),
            cross(columns.0, columns.1)
        ])
    }
}

extension simd_float4x4 {
    var upperLeft3x3: simd_float3x3 {
        return simd_float3x3(
            columns.0.xyz,
            columns.1.xyz,
            columns.2.xyz)
    }
}

extension AffineTransform3D {
    init(lookAt target: Point3D, from: Point3D, up: Vector3D) {
        let zNeg = normalize(target.vector - from.vector)
        let x = normalize(cross(zNeg, up.vector))
        let y = normalize(cross(x, zNeg))
        let M = simd_double4x4(SIMD4<Double>(x, 0.0),
                               SIMD4<Double>(y, 0.0),
                               SIMD4<Double>(-zNeg, 0.0),
                               SIMD4<Double>(from.vector, 1.0))
        self.init(truncating: M)
    }
}
