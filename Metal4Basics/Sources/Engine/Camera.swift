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

class PerspectiveCamera {
    var position: SIMD3<Double> = [0, 0, 0]
    var orientation: simd_quatd = simd_quatd(ix: 0.0, iy: 0.0, iz: 0.0, r: 1.0)
    var fieldOfView: Angle2D = .degrees(60)
    var nearZ = 0.1
    var farZ = 1000.0

    func setTransform(_ transform: AffineTransform3D) {
        self.position = transform.translation.vector
        self.orientation = transform.rotation!.quaternion
    }

    var transform: simd_float4x4 {
        let affineTransform = AffineTransform3D(rotation: Rotation3D(orientation),
                                                translation: Vector3D(vector: position))
        return simd_float4x4(affineTransform)
    }

    func viewMatrix() -> simd_float4x4 {
        let affineTransform = AffineTransform3D(rotation: Rotation3D(orientation),
                                                translation: Vector3D(vector: position))
        return simd_float4x4(affineTransform.inverse!)
    }

    func projectionMatrix(aspectRatio: Double) -> simd_float4x4 {
        let projectionTransform = ProjectiveTransform3D(fovY: fieldOfView,
                                                        aspectRatio: aspectRatio,
                                                        nearZ: nearZ,
                                                        farZ: farZ)
        return simd_float4x4(projectionTransform)
    }
}
