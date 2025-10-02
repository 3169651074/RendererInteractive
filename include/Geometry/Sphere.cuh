#ifndef RENDERERINTERACTIVE_SPHERE_CUH
#define RENDERERINTERACTIVE_SPHERE_CUH

#include <AS/BoundingBox.cuh>

namespace renderer {
    /*
     * 球体类，普通类型（非聚合类型）
     * 对象操作：
     *   碰撞检测
     *   获取几何中心
     *   构造包围盒
     */
    class Sphere {
    public:
        //球心和半径
        Point3 center;
        double radius;

        //材质索引
        MaterialType materialType;
        size_t materialIndex;

        Sphere(MaterialType materialType, size_t materialIndex, const Point3 & center, double radius)
        : center(center), radius(radius), materialType(materialType), materialIndex(materialIndex) {}

        //碰撞检测
        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & record) const;

        //构造包围盒
        BoundingBox constructBoundingBox() const;

        //获取几何中心
        Point3 centroid() const {
            return center;
        }
    };
}

#endif //RENDERERINTERACTIVE_SPHERE_CUH
