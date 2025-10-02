#ifndef RENDERERINTERACTIVE_BASICTYPES_CUH
#define RENDERERINTERACTIVE_BASICTYPES_CUH

#include <Basic/Point3.cuh>
#include <Util/Pair.cuh>

namespace renderer {
    //图元类型枚举
    typedef enum class PrimitiveType {
        SPHERE, PARALLELOGRAM
    } PrimitiveType;

    //材质类型枚举
    typedef enum class MaterialType {
        ROUGH, METAL
    } MaterialType;

    //碰撞信息
    typedef struct HitRecord {
        Point3 hitPoint;
        Vec3 normalVector;
        double t;
        bool hitFrontFace;

        //材质索引
        MaterialType materialType;
        size_t materialIndex;

        //纹理坐标
        Pair<double, double> uvPair;
    } HitRecord;
}

#endif //RENDERERINTERACTIVE_BASICTYPES_CUH
