#ifndef RENDERERINTERACTIVE_RENDER_CUH
#define RENDERERINTERACTIVE_RENDER_CUH

#include <Geometry/Sphere.cuh>
#include <Geometry/Parallelogram.cuh>

#include <Material/Rough.cuh>
#include <Material/Metal.cuh>

namespace renderer {
    //几何体数据
    typedef struct SceneGeometryData {
        Sphere * spheres;
        size_t sphereCount;

        Parallelogram * parallelograms;
        size_t parallelogramCount;
    } SceneGeometryData;

    //材质数据
    typedef struct SceneMaterialData {
        Rough * roughs;
        size_t roughCount;

        Metal * metals;
        size_t metalCount;
    } SceneMaterialData;

    //相机数据
    typedef struct Camera {
        //窗口尺寸，简化访问
        int windowWidth;
        int windowHeight;

        //相机
        Color3 backgroundColor;
        Point3 cameraCenter;
        Point3 cameraTarget;
        double fov;

        //视口
        Vec3 upDirection;
        double viewPortWidth;
        double viewPortHeight;
        Vec3 cameraU, cameraV, cameraW;
        Vec3 viewPortX, viewPortY;
        Vec3 viewPortPixelDx, viewPortPixelDy;
        Point3 viewPortOrigin;
        Point3 pixelOrigin;

        //采样
        double focusDiskRadius;
        double focusDistance;
        double sampleRange;
        size_t sampleCount;
        size_t sqrtSampleCount;
        double reciprocalSqrtSampleCount;
        size_t rayTraceDepth;
    } Camera;

    /*
     * 初始化：
     * 分配存放材质，几何体数据和两层加速结构的页面锁定内存
     * 在页面锁定内存中初始化材质，几何体，相机
     * 构建底层加速结构和顶层加速结构，计算几何体变换矩阵
     * 分配全局内存，初始化流
     * 拷贝材质，几何体和两层加速结构数据到全局内存
     * 拷贝相机到常量内存
     * 启动渲染核函数
     * 显示结果
     *
     * 更新：
     * CPU：更新几何体，相机数据
     * CPU：基于新的几何体数据计算新的变换矩阵，更新顶层加速结构
     * CUDA流1：拷贝相机和新的顶层加速结构到全局内存
     * CUDA流2：启动渲染核函数并显示结果
     * 流1和流2异步执行
     *
     * 清理：
     * 释放全局内存
     * 释放页面锁定内存
     * 释放其他资源
     */
    class Renderer {
    public:
        //分配页面锁定内存：传入的结构体需要设置每种几何体个数，函数将结构体的指针指向有效内存地址
        static void mallocPinnedMemory(SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & cameraData);

        //释放页面锁定内存
        static void freePinnedMemory(SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & cameraData);

        //计算相机参数
        static void constructCamera(Camera & hos_cameraData);

        /*
         * 分配全局内存，将场景数据拷贝到全局内存，将相机拷贝到常量内存
         * 传入初始化后的场景数据结构体，结构体内指针指向有效的页面锁定内存
         * 返回新的场景数据结构体，结构体内指针指向全局内存
         */
        static Pair<SceneGeometryData, SceneMaterialData> copyToGlobalMemory(const SceneGeometryData & geometryData, const SceneMaterialData & materialData, const Camera * cameraData);

        //释放全局内存
        static void freeGlobalMemory(SceneGeometryData & geometryDataWithDevPtr, SceneMaterialData & materialDataWithDevPtr);

        //渲染循环
        static void renderLoop(const SceneGeometryData & geometryDataWithDevPtr, const SceneMaterialData & materialDataWithDevPtr, const Camera * cameraData);
    };

    extern __constant__ Camera dev_camera[1];
}

#endif //RENDERERINTERACTIVE_RENDER_CUH
