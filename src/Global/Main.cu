#include <Global/Render.cuh>
using namespace renderer;

#undef main
int main(int argc, char * argv[]) {
    //结构体本身存放在普通内存中，结构体内部的指针指向页面锁定内存
    SceneGeometryData geometryDataWithPinPtr = {
            .sphereCount = 2,
            .parallelogramCount = 1
    };
    SceneMaterialData materialDataWithPinPtr = {
            .roughCount = 4,
            .metalCount = 1
    };
    Camera * pin_camera;
    Instance * pin_instances;
    const size_t instanceCount = 3;

    //让参数结构体的指针指向有效的页面锁定内存，并分配相机的页面锁定内存
    Renderer::mallocPinnedMemory(geometryDataWithPinPtr, materialDataWithPinPtr,
                                 pin_camera, pin_instances, instanceCount);

    //在分配好的相机页面锁定内存上写入相机基础信息
    pin_camera->windowWidth = 1200;
    pin_camera->windowHeight = 800;
    pin_camera->backgroundColor = {0.7, 0.8, 0.9};
    pin_camera->cameraCenter = {0.0, 2.0, 10.0};
    pin_camera->cameraTarget = {0.0, 2.0, 0.0};
    pin_camera->fov = 90;
    pin_camera->upDirection = {0.0, 1.0, 0.0};
    pin_camera->focusDiskRadius = 0.0;
    pin_camera->sampleRange = 0.5;
    pin_camera->sampleCount = 1;
    pin_camera->rayTraceDepth = 10;
    //计算相机剩余数据
    Renderer::constructCamera(pin_camera);

    //在分配好的几何页面锁定内存和材质页面锁定内存中写入几何体和材质信息
    //材质
    materialDataWithPinPtr.roughs[0] = {.65, .05, .05};
    materialDataWithPinPtr.roughs[1] = {.73, .73, .73};
    materialDataWithPinPtr.roughs[2] = {.12, .45, .15};
    materialDataWithPinPtr.roughs[3] = {.70, .60, .50};
    materialDataWithPinPtr.metals[0] = {0.8, 0.85, 0.88, 0.0};

    //物体
    geometryDataWithPinPtr.spheres[0] = {MaterialType::ROUGH, 3,
                                         {0.0, 0, 0.0}, 1000.0};
    geometryDataWithPinPtr.spheres[1] = {MaterialType::ROUGH, 0,
                                         {0.0, 0, 0.0}, 2.0};
    geometryDataWithPinPtr.parallelograms[0] = {MaterialType::ROUGH, 1,
                                                {0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 4.0, 0.0}};

    /*
     * 分配并拷贝几何体，材质信息到全局内存，获得指针指向有效全局内存的参数结构体
     * 此时相机对象存储于常量内存，物体和材质信息存储于GPU全局内存
     * geometryDataWithDevPtr，materialDataWithDevPtr和pin_camera作为指针，本身存储于普通内存
     */
    const auto sceneDataWithDevPtr = Renderer::copyToGlobalMemory(
            geometryDataWithPinPtr, materialDataWithPinPtr, pin_camera);
    SceneGeometryData geometryDataWithDevPtr = sceneDataWithDevPtr.first;
    SceneMaterialData materialDataWithDevPtr = sceneDataWithDevPtr.second;

    //构建实例列表
    pin_instances[0] = Instance(PrimitiveType::SPHERE, 0,
                                std::array<double, 3>{},
                                std::array<double, 3>{0.0, -1000.0, 0.0},
                                std::array<double, 3>{1.0, 1.0, 1.0});
    pin_instances[1] = Instance(PrimitiveType::SPHERE, 1,
                                std::array<double, 3>{},
                                std::array<double, 3>{0.0, 2.0, 0.0},
                                std::array<double, 3>{1.0, 1.0, 1.0});
    pin_instances[2] = Instance(PrimitiveType::PARALLELOGRAM, 0,
                                std::array<double, 3>{0.0, 0.0, 0.0},
                                std::array<double, 3>{-5.0, 0.0, 0.0},
                                std::array<double, 3>{1.0, 1.0, 1.0});

    //构建加速结构
    const auto asBuildResult = Renderer::buildAccelerationStructure(geometryDataWithPinPtr, pin_instances, instanceCount);
    auto asTraverseData = Renderer::copyAccelerationStructureToGlobalMemory(asBuildResult, pin_instances, instanceCount);

    //启动渲染
    Renderer::renderLoop(
            geometryDataWithDevPtr, materialDataWithDevPtr, pin_camera, asTraverseData);

    //释放加速结构内存
    Renderer::freeAccelerationStructureGlobalMemory(asTraverseData);

    //释放全局内存和页面锁定内存
    Renderer::freeGlobalMemory(geometryDataWithDevPtr, materialDataWithDevPtr);
    Renderer::freePinnedMemory(geometryDataWithPinPtr, materialDataWithPinPtr, pin_camera, pin_instances);

    return 0;
}
