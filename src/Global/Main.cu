#include <Global/Render.cuh>
using namespace renderer;

#undef main
int main(int argc, char * argv[]) {
    //结构体本身存放在普通内存中，结构体内部的指针指向页面锁定内存
    SceneGeometryData geometryData = {
            .sphereCount = 2,
            .parallelogramCount = 1
    };
    SceneMaterialData materialData = {
            .roughCount = 4,
            .metalCount = 1
    };

    //让参数结构体的指针指向有效的页面锁定内存，并分配相机的页面锁定内存
    Camera * pin_camera;
    Renderer::mallocPinnedMemory(geometryData, materialData, pin_camera);

    //在普通内存中初始化相机，完成对相机参数的计算，然后拷贝到页面锁定内存
    Camera hos_camera{
        .windowWidth = 900, .windowHeight = 600, .backgroundColor = {0.7, 0.8, 0.9},
        .cameraCenter = {0.0, 2.0, 10.0}, .cameraTarget = {0.0, 2.0, 0.0},
        .fov = 90, .upDirection = {0.0, 1.0, 0.0},
        .focusDiskRadius = 0.0, .sampleRange = 0.5, .sampleCount = 1, .rayTraceDepth = 10
    };
    Renderer::constructCamera(hos_camera);
    memcpy(pin_camera, &hos_camera, sizeof(Camera));

    //在页面锁定内存中写入几何体和材质信息
    //材质
    materialData.roughs[0] = {.65, .05, .05};
    materialData.roughs[1] = {.73, .73, .73};
    materialData.roughs[2] = {.12, .45, .15};
    materialData.roughs[3] = {.70, .60, .50};
    materialData.metals[0] = {0.8, 0.85, 0.88, 0.0};

    //物体
    geometryData.spheres[0] = {MaterialType::ROUGH, 3, {0.0, -1000.0, 0.0}, 1000.0};
    geometryData.spheres[1] = {MaterialType::ROUGH, 0, {0.0, 4.0, 0.0}, 2.0};
    geometryData.parallelograms[0] = {MaterialType::ROUGH, 1, {-4.0, 0.0, 0.0}, {0.5, 0.0, -1.0}, {0.0, 4.0, 0.0}};

    /*
     * 分配并拷贝几何体，材质信息到全局内存，获得指针指向有效全局内存的参数结构体
     * 此时相机对象存储于页面锁定内存，物体和材质信息存储于GPU全局内存
     * geometryDataWithDevPtr，materialDataWithDevPtr和pin_camera作为指针，本身存储于普通内存
     */
    auto dev_structs = Renderer::copyToGlobalMemory(geometryData, materialData, pin_camera);
    SceneGeometryData geometryDataWithDevPtr = dev_structs.first;
    SceneMaterialData materialDataWithDevPtr = dev_structs.second;
    Renderer::renderLoop(geometryDataWithDevPtr, materialDataWithDevPtr, pin_camera);

    Renderer::freeGlobalMemory(geometryDataWithDevPtr, materialDataWithDevPtr);
    Renderer::freePinnedMemory(geometryData, materialData, pin_camera);
    return 0;
}
