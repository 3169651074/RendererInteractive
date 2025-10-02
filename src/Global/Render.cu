#include <Global/Render.cuh>

namespace renderer {
    //核函数声明
    __global__ void render(
            SceneGeometryData * dev_geometryData,
            SceneMaterialData * dev_materialData,
            cudaSurfaceObject_t surfaceObject);

    //分配页面锁定内存
    /*
     * cudaHostAllocWriteCombined标志：写合并
     * CPU读取这块内存的性能差，仅适用于CPU主要负责写入、GPU主要负责读取的场景
     */
#define _mallocHost(structName, className, arrayName, countName) \
    do {                                    \
        if (structName.countName != 0) {    \
            cudaCheckError(cudaHostAlloc(&structName.arrayName, structName.countName * sizeof(className), cudaHostAllocWriteCombined));\
        }                                   \
    } while(false)
    void Renderer::mallocPinnedMemory(SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & cameraData) {
        _mallocHost(geometryData, Sphere, spheres, sphereCount);
        _mallocHost(geometryData, Parallelogram, parallelograms, parallelogramCount);

        _mallocHost(materialData, Rough, roughs, roughCount);
        _mallocHost(materialData, Metal, metals, metalCount);

        //分配相机空间
        cudaCheckError(cudaHostAlloc(&cameraData, sizeof(Camera), cudaHostAllocWriteCombined));
    }

    //释放页面锁定内存
#define _freeHost(structName, arrayName) cudaCheckError(cudaFreeHost(structName.arrayName))
    void Renderer::freePinnedMemory(SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & cameraData) {
        _freeHost(geometryData, spheres);
        _freeHost(geometryData, parallelograms);

        _freeHost(materialData, roughs);
        _freeHost(materialData, metals);

        //释放相机空间
        cudaCheckError(cudaFreeHost(cameraData));
    }

    //分配全局内存并拷贝数据
#define _mallocGlobalAndCopy(srcStructName, dstStructName, arrayName, countName, className)\
        do {                                        \
            if (srcStructName.countName != 0) {     \
                cudaCheckError(cudaMalloc(&dstStructName.arrayName, srcStructName.countName * sizeof(className)));\
                cudaCheckError(cudaMemcpyAsync(dstStructName.arrayName, srcStructName.arrayName, srcStructName.countName * sizeof(className), cudaMemcpyHostToDevice));\
            }                                       \
        } while (false)
    Pair<SceneGeometryData, SceneMaterialData> Renderer::copyToGlobalMemory(
            const SceneGeometryData & geometryData, const SceneMaterialData & materialData, const Camera * cameraData)
    {
        //拷贝数组长度信息
        SceneGeometryData geometryDataWithDevPtr = geometryData;
        SceneMaterialData materialDataWithDevPtr = materialData;

        _mallocGlobalAndCopy(geometryData, geometryDataWithDevPtr, spheres, sphereCount, Sphere);
        _mallocGlobalAndCopy(geometryData, geometryDataWithDevPtr, parallelograms, parallelogramCount, Parallelogram);

        _mallocGlobalAndCopy(materialData, materialDataWithDevPtr, roughs, roughCount, Rough);
        _mallocGlobalAndCopy(materialData, materialDataWithDevPtr, metals, metalCount, Metal);

        //拷贝相机到常量内存
        cudaCheckError(cudaMemcpyToSymbolAsync(dev_camera, cameraData, sizeof(Camera)));

        return {geometryDataWithDevPtr, materialDataWithDevPtr};
    }

    //释放全局内存，无需释放常量内存
#define _freeGlobal(structName, arrayName) cudaCheckError(cudaFree(structName.arrayName))
    void Renderer::freeGlobalMemory(SceneGeometryData & geometryDataWithDevPtr, SceneMaterialData & materialDataWithDevPtr) {
        _freeGlobal(geometryDataWithDevPtr, spheres);
        _freeGlobal(geometryDataWithDevPtr, parallelograms);

        _freeGlobal(materialDataWithDevPtr, roughs);
        _freeGlobal(materialDataWithDevPtr, metals);
    }

    void Renderer::renderLoop(const SceneGeometryData & geometryDataWithDevPtr, const SceneMaterialData & materialDataWithDevPtr, const Camera * cameraData) {
        const int w = cameraData->windowWidth;
        const int h = cameraData->windowHeight;

        //SDL
        SDL_Window * window = SDL_CreateWindow(
                "Test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                w, h, SDL_WINDOW_OPENGL);
        SDL_GLContext context = SDL_GL_CreateContext(window);

        //OGL
        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
            SDL_Log("Failed to init GLAD!"); return;
        }
        glViewport(0, 0, w, h);
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        constexpr float vertices[] = {
                -1.0f,  1.0f,  0.0f, 1.0f,
                -1.0f, -1.0f,  0.0f, 0.0f,
                1.0f, -1.0f,  1.0f, 0.0f,
                1.0f,  1.0f,  1.0f, 1.0f
        };
        constexpr GLuint indices[] = {
                0, 1, 2,
                0, 2, 3
        };
        GLuint VAO, VBO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        constexpr const char* vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        constexpr const char* fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D ourTexture;
            void main() {
                FragColor = texture(ourTexture, TexCoord);
            }
        )";
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);
        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        //CUDA
        cudaGraphicsResource_t cudaResource;
        cudaCheckError(cudaGraphicsGLRegisterImage(
                &cudaResource, textureID, GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsWriteDiscard));

        // ====== 启动参数 ======
        //将结构体本身拷贝到全局内存
        SceneGeometryData * dev_geometryData;
        SceneMaterialData * dev_materialData;
        cudaCheckError(cudaMalloc(&dev_geometryData, sizeof(SceneGeometryData)));
        cudaCheckError(cudaMalloc(&dev_materialData, sizeof(SceneMaterialData)));
        cudaCheckError(cudaMemcpyAsync(dev_geometryData, &geometryDataWithDevPtr, sizeof(SceneGeometryData), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(dev_materialData, &materialDataWithDevPtr, sizeof(SceneMaterialData), cudaMemcpyHostToDevice));

        const dim3 blocks(w % 16 == 0 ? w / 16 : w / 16 + 1,
                          h % 16 == 0 ? h / 16 : h / 16 + 1, 1);
        const dim3 threads(16, 16, 1);

        // ====== 渲染循环 ======
        bool quit = false;
        SDL_Event event;

        while (!quit) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    quit = true;
                    break;
                }
            }

            //a. 映射资源，让CUDA接管纹理
            cudaCheckError(cudaGraphicsMapResources(1, &cudaResource, nullptr));
            //b. 获取指向纹理的CUDA数组
            cudaArray_t cudaTextureArray;
            cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaResource, 0, 0));
            //c. 为CUDA数组创建一个 Surface Object，以便核函数写入
            cudaResourceDesc resDesc {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cudaTextureArray;
            cudaSurfaceObject_t surfaceObject;
            cudaCheckError(cudaCreateSurfaceObject(&surfaceObject, &resDesc));

            //d. 启动核函数
            render<<<blocks, threads>>>(dev_geometryData, dev_materialData, surfaceObject);
            cudaCheckError(cudaDeviceSynchronize());

            //e. 销毁 Surface Object
            cudaCheckError(cudaDestroySurfaceObject(surfaceObject));
            //f. 解除映射，将纹理控制权还给OpenGL
            cudaCheckError(cudaGraphicsUnmapResources(1, &cudaResource, nullptr));
            //g. OpenGL渲染并显示到SDL窗口
            glUseProgram(shaderProgram);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
            SDL_GL_SwapWindow(window);
        }

        //释放参数结构体
        cudaCheckError(cudaFree(dev_geometryData));
        cudaCheckError(cudaFree(dev_materialData));

        // ====== 清理资源 ======
        //~CUDA
        cudaCheckError(cudaGraphicsUnregisterResource(cudaResource));

        //~OGL
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(shaderProgram);
        glDeleteTextures(1, &textureID);

        //~SDL
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
    }

    void Renderer::constructCamera(Camera & hos_cameraData) {
        //需要使用引用，否则cam为拷贝的副本
        Camera & cam = hos_cameraData;

        cam.focusDistance = cam.cameraCenter.distance(cam.cameraTarget);
        const double thetaFOV = MathHelper::degreeToRadian(cam.fov);
        const double vWidth = 2.0 * tan(thetaFOV / 2.0) * cam.focusDistance;
        const double vHeight = vWidth / (cam.windowWidth * 1.0 / cam.windowHeight);

        cam.viewPortWidth = vWidth;
        cam.viewPortHeight = vHeight;
        cam.cameraW = Point3::constructVector(cam.cameraCenter, cam.cameraTarget).unitVector();
        cam.cameraU = Vec3::cross(cam.cameraW, cam.upDirection).unitVector();
        cam.cameraV = Vec3::cross(cam.cameraU, cam.cameraW).unitVector();

        cam.viewPortX = vWidth * cam.cameraU;
        cam.viewPortY = vHeight * cam.cameraV;
        cam.viewPortPixelDx = cam.viewPortX / cam.windowWidth;
        cam.viewPortPixelDy = cam.viewPortY / cam.windowHeight;
        cam.viewPortOrigin = cam.cameraCenter + cam.focusDistance * cam.cameraW - cam.viewPortX * 0.5 - cam.viewPortY * 0.5;
        cam.pixelOrigin = cam.viewPortOrigin + cam.viewPortPixelDx * 0.5 + cam.viewPortPixelDy * 0.5;
        cam.sqrtSampleCount = static_cast<size_t>(std::sqrt(cam.sampleCount));
        cam.reciprocalSqrtSampleCount = 1.0 / static_cast<double>(cam.sqrtSampleCount);
    }
}