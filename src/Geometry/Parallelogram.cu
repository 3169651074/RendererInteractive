#include <Geometry/Parallelogram.cuh>

namespace renderer {
    __device__ bool Parallelogram::hit(const Ray & ray, const Range & range, HitRecord & record) const {
        const double NDotD = Vec3::dot(normalVector, ray.direction);
        if (MathHelper::floatValueNearZero(NDotD)) {
            return false;
        }

        //计算光线和四边形所在无限平面的交点参数t
        double NDotP = 0.0;
        for (int i = 0; i < 3; i++) {
            NDotP += normalVector[i] * ray.origin[i];
        }
        const double t = (d - NDotP) / NDotD;
        if (!range.inRange(t)) {
            return false;
        }

        //计算用四边形边向量表示的交点的系数，判断两个系数是否在[0, 1]范围内
        const Point3 intersection = ray.at(t);
        const Vec3 p = Point3::constructVector(q, intersection);
        const Vec3 normal = Vec3::cross(u, v);
        const double denominator = normal.lengthSquared();

        if (MathHelper::floatValueNearZero(denominator)) {
            return false;
        }
        const double alpha = Vec3::dot(Vec3::cross(p, v), normal) / denominator;
        const double beta = Vec3::dot(Vec3::cross(u, p), normal) / denominator;

        static constexpr Range coefficientRange{0.0, 1.0};
        if (!coefficientRange.inRange(alpha) || !coefficientRange.inRange(beta)) {
            return false;
        }

        //记录碰撞信息
        record.t = t;
        record.hitPoint = intersection;
        record.materialType = materialType;
        record.materialIndex = materialIndex;
        record.hitFrontFace = Vec3::dot(ray.direction, normalVector) < 0.0;
        record.normalVector = record.hitFrontFace ? normalVector : -normalVector;
        record.uvPair = {alpha, beta};
        return true;
    }

    BoundingBox Parallelogram::constructBoundingBox() const {
        return {q + 0.5 * (u + v), q - 0.5 * (u + v)};
    }
}