// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_AGGREGATES_H
#define PBRT_CPU_AGGREGATES_H

#include <pbrt/pbrt.h>

#include <pbrt/cpu/primitive.h>
#include <pbrt/util/parallel.h>

#include <atomic>
#include <memory>
#include <vector>

namespace pbrt {

Primitive CreateAccelerator(std::vector<Primitive> prims);

Primitive CreateAccelerator(const std::string &name, std::vector<Primitive> prims,
                            const ParameterDictionary &parameters);

struct BVHBuildNode;
struct WideBVHBuildNode;
struct FourWideBVHBuildNode;
struct EightWideBVHBuildNode;
struct SixteenWideBVHBuildNode;
struct BVHPrimitive;
struct BVHSplitBucket;
struct LinearBVHNode;
struct WideLinearBVHNode;
struct SIMDFourWideLinearBVHNode;
struct SIMDEightWideLinearBVHNode;
struct SIMDSixteenWideLinearBVHNode;
struct MortonPrimitive;

struct ICostCalcable {
    virtual Bounds3f Bounds() const = 0;
    virtual int PrimCount() const = 0;
};

class BVHAggregate {
  public:
    enum SplitMethod : int {
        Undefined = -1,
        SAH = 1,
        EPO = 2,
        HLBVH = 3,
        Middle = 4,
        EqualCounts = 5
    };
    enum CreationMethod : char { UndefinedMethod = -1, Direct = 0, FromBVH = 1 };
    enum OptimizationStrategy : char {
        UndefinedStrat = -1,
        None = 0,
        MergeIntoParent = 1,
        MergeInnerChildren = 2,
        MergeLeaves = 4,
        All = 0x7F
    };
    static int WidthOverride;
    static Float epoRatioOverride;
    static int maxPrimsInNodeOverride;
    static int splitVariantOverride;
    static int collapseVariantOverride;
    static std::string splitMethodOverride;
    static std::string creationMethodOverride;
    static std::string optimizationStrategyOverride;
    static int SimdWidth;
    static constexpr Float RelativeInnerCost = 1.f;
    virtual Bounds3f Bounds() const = 0;
    virtual pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const = 0;
    virtual bool IntersectP(const Ray &ray, Float tMax) const = 0;
    bool metricsEnabled;
  protected:
    BVHAggregate(int maxPrimsInNode, std::vector<Primitive> p, SplitMethod splitMethod,
                 Float epoRatio);
    int maxPrimsInNode;
    std::vector<Primitive> primitives;
    SplitMethod splitMethod;
    Float epoRatio;
    Float penalizedHitProbability(int idx, int count, ICostCalcable **buckets, Bounds3f parent) const;
    Float childCost(int idx, int count, ICostCalcable **buckets, Bounds3f parent) const;
    Float splitCost(int count, ICostCalcable **buckets, Bounds3f parent) const;
    Float inline leafCost(int primCount) const;
};
class WideBVHAggregate : public BVHAggregate {
  public:
    Bounds3f Bounds() const override;
  protected:
    WideBVHAggregate(int TreeWidth, int maxPrimsInNode, std::vector<Primitive> p, SplitMethod splitMethod,
                 Float epoRatio);
    static int constexpr MaxTreeWidth = 16;
    int const TreeWidth;
    virtual int8_t relevantAxisIdx(int axis1, int axis2) const = 0;
    bool optimizeTree(WideBVHBuildNode *root, std::atomic<int> *totalNodes,
                      OptimizationStrategy strat = OptimizationStrategy::All);
    Float calcMetrics(WideBVHBuildNode *root, int depth);
    Bounds3f bounds;
};

class FourWideBVHAggregate : public WideBVHAggregate {
  public:
    FourWideBVHAggregate(std::vector<Primitive> p, int maxPrimsInNode = 1, 
        SplitMethod splitMethod = SplitMethod::SAH, Float epoRatio = 0.5f,
                     int splitVariant = 0, 
        CreationMethod method = CreationMethod::Direct, 
        OptimizationStrategy = OptimizationStrategy::All, int collapseVariant = 0);
    static FourWideBVHAggregate *Create(std::vector<Primitive> prims,
                                const ParameterDictionary &parameters);
    
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const override;
    bool IntersectP(const Ray &ray, Float tMax) const override;

  protected:
    int8_t relevantAxisIdx(int axis1, int axis2) const override;
  private:
    // FourWideBVHAggregate Private Methods
    FourWideBVHBuildNode *buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                 pstd::span<BVHPrimitive> bvhPrimitives,
                                 std::atomic<int> *totalNodes,
                                 std::atomic<int> *orderedPrimsOffset,
                                 std::vector<Primitive> &orderedPrims,
                                 int splitVariant);
    int flattenBVH(FourWideBVHBuildNode *node, int *offset);
    FourWideBVHBuildNode *collapseBinBVH(BVHBuildNode *root, std::atomic<int> *totalNodes,
                                       int collapseVariant);
    // FourWideBVHAggregate Private Members
    
    
    SIMDFourWideLinearBVHNode *nodes = nullptr;
    //pre computed traversal order for all combinations of axis being negative
    static constexpr uint8_t traversalOrder[2][2][2][4] = {
        {{{0, 1, 2, 3}, {0, 1, 3, 2}}, {{2, 3, 0, 1}, {2, 3, 1, 0}}},
        {{{1, 0, 2, 3}, {1, 0, 3, 2}}, {{3, 2, 0, 1}, {3, 2, 1, 0}}}};
    //pre computed index of the most relevant split axis between two child nodes
    static constexpr int8_t relevantAxisIdxArr[4][4] = {
        {-1, 0, 1, 1},
        { 0,-1, 1, 1},
        { 1, 1,-1, 2},
        { 1, 1, 2,-1},
    };
};
class EightWideBVHAggregate : public WideBVHAggregate {
  public:
    EightWideBVHAggregate(std::vector<Primitive> p, int maxPrimsInNode = 1,
                     SplitMethod splitMethod = SplitMethod::SAH, Float epoRatio = 0.5f,
                     int splitVariant = 0, CreationMethod method = CreationMethod::Direct,
                          OptimizationStrategy = OptimizationStrategy::All,
                          int collapseVariant = 0);
    
    static EightWideBVHAggregate *Create(std::vector<Primitive> prims,
                                         const ParameterDictionary &parameters);
    
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax) const override;
    bool IntersectP(const Ray &ray, Float tMax) const override;
    
  protected:
    int8_t relevantAxisIdx(int a1, int a2) const override {
        return relevantAxisIdxArr[a1][a2];
    }
  private:
    // FourWideBVHAggregate Private Methods
    EightWideBVHBuildNode *buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                     pstd::span<BVHPrimitive> bvhPrimitives,
                                     std::atomic<int> *totalNodes,
                                     std::atomic<int> *orderedPrimsOffset,
                                     std::vector<Primitive> &orderedPrims,
                                     int splitVariant);
    EightWideBVHBuildNode *collapseBinBVH(BVHBuildNode *root, std::atomic<int> *totalNodes, int collapseVariant);
    int flattenBVH(EightWideBVHBuildNode *node, int *offset);
    static int traversalIdx(int dirIsNeg[3],const SIMDEightWideLinearBVHNode *node, int idx);
    // FourWideBVHAggregate Private Members
    SIMDEightWideLinearBVHNode *nodes = nullptr;
    // pre computed index of the most relevant split axis between two child nodes
    static constexpr int8_t relevantAxisIdxArr[8][8] = {
        {-1, 0, 1, 1, 3, 3, 3, 3},
        { 0,-1, 1, 1, 3, 3, 3, 3},
        { 1, 1,-1, 2, 3, 3, 3, 3},
        { 1, 1, 2,-1, 3, 3, 3, 3},
        { 3, 3, 3, 1,-1, 4, 5, 5},
        { 3, 3, 3, 1, 4,-1, 5, 5},
        { 3, 3, 3, 1, 5, 5,-1, 6},
        { 3, 3, 3, 1, 5, 5, 6,-1},
    };
};
class SixteenWideBVHAggregate : public WideBVHAggregate {
  public:
    SixteenWideBVHAggregate(std::vector<Primitive> p, int maxPrimsInNode = 1,
                          SplitMethod splitMethod = SplitMethod::SAH,
                          Float epoRatio = 0.5f, int splitVariant = 0,
                          CreationMethod method = CreationMethod::Direct,
                          OptimizationStrategy = OptimizationStrategy::All,
                          int collapseVariant = 0);

    static SixteenWideBVHAggregate *Create(std::vector<Primitive> prims,
                                         const ParameterDictionary &parameters);

    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax) const override;
    bool IntersectP(const Ray &ray, Float tMax) const override;

  protected:
    int8_t relevantAxisIdx(int a1, int a2) const override;

  private:
    // FourWideBVHAggregate Private Methods
    SixteenWideBVHBuildNode *buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                          pstd::span<BVHPrimitive> bvhPrimitives,
                                          std::atomic<int> *totalNodes,
                                          std::atomic<int> *orderedPrimsOffset,
                                          std::vector<Primitive> &orderedPrims,
                                          int splitVariant);
    SixteenWideBVHBuildNode *collapseBinBVH(BVHBuildNode *root,
                                          std::atomic<int> *totalNodes,
                                          int collapseVariant);
    int flattenBVH(SixteenWideBVHBuildNode *node, int *offset);
    static int traversalIdx(int dirIsNeg[3], const SIMDSixteenWideLinearBVHNode *node,
                            int idx);
    // FourWideBVHAggregate Private Members
    SIMDSixteenWideLinearBVHNode *nodes = nullptr;
    // pre computed index of the most relevant split axis between two child nodes
};
// BinBVHAggregate Definition
class BinBVHAggregate : public BVHAggregate {
  public:
    // BinBVHAggregate Public Methods
    BinBVHAggregate(std::vector<Primitive> p, int maxPrimsInNode = 1,
                    SplitMethod splitMethod = SplitMethod::SAH,
                    Float epoRatio = 0.5f, bool skipCreation = false);

    static BinBVHAggregate *Create(std::vector<Primitive> prims,
                                const ParameterDictionary &parameters);

    Bounds3f Bounds() const override;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const override;
    bool IntersectP(const Ray &ray, Float tMax) const override;
    BVHBuildNode *buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                 pstd::span<BVHPrimitive> bvhPrimitives,
                                 std::atomic<int> *totalNodes,
                                 std::atomic<int> *orderedPrimsOffset,
                                 std::vector<Primitive> &orderedPrims);
  private:
    // BinBVHAggregate Private Methods
    BVHBuildNode *buildHLBVH(Allocator alloc,
                             const std::vector<BVHPrimitive> &primitiveInfo,
                             std::atomic<int> *totalNodes,
                             std::vector<Primitive> &orderedPrims);
    BVHBuildNode *emitLBVH(BVHBuildNode *&buildNodes,
                           const std::vector<BVHPrimitive> &primitiveInfo,
                           MortonPrimitive *mortonPrims, int nPrimitives, int *totalNodes,
                           std::vector<Primitive> &orderedPrims,
                           std::atomic<int> *orderedPrimsOffset, int bitIndex);
    BVHBuildNode *buildUpperSAH(Allocator alloc,
                                std::vector<BVHBuildNode *> &treeletRoots, int start,
                                int end, std::atomic<int> *totalNodes) const;
    int flattenBVH(BVHBuildNode *node, int *offset);
    Float splitCost(BVHSplitBucket left, BVHSplitBucket right, Bounds3f bounds) const;
    Float calcMetrics(BVHBuildNode *root) const;
    // BinBVHAggregate Private Members
    LinearBVHNode *nodes = nullptr;
};

struct KdTreeNode;
struct BoundEdge;

// KdTreeAggregate Definition
class KdTreeAggregate {
  public:
    // KdTreeAggregate Public Methods
    KdTreeAggregate(std::vector<Primitive> p, int isectCost = 5, int traversalCost = 1,
                    Float emptyBonus = 0.5, int maxPrims = 1, int maxDepth = -1);
    static KdTreeAggregate *Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters);
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;

    Bounds3f Bounds() const { return bounds; }

    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    // KdTreeAggregate Private Methods
    void buildTree(int nodeNum, const Bounds3f &bounds,
                   const std::vector<Bounds3f> &primBounds,
                   pstd::span<const int> primNums, int depth,
                   std::vector<BoundEdge> edges[3], pstd::span<int> prims0,
                   pstd::span<int> prims1, int badRefines);

    // KdTreeAggregate Private Members
    int isectCost, traversalCost, maxPrims;
    Float emptyBonus;
    std::vector<Primitive> primitives;
    std::vector<int> primitiveIndices;
    KdTreeNode *nodes;
    int nAllocedNodes, nextFreeNode;
    Bounds3f bounds;
};

}  // namespace pbrt

#endif  // PBRT_CPU_AGGREGATES_H
