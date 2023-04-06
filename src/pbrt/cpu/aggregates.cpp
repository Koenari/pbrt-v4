// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/aggregates.h>

#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/shapes.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <tuple>
#include <chrono>

namespace pbrt {
bool BVHEnableMetrics = true;

STAT_MEMORY_COUNTER("Memory/BVH", treeBytes);
//Tree Creation
STAT_RATIO("BVH Creation/Count Primitives per leaf", totalPrimitives, totalLeafNodes);
STAT_COUNTER("BVH Creation/Count Interior Nodes", interiorNodes);
STAT_PERCENT("BVH Creation/Count Empty Nodes", emptyNodes, totalNodes);
STAT_PERCENT("BVH Creation/Wrong Predictions", bvhWrongPrediction,bvhAllPrediction);
STAT_INT_DISTRIBUTION("BVH Creation/Depth Leaves", bvhLeafDepth);
STAT_INT_DISTRIBUTION("BVH Creation/Depth Interior", bvhIntNodeDepth);
//Traversal
STAT_INT_DISTRIBUTION("BVH Traversal/Avg Leaf size hit", bvhAvgLeaveSizeHit);
STAT_PIXEL_RATIO("BVH Traversal/Interior Nodes hit", bvhInteriorHit, bvhInteriorTested);
STAT_PIXEL_RATIO("BVH Traversal/Leaf Nodes hit", bvhLeavesHit, bvhLeavesTested);
STAT_PIXEL_COUNTER("BVH Traversal/SIMD Primitive intersections", bvhSimdTriangleTests);
STAT_PIXEL_COUNTER("BVH Traversal/SIMD Interior Nodes visited", bvhSimdNodesVisited);
STAT_PIXEL_COUNTER("BVH Traversal/SISD Nodes visited", bvhSisdNodesVisited);
STAT_PIXEL_COUNTER("BVH Traversal/SISD Primitive intersections", bvhSisdTriangleTests);
//optimization
STAT_COUNTER("BVH Opti/Merged Leaves", bvhOptiLeaves);
STAT_COUNTER("BVH Opti/Merged Into Parent", bvhOptiIntoParent);
STAT_COUNTER("BVH Opti/Merged Inner", bvhOptiInner);
// MortonPrimitive Definition
struct MortonPrimitive {
    int primitiveIndex;
    uint32_t mortonCode;
};

// LBVHTreelet Definition
struct LBVHTreelet {
    size_t startIndex, nPrimitives;
    BVHBuildNode *buildNodes;
};
int BVHAggregate::SimdWidth = 1;
int BVHAggregate::maxPrimsInNodeOverride = -1;
int BVHAggregate::splitVariantOverride = -1;
Float BVHAggregate::epoRatioOverride = -1;
std::string BVHAggregate::splitMethodOverride = std::string();
std::string BVHAggregate::creationMethodOverride = std::string();
std::string BVHAggregate::optimizationStrategyOverride = std::string();
    // BinBVHAggregate Utility Functions
static void RadixSort(std::vector<MortonPrimitive> *v) {
    std::vector<MortonPrimitive> tempVector(v->size());
    constexpr int bitsPerPass = 6;
    constexpr int nBits = 30;
    static_assert((nBits % bitsPerPass) == 0,
                  "Radix sort bitsPerPass must evenly divide nBits");
    constexpr int nPasses = nBits / bitsPerPass;
    for (int pass = 0; pass < nPasses; ++pass) {
        // Perform one pass of radix sort, sorting _bitsPerPass_ bits
        int lowBit = pass * bitsPerPass;
        // Set in and out vector references for radix sort pass
        std::vector<MortonPrimitive> &in = (pass & 1) ? tempVector : *v;
        std::vector<MortonPrimitive> &out = (pass & 1) ? *v : tempVector;

        // Count number of zero bits in array for current radix sort bit
        constexpr int nBuckets = 1 << bitsPerPass;
        int bucketCount[nBuckets] = {0};
        constexpr int bitMask = (1 << bitsPerPass) - 1;
        for (const MortonPrimitive &mp : in) {
            int bucket = (mp.mortonCode >> lowBit) & bitMask;
            CHECK_GE(bucket, 0);
            CHECK_LT(bucket, nBuckets);
            ++bucketCount[bucket];
        }

        // Compute starting index in output array for each bucket
        int outIndex[nBuckets];
        outIndex[0] = 0;
        for (int i = 1; i < nBuckets; ++i)
            outIndex[i] = outIndex[i - 1] + bucketCount[i - 1];

        // Store sorted values in output array
        for (const MortonPrimitive &mp : in) {
            int bucket = (mp.mortonCode >> lowBit) & bitMask;
            out[outIndex[bucket]++] = mp;
        }
    }
    // Copy final result from _tempVector_, if needed
    if (nPasses & 1)
        std::swap(*v, tempVector);
}

// BVHSplitBucket Definition
struct BVHSplitBucket : ICostCalcable {
    int count = 0;
    Bounds3f bounds;
    int PrimCount() const { return count; }
    Bounds3f Bounds() const { return bounds; }
};

// BVHPrimitive Definition
struct BVHPrimitive {
    BVHPrimitive() {}
    BVHPrimitive(size_t primitiveIndex, const Bounds3f &bounds)
        : primitiveIndex(primitiveIndex), bounds(bounds) {}
    size_t primitiveIndex;
    Bounds3f bounds;
    // BVHPrimitive Public Methods
    Point3f Centroid() const { return .5f * bounds.pMin + .5f * bounds.pMax; }
};
//These are data structures used in creation of the BVH trees they are 
//more optimized for easy usage and less for memory efficiency
//Some abstraction is used to facilitate reusing code for different tree widths 
#pragma region BUILD_DATASTRUCTS
//Binary BVHBuildNode Definition
struct BVHBuildNode : ICostCalcable {
    // BVHBuildNode Public Methods
    void InitLeaf(int first, int n, const Bounds3f &b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = nullptr;
        if (BVHEnableMetrics) {
            ++totalLeafNodes;
            totalPrimitives += n;
        }
    }

    void InitInterior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        splitAxis = axis;
        nPrimitives = 0;
        if (BVHEnableMetrics)
            ++interiorNodes;
    }

    Bounds3f bounds;
    BVHBuildNode *children[2];
    int splitAxis, firstPrimOffset, nPrimitives;

    Bounds3f Bounds() const override { return bounds; };
    virtual int PrimCount() const override { return nPrimitives; }
};
//General Wide BVHBuildNote defintion
struct WideBVHBuildNode : ICostCalcable {
    Bounds3f bounds;
    int firstPrimOffset, nPrimitives;
    Bounds3f Bounds() const {
        return bounds;
    }
    bool IsLeaf() const {
        return nPrimitives > 0;
    }
    virtual WideBVHBuildNode *getChild(int idx) const = 0;
    virtual int getAxis(int idx) const = 0;
    virtual void setChild(int idx, WideBVHBuildNode *child) = 0;
    virtual void setAxis(int idx, int axis) = 0;
    virtual int numChildren() const = 0;
};
// Wide BVHBuildNote defintion with 4 children
struct FourWideBVHBuildNode : WideBVHBuildNode {
    void InitLeaf(int first, int n, const Bounds3f &b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = children[2] = children[3] = nullptr;
        if (BVHEnableMetrics) {
            ++totalLeafNodes;
            totalPrimitives += n;
        }
    }

    void InitInterior(int a[3], FourWideBVHBuildNode *c[4]) {
        children[0] = c[0];
        children[1] = c[1];
        children[2] = c[2];
        children[3] = c[3];
        for (int i = 0; i < 4; ++i) {
            if (children[i])
                bounds = Union(bounds, children[i]->bounds);
            else if (BVHEnableMetrics)
                emptyNodes++;
        }
        splitAxis[0] = a[0];
        splitAxis[1] = a[1];
        splitAxis[2] = a[2];
        nPrimitives = 0;
        if (BVHEnableMetrics) {
            totalNodes += 4;
            ++interiorNodes;
        }
    }
    FourWideBVHBuildNode *children[4];
    int numChildren() const override {
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (children[i])
                count++;
        }
        return count;
    };
    int splitAxis[3];
    WideBVHBuildNode *getChild(int idx) const override { return children[idx]; }
    int getAxis(int idx) const override { return splitAxis[idx]; }
    void setAxis(int idx, int axis) override { splitAxis[idx] = axis; }
    void setChild(int idx, WideBVHBuildNode *child) override {
        children[idx] = (FourWideBVHBuildNode *)child;
    }
    int PrimCount() const override {
        if (nPrimitives > 0) {
            return nPrimitives;
        } else {
            int count = 0;
            for (int i = 0; i < 4; ++i) {
                if (children[i]) {
                    count += children[i]->PrimCount();
                }
            }
            return count;
        }
    }
};
// Wide BVHBuildNote defintion with 8 newChildren
struct EightWideBVHBuildNode : WideBVHBuildNode {
    EightWideBVHBuildNode *children[8];
    int splitAxis[7];
    void InitLeaf(int first, int n, const Bounds3f &b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = children[2] = children[3] = children[4] =
            children[5] = children[6] = children[7] = nullptr;
        if (BVHEnableMetrics) {
            ++totalLeafNodes;
            totalPrimitives += n;
        }
    }

    void InitInterior(int a[7], EightWideBVHBuildNode *c[8]) {
        for (int i = 0; i < 8; ++i) {
            children[i] = c[i];
            if (children[i])
                bounds = Union(bounds, children[i]->bounds);
            else if (BVHEnableMetrics)
                emptyNodes++;
            if (i < 7)
                splitAxis[0] = a[0];
        }
        nPrimitives = 0;
        if (BVHEnableMetrics) {
            totalNodes += 8;
            ++interiorNodes;
        }
    }
    int numChildren() const override {
        int count = 0;
        for (int i = 0; i < 8; ++i) {
            if (children[i])
                count++;
        }
        return count;
    };

    WideBVHBuildNode *getChild(int idx) const override { return children[idx]; }
    void setChild(int idx, WideBVHBuildNode *child) override {
        children[idx] = (EightWideBVHBuildNode *)child;
    }
    int getAxis(int idx) const override { return splitAxis[idx]; }
    void setAxis(int idx, int axis) override { splitAxis[idx] = axis; }

    int PrimCount() const override {
        if (nPrimitives > 0) {
            return nPrimitives;
        } else {
            int count = 0;
            for (int i = 0; i < 8; ++i) {
                if (children[i]) {
                    count += children[i]->PrimCount();
                }
            }
            return count;
        }
    }
};
#pragma endregion
// These strcutures do not have inheritances as they are more optimized in terms of size
// and fast memory access therefore each TreeWidth has a special struct
#pragma region TRAVELSAL_DATASTRUCTS
//Binary LinearBVHNode Definition
struct alignas(32) LinearBVHNode {
    Bounds3f bounds;
    union {
        int primitivesOffset;   // leaf
        int secondChildOffset;  // interior
    };
    uint16_t nPrimitives;  // 0 -> interior newNode
    uint8_t axis;          // interior newNode: xyz
};

//  Four Wide BVH Noide optimized for SIMD usage size < 128 byte
struct alignas(32) SIMDFourWideLinearBVHNode {
    //  x:0 y:1 z:2
    // mains split axis in bits 0+1
    // secondary exits in bit 2+3
    //  7 6 | 5 4     | 3 2     | 1 0
    //      | axis[2] | axis[1] | axis [0]
    uint8_t axis = 0;
    Bounds3f bounds[4];
    uint16_t nPrimitives[4] = {0};
    int offsets[4];

    inline uint8_t Axis0() const { return axis >> 0 & 3; }
    inline uint8_t Axis1() const { return axis >> 2 & 3; }
    inline uint8_t Axis2() const { return axis >> 4 & 3; }
    inline uint8_t ConstructAxis(int axis0, int axis1, int axis2) const {
        return axis0 & 3 + ((axis1 & 3) << 2) + ((axis2 & 3) << 4);
    }
};

// Eight Wide BVH Noide optimized for SIMD usage size < ???? byte
struct alignas(32) SIMDEightWideLinearBVHNode {
    //  x:0 y:1 z:2
    // mains split axis in bits 0+1
    // secondary exits in bit 2+3
    //  7 6 | 5 4     | 3 2     | 1 0
    //  axis[3] | axis[2] | axis[1] | axis [0]
    uint16_t axis = 0;
    Bounds3f bounds[8];
    uint16_t nPrimitives[8] = {0};
    int offsets[8];

    inline uint8_t Axis0() const { return axis >> 0 & 3; }
    inline uint8_t Axis1() const { return axis >> 2 & 3; }
    inline uint8_t Axis2() const { return axis >> 4 & 3; }
    inline uint8_t Axis3() const { return axis >> 6 & 3; }
    inline uint8_t Axis4() const { return axis >> 8 & 3; }
    inline uint8_t Axis5() const { return axis >> 10 & 3; }
    inline uint8_t Axis6() const { return axis >> 12 & 3; }
    inline uint16_t ConstructAxis(int axis0, int axis1, int axis2, int axis3, int axis4,
                                  int axis5, int axis6) const {
        return axis0 & 3 + ((axis1 & 3) << 2) + ((axis2 & 3) << 4) + ((axis3 & 3) << 6) +
                           ((axis4 & 3) << 8) + ((axis5 & 3) << 10) + ((axis6 & 3) << 12);
    }
};
#pragma endregion

#pragma region BVH_CONSTRUCTORS
BVHAggregate::BVHAggregate(int maxPrimsInNode, std::vector<Primitive> prims,
                           SplitMethod sm, Float epoRatio)
    : maxPrimsInNode(maxPrimsInNode),
      primitives(prims),
      splitMethod(sm),
      epoRatio(epoRatio) {}
WideBVHAggregate::WideBVHAggregate(int TreeWidth, int maxPrimsInNode, std::vector<Primitive> prims,
                                   SplitMethod sm, Float epoRatio)
    : BVHAggregate(maxPrimsInNode, prims, sm, epoRatio), TreeWidth(TreeWidth) {}
BinBVHAggregate::BinBVHAggregate(std::vector<Primitive> prims, int maxPrimsInNode,
                                 SplitMethod splitMethod, Float epoRatio,
                                 bool skipCreation)
    : BVHAggregate(std::min(255, maxPrimsInNode), std::move(prims), splitMethod,
                   epoRatio) {
    CHECK(!primitives.empty());
    // Build BVH from _primitives_
    // Initialize _bvhPrimitives_ array for primitives
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i].Bounds());
    if (skipCreation)
        return;
    // Build BVH for primitives using _bvhPrimitives_
    // Declare _Allocator_s used for BVH construction
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });

    std::vector<Primitive> orderedPrims(primitives.size());
    BVHBuildNode *root;
    // Build BVH according to selected _splitMethod_
    std::atomic<int> totalNodes{0};
    if (splitMethod == SplitMethod::HLBVH) {
        root = buildHLBVH(alloc, bvhPrimitives, &totalNodes, orderedPrims);
    } else {
        std::atomic<int> orderedPrimsOffset{0};
        root = buildRecursive(threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives),
                              &totalNodes, &orderedPrimsOffset, orderedPrims);
        CHECK_EQ(orderedPrimsOffset.load(), orderedPrims.size());
    }
    primitives.swap(orderedPrims);

    // Convert BVH into compact representation in _nodes_ array
    bvhPrimitives.resize(0);
    LOG_VERBOSE("BVH created with %d nodes for %d primitives (%.2f MB)",
                totalNodes.load(), (int)primitives.size(),
                float(totalNodes.load() * sizeof(LinearBVHNode)) / (1024.f * 1024.f));
    treeBytes += totalNodes * sizeof(LinearBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    Float treeCost = calcMetrics(root);
    LOG_VERBOSE("Calulated cost for BVH is: {%.3f}", treeCost);
    nodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    flattenBVH(root, &offset);
    CHECK_EQ(totalNodes.load(), offset);
}
FourWideBVHAggregate::FourWideBVHAggregate(std::vector<Primitive> prims,
                                           int maxPrimsInNode, SplitMethod splitMethod,
                                           Float epoRatio, int splitVariant,
                                           CreationMethod method,
                                           OptimizationStrategy opti)
    : WideBVHAggregate(4,std::min(255, maxPrimsInNode), std::move(prims), splitMethod,
                       epoRatio) {
    CHECK(!primitives.empty());
    LOG_VERBOSE("Creating 4 WideBVHAggregate: SPlitmethod %d (%d) - %s - %s",
                (int)splitMethod, splitVariant,
                method == CreationMethod::Direct ? "direct" : "fromBVH",
                opti != None ? std::to_string(opti) : "no opti");
    // Build BVH from _primitives_
    // Initialize _bvhPrimitives_ array for primitives
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i].Bounds());

    // Build BVH for primitives using _bvhPrimitives_
    // Declare _Allocator_s used for BVH construction
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });
    std::vector<Primitive> orderedPrims(primitives.size());
    FourWideBVHBuildNode *root;
    std::atomic<int> totalNodesLocal{0};
    std::atomic<int> orderedPrimsOffset{0};
    // Build BVH according to selected _splitMethod_
    auto start = std::chrono::high_resolution_clock::now();
    if (method == CreationMethod::FromBVH) {
        BVHEnableMetrics = false;
        BinBVHAggregate *binTree =
            new BinBVHAggregate(primitives, maxPrimsInNode, splitMethod, epoRatio, true);
        BVHBuildNode *binRoot = binTree->buildRecursive(
            threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives), &totalNodesLocal,
            &orderedPrimsOffset, orderedPrims);
        totalNodesLocal = 0;
        BVHEnableMetrics = true;
        root = buildFromBVH(binRoot, &totalNodesLocal);
    } else {
        root = buildRecursive(threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives),
                              &totalNodesLocal, &orderedPrimsOffset, orderedPrims,
                              splitVariant);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    CHECK_EQ(orderedPrimsOffset.load(), orderedPrims.size());
    primitives.swap(orderedPrims);
    
    bvhPrimitives.resize(0);
    LOG_VERBOSE(
        "WideBVH created with %d nodes for %d primitives (%.2f MB)",
        totalNodesLocal.load(), (int)primitives.size(),
        totalNodesLocal.load() * sizeof(SIMDFourWideLinearBVHNode) / (1024.f * 1024.f));
    LOG_VERBOSE("Creation took: %d ms", time.count());
    start = std::chrono::high_resolution_clock::now();
    if (optimizeTree(root, &totalNodesLocal, opti)) {
        end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        LOG_VERBOSE("WideBVH optimized to %d nodes for %d primitives (%.2f MB)",
                    totalNodesLocal.load(), (int)primitives.size(),
                    totalNodesLocal.load() * sizeof(SIMDFourWideLinearBVHNode) /
                        (1024.f * 1024.f));
        LOG_VERBOSE("Optimization took: %d ms", time.count());
    }
    Float treeCost = calcMetrics(root, 0);
    LOG_VERBOSE("Calulated cost for BVH is: {%.3f}", treeCost);
    if (totalNodes == 0)
        totalNodes++;
    treeBytes += totalNodesLocal * sizeof(SIMDFourWideLinearBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    nodes = new SIMDFourWideLinearBVHNode[totalNodesLocal];
    int offset = 0;
    // Convert BVH into compact representation in _nodes_ array
    flattenBVH(root, &offset);
    CHECK_EQ(totalNodesLocal.load(), offset);
}
EightWideBVHAggregate::EightWideBVHAggregate(std::vector<Primitive> prims,
                                             int maxPrimsInNode, SplitMethod splitMethod,
                                             Float epoRatio, int splitVariant,
                                             CreationMethod method,
                                             OptimizationStrategy opti)
    : WideBVHAggregate(8,std::min(255, maxPrimsInNode), std::move(prims), splitMethod,
                       epoRatio) {
    CHECK(!primitives.empty());
    LOG_VERBOSE("Creating 8 WideBVHAggregate: SPlitmethod %d (%d) - %s - %s",
                (int)splitMethod, splitVariant,
                method == CreationMethod::Direct ? "direct" : "fromBVH",
                opti != None ? std::to_string(opti) : "no opti");
    // Build BVH from _primitives_
    // Initialize _bvhPrimitives_ array for primitives
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i].Bounds());

    // Build BVH for primitives using _bvhPrimitives_
    // Declare _Allocator_s used for BVH construction
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });
    std::vector<Primitive> orderedPrims(primitives.size());
    EightWideBVHBuildNode *root;
    std::atomic<int> totalNodesLocal{0};
    std::atomic<int> orderedPrimsOffset{0};
    auto start = std::chrono::high_resolution_clock::now();
    if (method == CreationMethod::FromBVH) {
        BVHEnableMetrics = false;
        BinBVHAggregate *binTree =
            new BinBVHAggregate(primitives, maxPrimsInNode, splitMethod, epoRatio, true);
        BVHBuildNode *binRoot = binTree->buildRecursive(
            threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives), &totalNodesLocal,
            &orderedPrimsOffset, orderedPrims);
        totalNodesLocal = 0;
        BVHEnableMetrics = true;
        root = buildFromBVH(binRoot, &totalNodesLocal);
    } else {
        root = buildRecursive(threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives),
                              &totalNodesLocal, &orderedPrimsOffset, orderedPrims,
                              splitVariant);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Build BVH according to selected _splitMethod_

    CHECK_EQ(orderedPrimsOffset.load(), orderedPrims.size());

    primitives.swap(orderedPrims);
    // Convert BVH into compact representation in _nodes_ array
    bvhPrimitives.resize(0);
    LOG_VERBOSE(
        "WideBVH created with %d nodes for %d primitives (%.2f MB)",
        totalNodesLocal.load(), (int)primitives.size(),
        totalNodesLocal.load() * sizeof(SIMDEightWideLinearBVHNode) / (1024.f * 1024.f));
    LOG_VERBOSE("Creation took: %d ms", time.count());
    start = std::chrono::high_resolution_clock::now();
    if (optimizeTree(root, &totalNodesLocal, opti)) {
        end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        LOG_VERBOSE("WideBVH optimized to %d nodes for %d primitives (%.2f MB)",
                    totalNodesLocal.load(), (int)primitives.size(),
                    totalNodesLocal.load() * sizeof(SIMDEightWideLinearBVHNode) /
                        (1024.f * 1024.f));
        LOG_VERBOSE("Optimization took: %d ms", time.count());
    }
    Float treeCost = calcMetrics(root, 0);
    LOG_VERBOSE("Calulated cost for BVH is: {%.3f}", treeCost);
    if (totalNodes == 0)
        totalNodes++;
    treeBytes += totalNodesLocal * sizeof(SIMDEightWideLinearBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    nodes = new SIMDEightWideLinearBVHNode[totalNodesLocal];
    int offset = 0;
    flattenBVH(root, &offset);
    CHECK_EQ(totalNodesLocal.load(), offset);
}
#pragma endregion

#pragma region BVH_STATIC_CREATE
Primitive CreateAccelerator(std::vector<Primitive> prims) {
    return CreateAccelerator("bvh", prims, ParameterDictionary());
}
Primitive CreateAccelerator(const std::string &name, std::vector<Primitive> prims,
                            const ParameterDictionary &parameters) {
    Primitive accel = nullptr;
    if (name == "bvh" || name == "widebvh4")
        accel = FourWideBVHAggregate::Create(std::move(prims), parameters);
    else if (name == "widebvh8")
        accel = EightWideBVHAggregate::Create(std::move(prims), parameters);
    else if (name == "binbvh")
        accel = BinBVHAggregate::Create(std::move(prims), parameters);
    else if (name == "kdtree")
        accel = KdTreeAggregate::Create(std::move(prims), parameters);
    else
        ErrorExit("%s: accelerator type unknown.", name);

    if (!accel)
        ErrorExit("%s: unable to create accelerator.", name);

    parameters.ReportUnused();
    return accel;
}
BinBVHAggregate *BinBVHAggregate::Create(std::vector<Primitive> prims,
                                         const ParameterDictionary &parameters) {
    std::string splitMethodName = parameters.GetOneString("splitmethod", "sah");
    if (!splitMethodOverride.empty())
        splitMethodName = splitMethodOverride;
    SplitMethod splitMethod;
    if (splitMethodName == "sah")
        splitMethod = SplitMethod::SAH;
    else if (splitMethodName == "epo")
        splitMethod = SplitMethod::EPO;
    else if (splitMethodName == "hlbvh")
        splitMethod = SplitMethod::HLBVH;
    else if (splitMethodName == "middle")
        splitMethod = SplitMethod::Middle;
    else if (splitMethodName == "equal")
        splitMethod = SplitMethod::EqualCounts;
    else {
        Warning(R"(BVH split method "%s" unknown.  Using "sah".)", splitMethodName);
        splitMethod = SplitMethod::SAH;
    }
    int splitVariant = parameters.GetOneInt("splitVariant", 0);
    int maxPrimsInNode = parameters.GetOneInt("maxnodeprims", 4);
    if (maxPrimsInNodeOverride > 0)
        maxPrimsInNode = maxPrimsInNodeOverride;
    Float epoRatio = parameters.GetOneFloat("epoRatio", 1.f);
    if (epoRatioOverride >= 0)
        epoRatio = epoRatioOverride;
    return new BinBVHAggregate(std::move(prims), maxPrimsInNode, splitMethod, epoRatio);
}
FourWideBVHAggregate *FourWideBVHAggregate::Create(
    std::vector<Primitive> prims, const ParameterDictionary &parameters) {
    std::string splitMethodName = parameters.GetOneString("splitmethod", "sah");
    if (!splitMethodOverride.empty())
        splitMethodName = splitMethodOverride;
    SplitMethod splitMethod;
    if (splitMethodName == "sah")
        splitMethod = SplitMethod::SAH;
    else if (splitMethodName == "epo")
        splitMethod = SplitMethod::EPO;
    else if (splitMethodName == "hlbvh")
        splitMethod = SplitMethod::HLBVH;
    else if (splitMethodName == "middle")
        splitMethod = SplitMethod::Middle;
    else if (splitMethodName == "equal")
        splitMethod = SplitMethod::EqualCounts;
    else {
        Warning(R"(WideBVH split method "%s" unknown.  Using "sah".)", splitMethodName);
        splitMethod = SplitMethod::SAH;
    }
    std::string creationMethodName = parameters.GetOneString("creationmethod", "direct");
    CreationMethod creationMethod;
    if (!creationMethodOverride.empty())
        creationMethodName = creationMethodOverride;
    if (creationMethodName == "direct")
        creationMethod = CreationMethod::Direct;
    else if (creationMethodName == "frombvh")
        creationMethod = CreationMethod::FromBVH;
    else {
        Warning(R"(WideBVH creation method "%s" unknown.  Using "direct".)",
                splitMethodName);
        creationMethod = CreationMethod::Direct;
    }
    OptimizationStrategy optiStrat = None;
    auto optiNames = parameters.GetStringArray("optimizations");
    if (!optimizationStrategyOverride.empty()) {
        optiNames.clear();
        optiNames.push_back(optimizationStrategyOverride);
    }
    for each (auto optiName in optiNames) {
        if (optiName == "all") {
            optiStrat = All;
            break;
        } else if (optiName == "none") {
            optiStrat = None;
            break;
        } else if (optiName == "mergeInnerChildren") {
            optiStrat = (OptimizationStrategy)(optiStrat | MergeInnerChildren);
        } else if (optiName == "mergeIntoParent") {
            optiStrat = (OptimizationStrategy)(optiStrat | MergeIntoParent);
        } else if (optiName == "mergeLeaves") {
            optiStrat = (OptimizationStrategy)(optiStrat | MergeLeaves);
        } else {
            Warning(R"(WideBVH optimization strategy "%s" unknown.)", optiName);
        }
    };
    int maxPrimsInNode = parameters.GetOneInt("maxnodeprims", 16);
    if (maxPrimsInNodeOverride > 0)
        maxPrimsInNode = maxPrimsInNodeOverride;
    int splitVariant = parameters.GetOneInt("splitVariant", 0);
    if (splitVariantOverride >= 0)
        splitVariant = splitVariantOverride;
    Float epoRatio = parameters.GetOneFloat("epoRatio", 1.f);
    if (epoRatioOverride >= 0)
        epoRatio = epoRatioOverride;
    return new FourWideBVHAggregate(std::move(prims), maxPrimsInNode, splitMethod,
                                    epoRatio, splitVariant, creationMethod, optiStrat);
}
EightWideBVHAggregate *EightWideBVHAggregate::Create(
    std::vector<Primitive> prims, const ParameterDictionary &parameters) {
    std::string splitMethodName = parameters.GetOneString("splitmethod", "sah");
    if (!splitMethodOverride.empty())
        splitMethodName = splitMethodOverride;
    SplitMethod splitMethod;
    if (splitMethodName == "sah")
        splitMethod = SplitMethod::SAH;
    else if (splitMethodName == "epo")
        splitMethod = SplitMethod::EPO;
    else if (splitMethodName == "hlbvh")
        splitMethod = SplitMethod::HLBVH;
    else if (splitMethodName == "middle")
        splitMethod = SplitMethod::Middle;
    else if (splitMethodName == "equal")
        splitMethod = SplitMethod::EqualCounts;
    else {
        Warning(R"(WideBVH split method "%s" unknown.  Using "sah".)", splitMethodName);
        splitMethod = SplitMethod::SAH;
    }
    std::string creationMethodName = parameters.GetOneString("creationmethod", "direct");
    CreationMethod creationMethod;
    if (!creationMethodOverride.empty())
        creationMethodName = creationMethodOverride;
    if (creationMethodName == "direct")
        creationMethod = CreationMethod::Direct;
    else if (creationMethodName == "frombvh")
        creationMethod = CreationMethod::FromBVH;
    else {
        Warning(R"(WideBVH creation method "%s" unknown.  Using "direct".)",
                splitMethodName);
        creationMethod = CreationMethod::Direct;
    }
    OptimizationStrategy optiStrat = None;
    auto optiNames = parameters.GetStringArray("optimizations");
    if (!optimizationStrategyOverride.empty()) {
        optiNames.clear();
        optiNames.push_back(optimizationStrategyOverride);
    }
    for each (auto optiName in optiNames) {
        if (optiName == "all") {
            optiStrat = All;
            break;
        } else if (optiName == "none") {
            optiStrat = None;
            break;
        } else if (optiName == "mergeInnerChildren") {
            optiStrat = (OptimizationStrategy)(optiStrat | MergeInnerChildren);
        } else if (optiName == "mergeIntoParent") {
            optiStrat = (OptimizationStrategy)(optiStrat | MergeIntoParent);
        } else if (optiName == "mergeLeaves") {
            optiStrat = (OptimizationStrategy)(optiStrat | MergeLeaves);
        } else {
            Warning(R"(WideBVH optimization strategy "%s" unknown.)", optiName);
        }
    };
    int maxPrimsInNode = parameters.GetOneInt("maxnodeprims", 16);
    if (maxPrimsInNodeOverride > 0)
        maxPrimsInNode = maxPrimsInNodeOverride;
    int splitVariant = parameters.GetOneInt("splitVariant", 0);
    if (splitVariantOverride >= 0)
        splitVariant = splitVariantOverride;
    Float epoRatio = parameters.GetOneFloat("epoRatio", 1.f);
    if (epoRatioOverride >= 0)
        epoRatio = epoRatioOverride;
    return new EightWideBVHAggregate(std::move(prims), maxPrimsInNode, splitMethod,
                                     epoRatio, splitVariant, creationMethod, optiStrat);
}
#pragma endregion

#pragma region BVH_SHARED_FUNCS
//BVHAggregate
Float BVHAggregate::leafCost(int primCount) const {
    return pstd::ceil(primCount / (Float)SimdWidth);
};
Float BVHAggregate::penalizedHitProbability(int idx, int count,
                                            ICostCalcable **buckets) const {
    const ICostCalcable *bucket = buckets[idx];
    if (idx >= count || bucket == NULL)
        return 0.f;
    Bounds3f combinedBounds;
    for (int i = 0; i < count; ++i)
        if (buckets[i])
            combinedBounds = Union(combinedBounds, buckets[i]->Bounds());
    Float prob = bucket->Bounds().SurfaceArea() / combinedBounds.SurfaceArea();
    if (splitMethod == SplitMethod::EPO) {
        Bounds3f overlap;
        for (int j = 0; j < count; ++j) {
            if (j == idx || !(buckets[j]))
                continue;
            overlap =
                Union(overlap, pbrt::Intersect(bucket->Bounds(), buckets[j]->Bounds()));
        }
        if (!overlap.IsEmpty())
            prob *= (1.f + epoRatio * overlap.Volume() / combinedBounds.Volume());
    }
    return prob;
}
Float BVHAggregate::childCost(int idx, int count, ICostCalcable **buckets) const {
    // All metrics are 0  for empty buckets
    if (buckets[idx] && buckets[idx]->PrimCount()) {
        return leafCost(buckets[idx]->PrimCount()) *
               penalizedHitProbability(idx, count, buckets);
    } else {
        return 0.f;
    }
}
Float BVHAggregate::splitCost(int count, ICostCalcable **buckets) const {
    if (splitMethod != SplitMethod::SAH && splitMethod != SplitMethod::EPO)
        return 0.f;
    Float cost = 0.f;
    for (int i = 0; i < count; ++i) {
        cost += childCost(i, count, buckets);
    }
    return cost;
}
//WideBVHAggregate
Bounds3f WideBVHAggregate::Bounds() const {
    return bounds;
}
Float WideBVHAggregate::calcMetrics(WideBVHBuildNode *root, int depth) {
    if (root->IsLeaf()) {
        bvhLeafDepth << depth;
        return leafCost(root->PrimCount());
    } else {
        bvhIntNodeDepth << depth;
        ICostCalcable *nodes[MaxTreeWidth]{};
        for (int i = 0; i < TreeWidth; i++) {
            nodes[i] = root->getChild(i);
        }
        //Own box intersection cost
        Float cost = RelativeInnerCost * pstd::ceil(TreeWidth / (Float)SimdWidth);
        for (int i = 0; i < TreeWidth; ++i) {
            if (!(root->getChild(i)))
                continue;
            cost += penalizedHitProbability(i, TreeWidth, nodes) *
                    calcMetrics(root->getChild(i), depth + 1);
        }
        return cost;
    }
}
bool WideBVHAggregate::optimizeTree(WideBVHBuildNode *root, std::atomic<int> *totalNodes,
                                    OptimizationStrategy strat) {
    bool didOptimize = false;
    auto mergeChildIntoParent = [=](WideBVHBuildNode *parent) {
        bool opmiziationDone = false;
        for (int mergedChildIdx = 0; mergedChildIdx < TreeWidth; ++mergedChildIdx) {
            auto child = parent->getChild(mergedChildIdx);
            if (child == NULL || child->IsLeaf())
                continue;
            if ((parent->numChildren() + child->numChildren() - 1) <= TreeWidth) {
                bvhOptiIntoParent++;
                emptyNodes -= (TreeWidth - 1);
                --interiorNodes;
                --*totalNodes;
                opmiziationDone = true;
                parent->setChild(mergedChildIdx, NULL);
                WideBVHBuildNode *newChildren[MaxTreeWidth]{NULL};
                int newAxis[MaxTreeWidth-1]{0};
                int lastParIdx = -1;
                int curIdx = 0;
                for (int parIdx = 0; parIdx < mergedChildIdx; ++parIdx) {
                    if (!parent->getChild(parIdx))
                        continue;
                    DCHECK_GE(parent->getChild(parIdx)->nPrimitives, -1);
                    DCHECK_LT(curIdx, TreeWidth);
                    newChildren[curIdx] = parent->getChild(parIdx);
                    if (lastParIdx >= 0) {
                        newAxis[curIdx - 1] =
                            parent->getAxis(relevantAxisIdx(lastParIdx, parIdx));
                    }
                    lastParIdx = parIdx;
                    curIdx++;
                }
                if (curIdx > 0)
                    newAxis[curIdx - 1] =
                        parent->getAxis(relevantAxisIdx(lastParIdx, mergedChildIdx));
                int lastChildIdx = -1;
                for (int childIdx = 0; childIdx < TreeWidth; ++childIdx) {
                    if (!child->getChild(childIdx))
                        continue;
                    DCHECK_GE(child->getChild(childIdx)->nPrimitives, -1);
                    DCHECK_LT(curIdx, TreeWidth);
                    newChildren[curIdx] = child->getChild(childIdx);
                    if (lastChildIdx >= 0) {
                        newAxis[curIdx - 1] =
                            child->getAxis(relevantAxisIdx(lastChildIdx, childIdx));
                    }
                    lastChildIdx = childIdx;
                    curIdx++;
                }
                lastParIdx = mergedChildIdx;
                for (int parIdx = mergedChildIdx + 1; parIdx < TreeWidth; ++parIdx) {
                    if (!parent->getChild(parIdx))
                        continue;
                    DCHECK_GE(parent->getChild(parIdx)->nPrimitives, -1);
                    DCHECK_LT(curIdx, TreeWidth);
                    newChildren[curIdx] = parent->getChild(parIdx);
                    if (lastParIdx >= 0) {
                        newAxis[curIdx - 1] =
                            parent->getAxis(relevantAxisIdx(lastParIdx, parIdx));
                    }
                    lastParIdx = parIdx;
                    curIdx++;
                }
                DCHECK_LE(curIdx, TreeWidth);
                DCHECK_LT(lastParIdx, TreeWidth);
                DCHECK_LT(lastChildIdx, TreeWidth);
                for (int i = 0; i < TreeWidth; ++i) {
                    parent->setChild(i, newChildren[i]);
                }
                for (int i = 0; i < TreeWidth - 1; ++i) {
                    parent->setAxis(i, newAxis[i]);
                }
                mergedChildIdx = -1;
            }
        }
        return opmiziationDone;
    };
    // Will not implement for now as there is no benefit for now
    auto mergeLeafChildren = [=](WideBVHBuildNode *parent) {
        bool opmtimizationDone = false;
        // See if any merges are benficial
        for (int i = 0; i < TreeWidth; ++i) {
            auto child1 = parent->getChild(i);
            if (child1 == NULL || !child1->IsLeaf())
                continue;
            for (int j = i + 1; j < TreeWidth; ++j) {
                auto child2 = parent->getChild(j);
                if (child2 == NULL || !child2->IsLeaf())
                    continue;
                // Can only easily merge leaves with consecutive prims
                if (child1->firstPrimOffset + child1->nPrimitives !=
                    child2->firstPrimOffset)
                    continue;
                // Bounds3f newBounds = Union(child1->bounds, child2->bounds);
                ICostCalcable *oldNodes[MaxTreeWidth];
                for (int k = 0; k < TreeWidth; k++) {
                    oldNodes[k] = parent->getChild(k);
                }
                BVHSplitBucket newLeaf;
                newLeaf.bounds = Union(child1->bounds, child2->bounds);
                newLeaf.count = child1->nPrimitives + child2->nPrimitives;
                ICostCalcable *newNodes[MaxTreeWidth];
                for (int k = 0; k < TreeWidth; k++) {
                    if (k == j) {
                        newNodes[k] = NULL;
                    } else if (k == i) {
                        newNodes[k] = &newLeaf;
                    } else {
                        newNodes[k] = oldNodes[k];
                    }
                }
                float oldCost1 = childCost(i, TreeWidth, oldNodes);
                float oldCost2 = childCost(j, TreeWidth, oldNodes);
                float newCost = childCost(i, TreeWidth, newNodes);
                float costReduction = (oldCost1 + oldCost2) - newCost;
                if (costReduction > 0) {
                    bvhOptiLeaves++;
                    opmtimizationDone = true;
                    child1->nPrimitives = newLeaf.count;
                    child1->bounds = newLeaf.bounds;
                    parent->setChild(i,child1);
                    parent->setChild(j, NULL);
                    emptyNodes++;
                    totalLeafNodes--;
                    i = -1;
                    break;
                }
            }
        }
        return opmtimizationDone;
    };
    auto mergeInnerNodeChildren = [=](WideBVHBuildNode *parent) {
        bool opmtimizationDone = false;
        while (true) {
            int bestI = -1, bestJ = -1;
            float maxReduction = -1;
            for (int i = 0; i < TreeWidth; ++i) {
                auto child1 = parent->getChild(i);
                if (child1 == NULL || child1->nPrimitives > 0)
                    continue;
                for (int j = i + 1; j < TreeWidth; ++j) {
                    auto child2 = parent->getChild(j);
                    if (child2 == NULL || child2->nPrimitives > 0)
                        continue;
                    if (child1->numChildren() + child2->numChildren() <= TreeWidth) {
                        ICostCalcable *oldNodes[MaxTreeWidth];
                        for (int k = 0; k < TreeWidth; k++) {
                            oldNodes[k] = parent->getChild(k);
                        }
                        BVHSplitBucket newNode;
                        newNode.bounds = Union(child1->bounds, child2->bounds);
                        newNode.count = 0;
                        ICostCalcable *newNodes[MaxTreeWidth];
                        for (int k = 0; k < TreeWidth; k++) {
                            if (k == j) {
                                newNodes[k] = NULL;
                            } else if (k == i) {
                                newNodes[k] = &newNode;
                            } else {
                                newNodes[k] = oldNodes[k];
                            }
                        }
                        Float oldCost1 = childCost(i, TreeWidth, oldNodes);
                        Float oldCost2 = childCost(j, TreeWidth, oldNodes);
                        Float newCost = childCost(i, TreeWidth, newNodes);
                        float costReduction = (oldCost1 + oldCost2) - newCost;
                        if (costReduction > maxReduction) {
                            maxReduction = costReduction;
                            bestI = i;
                            bestJ = j;
                        }
                    }
                }
            }
            if (maxReduction <= 0)
                return opmtimizationDone;
            bvhOptiInner++;
            opmtimizationDone = true;
            --*totalNodes;
            --interiorNodes;
            emptyNodes -= (TreeWidth - 1);
            int planeBetweenChildren = parent->getAxis(relevantAxisIdx(bestI, bestJ));
            auto child1 = parent->getChild(bestI);
            auto child2 = parent->getChild(bestJ);
            WideBVHBuildNode *newChildren[MaxTreeWidth]{NULL};
            int newAxis[MaxTreeWidth-1]{0};
            int lastIdx = -1;
            int curIdx = 0;
            int j = 0;
            for (int i = 0; i < TreeWidth; ++i) {
                if (!child1->getChild(i))
                    continue;
                newChildren[curIdx] = child1->getChild(i);
                if (lastIdx >= 0) {
                    newAxis[curIdx - 1] = parent->getAxis(relevantAxisIdx(lastIdx, i));
                }
                lastIdx = i;
                curIdx++;
            }
            if (curIdx > 0)
                newAxis[curIdx - 1] = planeBetweenChildren;
            lastIdx = -1;
            for (int i = 0; i < TreeWidth; ++i) {
                if (!child2->getChild(i))
                    continue;
                newChildren[curIdx] = child2->getChild(i);
                if (lastIdx >= 0) {
                    newAxis[curIdx - 1] = parent->getAxis(relevantAxisIdx(lastIdx, i));
                }
                lastIdx = i;
                curIdx++;
            }
            CHECK_EQ(curIdx, child1->numChildren() + child2->numChildren());
            for (int i = 0; i < TreeWidth; ++i) {
                parent->getChild(bestI)->setChild(i, newChildren[i]);
            }
            for (int i = 0; i < TreeWidth - 1; ++i) {
                parent->getChild(bestI)->setAxis(i, newAxis[i]);
            }
            parent->getChild(bestI)->bounds = Union(child1->bounds, child2->bounds);

            parent->setChild(bestJ, NULL);
        }
    };

    bool didOptimizeThisRound = false;

    do {
        didOptimizeThisRound = false;
        if (strat & OptimizationStrategy::MergeIntoParent)
            didOptimizeThisRound |= mergeChildIntoParent(root);
        if (!didOptimizeThisRound && (strat & MergeLeaves))
            didOptimizeThisRound |= mergeLeafChildren(root);
        if (!didOptimizeThisRound && (strat & OptimizationStrategy::MergeInnerChildren))
            didOptimizeThisRound |= mergeInnerNodeChildren(root);
        if (!didOptimizeThisRound)
            for (int i = 0; i < TreeWidth; ++i)
                if (root->getChild(i))
                    didOptimizeThisRound |=
                        optimizeTree(root->getChild(i), totalNodes, strat);
        didOptimize |= didOptimizeThisRound;
    } while (didOptimizeThisRound);
    return didOptimize;
};
#pragma endregion

#pragma region BvhBinary
//vfuncs BVH
Bounds3f BinBVHAggregate::Bounds() const {
    CHECK(nodes);
    return nodes[0].bounds;
}
pstd::optional<ShapeIntersection> BinBVHAggregate::Intersect(const Ray &ray,
                                                             Float tMax) const {
    if (!nodes)
        return {};
    pstd::optional<ShapeIntersection> si;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    int nodesVisited = 0;
    int trianglesTested = 0;
    int interiorHit = 0;
    int interiorTested = 0;
    int leavesHit = 0;
    int leavesTested = 0;
    while (true) {
        ++nodesVisited;
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        // Check ray against BVH newNode
        if (node->nPrimitives > 0) {
            leavesTested++;
        } else {
            interiorTested++;
        }
        if (node->bounds.IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                trianglesTested += node->nPrimitives;
                bvhAvgLeaveSizeHit << (int)(node->nPrimitives);
                leavesHit++;
                // Intersect ray with primitives in leaf BVH newNode
                for (int i = 0; i < node->nPrimitives; ++i) {
                    // Check for intersection with primitive in BVH newNode
                    pstd::optional<ShapeIntersection> primSi =
                        primitives[node->primitivesOffset + i].Intersect(ray, tMax);
                    if (primSi) {
                        si = primSi;
                        tMax = si->tHit;
                    }
                }
                if (toVisitOffset == 0)
                    break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];

            } else {
                interiorHit++;
                // Put far BVH newNode on _nodesToVisit_ stack, advance to near newNode
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0)
                break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    bvhSisdTriangleTests += trianglesTested;
    bvhSisdNodesVisited += nodesVisited;
    bvhInteriorHit += interiorHit;
    bvhInteriorTested += interiorTested;
    bvhLeavesHit += leavesHit;
    bvhLeavesTested += leavesTested;
    return si;
}

bool BinBVHAggregate::IntersectP(const Ray &ray, Float tMax) const {
    if (!nodes)
        return false;
    Vector3f invDir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
    int dirIsNeg[3] = {static_cast<int>(invDir.x < 0), static_cast<int>(invDir.y < 0),
                       static_cast<int>(invDir.z < 0)};
    int nodesToVisit[64];
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesVisited = 0;
    int trianglesTested = 0;
    while (true) {
        ++nodesVisited;
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        if (node->bounds.IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
            // Process BVH newNode _node_ for traversal
            if (node->nPrimitives > 0) {
                trianglesTested += node->nPrimitives;
                for (int i = 0; i < node->nPrimitives; ++i) {
                    if (primitives[node->primitivesOffset + i].IntersectP(ray, tMax)) {
                        bvhSisdNodesVisited += nodesVisited;
                        return true;
                    }
                }
                if (toVisitOffset == 0)
                    break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                if (dirIsNeg[node->axis] != 0) {
                    /// second oldChild first
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0)
                break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    bvhSisdTriangleTests += trianglesTested;
    bvhSisdNodesVisited += nodesVisited;
    return false;
}
//Private Funcs
Float BinBVHAggregate::splitCost(BVHSplitBucket left, BVHSplitBucket right) const {
    ICostCalcable *buckets[2] = {&left, &right};
    return BVHAggregate::splitCost(2, buckets);
}
BVHBuildNode *BinBVHAggregate::buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                              pstd::span<BVHPrimitive> bvhPrimitives,
                                              std::atomic<int> *totalNodes,
                                              std::atomic<int> *orderedPrimsOffset,
                                              std::vector<Primitive> &orderedPrims) {
    DCHECK_NE(bvhPrimitives.size(), 0);
    Allocator alloc = threadAllocators.Get();
    BVHBuildNode *node = alloc.new_object<BVHBuildNode>();
    // Initialize _BVHBuildNode_ for primitive range
    ++*totalNodes;
    // Compute bounds of all primitives in BVH newNode
    Bounds3f bounds;
    for (const auto &prim : bvhPrimitives)
        bounds = Union(bounds, prim.bounds);

    if (bounds.SurfaceArea() == 0 || bvhPrimitives.size() == 1) {
        // Create leaf _BVHBuildNode_
        int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
            int index = bvhPrimitives[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[index];
        }
        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
        return node;

    } else {
        // Compute bound of primitive centroids and choose split dimension _dim_
        Bounds3f centroidBounds;
        for (const auto &prim : bvhPrimitives)
            centroidBounds = Union(centroidBounds, prim.Centroid());
        int dim = centroidBounds.MaxDimension();

        // Partition primitives into two sets and build newChildren
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            // Create leaf _BVHBuildNode_
            int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
            for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                int index = bvhPrimitives[i].primitiveIndex;
                orderedPrims[firstPrimOffset + i] = primitives[index];
            }
            node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
            return node;

        } else {
            int mid = bvhPrimitives.size() / 2;
            // Partition primitives based on _splitMethod_
            switch (splitMethod) {
            case SplitMethod::Middle: {
                // Partition primitives through newNode's midpoint
                Float pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
                auto midIter = std::partition(bvhPrimitives.begin(), bvhPrimitives.end(),
                                              [dim, pmid](const BVHPrimitive &pi) {
                                                  return pi.Centroid()[dim] < pmid;
                                              });
                mid = midIter - bvhPrimitives.begin();
                // For lots of prims with large overlapping bounding boxes, this
                // may fail to partition; in that case do not break and fall through
                // to EqualCounts.
                if (midIter != bvhPrimitives.begin() && midIter != bvhPrimitives.end())
                    break;
                __fallthrough;
            }
            case SplitMethod::EqualCounts: {
                // Partition primitives into equally sized subsets
                mid = bvhPrimitives.size() / 2;
                std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + mid,
                                 bvhPrimitives.end(),
                                 [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                     return a.Centroid()[dim] < b.Centroid()[dim];
                                 });

                break;
            }
            case SplitMethod::EPO:
            case SplitMethod::SAH:
            default: {
                // Partition primitives using approximate SAH
                if (bvhPrimitives.size() <= 2) {
                    // Partition primitives into equally sized subsets
                    mid = bvhPrimitives.size() / 2;
                    std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + mid,
                                     bvhPrimitives.end(),
                                     [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                         return a.Centroid()[dim] < b.Centroid()[dim];
                                     });

                } else {
                    // Allocate _BVHSplitBucket_ for SAH partition buckets
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];

                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : bvhPrimitives) {
                        int b = nBuckets * centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }

                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    BVHSplitBucket bucketsBelow[nSplits];
                    BVHSplitBucket bucketsAbove[nSplits];
                    // Partially initialize _costs_ using a forward scan over splits
                    int countBelow = 0;
                    Bounds3f boundBelow;
                    for (int i = 0; i < nSplits; ++i) {
                        boundBelow = Union(boundBelow, buckets[i].bounds);
                        countBelow += buckets[i].count;
                        bucketsBelow[i].count = countBelow;
                        bucketsBelow[i].bounds = boundBelow;
                    }

                    // Finish initializing _costs_ using a backwards scan over splits
                    int countAbove = 0;
                    Bounds3f boundAbove;
                    for (int i = nSplits; i >= 1; --i) {
                        boundAbove = Union(boundAbove, buckets[i].bounds);
                        countAbove += buckets[i].count;
                        bucketsAbove[i - 1].bounds = boundAbove;
                        bucketsAbove[i - 1].count = countAbove;
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    Float cost;
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        cost = splitCost(bucketsBelow[i], bucketsAbove[i]);
                        CHECK_GT(cost, 0);
                        if (cost < minCost) {
                            minCost = cost;
                            minCostSplitBucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    Float leafCost = BVHAggregate::leafCost(bvhPrimitives.size());
                    minCost = RelativeInnerCost + minCost;

                    // Either create leaf or split primitives at selected SAH bucket
                    if (bvhPrimitives.size() > maxPrimsInNode || minCost < leafCost) {
                        auto midIter = std::partition(
                            bvhPrimitives.begin(), bvhPrimitives.end(),
                            [=](const BVHPrimitive &bp) {
                                int b =
                                    nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                                if (b == nBuckets)
                                    b = nBuckets - 1;
                                return b <= minCostSplitBucket;
                            });
                        mid = midIter - bvhPrimitives.begin();
                    } else {
                        // Create leaf _BVHBuildNode_
                        int firstPrimOffset =
                            orderedPrimsOffset->fetch_add(bvhPrimitives.size());
                        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                            int index = bvhPrimitives[i].primitiveIndex;
                            orderedPrims[firstPrimOffset + i] = primitives[index];
                        }
                        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
                        return node;
                    }
                }

                break;
            }
            }

            BVHBuildNode *children[2];
            // Recursively build BVHs for _children_
            if (bvhPrimitives.size() > 128 * 1024) {
                // Recursively build oldChild BVHs in parallel
                ParallelFor(0, 2, [&](int i) {
                    if (i == 0)
                        children[0] = buildRecursive(
                            threadAllocators, bvhPrimitives.subspan(0, mid), totalNodes,
                            orderedPrimsOffset, orderedPrims);
                    else
                        children[1] =
                            buildRecursive(threadAllocators, bvhPrimitives.subspan(mid),
                                           totalNodes, orderedPrimsOffset, orderedPrims);
                });

            } else {
                // Recursively build oldChild BVHs sequentially
                children[0] =
                    buildRecursive(threadAllocators, bvhPrimitives.subspan(0, mid),
                                   totalNodes, orderedPrimsOffset, orderedPrims);
                children[1] =
                    buildRecursive(threadAllocators, bvhPrimitives.subspan(mid),
                                   totalNodes, orderedPrimsOffset, orderedPrims);
            }

            node->InitInterior(dim, children[0], children[1]);
        }
    }

    return node;
}

BVHBuildNode *BinBVHAggregate::buildHLBVH(Allocator alloc,
                                          const std::vector<BVHPrimitive> &bvhPrimitives,
                                          std::atomic<int> *totalNodes,
                                          std::vector<Primitive> &orderedPrims) {
    // Compute bounding box of all primitive centroids
    Bounds3f bounds;
    for (const BVHPrimitive &prim : bvhPrimitives)
        bounds = Union(bounds, prim.Centroid());

    // Compute Morton indices of primitives
    std::vector<MortonPrimitive> mortonPrims(bvhPrimitives.size());
    ParallelFor(0, bvhPrimitives.size(), [&](int64_t i) {
        // Initialize _mortonPrims[i]_ for _i_th primitive
        constexpr int mortonBits = 10;
        constexpr int mortonScale = 1 << mortonBits;
        mortonPrims[i].primitiveIndex = bvhPrimitives[i].primitiveIndex;
        Vector3f centroidOffset = bounds.Offset(bvhPrimitives[i].Centroid());
        Vector3f offset = centroidOffset * mortonScale;
        mortonPrims[i].mortonCode = EncodeMorton3(offset.x, offset.y, offset.z);
    });

    // Radix sort primitive Morton indices
    RadixSort(&mortonPrims);

    // Create LBVH treelets at bottom of BVH
    // Find intervals of primitives for each treelet
    std::vector<LBVHTreelet> treeletsToBuild;
    for (size_t start = 0, end = 1; end <= mortonPrims.size(); ++end) {
        uint32_t mask = 0b00111111111111000000000000000000;
        if (end == (int)mortonPrims.size() || ((mortonPrims[start].mortonCode & mask) !=
                                               (mortonPrims[end].mortonCode & mask))) {
            // Add entry to _treeletsToBuild_ for this treelet
            size_t nPrimitives = end - start;
            int maxBVHNodes = 2 * nPrimitives - 1;
            BVHBuildNode *nodes = alloc.allocate_object<BVHBuildNode>(maxBVHNodes);
            treeletsToBuild.push_back({start, nPrimitives, nodes});

            start = end;
        }
    }

    // Create LBVHs for treelets in parallel
    std::atomic<int> orderedPrimsOffset(0);
    ParallelFor(0, treeletsToBuild.size(), [&](int i) {
        // Generate _i_th LBVH treelet
        int nodesCreated = 0;
        const int firstBitIndex = 29 - 12;
        LBVHTreelet &tr = treeletsToBuild[i];
        tr.buildNodes = emitLBVH(
            tr.buildNodes, bvhPrimitives, &mortonPrims[tr.startIndex], tr.nPrimitives,
            &nodesCreated, orderedPrims, &orderedPrimsOffset, firstBitIndex);
        *totalNodes += nodesCreated;
    });

    // Create and return SAH BVH from LBVH treelets
    std::vector<BVHBuildNode *> finishedTreelets;
    finishedTreelets.reserve(treeletsToBuild.size());
    for (LBVHTreelet &treelet : treeletsToBuild)
        finishedTreelets.push_back(treelet.buildNodes);
    return buildUpperSAH(alloc, finishedTreelets, 0, finishedTreelets.size(), totalNodes);
}

BVHBuildNode *BinBVHAggregate::emitLBVH(BVHBuildNode *&buildNodes,
                                        const std::vector<BVHPrimitive> &bvhPrimitives,
                                        MortonPrimitive *mortonPrims, int nPrimitives,
                                        int *totalNodes,
                                        std::vector<Primitive> &orderedPrims,
                                        std::atomic<int> *orderedPrimsOffset,
                                        int bitIndex) {
    CHECK_GT(nPrimitives, 0);
    if (bitIndex == -1 || nPrimitives < maxPrimsInNode) {
        // Create and return leaf newNode of LBVH treelet
        ++*totalNodes;
        BVHBuildNode *node = buildNodes++;
        Bounds3f bounds;
        size_t firstPrimOffset = orderedPrimsOffset->fetch_add(nPrimitives);
        for (int i = 0; i < nPrimitives; ++i) {
            int primitiveIndex = mortonPrims[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[primitiveIndex];
            bounds = Union(bounds, bvhPrimitives[primitiveIndex].bounds);
        }
        node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
        return node;

    } else {
        int mask = 1 << bitIndex;
        // Advance to next subtree level if there is no LBVH split for this bit
        if ((mortonPrims[0].mortonCode & mask) ==
            (mortonPrims[nPrimitives - 1].mortonCode & mask))
            return emitLBVH(buildNodes, bvhPrimitives, mortonPrims, nPrimitives,
                            totalNodes, orderedPrims, orderedPrimsOffset, bitIndex - 1);

        // Find LBVH split point for this dimension
        int splitOffset = FindInterval(nPrimitives, [&](int index) {
            return ((mortonPrims[0].mortonCode & mask) ==
                    (mortonPrims[index].mortonCode & mask));
        });
        ++splitOffset;
        CHECK_LE(splitOffset, nPrimitives - 1);
        CHECK_NE(mortonPrims[splitOffset - 1].mortonCode & mask,
                 mortonPrims[splitOffset].mortonCode & mask);

        // Create and return interior LBVH newNode
        (*totalNodes)++;
        BVHBuildNode *node = buildNodes++;
        BVHBuildNode *lbvh[2] = {
            emitLBVH(buildNodes, bvhPrimitives, mortonPrims, splitOffset, totalNodes,
                     orderedPrims, orderedPrimsOffset, bitIndex - 1),
            emitLBVH(buildNodes, bvhPrimitives, &mortonPrims[splitOffset],
                     nPrimitives - splitOffset, totalNodes, orderedPrims,
                     orderedPrimsOffset, bitIndex - 1)};
        int axis = bitIndex % 3;
        node->InitInterior(axis, lbvh[0], lbvh[1]);
        return node;
    }
}

int BinBVHAggregate::flattenBVH(BVHBuildNode *node, int *offset) {
    LinearBVHNode *linearNode = &nodes[*offset];
    linearNode->bounds = node->bounds;
    int nodeOffset = (*offset)++;
    if (node->nPrimitives > 0) {
        CHECK(!node->children[0] && !node->children[1]);
        CHECK_LT(node->nPrimitives, 65536);
        linearNode->primitivesOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
    } else {
        // Create interior flattened BVH newNode
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;
        flattenBVH(node->children[0], offset);
        linearNode->secondChildOffset = flattenBVH(node->children[1], offset);
    }
    return nodeOffset;
}
BVHBuildNode *BinBVHAggregate::buildUpperSAH(Allocator alloc,
                                             std::vector<BVHBuildNode *> &treeletRoots,
                                             int start, int end,
                                             std::atomic<int> *totalNodes) const {
    CHECK_LT(start, end);
    int nNodes = end - start;
    if (nNodes == 1)
        return treeletRoots[start];
    (*totalNodes)++;
    BVHBuildNode *node = alloc.new_object<BVHBuildNode>();

    // Compute bounds of all nodes under this HLBVH newNode
    Bounds3f bounds;
    for (int i = start; i < end; ++i)
        bounds = Union(bounds, treeletRoots[i]->bounds);

    // Compute bound of HLBVH newNode centroids, choose split dimension _dim_
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i) {
        Point3f centroid =
            (treeletRoots[i]->bounds.pMin + treeletRoots[i]->bounds.pMax) * 0.5f;
        centroidBounds = Union(centroidBounds, centroid);
    }
    int dim = centroidBounds.MaxDimension();
    // FIXME: if this hits, what do we need to do?
    // Make sure the SAH split below does something... ?
    CHECK_NE(centroidBounds.pMax[dim], centroidBounds.pMin[dim]);

    // Allocate _BVHSplitBucket_ for SAH partition buckets
    constexpr int nBuckets = 12;
    struct BVHSplitBucket {
        int count = 0;
        Bounds3f bounds;
    };
    BVHSplitBucket buckets[nBuckets];

    // Initialize _BVHSplitBucket_ for HLBVH SAH partition buckets
    for (int i = start; i < end; ++i) {
        Float centroid =
            (treeletRoots[i]->bounds.pMin[dim] + treeletRoots[i]->bounds.pMax[dim]) *
            0.5f;
        int b = nBuckets * ((centroid - centroidBounds.pMin[dim]) /
                            (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
        if (b == nBuckets)
            b = nBuckets - 1;
        CHECK_GE(b, 0);
        CHECK_LT(b, nBuckets);
        buckets[b].count++;
        buckets[b].bounds = Union(buckets[b].bounds, treeletRoots[i]->bounds);
    }

    // Compute costs for splitting after each bucket
    Float cost[nBuckets - 1];
    for (int i = 0; i < nBuckets - 1; ++i) {
        Bounds3f b0, b1;
        int count0 = 0, count1 = 0;
        for (int j = 0; j <= i; ++j) {
            b0 = Union(b0, buckets[j].bounds);
            count0 += buckets[j].count;
        }
        for (int j = i + 1; j < nBuckets; ++j) {
            b1 = Union(b1, buckets[j].bounds);
            count1 += buckets[j].count;
        }
        cost[i] = .125f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) /
                              bounds.SurfaceArea();
    }

    // Find bucket to split at that minimizes SAH metric
    Float minCost = cost[0];
    int minCostSplitBucket = 0;
    for (int i = 1; i < nBuckets - 1; ++i) {
        if (cost[i] < minCost) {
            minCost = cost[i];
            minCostSplitBucket = i;
        }
    }

    // Split nodes and create interior HLBVH SAH newNode
    BVHBuildNode **pmid = std::partition(
        &treeletRoots[start], &treeletRoots[end - 1] + 1, [=](const BVHBuildNode *node) {
            Float centroid = (node->bounds.pMin[dim] + node->bounds.pMax[dim]) * 0.5f;
            int b = nBuckets * ((centroid - centroidBounds.pMin[dim]) /
                                (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
            if (b == nBuckets)
                b = nBuckets - 1;
            CHECK_GE(b, 0);
            CHECK_LT(b, nBuckets);
            return b <= minCostSplitBucket;
        });
    int mid = pmid - &treeletRoots[0];
    CHECK_GT(mid, start);
    CHECK_LT(mid, end);
    node->InitInterior(dim,
                       this->buildUpperSAH(alloc, treeletRoots, start, mid, totalNodes),
                       this->buildUpperSAH(alloc, treeletRoots, mid, end, totalNodes));
    return node;
}
Float BinBVHAggregate::calcMetrics(BVHBuildNode *root) const { 
    if (root->nPrimitives > 0) {
        return leafCost(root->nPrimitives);
    } else {
        Float cost = RelativeInnerCost * pstd::ceil(2 / (Float)SimdWidth);
        ICostCalcable *children[2];
        children[0] = root->children[0];
        children[1] = root->children[1];
        CHECK_GE(penalizedHitProbability(0, 2, children), root->children[0]->bounds.SurfaceArea() / root->bounds.SurfaceArea());
        cost += penalizedHitProbability(0, 2, children) * calcMetrics(root->children[0]);
        cost += penalizedHitProbability(1, 2, children) * calcMetrics(root->children[1]);
        return cost;
    }
}

#pragma endregion

#pragma region BvhFour
//vfuncs BVH
pstd::optional<ShapeIntersection> FourWideBVHAggregate::Intersect(const Ray &ray,
                                                                  Float tMax) const {
    pstd::optional<ShapeIntersection> si;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    if (!nodes)
        return {};
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    std::vector<int> nodesToVisit(64);
    int nodesVisited = 0;
    int simdNodesVisited = 0;
    int simdTriangleTests = 0;
    int triangleTests = 0;
    int leavesTested = 0;
    int leavesHit = 0;
    int interiorTested = 0;
    int interiorHit = 0;
    while (true) {
        ++simdNodesVisited;
        const SIMDFourWideLinearBVHNode *node = &nodes[currentNodeIndex];
        const uint8_t *idxArr =
            traversalOrder[dirIsNeg[node->Axis0()]][dirIsNeg[node->Axis1()]]
                          [dirIsNeg[node->Axis2()]];
        for (int i = 0; i < TreeWidth; ++i) {
            uint8_t idx = idxArr[i];
            if (node->offsets[idx] < 0)
                continue;
            ++nodesVisited;
            if (node->nPrimitives[idx] > 0) {
                leavesTested++;
            } else {
                interiorTested++;
            }
            if (node->bounds[idx].IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
                // Leaf oldChild
                if (node->nPrimitives[idx] > 0) {
                    leavesHit++;
                    bvhAvgLeaveSizeHit << (int)(node->nPrimitives[idx]);
                    simdTriangleTests +=
                        pstd::ceil(node->nPrimitives[idx] / (float)SimdWidth);
                    triangleTests += node->nPrimitives[idx];
                    // Intersect ray with primitives in leaf BVH newNode
                    for (size_t j = 0; j < node->nPrimitives[idx]; ++j) {
                        // Check for intersection with primitive in BVH newNode
                        pstd::optional<ShapeIntersection> primSi =
                            primitives[node->offsets[idx] + j].Intersect(ray, tMax);
                        if (primSi) {
                            si = primSi;
                            tMax = si->tHit;
                        }
                    }
                }
                // Interior oldChild
                else {
                    interiorHit++;
                    nodesToVisit[toVisitOffset++] = node->offsets[idx];
                }
            }
        }
        if (toVisitOffset == 0)
            break;
        currentNodeIndex = nodesToVisit[--toVisitOffset];
    }
    bvhSimdTriangleTests += simdTriangleTests;
    bvhSimdNodesVisited += simdNodesVisited;
    bvhSisdNodesVisited += nodesVisited;
    bvhSisdTriangleTests += triangleTests;
    bvhInteriorHit += interiorHit;
    bvhInteriorTested += interiorTested;
    bvhLeavesHit += leavesHit;
    bvhLeavesTested += leavesTested;
    return si;
}
bool FourWideBVHAggregate::IntersectP(const Ray &ray, Float tMax) const {
    Vector3f invDir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
    int dirIsNeg[3] = {static_cast<int>(invDir.x < 0), static_cast<int>(invDir.y < 0),
                       static_cast<int>(invDir.z < 0)};
    if (!nodes || !this->bounds.IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg))
        return false;
    std::vector<int> nodesToVisit = std::vector<int>(64);
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesVisited = 0;
    int simdNodesVisited = 0;
    int simdTriangleTests = 0;
    int triangleTests = 0;
    while (true) {
        ++simdNodesVisited;
        CHECK_GE(currentNodeIndex, 0);
        const SIMDFourWideLinearBVHNode *node = &nodes[currentNodeIndex];
        const uint8_t *idxArr =
            traversalOrder[dirIsNeg[node->Axis0()]][dirIsNeg[node->Axis1()]]
                          [dirIsNeg[node->Axis2()]];
        for (int i = 0; i < TreeWidth; ++i) {
            uint8_t idx = idxArr[i];
            if (node->offsets[idx] < 0)
                continue;
            ++nodesVisited;
            if (node->bounds[idx].IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
                // Leaf oldChild
                if (node->nPrimitives[idx] > 0) {
                    simdTriangleTests +=
                        pstd::ceil(node->nPrimitives[idx] / (float)SimdWidth);
                    triangleTests += node->nPrimitives[idx];
                    // Intersect ray with primitives in leaf BVH newNode
                    for (size_t j = 0; j < node->nPrimitives[idx]; ++j) {
                        // Check for intersection with primitive in BVH newNode
                        if (primitives[node->offsets[idx] + j].IntersectP(ray, tMax)) {
                            return true;
                        }
                    }
                }
                // Interior oldChild
                else {
                    nodesToVisit[toVisitOffset++] = node->offsets[idx];
                }
            }
        }
        if (toVisitOffset == 0)
            break;
        currentNodeIndex = nodesToVisit[--toVisitOffset];
    }
    bvhSisdNodesVisited += nodesVisited;
    bvhSimdNodesVisited += simdNodesVisited;
    bvhSisdTriangleTests += triangleTests;
    bvhSimdTriangleTests += simdTriangleTests;
    return false;
}
//vfuncs WideBVH
int8_t FourWideBVHAggregate::relevantAxisIdx(int axis1, int axis2) const {
    return relevantAxisIdxArr[axis1][axis2];
}
//Private functions
FourWideBVHBuildNode *FourWideBVHAggregate::buildFromBVH(BVHBuildNode *binRoot,
                                                         std::atomic<int> *totalNodes) {
    if (binRoot->nPrimitives > 0) {
        FourWideBVHBuildNode *leaf = new FourWideBVHBuildNode();
        leaf->InitLeaf(binRoot->firstPrimOffset, binRoot->nPrimitives, binRoot->bounds);
        return leaf;
    }
    FourWideBVHBuildNode *node = new FourWideBVHBuildNode();
    FourWideBVHBuildNode *children[4]{};
    int axis[3] = {binRoot->children[0]->splitAxis, binRoot->splitAxis,
                   binRoot->children[1]->splitAxis};
    for (int i = 0; i < 2; ++i) {
        auto child = binRoot->children[i];
        if (child->nPrimitives > 0) {
            children[i * 2] = buildFromBVH(child, totalNodes);
            children[i * 2 + 1] = NULL;
        } else {
            children[i * 2] = buildFromBVH(child->children[0], totalNodes);
            children[i * 2 + 1] = buildFromBVH(child->children[1], totalNodes);
        }
    }
    ++*totalNodes;
    node->InitInterior(axis, children);
    return node;
};
FourWideBVHBuildNode *FourWideBVHAggregate::buildRecursive(
    ThreadLocal<Allocator> &threadAllocators, pstd::span<BVHPrimitive> bvhPrimitives,
    std::atomic<int> *totalNodes, std::atomic<int> *orderedPrimsOffset,
    std::vector<Primitive> &orderedPrims, int splitVariant) {
    if (bvhPrimitives.size() < 1) {
        return nullptr;
    }
    Allocator alloc = threadAllocators.Get();
    FourWideBVHBuildNode *node = alloc.new_object<FourWideBVHBuildNode>();
    // Initialize _BVHBuildNode_ for primitive range
    // Compute bounds of all primitives in BVH newNode
    Bounds3f bounds;
    for (const auto &prim : bvhPrimitives)
        bounds = Union(bounds, prim.bounds);
    if (bounds.SurfaceArea() == 0 || bvhPrimitives.size() < 2) {
        // Create leaf _BVHBuildNode_
        int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
            int index = bvhPrimitives[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[index];
        }
        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
        return node;

    } else {
        // Compute bound of primitive centroids and choose split dimensions _dim_
        int axis[3]{};
        Bounds3f centroidBounds;
        for (const auto &prim : bvhPrimitives)
            centroidBounds = Union(centroidBounds, prim.Centroid());
        axis[1] = centroidBounds.MaxDimension();
        // Decide on Leaf or inner
        if (centroidBounds.pMax[axis[1]] == centroidBounds.pMin[axis[1]]) {
            // Create leaf _BVHBuildNode_
            int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
            for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                int index = bvhPrimitives[i].primitiveIndex;
                orderedPrims[firstPrimOffset + i] = primitives[index];
            }
            node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
            return node;
        }
        // Cut in 4
        int splits[3]{};
        switch (splitMethod) {
        case SplitMethod::Middle: {
            // Partition primitives through newNode's midpoint
            int dim = axis[1];
            Float pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
            auto midIter = std::partition(bvhPrimitives.begin(), bvhPrimitives.end(),
                                          [dim, pmid](const BVHPrimitive &pi) {
                                              return pi.Centroid()[dim] < pmid;
                                          });
            splits[1] = midIter - bvhPrimitives.begin();
            centroidBounds = Bounds3f();
            for (size_t i = 0; i < splits[1]; ++i) {
                auto &prim = bvhPrimitives.begin()[i];
                centroidBounds = Union(centroidBounds, prim.Centroid());
            }
            dim = axis[0] = centroidBounds.MaxDimension();
            pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
            // Split along secondary plane
            auto midIter2 = std::partition(bvhPrimitives.begin(), midIter,
                                           [dim, pmid](const BVHPrimitive &pi) {
                                               return pi.Centroid()[dim] < pmid;
                                           });
            splits[0] = midIter2 - bvhPrimitives.begin();
            centroidBounds = Bounds3f();
            for (size_t i = splits[1]; i < bvhPrimitives.size(); ++i) {
                auto &prim = bvhPrimitives.begin()[i];
                centroidBounds = Union(centroidBounds, prim.Centroid());
            }
            dim = axis[2] = centroidBounds.MaxDimension();
            pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
            midIter2 = std::partition(midIter + 1, bvhPrimitives.end(),
                                      [dim, pmid](const BVHPrimitive &pi) {
                                          return pi.Centroid()[dim] < pmid;
                                      });
            splits[2] = midIter2 - bvhPrimitives.begin();
            // For lots of prims with large overlapping bounding boxes, this
            // may fail to partition; in that case do not break and fall through
            // to EqualCounts.
            if (splits[0] <= splits[1] && splits[1] <= splits[2] &&
                (splits[0] < bvhPrimitives.size() || splits[2] > 0))
                break;
            // Fall back on equal counts if splits are malformed
            __fallthrough;
        }
        case SplitMethod::EqualCounts: {
            int dim = axis[1];
            int dim2 = axis[0] = axis[2] = ((centroidBounds.MaxDimensions() >> 2) & 3);
            splits[1] = bvhPrimitives.size() / 2;
            splits[0] = splits[1] / 2;
            splits[2] = splits[1] + (bvhPrimitives.size() - splits[1]) / 2;
            std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + splits[1],
                             bvhPrimitives.end(),
                             [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                 return a.Centroid()[dim] < b.Centroid()[dim];
                             });
            std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + splits[0],
                             bvhPrimitives.begin() + splits[1],
                             [dim2](const BVHPrimitive &a, const BVHPrimitive &b) {
                                 return a.Centroid()[dim2] < b.Centroid()[dim2];
                             });
            std::nth_element(bvhPrimitives.begin() + splits[1] + 1,
                             bvhPrimitives.begin() + splits[2], bvhPrimitives.end(),
                             [dim2](const BVHPrimitive &a, const BVHPrimitive &b) {
                                 return a.Centroid()[dim2] < b.Centroid()[dim2];
                             });
            break;
        }
        case SplitMethod::EPO:
        case SplitMethod::SAH:
        default: {
            switch (splitVariant) {
            case 1: {
                // Splitting on 1 Dimension at 3 points
                // Partition primitives using approximate SAH
                // Allocate _BVHSplitBucket_ for SAH partition buckets
                constexpr int nBuckets = 24;
                BVHSplitBucket buckets[nBuckets];
                int dim = axis[0] = axis[2] = axis[1];
                // Initialize _BVHSplitBucket_ for SAH partition buckets
                for (const auto &prim : bvhPrimitives) {
                    int b = nBuckets * centroidBounds.Offset(prim.Centroid())[dim];
                    if (b == nBuckets)
                        b = nBuckets - 1;
                    DCHECK_GE(b, 0);
                    DCHECK_LT(b, nBuckets);
                    buckets[b].count++;
                    buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                }

                // Compute costs for splitting after each bucket
                constexpr int nSplits = nBuckets - 1;
                BVHSplitBucket bucketsAbove[nSplits];
                BVHSplitBucket bucketsBelow[nSplits];
                // Partially initialize _costs_ using a forward scan over splits
                bucketsBelow[0].bounds = buckets[0].bounds;
                bucketsBelow[0].count = buckets[0].count;
                for (int i = 1; i < nSplits; ++i) {
                    bucketsBelow[i].bounds =
                        Union(bucketsBelow[i - 1].bounds, buckets[i].bounds);
                    bucketsBelow[i].count = bucketsBelow[i - 1].count + buckets[i].count;
                }
                // Finish initializing _costs_ using a backwards scan over splits

                bucketsAbove[nSplits - 1].bounds = buckets[nSplits].bounds;
                bucketsAbove[nSplits - 1].count = buckets[nSplits].count;
                for (int i = nSplits - 1; i >= 1; --i) {
                    bucketsAbove[i - 1].bounds =
                        Union(bucketsAbove[i].bounds, buckets[i].bounds);
                    bucketsAbove[i - 1].count = bucketsAbove[i].count + buckets[i].count;
                }
                for (int i = 0; i < nSplits; ++i) {
                    DCHECK_EQ(bucketsAbove[i].count + bucketsBelow[i].count,
                              bvhPrimitives.size());
                }
                // Find bucket to split at that minimizes SAH metric
                int minCostSplitBucket1 = -1;
                Float minCost = Infinity;
                ICostCalcable *curBuckets[4];
                for (int i = 0; i < nSplits; ++i) {
                    // Compute cost for candidate split and update minimum if
                    // necessary
                    curBuckets[0] = &bucketsBelow[i];
                    curBuckets[1] = &bucketsAbove[i];
                    float cost = splitCost(2, curBuckets);
                    if (cost < minCost) {
                        minCost = cost;
                        minCostSplitBucket1 = i;
                    }
                }
                CHECK_GE(minCostSplitBucket1, 0);
                CHECK_LT(minCostSplitBucket1, nSplits);
                // Initialize below buckets for both sides of split1
                bucketsBelow[minCostSplitBucket1] = {};
                for (int i = minCostSplitBucket1 + 1; i < nSplits; ++i) {
                    bucketsBelow[i].bounds =
                        Union(bucketsBelow[i - 1].bounds, buckets[i].bounds);
                    bucketsBelow[i].count = bucketsBelow[i - 1].count + buckets[i].count;
                }
                bucketsBelow[0].bounds = buckets[0].bounds;
                bucketsBelow[0].count = buckets[0].count;
                for (int i = 1; i <= minCostSplitBucket1; ++i) {
                    bucketsBelow[i].bounds =
                        Union(bucketsBelow[i - 1].bounds, buckets[i].bounds);
                    bucketsBelow[i].count = bucketsBelow[i - 1].count + buckets[i].count;
                }

                // Initialize above buckets for both sides of split1
                bucketsAbove[minCostSplitBucket1 - 1].bounds =
                    buckets[minCostSplitBucket1].bounds;
                bucketsAbove[minCostSplitBucket1 - 1].count =
                    buckets[minCostSplitBucket1].count;
                for (int i = minCostSplitBucket1 - 1; i >= 1; --i) {
                    bucketsAbove[i - 1].bounds =
                        Union(bucketsAbove[i].bounds, buckets[i].bounds);
                    bucketsAbove[i - 1].count = bucketsAbove[i].count + buckets[i].count;
                }

                // Finish initializing _costs_ using a backwards scan over splits
                bucketsAbove[nSplits - 1].bounds = buckets[nSplits].bounds;
                bucketsAbove[nSplits - 1].count = buckets[nSplits].count;
                for (int i = nSplits - 1; i > minCostSplitBucket1; --i) {
                    bucketsAbove[i - 1].bounds =
                        Union(bucketsAbove[i].bounds, buckets[i].bounds);
                    bucketsAbove[i - 1].count = bucketsAbove[i].count + buckets[i].count;
                }
                BVHSplitBucket emptyBucket = {};
                minCost = Infinity;
                int minCostSplitBucket0 = -1;
                int minCostSplitBucket2 = -1;
                for (int split0 = 0; split0 <= minCostSplitBucket1; ++split0) {
                    for (int split2 = minCostSplitBucket1; split2 < nSplits; split2++) {
                        curBuckets[0] = &bucketsBelow[split0];
                        curBuckets[1] = split0 == minCostSplitBucket1
                                            ? &emptyBucket
                                            : &bucketsAbove[split0];
                        curBuckets[2] = split2 == minCostSplitBucket1
                                            ? &emptyBucket
                                            : &bucketsBelow[split2];
                        curBuckets[3] = &bucketsAbove[split2];
                        DCHECK_EQ(
                            curBuckets[0]->PrimCount() + curBuckets[1]->PrimCount() +
                                curBuckets[2]->PrimCount() + curBuckets[3]->PrimCount(),
                            bvhPrimitives.size());
                        float cost = splitCost(4, curBuckets);
                        if (cost < minCost) {
                            minCost = cost;
                            minCostSplitBucket0 = split0;
                            minCostSplitBucket2 = split2;
                        }
                    }
                }
                if (minCostSplitBucket0 < 0)
                    minCostSplitBucket0 = 0;
                if (minCostSplitBucket2 < 0)
                    minCostSplitBucket2 = minCostSplitBucket1;

                // Compute leaf cost and SAH split cost for chosen split
                minCost = RelativeInnerCost + minCost;
                // Either create leaf or split primitives at selected SAH bucket
                if (bvhPrimitives.size() > maxPrimsInNode ||
                    minCost < leafCost(bvhPrimitives.size())) {
                    auto midIter = std::partition(
                        bvhPrimitives.begin(), bvhPrimitives.end(),
                        [=](const BVHPrimitive &bp) {
                            int b = nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                            if (b == nBuckets)
                                b = nBuckets - 1;
                            return b <= minCostSplitBucket1;
                        });
                    splits[1] = midIter - bvhPrimitives.begin();
                    auto midIter2 = std::partition(
                        bvhPrimitives.begin(), midIter, [=](const BVHPrimitive &bp) {
                            int b = nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                            if (b == nBuckets)
                                b = nBuckets - 1;
                            return b <= minCostSplitBucket0;
                        });
                    splits[0] = midIter2 - bvhPrimitives.begin();
                    midIter2 = std::partition(
                        midIter, bvhPrimitives.end(), [=](const BVHPrimitive &bp) {
                            int b = nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                            if (b == nBuckets)
                                b = nBuckets - 1;
                            return b <= minCostSplitBucket2;
                        });
                    splits[2] = midIter2 - bvhPrimitives.begin();
                } else {
                    // Create leaf _BVHBuildNode_
                    int firstPrimOffset =
                        orderedPrimsOffset->fetch_add(bvhPrimitives.size());
                    for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                        int index = bvhPrimitives[i].primitiveIndex;
                        orderedPrims[firstPrimOffset + i] = primitives[index];
                    }
                    node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
                    return node;
                }
            } break;
            case 3:
            case 2: {
                // Splitting on a 2 dimensional array
                // Have a bucket array on 2 dimenions
                //    Allocate _BVHSplitBucket_ for SAH partition buckets
                constexpr int nBuckets = 12;
                constexpr int nSplits = nBuckets - 1;
                Bounds3f cumCentroidBoundsAbv[nSplits];
                Bounds3f cumCentroidBoundsBel[nSplits];
                Bounds3f centroidBoundsAbv;
                Bounds3f centroidBoundsBel;
                int dim1ProposedSplit = -1;
                // The difference in these variants lies in the dimensions used for the
                // columns
                bool complexSplit = splitVariant == 3;
                if (!complexSplit) {
                    axis[0] = axis[2] = ((centroidBounds.MaxDimensions() >> 2) & 3);
                    centroidBoundsAbv = centroidBounds;
                    centroidBoundsBel = centroidBounds;
                } else {
                    // splitVarian == 3
                    Bounds3f centroidBoundsDim1[nBuckets];
                    BVHSplitBucket dim1Buckets[nBuckets];
                    // Compute proposed split on main dimension
                    for (const auto &prim : bvhPrimitives) {
                        int b =
                            nBuckets * centroidBounds.Offset(prim.Centroid())[axis[1]];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        dim1Buckets[b].count++;
                        dim1Buckets[b].bounds = Union(dim1Buckets[b].bounds, prim.bounds);
                        centroidBoundsDim1[b] =
                            Union(centroidBoundsDim1[b], prim.Centroid());
                    }
                    // Compute costs for splitting after each bucket
                    BVHSplitBucket tempBucketsAbv[nBuckets];
                    cumCentroidBoundsAbv[0] = centroidBoundsDim1[0];
                    tempBucketsAbv[0].bounds = dim1Buckets[0].bounds;
                    tempBucketsAbv[0].count = dim1Buckets[0].count;
                    for (int i = 1; i < nSplits; ++i) {
                        tempBucketsAbv[i].bounds =
                            Union(tempBucketsAbv[i - 1].bounds, dim1Buckets[i].bounds);
                        tempBucketsAbv[i].count =
                            tempBucketsAbv[i - 1].count + dim1Buckets[i].count;
                        cumCentroidBoundsAbv[i] =
                            Union(cumCentroidBoundsAbv[i - 1], centroidBoundsDim1[i]);
                    }
                    // Finish initializing _costs_ using a backwards scan over splits
                    BVHSplitBucket tempBucketsBel[nSplits];
                    cumCentroidBoundsBel[nSplits - 1] = centroidBoundsDim1[nSplits];
                    tempBucketsBel[nSplits - 1].bounds = dim1Buckets[nSplits].bounds;
                    tempBucketsBel[nSplits - 1].count = dim1Buckets[nSplits].count;
                    for (int i = nSplits - 1; i >= 1; --i) {
                        tempBucketsBel[i - 1].bounds =
                            Union(tempBucketsBel[i].bounds, dim1Buckets[i].bounds);
                        tempBucketsBel[i - 1].count =
                            tempBucketsBel[i].count + dim1Buckets[i].count;
                        cumCentroidBoundsBel[i - 1] =
                            Union(cumCentroidBoundsBel[i], centroidBoundsDim1[i]);
                    }
                    // Find bucket to split at that minimizes metric
                    Float minCost = Infinity;
                    ICostCalcable *curBuckets[2];
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        curBuckets[0] = &tempBucketsBel[i];
                        curBuckets[1] = &tempBucketsAbv[i];
                        float cost = splitCost(2, curBuckets);
                        if (cost < minCost) {
                            minCost = cost;
                            dim1ProposedSplit = i;
                        }
                    }
                    CHECK_GE(dim1ProposedSplit, 0);
                    centroidBoundsAbv = cumCentroidBoundsAbv[dim1ProposedSplit];
                    centroidBoundsBel = cumCentroidBoundsBel[dim1ProposedSplit];
                    // Approximate best sunbsequent split axis by using the max dimension
                    // for most probable split on main dimension
                    axis[0] = cumCentroidBoundsAbv[dim1ProposedSplit].MaxDimension();
                    axis[2] = cumCentroidBoundsBel[dim1ProposedSplit].MaxDimension();
                }
                Float minCost = Infinity;
                int minCostRow = -1;
                int minCostColAbove = -1;
                int minCostColBelow = -1;
                do {
                    // Update bounds and split row if this is not the first iteration
                    if (minCostRow != -1) {
                        dim1ProposedSplit = minCostRow;
                        centroidBoundsAbv = cumCentroidBoundsAbv[dim1ProposedSplit];
                        centroidBoundsBel = cumCentroidBoundsBel[dim1ProposedSplit];
                        // Approximate best sunbsequent split axis by using the max
                        // dimension for most probable split on main dimension
                        axis[0] = cumCentroidBoundsAbv[dim1ProposedSplit].MaxDimension();
                        axis[2] = cumCentroidBoundsBel[dim1ProposedSplit].MaxDimension();
                    }
                    BVHSplitBucket buckets0[nBuckets][nBuckets];
                    BVHSplitBucket buckets2[nBuckets][nBuckets];
                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : bvhPrimitives) {
                        int b1 =
                            nBuckets * centroidBounds.Offset(prim.Centroid())[axis[1]];
                        if (b1 == nBuckets)
                            b1 = nBuckets - 1;
                        DCHECK_GE(b1, 0);
                        DCHECK_LT(b1, nBuckets);
                        int b0 =
                            nBuckets * centroidBoundsAbv.Offset(prim.Centroid())[axis[0]];
                        if (b0 >= nBuckets)
                            b0 = nBuckets - 1;
                        if (b0 < 0)
                            b0 = 0;
                        int b2 =
                            nBuckets * centroidBoundsBel.Offset(prim.Centroid())[axis[2]];
                        if (b2 >= nBuckets)
                            b2 = nBuckets - 1;
                        if (b2 < 0)
                            b2 = 0;
                        DCHECK_GE(b0, 0);
                        DCHECK_LT(b0, nBuckets);
                        DCHECK_GE(b2, 0);
                        DCHECK_LT(b2, nBuckets);
                        buckets0[b1][b0].count++;
                        buckets0[b1][b0].bounds =
                            Union(buckets0[b1][b2].bounds, prim.bounds);
                        buckets2[b1][b2].count++;
                        buckets2[b1][b2].bounds =
                            Union(buckets2[b1][b2].bounds, prim.bounds);
                    }

                    //[dim1Idx][dim2Idx]
                    std::vector<std::vector<BVHSplitBucket>> dim1Above =
                        std::vector(nSplits, std::vector<BVHSplitBucket>(nBuckets));
                    // BVHSplitBucket dim1Above[nSplits][nBuckets] = {};
                    std::vector<std::vector<BVHSplitBucket>> dim1Below =
                        std::vector(nSplits, std::vector<BVHSplitBucket>(nBuckets));
                    std::vector<std::vector<BVHSplitBucket>> bucketsAboveL =
                        std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits));
                    std::vector<std::vector<BVHSplitBucket>> bucketsBelowL =
                        std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits));
                    std::vector<std::vector<BVHSplitBucket>> bucketsAboveR =
                        std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits));
                    std::vector<std::vector<BVHSplitBucket>> bucketsBelowR =
                        std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits));
                    // Compute costs for potential splits
                    // Compute dimensions independent
                    // For each column precompute rows O(2 * nBuckets * (nBuckets-1))
                    for (int col = 0; col < nBuckets; ++col) {
                        // precompute above
                        dim1Above[0][col].bounds = buckets0[0][col].bounds;
                        dim1Above[0][col].count = buckets0[0][col].count;
                        for (size_t i = 1; i < nSplits; ++i) {
                            dim1Above[i][col].bounds = Union(dim1Above[i - 1][col].bounds,
                                                             buckets0[i][col].bounds);
                            dim1Above[i][col].count =
                                dim1Above[i - 1][col].count + buckets0[i][col].count;
                        }
                        // precompute below
                        dim1Below[nSplits - 1][col].bounds =
                            buckets2[nSplits][col].bounds;
                        dim1Below[nSplits - 1][col].count = buckets2[nSplits][col].count;
                        for (size_t i = nSplits - 1; i >= 1; --i) {
                            dim1Below[i - 1][col].bounds =
                                Union(dim1Below[i][col].bounds, buckets2[i][col].bounds);
                            dim1Below[i - 1][col].count =
                                dim1Below[i][col].count + buckets2[i][col].count;
                        }
                    }
                    // Compute costs for splits above(rows < splitRow) by summing
                    // precomputed values
                    for (size_t row = 0; row < nSplits; row++) {
                        // Comute splits above (j<i)
                        bucketsAboveL[row][0].bounds = dim1Above[row][0].bounds;
                        bucketsAboveL[row][0].count = dim1Above[row][0].count;
                        for (size_t col = 1; col < nSplits; ++col) {
                            bucketsAboveL[row][col].bounds =
                                Union(bucketsAboveL[row][col - 1].bounds,
                                      dim1Above[row][col].bounds);
                            bucketsAboveL[row][col].count =
                                bucketsAboveL[row][col - 1].count +
                                dim1Above[row][col].count;
                        }
                        bucketsAboveR[row][nSplits - 1].bounds =
                            dim1Above[row][nSplits].bounds;
                        bucketsAboveR[row][nSplits - 1].count =
                            dim1Above[row][nSplits].count;
                        for (size_t col = nSplits - 1; col >= 1; --col) {
                            bucketsAboveR[row][col - 1].bounds =
                                Union(bucketsAboveR[row][col].bounds,
                                      dim1Above[row][col].bounds);
                            bucketsAboveR[row][col - 1].count =
                                bucketsAboveR[row][col].count + dim1Above[row][col].count;
                        }
                    }
                    // Compute costs for splits below(rows > splitRow) by summing
                    // precomputed values
                    for (size_t row = 0; row < nSplits; row++) {
                        // Comute splits below (j>i)
                        bucketsBelowL[row][0].bounds = dim1Below[row][0].bounds;
                        bucketsBelowL[row][0].count = dim1Below[row][0].count;
                        for (size_t col = 1; col < nSplits; ++col) {
                            bucketsBelowL[row][col].bounds =
                                Union(bucketsBelowL[row][col - 1].bounds,
                                      dim1Below[row][col].bounds);
                            bucketsBelowL[row][col].count =
                                bucketsBelowL[row][col - 1].count +
                                dim1Below[row][col].count;
                        }
                        bucketsBelowR[row][nSplits - 1].bounds =
                            dim1Below[row][nSplits].bounds;
                        bucketsBelowR[row][nSplits - 1].count =
                            dim1Below[row][nSplits].count;
                        for (size_t col = nSplits - 1; col >= 1; --col) {
                            bucketsBelowR[row][col - 1].bounds =
                                Union(bucketsBelowR[row][col].bounds,
                                      dim1Below[row][col].bounds);
                            bucketsBelowR[row][col - 1].count =
                                bucketsBelowR[row][col].count + dim1Below[row][col].count;
                        }
                    }
                    ICostCalcable *curBuckets[4];
                    for (size_t row = 0; row < nSplits; ++row) {
                        for (size_t colAbove = 0; colAbove < nSplits; ++colAbove) {
                            for (size_t colBelow = 0; colBelow < nSplits; ++colBelow) {
                                // Compute cost for candidate split and update minimum if
                                // necessary
                                curBuckets[0] = &bucketsAboveL[row][colAbove];
                                curBuckets[1] = &bucketsAboveR[row][colAbove];
                                curBuckets[2] = &bucketsBelowL[row][colBelow];
                                curBuckets[3] = &bucketsBelowR[row][colBelow];
                                DCHECK_EQ(bvhPrimitives.size(),
                                          curBuckets[0]->PrimCount() +
                                              curBuckets[1]->PrimCount() +
                                              curBuckets[2]->PrimCount() +
                                              curBuckets[3]->PrimCount());

                                float cost = splitCost(4, curBuckets);
                                DCHECK_GT(cost, 0);
                                if (cost < minCost) {
                                    minCost = cost;
                                    minCostRow = row;
                                    minCostColAbove = colAbove;
                                    minCostColBelow = colBelow;
                                }
                            }
                        }
                    }
                    if (complexSplit) {
                        ++bvhAllPrediction;
                        if (dim1ProposedSplit != minCostRow)
                            ++bvhWrongPrediction;
                    }
                } while (complexSplit && minCostRow != dim1ProposedSplit);

                // Can free resources here
                // Compute leaf cost and split cost for chosen split
                minCost = RelativeInnerCost + minCost;
                if (leafCost(bvhPrimitives.size()) <= minCost &&
                    bvhPrimitives.size() <= maxPrimsInNode) {
                    int firstPrimOffset =
                        orderedPrimsOffset->fetch_add(bvhPrimitives.size());
                    for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                        int index = bvhPrimitives[i].primitiveIndex;
                        orderedPrims[firstPrimOffset + i] = primitives[index];
                    }
                    node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
                    return node;
                }
                auto midIter = std::partition(
                    bvhPrimitives.begin(), bvhPrimitives.end(),
                    [=](const BVHPrimitive &bp) {
                        int b = nBuckets * centroidBounds.Offset(bp.Centroid())[axis[1]];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        return b <= minCostRow;
                    });
                splits[1] = midIter - bvhPrimitives.begin();
                auto midIter2 = std::partition(
                    bvhPrimitives.begin(), midIter, [=](const BVHPrimitive &bp) {
                        int b =
                            nBuckets * centroidBoundsAbv.Offset(bp.Centroid())[axis[0]];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        return b <= minCostColAbove;
                    });
                splits[0] = midIter2 - bvhPrimitives.begin();
                midIter2 = std::partition(
                    midIter, bvhPrimitives.end(), [=](const BVHPrimitive &bp) {
                        int b =
                            nBuckets * centroidBoundsBel.Offset(bp.Centroid())[axis[2]];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        return b <= minCostColBelow;
                    });
                splits[2] = midIter2 - bvhPrimitives.begin();
                CHECK(splits[0] || splits[1] || splits[2]);
            } break;
            case 0:
            default:
                // splitting on 3 independent dimensions
                for (int i = 0; i < 3; i++) {
                    // Consider leaf only for whole newNode
                    bool leafAllowed = i == 0;
                    int mid;
                    int count =
                        i == 0 ? bvhPrimitives.size()
                               : (i == 1 ? splits[1] : bvhPrimitives.size() - splits[1]);
                    pstd::span<BVHPrimitive> currentPrimitives =
                        bvhPrimitives.subspan(i == 2 ? splits[1] : 0, count);
                    Bounds3f curCentroidBounds;
                    for (const auto &prim : currentPrimitives)
                        curCentroidBounds = Union(curCentroidBounds, prim.Centroid());
                    int dim = curCentroidBounds.MaxDimension();
                    // Compute Splits on local subset
                    //  Allocate _BVHSplitBucket_ for SAH partition buckets
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];
                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : currentPrimitives) {
                        int b = nBuckets * centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }
                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    BVHSplitBucket bucketsBelow[nSplits];
                    BVHSplitBucket bucketsAbove[nSplits];

                    bucketsBelow[0].bounds = buckets[0].bounds;
                    bucketsBelow[0].count = buckets[0].count;
                    for (int i = 1; i < nSplits; ++i) {
                        bucketsBelow[i].bounds =
                            Union(bucketsBelow[i - 1].bounds, buckets[i].bounds);
                        bucketsBelow[i].count =
                            bucketsBelow[i - 1].count + buckets[i].count;
                    }
                    // Finish initializing _costs_ using a backwards scan over splits

                    bucketsAbove[nSplits - 1].bounds = buckets[nSplits].bounds;
                    bucketsAbove[nSplits - 1].count = buckets[nSplits].count;
                    for (int i = nSplits - 1; i >= 1; --i) {
                        bucketsAbove[i - 1].bounds =
                            Union(bucketsAbove[i].bounds, buckets[i].bounds);
                        bucketsAbove[i - 1].count =
                            bucketsAbove[i].count + buckets[i].count;
                    }
                    // Find bucket to split at that minimizes metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    ICostCalcable *curBuckets[2];
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        curBuckets[0] = &bucketsBelow[i];
                        curBuckets[1] = &bucketsAbove[i];
                        float cost = splitCost(2, curBuckets);
                        if (cost < minCost) {
                            minCost = cost;
                            minCostSplitBucket = i;
                        }
                    }
                    // Consider leaf only for first iteration
                    if (leafAllowed && currentPrimitives.size() <= maxPrimsInNode) {
                        // Compute leaf cost and SAH split cost for chosen
                        minCost = RelativeInnerCost + minCost;
                        if (leafCost(count) <= minCost) {
                            int firstPrimOffset = orderedPrimsOffset->fetch_add(count);
                            for (size_t i = 0; i < count; ++i) {
                                int index = currentPrimitives[i].primitiveIndex;
                                orderedPrims[firstPrimOffset + i] = primitives[index];
                            }
                            node->InitLeaf(firstPrimOffset, count, bounds);
                            return node;
                        }
                    }
                    auto midIter = std::partition(
                        currentPrimitives.begin(), currentPrimitives.end(),
                        [=](const BVHPrimitive &bp) {
                            int b = nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                            if (b == nBuckets)
                                b = nBuckets - 1;
                            return b <= minCostSplitBucket;
                        });
                    mid = midIter - currentPrimitives.begin();

                    // Translate local split into offsets in complete bvhPrims
                    // last iterations subset begins at splits[1]
                    int offsetInBVHPrims = i == 2 ? splits[1] : 0;
                    // 1,0,2
                    int splitNr = i == 0 ? 1 : (i == 1 ? 0 : 2);
                    splits[splitNr] = offsetInBVHPrims + mid;
                    axis[splitNr] = dim;
                }
                break;
            }
            break;
        }
        }
        size_t splitpoints[5] = {0, splits[0], splits[1], splits[2],
                                 bvhPrimitives.size()};
        FourWideBVHBuildNode *children[MaxTreeWidth];
        // Recursively build BVHs for _children_
        if (bvhPrimitives.size() > 128 * 1024) {
            // Recursively build oldChild BVHs in parallel
            ParallelFor(0, TreeWidth, [&](int i) {
                DCHECK_LE(splitpoints[i], splitpoints[i + 1]);
                children[i] = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(splitpoints[i],
                                          splitpoints[i + 1] - splitpoints[i]),
                    totalNodes, orderedPrimsOffset, orderedPrims, splitVariant);
            });

        } else {
            // Recursively build oldChild BVHs sequentially
            for (int i = 0; i < TreeWidth; i++) {
                DCHECK_LE(splitpoints[i], splitpoints[i + 1]);
                children[i] = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(splitpoints[i],
                                          splitpoints[i + 1] - splitpoints[i]),
                    totalNodes, orderedPrimsOffset, orderedPrims, splitVariant);
            }
        }
        ++*totalNodes;
        node->InitInterior(axis, children);
        return node;
    }
}
int FourWideBVHAggregate::flattenBVH(FourWideBVHBuildNode *node, int *offset) {
    SIMDFourWideLinearBVHNode *linearNode = &nodes[*offset];
    // Root newNode
    if (*offset == 0) {
        this->bounds = node->bounds;
        // Root is leaf -> create interior with 1 leaf oldChild
        if (node->nPrimitives > 0) {
            CHECK(!node->children[0] && !node->children[1] && !node->children[2] &&
                  !node->children[3]);
            CHECK_LT(node->nPrimitives, 65536);
            linearNode->bounds[0] = node->bounds;
            linearNode->offsets[0] = node->firstPrimOffset;
            linearNode->nPrimitives[0] = node->nPrimitives;
            linearNode->offsets[1] = -1;
            linearNode->offsets[2] = -1;
            linearNode->offsets[3] = -1;
            (*offset)++;
            return 1;
        }
    }
    CHECK_EQ(node->nPrimitives, 0);
    int nodeOffset = (*offset)++;
    // Create interior flattened BVH newNode
    linearNode->axis = linearNode->ConstructAxis(node->splitAxis[0], node->splitAxis[1],
                                                 node->splitAxis[2]);
    for (int i = 0; i < TreeWidth; i++) {
        if (node->children[i]) {
            linearNode->bounds[i] = node->children[i]->bounds;
            linearNode->nPrimitives[i] = node->children[i]->nPrimitives;
            if (linearNode->nPrimitives[i] > 0) {
                linearNode->offsets[i] = node->children[i]->firstPrimOffset;
            } else {
                linearNode->offsets[i] = flattenBVH(node->children[i], offset);
            }

        } else {
            linearNode->offsets[i] = -1;
        }
    }
    return nodeOffset;
}
#pragma endregion

#pragma region BvhEight
//vfuncs BVH
pstd::optional<ShapeIntersection> EightWideBVHAggregate::Intersect(const Ray &ray,
                                                                   Float tMax) const {
    pstd::optional<ShapeIntersection> si;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    if (!nodes)
        return {};
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    std::vector<int> nodesToVisit(64);
    int nodesVisited = 0;
    int simdNodesVisited = 0;
    int simdTriangleTests = 0;
    int triangleTests = 0;
    int leavesTested = 0;
    int leavesHit = 0;
    int interiorTested = 0;
    int interiorHit = 0;
    while (true) {
        simdNodesVisited += std::ceil(TreeWidth / (float)SimdWidth);
        const SIMDEightWideLinearBVHNode *node = &nodes[currentNodeIndex];
        for (int i = 0; i < TreeWidth; ++i) {
            int idx = traversalIdx(dirIsNeg, node, i);
            if (node->offsets[idx] < 0)
                continue;
            ++nodesVisited;
            if (node->nPrimitives[idx] > 0) {
                leavesTested++;
            } else {
                interiorTested++;
            }
            if (node->bounds[idx].IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
                // Leaf oldChild
                if (node->nPrimitives[idx] > 0) {
                    leavesHit++;
                    bvhAvgLeaveSizeHit << (int)(node->nPrimitives[idx]);
                    simdTriangleTests +=
                        pstd::ceil(node->nPrimitives[idx] / (float)SimdWidth);
                    triangleTests += node->nPrimitives[idx];
                    // Intersect ray with primitives in leaf BVH newNode
                    for (size_t j = 0; j < node->nPrimitives[idx]; ++j) {
                        // Check for intersection with primitive in BVH newNode
                        pstd::optional<ShapeIntersection> primSi =
                            primitives[node->offsets[idx] + j].Intersect(ray, tMax);
                        if (primSi) {
                            si = primSi;
                            tMax = si->tHit;
                        }
                    }
                }
                // Interior oldChild
                else {
                    interiorHit++;
                    nodesToVisit[toVisitOffset++] = node->offsets[idx];
                }
            }
        }
        if (toVisitOffset == 0)
            break;
        currentNodeIndex = nodesToVisit[--toVisitOffset];
    }
    bvhSimdTriangleTests += simdTriangleTests;
    bvhSimdNodesVisited += simdNodesVisited;
    bvhSisdNodesVisited += nodesVisited;
    bvhSisdTriangleTests += triangleTests;
    bvhInteriorHit += interiorHit;
    bvhInteriorTested += interiorTested;
    bvhLeavesHit += leavesHit;
    bvhLeavesTested += leavesTested;
    return si;
}
bool EightWideBVHAggregate::IntersectP(const Ray &ray, Float tMax) const {
    pstd::optional<ShapeIntersection> si;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    if (!nodes)
        return false;
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    std::vector<int> nodesToVisit(64);
    int nodesVisited = 0;
    int simdNodesVisited = 0;
    int simdTriangleTests = 0;
    int triangleTests = 0;
    int leavesTested = 0;
    int leavesHit = 0;
    int interiorTested = 0;
    int interiorHit = 0;
    while (true) {
        simdNodesVisited += std::ceil(TreeWidth / (float)SimdWidth);
        const SIMDEightWideLinearBVHNode *node = &nodes[currentNodeIndex];
        for (int i = 0; i < TreeWidth; ++i) {
            int idx = traversalIdx(dirIsNeg, node, i);
            if (node->offsets[idx] < 0)
                continue;
            ++nodesVisited;
            if (node->nPrimitives[idx] > 0) {
                leavesTested++;
            } else {
                interiorTested++;
            }
            if (node->bounds[idx].IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
                // Leaf oldChild
                if (node->nPrimitives[idx] > 0) {
                    leavesHit++;
                    bvhAvgLeaveSizeHit << (int)(node->nPrimitives[idx]);
                    simdTriangleTests +=
                        pstd::ceil(node->nPrimitives[idx] / (float)SimdWidth);
                    triangleTests += node->nPrimitives[idx];
                    // Intersect ray with primitives in leaf BVH newNode
                    for (size_t j = 0; j < node->nPrimitives[idx]; ++j) {
                        // Check for intersection with primitive in BVH newNode
                        if (primitives[node->offsets[idx] + j].IntersectP(ray, tMax))
                            return true;
                    }
                }
                // Interior oldChild
                else {
                    interiorHit++;
                    nodesToVisit[toVisitOffset++] = node->offsets[idx];
                }
            }
        }
        if (toVisitOffset == 0)
            break;
        currentNodeIndex = nodesToVisit[--toVisitOffset];
    }
    bvhSimdTriangleTests += simdTriangleTests;
    bvhSimdNodesVisited += simdNodesVisited;
    bvhSisdNodesVisited += nodesVisited;
    bvhSisdTriangleTests += triangleTests;
    bvhInteriorHit += interiorHit;
    bvhInteriorTested += interiorTested;
    bvhLeavesHit += leavesHit;
    bvhLeavesTested += leavesTested;
    return false;
}
//vfuncs WideBVH
//Private funcs
EightWideBVHBuildNode *EightWideBVHAggregate::buildFromBVH(BVHBuildNode *binRoot,
                                                           std::atomic<int> *totalNodes) {
    if (binRoot->nPrimitives > 0) {
        EightWideBVHBuildNode *leaf = new EightWideBVHBuildNode();
        leaf->InitLeaf(binRoot->firstPrimOffset, binRoot->nPrimitives, binRoot->bounds);
        return leaf;
    }
    EightWideBVHBuildNode *newNode = new EightWideBVHBuildNode();
    BVHBuildNode *intermediateChildren[4]{};
    EightWideBVHBuildNode *newChildren[8]{};
    int axis[3] = {binRoot->children[0]->splitAxis, binRoot->splitAxis,
                   binRoot->children[1]->splitAxis};
    // First iteration (Flatten to 4 nodes)
    for (int i = 0; i < 2; ++i) {
        auto oldChild = binRoot->children[i];
        if (oldChild->nPrimitives > 0) {
            intermediateChildren[i * 2] = oldChild;
            intermediateChildren[i * 2 + 1] = NULL;
        } else {
            intermediateChildren[i * 2] = oldChild->children[0];
            intermediateChildren[i * 2 + 1] = oldChild->children[1];
        }
    }
    // merge these four's children to eight
    for (int i = 0; i < 4; ++i) {
        auto oldChild = intermediateChildren[i];
        if (!oldChild) {
            newChildren[i * 2] = NULL;
            newChildren[i * 2 + 1] = NULL;
        } else if (oldChild->nPrimitives > 0) {
            newChildren[i * 2] = buildFromBVH(oldChild, totalNodes);
            newChildren[i * 2 + 1] = NULL;
        } else {
            newChildren[i * 2] = buildFromBVH(oldChild->children[0], totalNodes);
            newChildren[i * 2 + 1] = buildFromBVH(oldChild->children[1], totalNodes);
        }
    }
    ++*totalNodes;
    newNode->InitInterior(axis, newChildren);
    return newNode;
};
//ToDo: Finish
EightWideBVHBuildNode *EightWideBVHAggregate::buildRecursive(
    ThreadLocal<Allocator> &threadAllocators, pstd::span<BVHPrimitive> bvhPrimitives,
    std::atomic<int> *totalNodes, std::atomic<int> *orderedPrimsOffset,
    std::vector<Primitive> &orderedPrims, int splitVariant) {
    if (bvhPrimitives.size() < 1) {
        return nullptr;
    }
    Allocator alloc = threadAllocators.Get();
    EightWideBVHBuildNode *node = alloc.new_object<EightWideBVHBuildNode>();
    // Initialize _BVHBuildNode_ for primitive range
    // Compute bounds of all primitives in BVH newNode
    Bounds3f bounds;
    for (const auto &prim : bvhPrimitives)
        bounds = Union(bounds, prim.bounds);
    if (bounds.SurfaceArea() == 0 || bvhPrimitives.size() < 2) {
        // Create leaf _BVHBuildNode_
        int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
            int index = bvhPrimitives[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[index];
        }
        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
        return node;

    } else {
        // Compute bound of primitive centroids and choose split dimensions _dim_
        int axis[7]{};
        Bounds3f centroidBounds;
        for (const auto &prim : bvhPrimitives)
            centroidBounds = Union(centroidBounds, prim.Centroid());
        axis[3] = centroidBounds.MaxDimension();
        // Decide on Leaf or inner
        if (centroidBounds.pMax[axis[1]] == centroidBounds.pMin[axis[1]]) {
            // Create leaf _BVHBuildNode_
            int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
            for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                int index = bvhPrimitives[i].primitiveIndex;
                orderedPrims[firstPrimOffset + i] = primitives[index];
            }
            node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
            return node;
        }
        // For simplicity only metrics with a splitCost() are allowed
        CHECK(splitMethod == EPO || splitMethod == SAH);
        // Aditioanlly we only use parts of the splitvariants
        CHECK(splitVariant == 0 || splitVariant == 2);
        // Cut in 8
        constexpr size_t nBuckets = 12;
        constexpr size_t nSplits = nBuckets - 1;
        size_t splitPoints[9]{};
        splitPoints[0] = 0;
        switch (splitVariant) {
        case 2:
        case 3: {
            int dims = centroidBounds.MaxDimensions();
            int dim1 = axis[3] = (dims >> 0) & 3;
            int dim2 = axis[1] = axis[5] = (dims >> 2) & 3;
            int dim3 = axis[0] = axis[2] = axis[4] = axis[6] = (dims >> 4) & 3;
            int count1 = 0;
            BVHSplitBucket buckets[nBuckets][nBuckets][nBuckets];
            for (const auto &prim : bvhPrimitives) {
                int b1 = nBuckets * centroidBounds.Offset(prim.Centroid())[dim1];
                if (b1 == nBuckets)
                    b1 = nBuckets - 1;
                int b2 = nBuckets * centroidBounds.Offset(prim.Centroid())[dim2];
                if (b2 == nBuckets)
                    b2 = nBuckets - 1;
                int b3 = nBuckets * centroidBounds.Offset(prim.Centroid())[dim3];
                if (b3 == nBuckets)
                    b3 = nBuckets - 1;
                CHECK_GE(b1, 0);
                CHECK_LT(b1, nBuckets);
                CHECK_GE(b2, 0);
                CHECK_LT(b2, nBuckets);
                CHECK_GE(b3, 0);
                CHECK_LT(b3, nBuckets);
                buckets[b1][b2][b3].count++;
                buckets[b1][b2][b3].bounds =
                    Union(buckets[b1][b2][b3].bounds, prim.bounds);
                count1++;
            }
            // Collapse on dim1
            auto bucketsA = std::vector(
                nSplits, std::vector(nBuckets, std::vector<BVHSplitBucket>(nBuckets)));
            auto bucketsB = std::vector(
                nSplits, std::vector(nBuckets, std::vector<BVHSplitBucket>(nBuckets)));
            for (size_t idx2 = 0; idx2 < nBuckets; idx2++) {
                for (size_t idx3 = 0; idx3 < nBuckets; idx3++) {
                    // Above collapse
                    bucketsA[0][idx2][idx3].bounds = buckets[0][idx2][idx3].bounds;
                    bucketsA[0][idx2][idx3].count = buckets[0][idx2][idx3].count;
                    for (size_t idx1 = 1; idx1 < nSplits; idx1++) {
                        bucketsA[idx1][idx2][idx3].bounds =
                            Union(bucketsA[idx1 - 1][idx2][idx3].bounds,
                                  buckets[idx1][idx2][idx3].bounds);
                        bucketsA[idx1][idx2][idx3].count =
                            bucketsA[idx1 - 1][idx2][idx3].count +
                            buckets[idx1][idx2][idx3].count;
                    }
                    // Below collapse
                    bucketsB[nSplits - 1][idx2][idx3].bounds =
                        buckets[nSplits][idx2][idx3].bounds;
                    bucketsB[nSplits - 1][idx2][idx3].count =
                        buckets[nSplits][idx2][idx3].count;
                    for (size_t idx1 = nSplits - 1; idx1 > 0; --idx1) {
                        bucketsB[idx1 - 1][idx2][idx3].bounds =
                            Union(bucketsB[idx1][idx2][idx3].bounds,
                                  buckets[idx1][idx2][idx3].bounds);
                        bucketsB[idx1 - 1][idx2][idx3].count =
                            bucketsB[idx1][idx2][idx3].count +
                            buckets[idx1][idx2][idx3].count;
                    }
                }
            }

            // Collapse on dim2
            auto bucketsAA = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nBuckets)));
            auto bucketsAB = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nBuckets)));
            auto bucketsBA = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nBuckets)));
            auto bucketsBB = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nBuckets)));
            for (size_t idx1 = 0; idx1 < nSplits; idx1++) {
                for (size_t idx3 = 0; idx3 < nBuckets; idx3++) {
                    // forward collapses
                    bucketsAA[idx1][0][idx3].bounds = bucketsA[idx1][0][idx3].bounds;
                    bucketsAA[idx1][0][idx3].count = bucketsA[idx1][0][idx3].count;

                    bucketsBA[idx1][0][idx3].bounds = bucketsB[idx1][0][idx3].bounds;
                    bucketsBA[idx1][0][idx3].count = bucketsB[idx1][0][idx3].count;
                    for (size_t idx2 = 1; idx2 < nSplits; idx2++) {
                        bucketsAA[idx1][idx2][idx3].bounds =
                            Union(bucketsAA[idx1][idx2 - 1][idx3].bounds,
                                  bucketsA[idx1][idx2][idx3].bounds);
                        bucketsAA[idx1][idx2][idx3].count =
                            bucketsAA[idx1][idx2 - 1][idx3].count +
                            bucketsA[idx1][idx2][idx3].count;

                        bucketsBA[idx1][idx2][idx3].bounds =
                            Union(bucketsBA[idx1][idx2 - 1][idx3].bounds,
                                  bucketsB[idx1][idx2][idx3].bounds);
                        bucketsBA[idx1][idx2][idx3].count =
                            bucketsBA[idx1][idx2 - 1][idx3].count +
                            bucketsB[idx1][idx2][idx3].count;
                    }
                    // backward collapses
                    bucketsAB[idx1][nSplits - 1][idx3].bounds =
                        bucketsA[idx1][nSplits][idx3].bounds;
                    bucketsAB[idx1][nSplits - 1][idx3].count =
                        bucketsA[idx1][nSplits][idx3].count;

                    bucketsBB[idx1][nSplits - 1][idx3].bounds =
                        bucketsB[idx1][nSplits][idx3].bounds;
                    bucketsBB[idx1][nSplits - 1][idx3].count =
                        bucketsB[idx1][nSplits][idx3].count;
                    for (size_t idx2 = nSplits - 1; idx2 > 0; --idx2) {
                        bucketsAB[idx1][idx2 - 1][idx3].bounds =
                            Union(bucketsAB[idx1][idx2][idx3].bounds,
                                  bucketsA[idx1][idx2][idx3].bounds);
                        bucketsAB[idx1][idx2 - 1][idx3].count =
                            bucketsAB[idx1][idx2][idx3].count +
                            bucketsA[idx1][idx2][idx3].count;

                        bucketsBB[idx1][idx2 - 1][idx3].bounds =
                            Union(bucketsBB[idx1][idx2][idx3].bounds,
                                  bucketsB[idx1][idx2][idx3].bounds);
                        bucketsBB[idx1][idx2 - 1][idx3].count =
                            bucketsBB[idx1][idx2][idx3].count +
                            bucketsB[idx1][idx2][idx3].count;
                    }
                }
            }
            // Collapse on dim3
            // Creates final canidates
            auto bucketsAAA = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            auto bucketsABA = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            auto bucketsBAA = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            auto bucketsBBA = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            auto bucketsAAB = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            auto bucketsABB = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            auto bucketsBAB = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            auto bucketsBBB = std::vector(
                nSplits, std::vector(nSplits, std::vector<BVHSplitBucket>(nSplits)));
            for (size_t idx1 = 0; idx1 < nSplits; idx1++) {
                for (size_t idx2 = 0; idx2 < nSplits; idx2++) {
                    // forward collapses
                    bucketsAAA[idx1][idx2][0].bounds = bucketsAA[idx1][idx2][0].bounds;
                    bucketsAAA[idx1][idx2][0].count = bucketsAA[idx1][idx2][0].count;
                    bucketsABA[idx1][idx2][0].bounds = bucketsAB[idx1][idx2][0].bounds;
                    bucketsABA[idx1][idx2][0].count = bucketsAB[idx1][idx2][0].count;
                    bucketsBAA[idx1][idx2][0].bounds = bucketsBA[idx1][idx2][0].bounds;
                    bucketsBAA[idx1][idx2][0].count = bucketsBA[idx1][idx2][0].count;
                    bucketsBBA[idx1][idx2][0].bounds = bucketsBB[idx1][idx2][0].bounds;
                    bucketsBBA[idx1][idx2][0].count = bucketsBB[idx1][idx2][0].count;
                    for (size_t idx3 = 1; idx3 < nSplits; idx3++) {
                        bucketsAAA[idx1][idx2][idx3].bounds =
                            Union(bucketsAAA[idx1][idx2][idx3 - 1].bounds,
                                  bucketsAA[idx1][idx2][idx3].bounds);
                        bucketsAAA[idx1][idx2][idx3].count =
                            bucketsAAA[idx1][idx2][idx3 - 1].count +
                            bucketsAA[idx1][idx2][idx3].count;

                        bucketsABA[idx1][idx2][idx3].bounds =
                            Union(bucketsABA[idx1][idx2][idx3 - 1].bounds,
                                  bucketsAB[idx1][idx2][idx3].bounds);
                        bucketsABA[idx1][idx2][idx3].count =
                            bucketsABA[idx1][idx2][idx3 - 1].count +
                            bucketsAB[idx1][idx2][idx3].count;

                        bucketsBAA[idx1][idx2][idx3].bounds =
                            Union(bucketsBAA[idx1][idx2][idx3 - 1].bounds,
                                  bucketsBA[idx1][idx2][idx3].bounds);
                        bucketsBAA[idx1][idx2][idx3].count =
                            bucketsBAA[idx1][idx2][idx3 - 1].count +
                            bucketsBA[idx1][idx2][idx3].count;

                        bucketsBBA[idx1][idx2][idx3].bounds =
                            Union(bucketsBBA[idx1][idx2][idx3 - 1].bounds,
                                  bucketsBB[idx1][idx2][idx3].bounds);
                        bucketsBBA[idx1][idx2][idx3].count =
                            bucketsBBA[idx1][idx2][idx3 - 1].count +
                            bucketsBB[idx1][idx2][idx3].count;
                    }
                    // backward collapses
                    bucketsAAB[idx1][idx2][nSplits - 1].bounds =
                        bucketsAA[idx1][idx2][nSplits].bounds;
                    bucketsAAB[idx1][idx2][nSplits - 1].count =
                        bucketsAA[idx1][idx2][nSplits].count;

                    bucketsABB[idx1][idx2][nSplits - 1].bounds =
                        bucketsAB[idx1][idx2][nSplits].bounds;
                    bucketsABB[idx1][idx2][nSplits - 1].count =
                        bucketsAB[idx1][idx2][nSplits].count;

                    bucketsBAB[idx1][idx2][nSplits - 1].bounds =
                        bucketsBA[idx1][idx2][nSplits].bounds;
                    bucketsBAB[idx1][idx2][nSplits - 1].count =
                        bucketsBA[idx1][idx2][nSplits].count;

                    bucketsBBB[idx1][idx2][nSplits - 1].bounds =
                        bucketsBB[idx1][idx2][nSplits].bounds;
                    bucketsBBB[idx1][idx2][nSplits - 1].count =
                        bucketsBB[idx1][idx2][nSplits].count;
                    for (size_t idx3 = nSplits - 1; idx3 > 0; --idx3) {
                        bucketsAAB[idx1][idx2][idx3 - 1].bounds =
                            Union(bucketsAAB[idx1][idx2][idx3].bounds,
                                  bucketsAA[idx1][idx2][idx3].bounds);
                        bucketsAAB[idx1][idx2][idx3 - 1].count =
                            bucketsAAB[idx1][idx2][idx3].count +
                            bucketsAA[idx1][idx2][idx3].count;

                        bucketsABB[idx1][idx2][idx3 - 1].bounds =
                            Union(bucketsABB[idx1][idx2][idx3].bounds,
                                  bucketsAB[idx1][idx2][idx3].bounds);
                        bucketsABB[idx1][idx2][idx3 - 1].count =
                            bucketsABB[idx1][idx2][idx3].count +
                            bucketsAB[idx1][idx2][idx3].count;

                        bucketsBAB[idx1][idx2][idx3 - 1].bounds =
                            Union(bucketsBAB[idx1][idx2][idx3].bounds,
                                  bucketsBA[idx1][idx2][idx3].bounds);
                        bucketsBAB[idx1][idx2][idx3 - 1].count =
                            bucketsBAB[idx1][idx2][idx3].count +
                            bucketsBA[idx1][idx2][idx3].count;

                        bucketsBBB[idx1][idx2][idx3 - 1].bounds =
                            Union(bucketsBBB[idx1][idx2][idx3].bounds,
                                  bucketsBB[idx1][idx2][idx3].bounds);
                        bucketsBBB[idx1][idx2][idx3 - 1].count =
                            bucketsBBB[idx1][idx2][idx3].count +
                            bucketsBB[idx1][idx2][idx3].count;
                    }
                }
            }
            // Compute cost for all possible splits and select best
            ICostCalcable *curBuckets[8];
            Float minCost = Infinity;
            int minCostIdx1 = -1;
            int minCostIdx2 = -1;
            int minCostIdx3 = -1;
            for (size_t idx1 = 0; idx1 < nSplits; ++idx1) {
                for (size_t idx2 = 0; idx2 < nSplits; ++idx2) {
                    for (size_t idx3 = 0; idx3 < nSplits; ++idx3) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        curBuckets[0] = &bucketsAAA[idx1][idx2][idx3];
                        curBuckets[1] = &bucketsAAB[idx1][idx2][idx3];
                        curBuckets[2] = &bucketsABA[idx1][idx2][idx3];
                        curBuckets[3] = &bucketsABB[idx1][idx2][idx3];
                        curBuckets[4] = &bucketsBAA[idx1][idx2][idx3];
                        curBuckets[5] = &bucketsBAB[idx1][idx2][idx3];
                        curBuckets[6] = &bucketsBBA[idx1][idx2][idx3];
                        curBuckets[7] = &bucketsBBB[idx1][idx2][idx3];

                        CHECK_EQ(
                            bvhPrimitives.size(),
                            curBuckets[0]->PrimCount() + curBuckets[1]->PrimCount() +
                                curBuckets[2]->PrimCount() + curBuckets[3]->PrimCount() +
                                curBuckets[4]->PrimCount() + curBuckets[5]->PrimCount() +
                                curBuckets[6]->PrimCount() + curBuckets[7]->PrimCount());

                        Float cost = splitCost(8, curBuckets);
                        DCHECK_GT(cost, 0);
                        if (cost < minCost) {
                            minCost = cost;
                            minCostIdx1 = idx1;
                            minCostIdx2 = idx2;
                            minCostIdx3 = idx3;
                        }
                    }
                }
            }
            // Sort prims and populate splipoints
            //   Can free resources here
            //   Compute leaf cost and split cost for chosen split
            minCost = RelativeInnerCost + minCost;

            if (leafCost(bvhPrimitives.size()) <= minCost &&
                bvhPrimitives.size() <= maxPrimsInNode) {
                int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
                for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                    int index = bvhPrimitives[i].primitiveIndex;
                    orderedPrims[firstPrimOffset + i] = primitives[index];
                }
                node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
                return node;
            }
            BVHPrimitive *iters[7];
            auto primPart = [=](BVHPrimitive *start, BVHPrimitive *end, int dim,
                                int splitIdx) {
                return std::partition(start, end, [=](const BVHPrimitive &bp) {
                    int b = nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                    if (b == nBuckets)
                        b = nBuckets - 1;
                    return b <= splitIdx;
                });
            };
            iters[3] =
                primPart(bvhPrimitives.begin(), bvhPrimitives.end(), dim1, minCostIdx1);
            iters[1] = primPart(bvhPrimitives.begin(), iters[3], dim2, minCostIdx2);
            iters[5] = primPart(iters[3], bvhPrimitives.end(), dim2, minCostIdx2);
            
            iters[0] = primPart(bvhPrimitives.begin(), iters[1], dim3, minCostIdx3);
            iters[2] = primPart(iters[1], iters[3], dim3, minCostIdx3);
            iters[4] = primPart(iters[3], iters[5], dim3, minCostIdx3);
            iters[6] = primPart(iters[5], bvhPrimitives.end(), dim3, minCostIdx3);
            for (int i = 0; i < 7;++i)
                splitPoints[i+1] = iters[i] - bvhPrimitives.begin();
            break;
        } 
        case 0:
        default:
            BVHSplitBucket *proposedBuckets[8]{NULL};
            proposedBuckets[0] = new BVHSplitBucket();
            proposedBuckets[0]->count = bvhPrimitives.size();
            proposedBuckets[0]->bounds = bounds;
            splitPoints[8] = splitPoints[1] = bvhPrimitives.size();
            // do 7 binary splits
            for (int i = 0; i < 7; i++) {
                // Consider leaf only for whole newNode
                bool firstIteration = i == 0;
                int mid;
                Float highestCost = 0;
                int bestSplit = 1;
                // find child to split
                for (int j = 1; j < i + 2; ++j) {
                    Float curCost =
                        childCost(j - 1, 7, (ICostCalcable **)proposedBuckets);
                    if (curCost > highestCost) {
                        highestCost = curCost;
                        bestSplit = j;
                    }
                }
                // move splits after current one back to make space
                for (int j = 6; j >= bestSplit; --j) {
                    proposedBuckets[j] = proposedBuckets[j - 1];
                    axis[j] = axis[j - 1];
                    splitPoints[j + 1] = splitPoints[j];
                }
                // prepare current subset to split
                int count = splitPoints[bestSplit + 1] - splitPoints[bestSplit - 1];
                pstd::span<BVHPrimitive> currentPrimitives =
                    bvhPrimitives.subspan(splitPoints[bestSplit - 1], count);
                Bounds3f curCentroidBounds;
                if (firstIteration) {
                    curCentroidBounds = centroidBounds;
                } else {
                    for (const auto &prim : currentPrimitives)
                        curCentroidBounds = Union(curCentroidBounds, prim.Centroid());
                }
                int dim = curCentroidBounds.MaxDimension();
                // Compute Splits on local subset
                //  Allocate _BVHSplitBucket_ for SAH partition buckets
                constexpr int nBuckets = 12;
                BVHSplitBucket buckets[nBuckets];
                // Initialize _BVHSplitBucket_ for SAH partition buckets
                for (const auto &prim : currentPrimitives) {
                    int b = nBuckets * curCentroidBounds.Offset(prim.Centroid())[dim];
                    if (b == nBuckets)
                        b = nBuckets - 1;
                    DCHECK_GE(b, 0);
                    DCHECK_LT(b, nBuckets);
                    buckets[b].count++;
                    buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                }
                // Compute costs for splitting after each bucket
                constexpr int nSplits = nBuckets - 1;
                BVHSplitBucket bucketsBelow[nSplits];
                BVHSplitBucket bucketsAbove[nSplits];

                bucketsBelow[0].bounds = buckets[0].bounds;
                bucketsBelow[0].count = buckets[0].count;
                for (int i = 1; i < nSplits; ++i) {
                    bucketsBelow[i].bounds =
                        Union(bucketsBelow[i - 1].bounds, buckets[i].bounds);
                    bucketsBelow[i].count = bucketsBelow[i - 1].count + buckets[i].count;
                }
                // Finish initializing _costs_ using a backwards scan over splits

                bucketsAbove[nSplits - 1].bounds = buckets[nSplits].bounds;
                bucketsAbove[nSplits - 1].count = buckets[nSplits].count;
                for (int i = nSplits - 1; i >= 1; --i) {
                    bucketsAbove[i - 1].bounds =
                        Union(bucketsAbove[i].bounds, buckets[i].bounds);
                    bucketsAbove[i - 1].count = bucketsAbove[i].count + buckets[i].count;
                }
                // Find bucket to split at that minimizes metric
                int minCostBucket = -1;
                Float minCost = Infinity;
                ICostCalcable *curBuckets[2];
                for (int i = 0; i < nSplits; ++i) {
                    // Compute cost for candidate split and update minimum if
                    // necessary
                    curBuckets[0] = &bucketsBelow[i];
                    curBuckets[1] = &bucketsAbove[i];
                    float cost = splitCost(2, curBuckets);
                    if (cost < minCost) {
                        minCost = cost;
                        minCostBucket = i;
                    }
                }
                // Consider leaf only for first iteration
                if (firstIteration && currentPrimitives.size() <= maxPrimsInNode) {
                    // Compute leaf cost and SAH split cost for chosen
                    minCost = RelativeInnerCost + minCost;
                    if (leafCost(count) <= minCost) {
                        int firstPrimOffset = orderedPrimsOffset->fetch_add(count);
                        for (size_t i = 0; i < count; ++i) {
                            int index = currentPrimitives[i].primitiveIndex;
                            orderedPrims[firstPrimOffset + i] = primitives[index];
                        }
                        node->InitLeaf(firstPrimOffset, count, bounds);
                        return node;
                    }
                }
                auto midIter = std::partition(
                    currentPrimitives.begin(), currentPrimitives.end(),
                    [=](const BVHPrimitive &bp) {
                        int b = nBuckets * curCentroidBounds.Offset(bp.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        return b <= minCostBucket;
                    });
                mid = midIter - currentPrimitives.begin();

                // Translate local split into offsets in complete bvhPrims
                splitPoints[bestSplit] = splitPoints[bestSplit - 1] + mid;
                axis[bestSplit - 1] = dim;
                proposedBuckets[bestSplit - 1] = new BVHSplitBucket();
                proposedBuckets[bestSplit - 1]->count = bucketsBelow[minCostBucket].count;
                proposedBuckets[bestSplit - 1]->bounds =
                    bucketsBelow[minCostBucket].bounds;
                // Can reuse old bucket here
                proposedBuckets[bestSplit] = new BVHSplitBucket();
                proposedBuckets[bestSplit]->count = bucketsAbove[minCostBucket].count;
                proposedBuckets[bestSplit]->bounds = bucketsAbove[minCostBucket].bounds;
            }
            break;
        }
        EightWideBVHBuildNode *children[MaxTreeWidth];
        if (bvhPrimitives.size() > 128 * 1024) {
            // Recursively build oldChild BVHs in parallel
            ParallelFor(0, TreeWidth, [&](int i) {
                DCHECK_LE(splitPoints[i], splitPoints[i + 1]);
                children[i] = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(splitPoints[i],
                                          splitPoints[i + 1] - splitPoints[i]),
                    totalNodes, orderedPrimsOffset, orderedPrims, splitVariant);
            });

        } else {
            // Recursively build oldChild BVHs sequentially
            for (int i = 0; i < TreeWidth; i++) {
                DCHECK_LE(splitPoints[i], splitPoints[i + 1]);
                children[i] = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(splitPoints[i],
                                          splitPoints[i + 1] - splitPoints[i]),
                    totalNodes, orderedPrimsOffset, orderedPrims, splitVariant);
            }
        }
        ++*totalNodes;
        node->InitInterior(axis, children);
        return node;
    }
}
int EightWideBVHAggregate::flattenBVH(EightWideBVHBuildNode *node, int *offset) {
    SIMDEightWideLinearBVHNode *linearNode = &nodes[*offset];
    // Root newNode
    if (*offset == 0) {
        this->bounds = node->bounds;
        // Root is leaf -> create interior with 1 leaf oldChild
        if (node->nPrimitives > 0) {
            CHECK(!node->children[0] && !node->children[1] && !node->children[2] &&
                  !node->children[3]);
            CHECK_LT(node->nPrimitives, 65536);
            linearNode->bounds[0] = node->bounds;
            linearNode->offsets[0] = node->firstPrimOffset;
            linearNode->nPrimitives[0] = node->nPrimitives;
            for (int i = 1; i < TreeWidth; ++i) {
                linearNode->offsets[i] = -1;
            }
            (*offset)++;
            return 1;
        }
    }
    CHECK_EQ(node->nPrimitives, 0);
    int nodeOffset = (*offset)++;
    // Create interior flattened BVH newNode
    linearNode->axis = linearNode->ConstructAxis(
        node->splitAxis[0], node->splitAxis[1], node->splitAxis[2], node->splitAxis[3],
        node->splitAxis[4], node->splitAxis[5], node->splitAxis[6]);

    for (int i = 0; i < TreeWidth; i++) {
        if (node->children[i]) {
            linearNode->bounds[i] = node->children[i]->bounds;
            linearNode->nPrimitives[i] = node->children[i]->nPrimitives;
            if (linearNode->nPrimitives[i] > 0) {
                linearNode->offsets[i] = node->children[i]->firstPrimOffset;
            } else {
                linearNode->offsets[i] = flattenBVH(node->children[i], offset);
            }

        } else {
            linearNode->offsets[i] = -1;
        }
    }
    return nodeOffset;
}
int EightWideBVHAggregate::traversalIdx(int dirIsNeg[3],
                                        const SIMDEightWideLinearBVHNode *node, int idx) {
    int res = 0;  
    res |= ((idx & 4) ^ (dirIsNeg[node->Axis3()] << 2));
    int bit2 = (res >> 2) & 1;
    res |= ((idx & 2) ^
           (((dirIsNeg[node->Axis1()] & (1-bit2)) | (dirIsNeg[node->Axis5()] & bit2))<<1));
    int bit1 = (res >> 1) & 1;
    res |=
        ((idx & 1) ^
            ((((dirIsNeg[node->Axis0()] & (1-bit1)) | (dirIsNeg[node->Axis2()] & bit1)) &
             (1-bit2)) |
        (((dirIsNeg[node->Axis4()] & (1-bit1)) | (dirIsNeg[node->Axis6()] & bit1)) & bit2)));
    //res += idx & 4;
    CHECK_GE(res, 0);
    CHECK_LE(res, 7);
    return res;
}
#pragma endregion

#pragma region KdTree

// KdNodeToVisit Definition
struct KdNodeToVisit {
    const KdTreeNode *node;
    Float tMin, tMax;
};

// KdTreeNode Definition
struct alignas(8) KdTreeNode {
    // KdTreeNode Methods
    void InitLeaf(pstd::span<const int> primNums, std::vector<int> *primitiveIndices);

    void InitInterior(int axis, int aboveChild, Float s) {
        split = s;
        flags = axis | (aboveChild << 2);
    }

    Float SplitPos() const { return split; }
    int nPrimitives() const { return flags >> 2; }
    int SplitAxis() const { return flags & 3; }
    bool IsLeaf() const { return (flags & 3) == 3; }
    int AboveChild() const { return flags >> 2; }

    union {
        Float split;                 // Interior
        int onePrimitiveIndex;       // Leaf
        int primitiveIndicesOffset;  // Leaf
    };

  private:
    uint32_t flags;
};

// EdgeType Definition
enum class EdgeType { Start, End };

// BoundEdge Definition
struct BoundEdge {
    // BoundEdge Public Methods
    BoundEdge() {}

    BoundEdge(Float t, int primNum, bool starting) : t(t), primNum(primNum) {
        type = starting ? EdgeType::Start : EdgeType::End;
    }

    Float t;
    int primNum;
    EdgeType type;
};

STAT_PIXEL_COUNTER("Kd-Tree/Nodes visited", kdNodesVisited);

// KdTreeAggregate Method Definitions
KdTreeAggregate::KdTreeAggregate(std::vector<Primitive> p, int isectCost,
                                 int traversalCost, Float emptyBonus, int maxPrims,
                                 int maxDepth)
    : isectCost(isectCost),
      traversalCost(traversalCost),
      maxPrims(maxPrims),
      emptyBonus(emptyBonus),
      primitives(std::move(p)) {
    // Build kd-tree aggregate
    nextFreeNode = nAllocedNodes = 0;
    if (maxDepth <= 0)
        maxDepth = std::round(8 + 1.3f * Log2Int(int64_t(primitives.size())));
    // Compute bounds for kd-tree construction
    std::vector<Bounds3f> primBounds;
    primBounds.reserve(primitives.size());
    for (Primitive &prim : primitives) {
        Bounds3f b = prim.Bounds();
        bounds = Union(bounds, b);
        primBounds.push_back(b);
    }

    // Allocate working memory for kd-tree construction
    std::vector<BoundEdge> edges[3];
    for (int i = 0; i < 3; ++i)
        edges[i].resize(2 * primitives.size());

    std::vector<int> prims0(primitives.size());
    std::vector<int> prims1((maxDepth + 1) * primitives.size());

    // Initialize _primNums_ for kd-tree construction
    std::vector<int> primNums(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        primNums[i] = i;

    // Start recursive construction of kd-tree
    buildTree(0, bounds, primBounds, primNums, maxDepth, edges, pstd::span<int>(prims0),
              pstd::span<int>(prims1), 0);
}

void KdTreeNode::InitLeaf(pstd::span<const int> primNums,
                          std::vector<int> *primitiveIndices) {
    flags = 3 | (primNums.size() << 2);
    // Store primitive ids for leaf newNode
    if (primNums.size() == 0)
        onePrimitiveIndex = 0;
    else if (primNums.size() == 1)
        onePrimitiveIndex = primNums[0];
    else {
        primitiveIndicesOffset = primitiveIndices->size();
        for (int pn : primNums)
            primitiveIndices->push_back(pn);
    }
}

void KdTreeAggregate::buildTree(int nodeNum, const Bounds3f &nodeBounds,
                                const std::vector<Bounds3f> &allPrimBounds,
                                pstd::span<const int> primNums, int depth,
                                std::vector<BoundEdge> edges[3], pstd::span<int> prims0,
                                pstd::span<int> prims1, int badRefines) {
    CHECK_EQ(nodeNum, nextFreeNode);
    // Get next free newNode from _nodes_ array
    if (nextFreeNode == nAllocedNodes) {
        int nNewAllocNodes = std::max(2 * nAllocedNodes, 512);
        KdTreeNode *n = new KdTreeNode[nNewAllocNodes];
        if (nAllocedNodes > 0) {
            std::memcpy(n, nodes, nAllocedNodes * sizeof(KdTreeNode));
            delete[] nodes;
        }
        nodes = n;
        nAllocedNodes = nNewAllocNodes;
    }
    ++nextFreeNode;

    // Initialize leaf newNode if termination criteria met
    if (primNums.size() <= maxPrims || depth == 0) {
        nodes[nodeNum].InitLeaf(primNums, &primitiveIndices);
        return;
    }

    // Initialize interior newNode and continue recursion
    // Choose split axis position for interior newNode
    int bestAxis = -1, bestOffset = -1;
    Float bestCost = Infinity, leafCost = isectCost * primNums.size();
    Float invTotalSA = 1 / nodeBounds.SurfaceArea();
    // Choose which axis to split along
    int axis = nodeBounds.MaxDimension();

    // Choose split along axis and attempt to partition primitives
    int retries = 0;
    size_t nPrimitives = primNums.size();
retrySplit:
    // Initialize edges for _axis_
    for (size_t i = 0; i < nPrimitives; ++i) {
        int pn = primNums[i];
        const Bounds3f &bounds = allPrimBounds[pn];
        edges[axis][2 * i] = BoundEdge(bounds.pMin[axis], pn, true);
        edges[axis][2 * i + 1] = BoundEdge(bounds.pMax[axis], pn, false);
    }
    // Sort _edges_ for _axis_
    std::sort(edges[axis].begin(), edges[axis].begin() + 2 * nPrimitives,
              [](const BoundEdge &e0, const BoundEdge &e1) -> bool {
                  return std::tie(e0.t, e0.type) < std::tie(e1.t, e1.type);
              });

    // Compute cost of all splits for _axis_ to find best
    int nBelow = 0, nAbove = primNums.size();
    for (size_t i = 0; i < 2 * primNums.size(); ++i) {
        if (edges[axis][i].type == EdgeType::End)
            --nAbove;
        Float edgeT = edges[axis][i].t;
        if (edgeT > nodeBounds.pMin[axis] && edgeT < nodeBounds.pMax[axis]) {
            // Compute oldChild surface areas for split at _edgeT_
            Vector3f d = nodeBounds.pMax - nodeBounds.pMin;
            int otherAxis0 = (axis + 1) % 3, otherAxis1 = (axis + 2) % 3;
            Float belowSA =
                2 * (d[otherAxis0] * d[otherAxis1] +
                     (edgeT - nodeBounds.pMin[axis]) * (d[otherAxis0] + d[otherAxis1]));
            Float aboveSA =
                2 * (d[otherAxis0] * d[otherAxis1] +
                     (nodeBounds.pMax[axis] - edgeT) * (d[otherAxis0] + d[otherAxis1]));

            // Compute cost for split at _i_th edge
            Float pBelow = belowSA * invTotalSA, pAbove = aboveSA * invTotalSA;
            Float eb = (nAbove == 0 || nBelow == 0) ? emptyBonus : 0;
            Float cost = traversalCost +
                         isectCost * (1 - eb) * (pBelow * nBelow + pAbove * nAbove);
            // Update best split if this is lowest cost so far
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestOffset = i;
            }
        }
        if (edges[axis][i].type == EdgeType::Start)
            ++nBelow;
    }
    CHECK(nBelow == nPrimitives && nAbove == 0);

    // Try to split along another axis if no good splits were found
    if (bestAxis == -1 && retries < 2) {
        ++retries;
        axis = (axis + 1) % 3;
        goto retrySplit;
    }

    // Create leaf if no good splits were found
    if (bestCost > leafCost)
        ++badRefines;
    if ((bestCost > 4 * leafCost && nPrimitives < 16) || bestAxis == -1 ||
        badRefines == 3) {
        nodes[nodeNum].InitLeaf(primNums, &primitiveIndices);
        return;
    }

    // Classify primitives with respect to split
    int n0 = 0, n1 = 0;
    for (int i = 0; i < bestOffset; ++i)
        if (edges[bestAxis][i].type == EdgeType::Start)
            prims0[n0++] = edges[bestAxis][i].primNum;
    for (int i = bestOffset + 1; i < 2 * nPrimitives; ++i)
        if (edges[bestAxis][i].type == EdgeType::End)
            prims1[n1++] = edges[bestAxis][i].primNum;

    // Recursively initialize kd-tree newNode's newChildren
    Float tSplit = edges[bestAxis][bestOffset].t;
    Bounds3f bounds0 = nodeBounds, bounds1 = nodeBounds;
    bounds0.pMax[bestAxis] = bounds1.pMin[bestAxis] = tSplit;
    buildTree(nodeNum + 1, bounds0, allPrimBounds, prims0.subspan(0, n0), depth - 1,
              edges, prims0, prims1.subspan(n1), badRefines);
    int aboveChild = nextFreeNode;
    nodes[nodeNum].InitInterior(bestAxis, aboveChild, tSplit);
    buildTree(aboveChild, bounds1, allPrimBounds, prims1.subspan(0, n1), depth - 1, edges,
              prims0, prims1.subspan(n1), badRefines);
}

pstd::optional<ShapeIntersection> KdTreeAggregate::Intersect(const Ray &ray,
                                                             Float rayTMax) const {
    // Compute initial parametric range of ray inside kd-tree extent
    Float tMin, tMax;
    if (!bounds.IntersectP(ray.o, ray.d, rayTMax, &tMin, &tMax))
        return {};

    // Prepare to traverse kd-tree for ray
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    constexpr int maxToVisit = 64;
    KdNodeToVisit toVisit[maxToVisit];
    int toVisitIndex = 0;
    int nodesVisited = 0;

    // Traverse kd-tree nodes in order for ray
    pstd::optional<ShapeIntersection> si;
    const KdTreeNode *node = &nodes[0];
    while (node) {
        // Bail out if we found a hit closer than the current newNode
        if (rayTMax < tMin)
            break;

        ++nodesVisited;
        if (!node->IsLeaf()) {
            // Visit kd-tree interior newNode
            // Compute parametric distance along ray to split plane
            int axis = node->SplitAxis();
            Float tSplit = (node->SplitPos() - ray.o[axis]) * invDir[axis];

            // Get newNode oldChild pointers for ray
            const KdTreeNode *firstChild, *secondChild;
            int belowFirst = (ray.o[axis] < node->SplitPos()) ||
                             (ray.o[axis] == node->SplitPos() && ray.d[axis] <= 0);
            if (belowFirst) {
                firstChild = node + 1;
                secondChild = &nodes[node->AboveChild()];
            } else {
                firstChild = &nodes[node->AboveChild()];
                secondChild = node + 1;
            }

            // Advance to next oldChild newNode, possibly enqueue other oldChild
            if (tSplit > tMax || tSplit <= 0)
                node = firstChild;
            else if (tSplit < tMin)
                node = secondChild;
            else {
                // Enqueue _secondChild_ in todo list
                toVisit[toVisitIndex].node = secondChild;
                toVisit[toVisitIndex].tMin = tSplit;
                toVisit[toVisitIndex].tMax = tMax;
                ++toVisitIndex;

                node = firstChild;
                tMax = tSplit;
            }

        } else {
            // Check for intersections inside leaf newNode
            int nPrimitives = node->nPrimitives();
            if (nPrimitives == 1) {
                const Primitive &p = primitives[node->onePrimitiveIndex];
                // Check one primitive inside leaf newNode
                pstd::optional<ShapeIntersection> primSi = p.Intersect(ray, rayTMax);
                if (primSi) {
                    si = primSi;
                    rayTMax = si->tHit;
                }

            } else {
                for (int i = 0; i < nPrimitives; ++i) {
                    int index = primitiveIndices[node->primitiveIndicesOffset + i];
                    const Primitive &p = primitives[index];
                    // Check one primitive inside leaf newNode
                    pstd::optional<ShapeIntersection> primSi = p.Intersect(ray, rayTMax);
                    if (primSi) {
                        si = primSi;
                        rayTMax = si->tHit;
                    }
                }
            }

            // Grab next newNode to visit from todo list
            if (toVisitIndex > 0) {
                --toVisitIndex;
                node = toVisit[toVisitIndex].node;
                tMin = toVisit[toVisitIndex].tMin;
                tMax = toVisit[toVisitIndex].tMax;
            } else
                break;
        }
    }
    kdNodesVisited += nodesVisited;
    return si;
}

bool KdTreeAggregate::IntersectP(const Ray &ray, Float raytMax) const {
    // Compute initial parametric range of ray inside kd-tree extent
    Float tMin, tMax;
    if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
        return false;

    // Prepare to traverse kd-tree for ray
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    constexpr int maxTodo = 64;
    KdNodeToVisit toVisit[maxTodo];
    int toVisitIndex = 0;
    int nodesVisited = 0;
    const KdTreeNode *node = &nodes[0];
    while (node) {
        ++nodesVisited;
        if (node->IsLeaf()) {
            // Check for shadow ray intersections inside leaf newNode
            int nPrimitives = node->nPrimitives();
            if (nPrimitives == 1) {
                const Primitive &p = primitives[node->onePrimitiveIndex];
                if (p.IntersectP(ray, raytMax)) {
                    kdNodesVisited += nodesVisited;
                    return true;
                }
            } else {
                for (int i = 0; i < nPrimitives; ++i) {
                    int primitiveIndex =
                        primitiveIndices[node->primitiveIndicesOffset + i];
                    const Primitive &prim = primitives[primitiveIndex];
                    if (prim.IntersectP(ray, raytMax)) {
                        kdNodesVisited += nodesVisited;
                        return true;
                    }
                }
            }

            // Grab next newNode to process from todo list
            if (toVisitIndex > 0) {
                --toVisitIndex;
                node = toVisit[toVisitIndex].node;
                tMin = toVisit[toVisitIndex].tMin;
                tMax = toVisit[toVisitIndex].tMax;
            } else
                break;
        } else {
            // Process kd-tree interior newNode

            // Compute parametric distance along ray to split plane
            int axis = node->SplitAxis();
            Float tSplit = (node->SplitPos() - ray.o[axis]) * invDir[axis];

            // Get newNode newChildren pointers for ray
            const KdTreeNode *firstChild, *secondChild;
            int belowFirst = (ray.o[axis] < node->SplitPos()) ||
                             (ray.o[axis] == node->SplitPos() && ray.d[axis] <= 0);
            if (belowFirst != 0) {
                firstChild = node + 1;
                secondChild = &nodes[node->AboveChild()];
            } else {
                firstChild = &nodes[node->AboveChild()];
                secondChild = node + 1;
            }

            // Advance to next oldChild newNode, possibly enqueue other oldChild
            if (tSplit > tMax || tSplit <= 0)
                node = firstChild;
            else if (tSplit < tMin)
                node = secondChild;
            else {
                // Enqueue _secondChild_ in todo list
                toVisit[toVisitIndex].node = secondChild;
                toVisit[toVisitIndex].tMin = tSplit;
                toVisit[toVisitIndex].tMax = tMax;
                ++toVisitIndex;
                node = firstChild;
                tMax = tSplit;
            }
        }
    }
    kdNodesVisited += nodesVisited;
    return false;
}

KdTreeAggregate *KdTreeAggregate::Create(std::vector<Primitive> prims,
                                         const ParameterDictionary &parameters) {
    int isectCost = parameters.GetOneInt("intersectcost", 5);
    int travCost = parameters.GetOneInt("traversalcost", 1);
    Float emptyBonus = parameters.GetOneFloat("emptybonus", 0.5f);
    int maxPrims = parameters.GetOneInt("maxprims", 1);
    int maxDepth = parameters.GetOneInt("maxdepth", -1);
    return new KdTreeAggregate(std::move(prims), isectCost, travCost, emptyBonus,
                               maxPrims, maxDepth);
}
#pragma endregion

}  // namespace pbrt