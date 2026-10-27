// NVIDIA Real-time RE-USE adaptation. NSCLv1, noncommercial research/education.
// New Vulkan adaptation (c) 2026 quangng2000.
#ifndef REUSE_VULKAN_COMMON
#define REUSE_VULKAN_COMMON
layout(set=0,binding=0,std430) buffer Arena { float data[]; } arena;
layout(push_constant) uniform Parameters { uint v[32]; } pc;

// Every address and stride is measured in FP32 elements. Dimension arrays
// always contain six dimensions, with leading ones for smaller ranks.
uint address6(uint linearIndex, uint firstStride) {
    int address=0;
    for (int axis=5;axis>=0;--axis) {
        uint size=pc.v[8+axis];
        if(size==1)continue;
        uint coordinate=linearIndex%size;
        linearIndex/=size;
        address += int(coordinate)*int(pc.v[firstStride+axis]);
    }
    return uint(address);
}

float reuseSoftplus(float value);
// MatMul's optional following Add/Softplus retain their rounded FP32 steps.
float matmulBias(float dotValue,uint outputIndex,uint column) {
    precise float result=dotValue;
    if(pc.v[14]==1)result=result+arena.data[pc.v[5]+column];
    else if(pc.v[14]==2)result=result+arena.data[pc.v[5]+outputIndex];
    if(pc.v[15]!=0)result=reuseSoftplus(result);
    return result;
}

// ARM64 ORT 1.23.2 MLAS Exp arithmetic, expressed in FP32 GLSL. Explicit
// fma calls reproduce the polynomial's fused operations; other expressions
// use precise intermediates to forbid implicit contraction. See MLAS-MIT.txt.
float reuseExp(float inputValue) {
    if (isnan(inputValue)) return inputValue;
    precise float value=clamp(inputValue,-103.9720840454,88.7762626647950);
    precise float biased=fma(value,1.44269504088896341,12582912.0);
    precise float m=biased-12582912.0;
    value=fma(m,-6.93145752e-1,value);
    value=fma(m,-1.42860677e-6,value);
    int overflow=int(floatBitsToUint(biased)<<23);
    int normal=clamp(overflow,int(0xC1000000u),int(0x3F800000u));
    overflow=int(uint(overflow)-uint(normal)+0x3F800000u);
    normal=int(uint(normal)+0x3F800000u);
    precise float p=uintBitsToFloat(0x3ab4a000u);
    p=fma(p,value,uintBitsToFloat(0x3c092f6eu));
    p=fma(p,value,uintBitsToFloat(0x3d2aadadu));
    p=fma(p,value,uintBitsToFloat(0x3e2aaa28u));
    p=fma(p,value,uintBitsToFloat(0x3efffffbu));
    p=fma(p,value,1.0);
    value=value*intBitsToFloat(overflow);
    p=fma(p,value,intBitsToFloat(overflow));
    return p*intBitsToFloat(normal);
}

float reuseSoftplus(float value) {
    precise float small=reuseExp(-abs(value));
    precise float sum=1.0+small;
    // Kahan correction retains log1p's small input rather than rounding it
    // to zero in the initial 1+x. All operations and buffers remain FP32.
    precise float correction=((sum-1.0)-small)/sum;
    precise float logarithm=sum==1.0 ? small : log(sum)-correction;
    return max(value,0.0)+logarithm;
}
#endif
