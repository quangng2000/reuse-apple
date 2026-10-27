#include "../common.glsl"

// Tensor extents are fixed for the model, but push constants prevent the
// compiler from strength-reducing generic division. Common feature dimensions
// are powers of two; the highest nontrivial dimension needs no division at
// all. This mapping is exactly the same integer address calculation.
uint address6Fast(uint linearIndex,uint firstStride) {
    int address=0;
    for(int axis=5;axis>=0;--axis) {
        uint size=pc.v[8+axis];
        if(size==1)continue;
        uint coordinate;
        if(linearIndex<size) {
            coordinate=linearIndex;
            linearIndex=0;
        } else if((size&(size-1))==0) {
            coordinate=linearIndex&(size-1);
            linearIndex>>=findMSB(size);
        } else {
            uint quotient=linearIndex/size;
            coordinate=linearIndex-quotient*size;
            linearIndex=quotient;
        }
        address+=int(coordinate)*int(pc.v[firstStride+axis]);
    }
    return uint(address);
}
