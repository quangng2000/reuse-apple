// Reduction order is exactly ci,ky,kx as in the generic FMA kernel. K tiles
// cover four/eight whole input channels, so input indexing uses constants
// and fewer workgroup barriers; no im2col buffer or changed weights.
void dense3x3() {
    uint x=gl_LocalInvocationID.x,y=gl_LocalInvocationID.y;
    uint lane=y*16+x;
    uint width=pc.v[13],cin=pc.v[11],cout=pc.v[7],height=pc.v[12];
    uint spatialBase=gl_WorkGroupID.x*32,channelBase=gl_WorkGroupID.y*16;
    uint batch=gl_WorkGroupID.z,fullK=cin*9;
    uint inputBatch=pc.v[3]+batch*cin*height*width;
    precise float c00=0.0,c01=0.0,c10=0.0,c11=0.0;
    for(uint start=0;start<fullK;start+=DENSE_K) {
        for(uint pos=lane;pos<16*DENSE_K;pos+=128) {
            uint oc=channelBase+pos/DENSE_K,ki=start+pos%DENSE_K;
            weights[pos/DENSE_K][pos%DENSE_K]=(oc<cout && ki<fullK)?arena.data[pc.v[4]+oc*fullK+ki]:0.0;
        }
        for(uint pos=lane;pos<DENSE_K*32;pos+=128) {
            uint localK=pos/32,spatial=spatialBase+(pos%32);
            uint ki=start+localK;
            uint ic=ki/9,ky=(localK/3)%3,kx=localK%3;
            int ix=int(spatial+kx)-1;
            uint inputIndex=inputBatch+(ic*height+ky*pc.v[20])*width+uint(max(ix,0));
            samples[localK][pos%32]=(ki<fullK && spatial<width && ix>=0 && ix<int(width))?arena.data[inputIndex]:0.0;
        }
        barrier();
        uint remaining=min(uint(DENSE_K),fullK-start);
        for(uint j=0;j<remaining;++j) {
            c00=fma(weights[y][j],samples[j][x],c00);
            c01=fma(weights[y][j],samples[j][x+16],c01);
            c10=fma(weights[y+8][j],samples[j][x],c10);
            c11=fma(weights[y+8][j],samples[j][x+16],c11);
        }
        barrier();
    }
    uint oc0=channelBase+y,oc1=oc0+8,s0=spatialBase+x,s1=s0+16;
    if(pc.v[23]!=0) {
        float bias0=oc0<cout?arena.data[pc.v[5]+oc0]:0.0;
        float bias1=oc1<cout?arena.data[pc.v[5]+oc1]:0.0;
        c00=c00+bias0;c01=c01+bias0;c10=c10+bias1;c11=c11+bias1;
    }
    if(oc0<cout && s0<width)arena.data[pc.v[1]+(batch*cout+oc0)*width+s0]=c00;
    if(oc0<cout && s1<width)arena.data[pc.v[1]+(batch*cout+oc0)*width+s1]=c01;
    if(oc1<cout && s0<width)arena.data[pc.v[1]+(batch*cout+oc1)*width+s0]=c10;
    if(oc1<cout && s1<width)arena.data[pc.v[1]+(batch*cout+oc1)*width+s1]=c11;
}
