/***********************************************************************
 
 TODO:
 
 - create kernels for each combinaiton of pre processing (flipx, flipy, greyscale)
 
 Copyright (c) 2008, 2009, Memo Akten, www.memo.tv
 *** The Mega Super Awesome Visuals Company ***
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of MSA Visuals nor the names of its contributors 
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE. 
 *
 * ***********************************************************************/ 



//--------------------------------------------------------------
__kernel void msacv_addImage(__write_only image2d_t dstImage, __read_only image2d_t srcImage1, __read_only image2d_t srcImage2, sampler_t smp) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 c2	= read_imagef(srcImage2, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	write_imagef(dstImage, coords, c1+c2);
}

//--------------------------------------------------------------
__kernel void msacv_mixImage(__write_only image2d_t dstImage, __read_only image2d_t srcImage1, __read_only image2d_t srcImage2, const float weight1, const float weight2, sampler_t smp) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 c2	= read_imagef(srcImage2, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	write_imagef(dstImage, coords, (c1 * weight1) + (c2 * weight2));
}


// multiple image by number
//--------------------------------------------------------------
__kernel void msacv_multiplyf(write_only image2d_t dstImage, read_only image2d_t srcImage, const float f, sampler_t smp) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 color	= read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	write_imagef(dstImage, coords, color * f);   
}


//--------------------------------------------------------------
__kernel void msacv_boxblur(write_only image2d_t dstImage, read_only image2d_t srcImage, const int offset, sampler_t smp) {                                                                                            
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dx		= (int2)(offset, 0);
	int2 dy		= (int2)(0, offset);
	
	float4 color1 = read_imagef(srcImage, smp, coords) * 0.2042f;
	float4 color2 = read_imagef(srcImage, smp, coords + dx) * 0.1238f;
	float4 color3 = read_imagef(srcImage, smp, coords - dx) * 0.1238f;
	float4 color4 = read_imagef(srcImage, smp, coords + dy) * 0.1238f;
	float4 color5 = read_imagef(srcImage, smp, coords - dy) * 0.1238f;
	float4 color6 = read_imagef(srcImage, smp, coords + dx + dy) * 0.0751f;
	float4 color7 = read_imagef(srcImage, smp, coords - dx + dy) * 0.0751f;
	float4 color8 = read_imagef(srcImage, smp, coords + dx - dy) * 0.0751f;
	float4 color9 = read_imagef(srcImage, smp, coords - dx - dy) * 0.0751f;
	float4 color = (color1 + color2 + color3 + color4 + color5 + color6 + color7 + color8 + color9);// * 1.0f/9.0f;
	write_imagef(dstImage, coords, color);
}  



//--------------------------------------------------------------
__kernel void msacv_dilate(write_only image2d_t dstImage, read_only image2d_t srcImage, sampler_t smp) {                                                                                            
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dx		= (int2)(1, 0);
	int2 dy		= (int2)(0, 1);
	
	float4 color	=	read_imagef(srcImage, smp, coords);
	color = fmax(color, read_imagef(srcImage, smp, coords + dx));
	color = fmax(color, read_imagef(srcImage, smp, coords - dx));
	color = fmax(color, read_imagef(srcImage, smp, coords + dy));
	color = fmax(color, read_imagef(srcImage, smp, coords - dy));
	color = fmax(color, read_imagef(srcImage, smp, coords + dx + dy));
	color = fmax(color, read_imagef(srcImage, smp, coords - dx + dy));
	color = fmax(color, read_imagef(srcImage, smp, coords + dx - dy));
	color = fmax(color, read_imagef(srcImage, smp, coords - dx - dy));
	write_imagef(dstImage, coords, color);
}




//--------------------------------------------------------------
__kernel void msacv_erode(write_only image2d_t dstImage, read_only image2d_t srcImage, sampler_t smp) {                                                                                            
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dx		= (int2)(1, 0);
	int2 dy		= (int2)(0, 1);
	
	float4 color	=	read_imagef(srcImage, smp, coords);
	color = fmin(color, read_imagef(srcImage, smp, coords + dx));
	color = fmin(color, read_imagef(srcImage, smp, coords - dx));
	color = fmin(color, read_imagef(srcImage, smp, coords + dy));
	color = fmin(color, read_imagef(srcImage, smp, coords - dy));
	color = fmin(color, read_imagef(srcImage, smp, coords + dx + dy));
	color = fmin(color, read_imagef(srcImage, smp, coords - dx + dy));
	color = fmin(color, read_imagef(srcImage, smp, coords + dx - dy));
	color = fmin(color, read_imagef(srcImage, smp, coords - dx - dy));
	write_imagef(dstImage, coords, color);
}


//--------------------------------------------------------------
__kernel void msacv_findEdges(write_only image2d_t dstImage, read_only image2d_t srcImage, sampler_t smp) {                                                                                            
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dx		= (int2)(1, 0);
	int2 dy		= (int2)(0, 1);
	
	//	float4 p11 = read_imagef(srcImage, smp, coords);
	float4 p00 = read_imagef(srcImage, smp, coords - dx - dy);
	float4 p01 = read_imagef(srcImage, smp, coords      - dy);
	float4 p02 = read_imagef(srcImage, smp, coords + dx - dy);
	float4 p10 = read_imagef(srcImage, smp, coords - dx);
	float4 p12 = read_imagef(srcImage, smp, coords + dx);
	float4 p20 = read_imagef(srcImage, smp, coords - dx + dy);
	float4 p21 = read_imagef(srcImage, smp, coords      + dy);
	float4 p22 = read_imagef(srcImage, smp, coords + dx + dy);
	
	float4 sumX = (p22+p02-p20-p00) + 2.0f*(p12-p10);
	float4 sumY = (p20+p22-p00-p02) + 2.0f*(p21-p01);
	
	float sum = sqrt(dot(sumX,sumX) + dot(sumY,sumY));
	
	write_imagef(dstImage, coords, (float4)(sum, sum, sum, sum));
}


//--------------------------------------------------------------
__kernel void msacv_flipx(write_only image2d_t dstImage, read_only image2d_t srcImage) {                                                                                            
	int i = get_global_id(0);
	int j = get_global_id(1);
	int2 coords1 = (int2)(i, j);
	int2 coords2 = (int2)(get_image_width(srcImage) - i - 1.0f, j);
	float4 color = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords1);
	write_imagef(dstImage, coords2, color);
}  


//--------------------------------------------------------------
__kernel void msacv_flipy(write_only image2d_t dstImage, read_only image2d_t srcImage) {                                                                                            
	int i = get_global_id(0);
	int j = get_global_id(1);
	int2 coords1 = (int2)(i, j);
	int2 coords2 = (int2)(i, get_image_height(srcImage) - j - 1.0f);
	float4 color = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords1);
	write_imagef(dstImage, coords2, color);
}  

//--------------------------------------------------------------
__kernel void msacv_greyscale(write_only image2d_t dstImage, read_only image2d_t srcImage) {                                                                                            
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 color = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float luminance = 0.3f * color.x + 0.59 * color.y + 0.11 * color.z;
	color = (float4)(luminance, luminance, luminance, 1.0f);
	write_imagef(dstImage, coords, color);                                     
} 

//--------------------------------------------------------------
__kernel void msacv_flipxAndGreyscale(write_only image2d_t dstImage, read_only image2d_t srcImage) {                                                                                            
	int i = get_global_id(0);
	int j = get_global_id(1);
	int2 coords1 = (int2)(i, j);
	int2 coords2 = (int2)(get_image_width(srcImage) - i - 1.0f, j);
	float4 color = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords1);
	float luminance = 0.3f * color.x + 0.59 * color.y + 0.11 * color.z;
	write_imagef(dstImage, coords2, (float4)(luminance, luminance, luminance, 1.0f));
}  


//--------------------------------------------------------------
__kernel void msacv_invert(write_only image2d_t dstImage, read_only image2d_t srcImage) {                                                                                            
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 color = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	color = (float4)(1.0f, 1.0f, 1.0f, 1.0f) - color;
	write_imagef(dstImage, coords, color);
}  


//--------------------------------------------------------------
__kernel void msacv_threshold(write_only image2d_t dstImage, read_only image2d_t srcImage, const float thresholdLevel) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 color	= read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	write_imagei(dstImage, coords, isgreaterequal(color, thresholdLevel));   
}



//--------------------------------------------------------------
__kernel void msacv_absDiff(write_only image2d_t dstImage, read_only image2d_t srcImage1, read_only image2d_t srcImage2) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 c2	= read_imagef(srcImage2, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 diff = fabs(c1-c2);
	write_imagef(dstImage, coords, (float4)(diff.x, diff.y, diff.z, 1.0f));
}

//--------------------------------------------------------------
__kernel void msacv_min(write_only image2d_t dstImage, read_only image2d_t srcImage1, read_only image2d_t srcImage2) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 c2	= read_imagef(srcImage2, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	write_imagef(dstImage, coords, fmin(c1, c2));
}


//--------------------------------------------------------------
__kernel void msacv_max(write_only image2d_t dstImage, read_only image2d_t srcImage1, read_only image2d_t srcImage2) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 c2	= read_imagef(srcImage2, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	write_imagef(dstImage, coords, fmax(c2, c1));
}


// returns binary BW image keeping only darker pixels
//--------------------------------------------------------------
__kernel void msacv_maskDarker(write_only image2d_t dstImage, read_only image2d_t srcImage1, read_only image2d_t srcImageBG, const float thresholdLevel) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 cBG	= read_imagef(srcImageBG, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 diff = cBG-c1;
	int4 comp	= isgreaterequal(diff, thresholdLevel);
	write_imagei(dstImage, coords, comp);
}

// returns binary BW image keeping only brighter pixels
//--------------------------------------------------------------
__kernel void msacv_maskLighter(write_only image2d_t dstImage, read_only image2d_t srcImage1, read_only image2d_t srcImageBG, const float thresholdLevel) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 cBG	= read_imagef(srcImageBG, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 diff = c1-cBG;
	int4 comp	= isgreaterequal(diff, thresholdLevel);
	write_imagei(dstImage, coords, comp);
}

// returns binary BW image keeping changed pixels
//--------------------------------------------------------------
__kernel void msacv_maskChanged(write_only image2d_t dstImage, read_only image2d_t srcImage1, read_only image2d_t srcImageBG, const float thresholdLevel) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 c1	= read_imagef(srcImage1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 cBG	= read_imagef(srcImageBG, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
	float4 diff = fabs(c1-cBG);
	int4 comp	= isgreaterequal(diff, thresholdLevel);
	write_imagei(dstImage, coords, comp);
}



//--------------------------------------------------------------
// OpenCL Optical Flow Kernel based on Andrew Benson's GLSL Shader andrewb@cycling74.com
// ported to OpenCL by Mehmet Akten. www.memo.tv

__kernel void msacv_opticalFlow(write_only image2d_t image, read_only image2d_t tex0, read_only image2d_t tex1, const float thresholdLevel, const float offset, const float lambda, const float scale) {
  	int xid = get_global_id(0), yid = get_global_id(1);
  	
   	int2	 ipos = make_int2(xid, yid);
	float2 texcoord = make_float2(xid + 0.5, yid + 0.5);
	
	float4 a = read_imagef(tex0, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord);
	float4 b = read_imagef(tex1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord);
	float2 x1 = make_float2(offset, 0.0);
	float2 y1 = make_float2(0.,offset);
	
	//get the difference
	float4 curdif = b-a;
	
	// mask out noise
	curdif *= convert_float4(isgreaterequal(fabs(curdif), thresholdLevel));
	
	if(dot(curdif, curdif) > 0.0) {
		
		//calculate the gradient
		float4 gradx = read_imagef(tex1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord+x1) - read_imagef(tex1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord-x1);
		gradx += read_imagef(tex0, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord+x1) - read_imagef(tex0, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord-x1);
		float4 grady = read_imagef(tex1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord+y1)- read_imagef(tex1, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord-y1);
		grady += read_imagef(tex0, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord+y1) - read_imagef(tex0, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, texcoord-y1);
		float4 gradmag = sqrt((gradx*gradx)+(grady*grady)+make_float4(lambda, lambda, lambda, lambda));
		
		float4 vx = curdif*(gradx/gradmag);
		float vxd = vx.x;//assumes greyscale
		//format output for flowrepos, out(-x,+x,-y,+y)
		float2 xout = make_float2(fmax(vxd, 0), fabs(fmin(vxd, 0)));
		float4 vy = curdif*(grady/gradmag);
		float vyd = vy.x;//assumes greyscale
		//format output for flowrepos, out(-x,+x,-y,+y)
		float2 yout = make_float2(fmax(vyd, 0), fabs(fmin(vyd, 0)));
		
		//	float4 outputColor = make_float4(xout.x, xout.y, yout.x, yout.y);
		float4 outputColor = make_float4((xout.y-xout.x) * scale, (yout.y-yout.x) * scale, 0, 1);
		write_imagef(image, ipos, outputColor);
	} else {
		write_imagef(image, ipos, (float4)(0., 0., 0., 0.));
	}
}


__kernel void msacv_advect(write_only image2d_t dstImage, read_only image2d_t srcImage, float amount) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dx		= (int2)(1, 0);
	int2 dy		= (int2)(0, 1);
	float2 dxf	= (float2)(1., 0);
	float2 dyf	= (float2)(0, 1.);
	float2 zero = (float2)(0., 0.);
	
	sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	
//	float2 color1 = zero;//read_imagef(srcImage, smp, coords).xy;
	float2 vel2 = read_imagef(srcImage, smp, coords + dx).xy;
	float2 vel3 = read_imagef(srcImage, smp, coords - dx).xy;
	float2 vel4 = read_imagef(srcImage, smp, coords + dy).xy;
	float2 vel5 = read_imagef(srcImage, smp, coords - dy).xy;
	float2 vel6 = read_imagef(srcImage, smp, coords + dx + dy).xy;
	float2 vel7 = read_imagef(srcImage, smp, coords - dx + dy).xy;
	float2 vel8 = read_imagef(srcImage, smp, coords + dx - dy).xy;
	float2 vel9 = read_imagef(srcImage, smp, coords - dx - dy).xy;

	float m = 0.0f;//
	vel2 *= max(m, dot(vel2, -dxf));
	vel3 *= max(m, dot(vel3, +dxf));
	vel4 *= max(m, dot(vel4, -dyf));
	vel5 *= max(m, dot(vel5, +dyf));
	vel6 *= max(m, dot(vel6, -( dxf + dyf) ));
	vel7 *= max(m, dot(vel7, -(-dxf + dyf) ));
	vel8 *= max(m, dot(vel8, -( dxf - dyf) ));
	vel9 *= max(m, dot(vel9, -(-dxf - dyf) ));
	float2 color = (vel2 + vel3 + vel4 + vel5 + vel6 + vel7 + vel8 + vel9) * 1.0f/8.0f;
	color = amount * clamp(color, (float)-1., (float)1.);
	write_imagef(dstImage, coords, (float4)(color.x, color.y, 0, 0));
}


/*
// vade opflow blur
__kernel void msacv_advect(write_only image2d_t dstImage, read_only image2d_t srcImage, float amount) {
	int i				= get_global_id(0);
	int j				= get_global_id(1);
	int2 coords			= (int2)(i, j);
	float4 col			= read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);

	float2 amount1		= col.xy * amount;// + blurAmount1 + blurAmount2;
	float2 amount2 = amount1 *1.5;
	float2 amount3 = amount1 *3.0;
	float2 amount4 = amount1 *6.0;
	
	float2 amount5 = amount1 * 8.0;
	float2 amount6 = amount1 * 10.0;
	float2 amount7 = amount1 * 12.0;
	float2 amount8 = amount1 * 18.0;
	
	// sample our textures
	float4 sample0 = col;
	float4 sample1 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount1);
	float4 sample2 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount2);
	float4 sample3 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount3);
	float4 sample4 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount4);
	float4 sample5 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount5);
	float4 sample6 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount6);
	float4 sample7 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount7);
	float4 sample8 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords + (int2)amount8);
	
	write_imagef(dstImage, coords, (sample0 + sample1 + sample2 + sample3 + sample4 + sample5 + sample6 + sample7 + sample8) / 9.0);
}
*/


// fluid advect style
//__kernel void msacv_advect(write_only image2d_t dstImage, read_only image2d_t srcImage, float amount) {
//	int i			= get_global_id(0);
//	int j			= get_global_id(1);
//	int2 coords		= (int2)(i, j);
//	float4 c		= read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
//	
//	
//	float dt0x	= amount;
//	float dt0y	= amount;
//	float x		= i - dt0x * c.x;
//	float y		= j - dt0y * c.y;
//	int2 size	= get_image_dim(srcImage) - (int2)(1, 1);
//	
//	if (x > size.x - 0.5) x = size.x - 0.5;
//	else if (x < 0.5)     x = 0.5f;
//	
//	int i0 = (int) x;
//	
//	if (y > size.y - 0.5) x = size.y - 0.5;
//	else if (y < 0.5)     y = 0.5f;
//	
//	int j0 = (int) y;
//	
//	float s1 = x - i0;
//	float s0 = 1. - s1;
//	float t1 = y - j0;
//	float t0 = 1. - t1;
//	
//	float4 c_i0  = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (int2)(i0, j0));
//	float4 c_j0  = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (int2)(i0, j0));
//	float4 c_i01 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (int2)(i0+1, j0));
//	float4 c_j01 = read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (int2)(i0, j0+1));
//	
//	float4 col	= s0 * ( t0 * c_i0     + t1 * c_j0     ) + s1 * ( t0 * c_i01      + t1 * c_j01 );
//	
//	write_imagef(dstImage, coords, col);
//}



 // my advect
//__kernel void msacv_advect(write_only image2d_t dstImage, read_only image2d_t srcImage, float amount) {
//	int2 coords		= (int2)(get_global_id(0), get_global_id(1));
//	float4 c		= read_imagef(srcImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, coords);
//	int2 coords2	= coords + (int2)(c.xy * amount);
//	
//	//	coords2			%= get_image_dim(srcImage);
//	//	coords2			= max((int2)(0, 0), coords);
//	int2 imageRange		= get_image_dim(srcImage) - (int2)(1, 1);
//	
//	if(coords2.x < 0) {
//		coords2.x = 0;
//		c.x *= -.1;
//	} else if(coords2.x > imageRange.x) {
//		coords2.x = imageRange.x;
//		c.x *= -.1;
//	}
//	if(coords2.y < 0) {
//		coords2.y = 0;
//		c.y *= -.1;
//	} else if(coords2.y > imageRange.y) {
//		coords2.y = imageRange.y;
//		c.y *= -.1;
//	}
//	
//	write_imagef(dstImage, coords2, c);
//}
