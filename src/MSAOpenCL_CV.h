/*
 *  OpenCL_CV.h
 *  OpenCL CV
 *
 *  Created by Mehmet Akten on 20/11/2009.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "MSAOpenCL.h"



#define kOpenCL_Arg_CV_dstImage					0
#define kOpenCL_Arg_CV_srcImage0				1
#define kOpenCL_Arg_CV_srcImage1				2

#define kOpenCL_Arg_CV_OpticalFlow_threshLevel	3
#define kOpenCL_Arg_CV_OpticalFlow_offset		4
#define kOpenCL_Arg_CV_OpticalFlow_lambda		5
#define kOpenCL_Arg_CV_OpticalFlow_scale		6

namespace msa {
	
	typedef struct {
		bool			enabled;
		cl_float		threshLevel;
		cl_float		offset;
		cl_float		lambda;
		cl_float		scale;
		int				blurAmount;
	} OpticalFlowSettings;
	
	
	class OpenCL_CV {
	public:
		
		OpenCL	*openCL;
		
		void setup();
		
		void addImage(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1);
		void mixImage(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1, cl_float weight1, cl_float weight2);
		void multiplyf(OpenCLImagePingPong& image, cl_float f);
		void boxblur(OpenCLImagePingPong& image, int iterations = 1);
		void dilate(OpenCLImagePingPong& image, int iterations = 1);
		void erode(OpenCLImagePingPong& image, int iterations = 1);
		void flipx(OpenCLImagePingPong& image);
		void flipy(OpenCLImagePingPong& image);
		void greyscale(OpenCLImagePingPong& image);
		void flipxAndGreyscale(OpenCLImagePingPong& image);
		void invert(OpenCLImagePingPong& image);
		void threshold(OpenCLImagePingPong& image, cl_float thresholdLevel);
		void absDiff(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1);
		void findEdges(OpenCLImage& dstImage, OpenCLImage& srcImage);
		void min(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG);
		void max(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG);
		void maskDarker(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG, cl_float thresholdLevel = 0);
		void maskLighter(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG, cl_float thresholdLevel = 0);
		void maskChanged(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG, cl_float thresholdLevel = 0);
		void opticalFlow(OpenCLImagePingPong& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1, OpticalFlowSettings &settings);
		void advect(OpenCLImagePingPong &image, cl_float amount);
		
	protected:
		cl_sampler	clSampler;
		
	};
	
	
	extern 	OpenCL_CV	cv;
}
