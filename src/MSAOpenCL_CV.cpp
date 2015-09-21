/*
 *  OpenCL_CV.cpp
 *  OpenCL CV
 *
 *  Created by Mehmet Akten on 20/11/2009.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "MSAOpenCL_CV.h"
#include "MSAUtils.h"

namespace MSA {
	
	OpenCL_CV	cv;
	
	void OpenCL_CV::setup() {
		// init OpenCL from OpenGL context to enable GL-CL data sharing
		if(OpenCL::currentOpenCL == NULL) {
			openCL = new OpenCL;
			openCL->setupFromOpenGL();
			
		} else {
			openCL = OpenCL::currentOpenCL;
		}
		
		// load and compile OpenCL program
		setDataPathToBundle();
		openCL->loadProgramFromFile("MSAOpenCL_CV.cl");
		restoreDataPath();
		
		// load kernels
		openCL->loadKernel("msacv_addImage");
		openCL->loadKernel("msacv_mixImage");
		openCL->loadKernel("msacv_multiplyf");
		openCL->loadKernel("msacv_boxblur");
		openCL->loadKernel("msacv_dilate");
		openCL->loadKernel("msacv_erode");
		openCL->loadKernel("msacv_flipx");
		openCL->loadKernel("msacv_flipy");
		openCL->loadKernel("msacv_greyscale");
		openCL->loadKernel("msacv_flipxAndGreyscale");
		openCL->loadKernel("msacv_invert");
		openCL->loadKernel("msacv_threshold");
		openCL->loadKernel("msacv_absDiff");
		openCL->loadKernel("msacv_findEdges");
		openCL->loadKernel("msacv_min");
		openCL->loadKernel("msacv_max");
		openCL->loadKernel("msacv_maskDarker");
		openCL->loadKernel("msacv_maskLighter");
		openCL->loadKernel("msacv_maskChanged");
		openCL->loadKernel("msacv_opticalFlow");
		openCL->loadKernel("msacv_advect");
		
		// create image sampler
		clSampler = clCreateSampler(openCL->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, NULL);
	}
	
	
	void OpenCL_CV::addImage(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1) {
		OpenCLKernel *kernel = openCL->kernel("msacv_addImage");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImage1.getCLMem());
		kernel->setArg(3, clSampler);
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::mixImage(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1, cl_float weight1, cl_float weight2) {
		OpenCLKernel *kernel = openCL->kernel("msacv_mixImage");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImage1.getCLMem());
		kernel->setArg(3, weight1);
		kernel->setArg(4, weight2);
		kernel->setArg(5, clSampler);
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	
	void OpenCL_CV::multiplyf(OpenCLImagePingPong& image, cl_float f) {
		OpenCLKernel *kernel = openCL->kernel("msacv_multiplyf");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->setArg(2, f);
		kernel->setArg(3, clSampler);
		
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}	
	
	
	
	void OpenCL_CV::boxblur(OpenCLImagePingPong& image, int iterations) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_boxblur");
		kernel->setArg(3, clSampler);
		for(int i=0; i<iterations; i++) {
			cl_int offset = i + 1;
			kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
			kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
			kernel->setArg(2, offset);
			kernel->run2D(image.getWidth(), image.getHeight());
			image.swap();
		}
	}
	
	void OpenCL_CV::dilate(OpenCLImagePingPong& image, int iterations) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_dilate");
		kernel->setArg(2, clSampler);
		for(int i=0; i<iterations; i++) {
			kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
			kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
			kernel->run2D(image.getWidth(), image.getHeight());
			image.swap();
		}}
	
	void OpenCL_CV::erode(OpenCLImagePingPong& image, int iterations) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_erode");
		kernel->setArg(2, clSampler);
		for(int i=0; i<iterations; i++) {
			kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
			kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
			kernel->run2D(image.getWidth(), image.getHeight());
			image.swap();
		}
	}
	
	void OpenCL_CV::flipx(OpenCLImagePingPong& image) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_flipx");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}
	
	
	void OpenCL_CV::flipy(OpenCLImagePingPong& image) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_flipy");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}
	
	void OpenCL_CV::greyscale(OpenCLImagePingPong& image) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_greyscale");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}
	
	void OpenCL_CV::flipxAndGreyscale(OpenCLImagePingPong& image) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_flipxAndGreyscale");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}
	
	void OpenCL_CV::invert(OpenCLImagePingPong& image) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_invert");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}
	
	void OpenCL_CV::threshold(OpenCLImagePingPong& image, float thresholdLevel) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_threshold");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->setArg(2, thresholdLevel);
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}
	
	void OpenCL_CV::absDiff(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_absDiff");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImage1.getCLMem());
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::findEdges(OpenCLImage& dstImage, OpenCLImage& srcImage) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_findEdges");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage.getCLMem());
		kernel->setArg(2, clSampler);
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::min(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_min");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImageBG.getCLMem());
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::max(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_max");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImageBG.getCLMem());
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::maskDarker(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG, cl_float thresholdLevel) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_maskDarker");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImageBG.getCLMem());
		kernel->setArg(3, thresholdLevel);
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::maskLighter(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG, cl_float thresholdLevel) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_maskLighter");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImageBG.getCLMem());
		kernel->setArg(3, thresholdLevel);
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::maskChanged(OpenCLImage& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImageBG, cl_float thresholdLevel) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_maskChanged");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImageBG.getCLMem());
		kernel->setArg(3, thresholdLevel);
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
	}
	
	void OpenCL_CV::opticalFlow(OpenCLImagePingPong& dstImage, OpenCLImage& srcImage0, OpenCLImage& srcImage1, OpticalFlowSettings& settings) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_opticalFlow");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, srcImage0.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage1, srcImage1.getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_OpticalFlow_threshLevel, settings.threshLevel);
		kernel->setArg(kOpenCL_Arg_CV_OpticalFlow_offset, settings.offset);
		kernel->setArg(kOpenCL_Arg_CV_OpticalFlow_lambda, settings.lambda);
		kernel->setArg(kOpenCL_Arg_CV_OpticalFlow_scale, settings.scale);
		kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
		
		if(settings.blurAmount) {
			boxblur(dstImage, settings.blurAmount);
			//		OpenCLKernel *kernel = openCL->kernel("msacv_boxblur");
			//		kernel->setArg(3, clSampler);
			//		for(int i=0; i<settings.blurAmount; i++) {
			//			cl_int offset = i + 1;
			//			kernel->setArg(kOpenCL_Arg_CV_dstImage, dstImage.getBack().getCLMem());
			//			kernel->setArg(kOpenCL_Arg_CV_srcImage0, dstImage.getFront().getCLMem());
			//			kernel->setArg(2, offset);
			//			kernel->run2D(dstImage.getWidth(), dstImage.getHeight());
			//			dstImage.swap();
			//		}
		}
		
	}
	
	void OpenCL_CV::advect(OpenCLImagePingPong& image, cl_float amount) { 
		OpenCLKernel *kernel = openCL->kernel("msacv_advect");
		kernel->setArg(kOpenCL_Arg_CV_dstImage, image.getBack().getCLMem());
		kernel->setArg(kOpenCL_Arg_CV_srcImage0, image.getFront().getCLMem());
		kernel->setArg(2, amount);
		kernel->run2D(image.getWidth(), image.getHeight());
		image.swap();
	}
	
}