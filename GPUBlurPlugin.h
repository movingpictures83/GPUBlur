#ifndef GPUBLURPLUGIN_H
#define GPUBLURPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUBlurPlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};


#define BLUR_SIZE 5

//@@ INSERT CODE HERE
#define TILE_WIDTH 16
__global__ void blurKernel(float *out, float *in, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float pixVal = 0;
    float pixels = 0;

    // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
    for (int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE + 1; ++blurrow) {
      for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1; ++blurcol) {

        int currow = row + blurrow;
        int curcol = col + blurcol;
        // Verify we have a valid image pixel
        if (currow > -1 && currow < height && curcol > -1 &&
            curcol < width) {
          pixVal += in[currow * width + curcol];
          pixels++; // Keep track of number of pixels in the avg
        }
      }
    }

    // Write our new pixel value out
    out[row * width + col] = (pixVal / pixels);
  }
}

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

static float *readPPM(const char *filename, int* width, int* height)
{
         char buff[16];
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    /*if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }*/

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    int myX, myY;
    //read image size information
    if (fscanf(fp, "%d %d", &myX, &myY) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }
    *height = myY;
    *width = myX;

    float* retval = (float*) malloc(3*myY*myX*sizeof(float));
    unsigned char* mychararr = (unsigned char*) malloc(3*myY*myX*sizeof(unsigned char));
    fread(mychararr, 1, 3*myX*myY, fp);
    for (int i = 0; i < 3*myX*myY; i++) {
       retval[i] = (float) mychararr[i];
    }
    fclose(fp);
    return retval;
}
void writePPM(const char *filename, int width, int height, float *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",width, height);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    unsigned char* result = (unsigned char*) malloc(3*width*height);
    for (int i = 0; i < 3*width*height; i++) {
       result[i] = img[i];
    }
    // pixel data
    fwrite(img, 1, 3 * width * height, fp);
    fclose(fp);
}

#endif
