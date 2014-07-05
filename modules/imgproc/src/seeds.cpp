/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, Beat Kueng (beat-kueng@gmx.net), Lukas Vogel, Morten Lysgaard
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

/******************************************************************************\
*                            SEEDS Superpixels                                *
*  This code implements the superpixel method described in:                   *
*  M. Van den Bergh, X. Boix, G. Roig, B. de Capitani and L. Van Gool,        *
*  "SEEDS: Superpixels Extracted via Energy-Driven Sampling", ECCV 2012       *
\******************************************************************************/

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
using namespace std;


//TODO:
//labels: convert to cv Mat (or something) -> remove LABELINT
//define public interface
// -> ifdefs configuration
//code comments -> [y*width+x], etc
//compiler warnings



/* algorithm configuration */


// If enabled, each block-updating level will be iterated twice.
// Improves the segmentation. Disable this to speed up the algorithm (about 10%).
#define DOUBLE_STEPS
#define REQ_CONF 0.1

// Enables 3x3 smoothing prior
#define PRIOR

#define MINIMUM_NR_SUBLABELS 1



// the integer type of the labels array
typedef int LABELINT;

// the type of the histogram and the T array
typedef float HISTN;


namespace cv {

class SuperpixelSEEDSImpl : public SuperpixelSEEDS
{
public:

    SuperpixelSEEDSImpl(int width, int height, int nr_channels = 3, int nr_bins = 5);

    virtual ~SuperpixelSEEDSImpl();

    void initialize(int seeds_w, int seeds_h, int nr_levels);

    // calculate the superpixels
    void iterate(InputArray img, int num_iterations = 4);

    //get the result. format: label of pixel(x,y) = labels[y*width + x]
    int* getLabels() { return labels; }

    void drawContoursAroundLabels(InputOutputArray image, int color = 0xff0000,
            bool thick_line = false);

    // evaluation
    int countSuperpixels();
    void saveLabelsToTextFile(std::string filename);

private:
    void deinitialize();

    /* initialization */
    void initImage(InputArray img);
    void assignLabels();
    void computeHistograms(int until_level = -1);
    template<typename _Tp>
    inline void initImageBins(const Mat& img, int max_value);


    /* pixel operations */
    inline void update(int label_new, int image_idx, int label_old);
    //image_idx = y*width+x
    inline void addPixel(int level, int label, int image_idx);
    inline void deletePixel(int level, int label, int image_idx);
    inline bool probability(int image_idx, int label1, int label2, int prior1, int prior2);
    inline int threebyfour(int x, int y, unsigned int label);
    inline int fourbythree(int x, int y, unsigned int label);

    inline void updateLabels();
    // main loop for border updating
    void updatePixels();


    /* block operations */
    void addBlock(int level, int label, int sublevel, int sublabel);
    inline void addBlockToplevel(int label, int sublevel, int sublabel);
    void deleteBlockToplevel(int label, int sublevel, int sublabel);

    // intersection on label1A and intersection_del on label1B
    // returns intA - intB
    float intersectConf(int level1, int label1A, int label1B, int level2, int label2);

    //main loop for block updates
    void updateBlocks(int level, float req_confidence = 0.0);

    /* go to next block level */
    int goDownOneLevel();

    //make sure a superpixel stays connected (h=horizontal,v=vertical, f=forward,b=backward)
    inline bool checkSplit_hf(int a11, int a12, int a21, int a22, int a31, int a32);
    inline bool checkSplit_hb(int a12, int a13, int a22, int a23, int a32, int a33);
    inline bool checkSplit_vf(int a11, int a12, int a13, int a21, int a22, int a23);
    inline bool checkSplit_vb(int a21, int a22, int a23, int a31, int a32, int a33);

    //compute initial label for sublevels: level <= seeds_top_level
    //this is an equally sized grid with size nr_h[level]*nr_w[level]
    int computeLabel(int level, int x, int y) {
        return std::min(y / (height / nr_wh[2 * level + 1]), nr_wh[2 * level + 1] - 1) * nr_wh[2 * level]
                + std::min((x / (width / nr_wh[2 * level])), nr_wh[2 * level] - 1);
    }
    inline int nrLabels(int level) const {
        return nr_wh[2 * level + 1] * nr_wh[2 * level];
    }

    int width, height; //image size
    int nr_bins; //number of histogram bins per channel
    int nr_channels;
    bool initialized;
    bool forwardbackward;

    int seeds_w; //grid size of lowest level 0
    int seeds_h;
    int seeds_nr_levels;
    int seeds_top_level; // == seeds_nr_levels-1 (const)
    int seeds_current_level; //start with level seeds_top_level-1, then go down

    // keep one labeling for each level
    LABELINT* nr_wh; // [2*level]/[2*level+1] number of label in x-direction/y-direction

    /* pre-initialized arrays. they are not modified afterwards */
    LABELINT* labels_bottom; //labels of level==0
    LABELINT** parent_pre_init;

    unsigned int* image_bins; //[y*width + x] bin index (histogram) of each image pixel

    LABELINT** parent; //[level][label] = corresponding label of block with level+1
    int* labels; //[y * width + x] output labels: labels of level==seeds_top_level
    unsigned int* nr_partitions; //[label] how many partitions label has on toplevel

    int histogram_size; //= nr_bins ^ 3 (3 channels)
    int histogram_size_aligned;
    HISTN** histogram; //[level][label * histogram_size_aligned + j]
    HISTN** T; //[level][label] how many pixels with this label
};

CV_EXPORTS Ptr<SuperpixelSEEDS> createSuperpixelSEEDS(int width, int height,
        int nr_channels, int nr_bins)
{
    return makePtr<SuperpixelSEEDSImpl>(width, height, nr_channels, nr_bins);
}

void SuperpixelSEEDSImpl::iterate(InputArray img, int num_iterations)
{
    initImage(img);

    // block updates
    while (seeds_current_level >= 0)
    {
#ifdef DOUBLE_STEPS
        updateBlocks(seeds_current_level, REQ_CONF); //this call is optional, improves results
#endif
        updateBlocks(seeds_current_level);
        seeds_current_level = goDownOneLevel();
    }
    updateLabels();

    for (int i = 0; i < num_iterations; ++i)
        updatePixels();
}

SuperpixelSEEDSImpl::SuperpixelSEEDSImpl(int width, int height, int nr_channels,
        int nr_bins)
{
    this->width = width;
    this->height = height;
    this->nr_bins = nr_bins;
    this->nr_channels = nr_channels;

    image_bins = new unsigned int[width * height];
    histogram_size = nr_bins;
    for (int i = 1; i < nr_channels; ++i)
        histogram_size *= nr_bins;
    histogram_size_aligned = (histogram_size
        + ((CV_MALLOC_ALIGN / sizeof(HISTN)) - 1)) & -(CV_MALLOC_ALIGN / sizeof(HISTN));
    initialized = false;
}

SuperpixelSEEDSImpl::~SuperpixelSEEDSImpl()
{
    deinitialize();
    delete[] image_bins;
}

void SuperpixelSEEDSImpl::deinitialize()
{
    if( initialized )
    {
        initialized = false;
        for (int level = 0; level < seeds_nr_levels; level++)
        {
            fastFree(histogram[level]);
            fastFree(T[level]);
            delete[] parent_pre_init[level];
            delete[] parent[level];
        }
        fastFree(histogram);
        fastFree(T);
        fastFree(labels);
        delete[] labels_bottom;
        delete[] parent;
        delete[] parent_pre_init;
        delete[] nr_partitions;
        delete[] nr_wh;
    }
}

void SuperpixelSEEDSImpl::initialize(int seeds_w, int seeds_h, int nr_levels)
{
    deinitialize();

    this->seeds_w = seeds_w;
    this->seeds_h = seeds_h;
    this->seeds_nr_levels = nr_levels;
    this->seeds_top_level = nr_levels - 1;

    // init labels
    labels = (LABELINT*) fastMalloc(sizeof(LABELINT) * width * height);
    labels_bottom = new LABELINT[width * height];
    parent = new LABELINT*[seeds_nr_levels];
    parent_pre_init = new LABELINT*[seeds_nr_levels];
    nr_wh = new LABELINT[2 * seeds_nr_levels];
    int level = 0;
    int nr_seeds_w = floor(width / seeds_w);
    int nr_seeds_h = floor(height / seeds_h);
    int nr_seeds = nr_seeds_w * nr_seeds_h;
    nr_wh[2 * level] = nr_seeds_w;
    nr_wh[2 * level + 1] = nr_seeds_h;
    parent[level] = new LABELINT[nr_seeds];
    parent_pre_init[level] = new LABELINT[nr_seeds];
    for (level = 1; level < seeds_nr_levels; level++)
    {
        nr_seeds_w /= 2; // always partitioned in 2x2 sub-blocks
        nr_seeds_h /= 2; // always partitioned in 2x2 sub-blocks
        nr_seeds = nr_seeds_w * nr_seeds_h;
        parent[level] = new LABELINT[nr_seeds];
        parent_pre_init[level] = new LABELINT[nr_seeds];
        nr_wh[2 * level] = nr_seeds_w;
        nr_wh[2 * level + 1] = nr_seeds_h;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                parent_pre_init[level - 1][computeLabel(level - 1, x, y)] =
                        computeLabel(level, x, y); // set parent
            }
        }
    }
    nr_partitions = new unsigned int[nrLabels(seeds_top_level)];

    //preinit the labels (these are not changed anymore later)
    int i = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            labels_bottom[i] = computeLabel(0, x, y);
            ++i;
        }
    }

    // create histogram buffers
    histogram = (HISTN**) fastMalloc(sizeof(HISTN*) * seeds_nr_levels);
    T = (HISTN**) fastMalloc(sizeof(HISTN*) * seeds_nr_levels);
    for (level = 0; level < seeds_nr_levels; level++)
    {
        int nr_labels = nrLabels(level);
        histogram[level] = (HISTN*) fastMalloc(
                sizeof(HISTN) * nr_labels * histogram_size_aligned);
        T[level] = (HISTN*) fastMalloc(sizeof(HISTN) * nr_labels);
    }

    initialized = true;
}


template<typename _Tp>
void SuperpixelSEEDSImpl::initImageBins(const Mat& img, int max_value)
{
    int width = img.size().width;
    int height = img.size().height;
    int channels = img.channels();

    for(int y = 0; y<height; ++y)
    {
        for(int x=0; x<width; ++x)
        {
            const _Tp* ptr = img.ptr<_Tp>(y, x);
            int bin = 0;
            for(int i=0; i<channels; ++i)
                bin = bin * nr_bins + (int)ptr[i] * nr_bins / max_value;
            image_bins[y*width + x] = bin;
        }
    }
}

/* specialization for float: max_value is assumed to be 1.0f */
template<>
void SuperpixelSEEDSImpl::initImageBins<float>(const Mat& img, int)
{
    int width = img.size().width;
    int height = img.size().height;
    int channels = img.channels();

    for(int y = 0; y<height; ++y)
    {
        for(int x=0; x<width; ++x)
        {
            const float* ptr = img.ptr<float>(y, x);
            int bin = 0;
            for(int i=0; i<channels; ++i)
                bin = bin * nr_bins + std::min((int)(ptr[i] * (float)nr_bins), nr_bins-1);
            image_bins[y*width + x] = bin;
        }
    }
}

void SuperpixelSEEDSImpl::initImage(InputArray img)
{
    Mat src = img.getMat();
    int depth = src.depth();
    seeds_current_level = seeds_nr_levels - 2;
    forwardbackward = true;

    assignLabels();

    CV_Assert(src.size().width == width && src.size().height == height);
    CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);
    CV_Assert(src.channels() == nr_channels);

    // initialize the histogram bins from the image
    switch (depth)
    {
    case CV_8U:
        initImageBins<uchar>(src, 1 << 8);
        break;
    case CV_16U:
        initImageBins<ushort>(src, 1 << 16);
        break;
    case CV_32F:
        initImageBins<float>(src, 1);
        break;
    }

    computeHistograms();
}

// adds labeling to all the blocks at all levels and sets the correct parents
void SuperpixelSEEDSImpl::assignLabels()
{
    /* each top level label is partitioned into 4 elements */
    int nr_labels_toplevel = nrLabels(seeds_top_level);
    for (int i = 0; i < nr_labels_toplevel; ++i)
        nr_partitions[i] = 4;

    for (int level = 1; level < seeds_nr_levels; level++)
    {
        memcpy(parent[level - 1], parent_pre_init[level - 1],
                sizeof(LABELINT) * nrLabels(level - 1));
    }
}

void SuperpixelSEEDSImpl::computeHistograms(int until_level)
{
    if( until_level == -1 )
        until_level = seeds_nr_levels - 1;
    until_level++;

    // clear histograms
    for (int level = 0; level < seeds_nr_levels; level++)
    {
        int nr_labels = nrLabels(level);
        memset(histogram[level], 0,
                sizeof(HISTN) * histogram_size_aligned * nr_labels);
        memset(T[level], 0, sizeof(HISTN) * nr_labels);
    }

    // build histograms on the first level by adding the pixels to the blocks
    for (int i = 0; i < width * height; ++i)
        addPixel(0, labels_bottom[i], i);

    // build histograms on the upper levels by adding the histogram from the level below
    for (int level = 1; level < until_level; level++)
    {
        for (unsigned int label = 0; label < nrLabels(level - 1); label++)
        {
            addBlock(level, parent[level - 1][label], level - 1, label);
        }
    }
}

void SuperpixelSEEDSImpl::updateBlocks(int level, float req_confidence)
{
    int labelA;
    int labelB;
    int sublabel;
    bool done;
    int step = nr_wh[2 * level];

    // horizontal bidirectional block updating
    for (int y = 1; y < nr_wh[2 * level + 1] - 1; y++)
    {
        for (int x = 1; x < nr_wh[2 * level] - 2; x++)
        {
            // choose a label at the current level
            sublabel = y * step + x;
            // get the label at the top level (= superpixel label)
            labelA = parent[level][y * step + x];
            // get the neighboring label at the top level (= superpixel label)
            labelB = parent[level][y * step + x + 1];

            if( labelA == labelB )
                continue;

            // get the surrounding labels at the top level, to check for splitting
            int a11 = parent[level][(y - 1) * step + (x - 1)];
            int a12 = parent[level][(y - 1) * step + (x)];
            int a21 = parent[level][(y) * step + (x - 1)];
            int a22 = parent[level][(y) * step + (x)];
            int a31 = parent[level][(y + 1) * step + (x - 1)];
            int a32 = parent[level][(y + 1) * step + (x)];
            done = false;

            if( nr_partitions[labelA] == 2 || (nr_partitions[labelA] > 2 // 3 or more partitions
                    && checkSplit_hf(a11, a12, a21, a22, a31, a32)) )
            {
                // run algorithm as usual
                float conf = intersectConf(seeds_top_level, labelB, labelA, level, sublabel);
                if( conf > req_confidence )
                {
                    deleteBlockToplevel(labelA, level, sublabel);
                    addBlockToplevel(labelB, level, sublabel);
                    done = true;
                }
            }

            if( !done && (nr_partitions[labelB] > MINIMUM_NR_SUBLABELS) )
            {
                // try opposite direction
                sublabel = y * step + x + 1;
                int a13 = parent[level][(y - 1) * step + (x + 1)];
                int a14 = parent[level][(y - 1) * step + (x + 2)];
                int a23 = parent[level][(y) * step + (x + 1)];
                int a24 = parent[level][(y) * step + (x + 2)];
                int a33 = parent[level][(y + 1) * step + (x + 1)];
                int a34 = parent[level][(y + 1) * step + (x + 2)];
                if( nr_partitions[labelB] <= 2 // == 2
                        || (nr_partitions[labelB] > 2 && checkSplit_hb(a13, a14, a23, a24, a33, a34)) )
                {
                    // run algorithm as usual
                    float conf = intersectConf(seeds_top_level, labelA, labelB, level, sublabel);
                    if( conf > req_confidence )
                    {
                        deleteBlockToplevel(labelB, level, sublabel);
                        addBlockToplevel(labelA, level, sublabel);
                        x++;
                    }
                }
            }
        }
    }

    // vertical bidirectional
    for (int x = 1; x < nr_wh[2 * level] - 1; x++)
    {
        for (int y = 1; y < nr_wh[2 * level + 1] - 2; y++)
        {
            // choose a label at the current level
            sublabel = y * step + x;
            // get the label at the top level (= superpixel label)
            labelA = parent[level][y * step + x];
            // get the neighboring label at the top level (= superpixel label)
            labelB = parent[level][(y + 1) * step + x];

            if( labelA == labelB )
                continue;

            int a11 = parent[level][(y - 1) * step + (x - 1)];
            int a12 = parent[level][(y - 1) * step + (x)];
            int a13 = parent[level][(y - 1) * step + (x + 1)];
            int a21 = parent[level][(y) * step + (x - 1)];
            int a22 = parent[level][(y) * step + (x)];
            int a23 = parent[level][(y) * step + (x + 1)];

            done = false;
            if( nr_partitions[labelA] == 2 || (nr_partitions[labelA] > 2 // 3 or more partitions
                    && checkSplit_vf(a11, a12, a13, a21, a22, a23)) )
            {
                // run algorithm as usual
                float conf = intersectConf(seeds_top_level, labelB, labelA, level, sublabel);
                if( conf > req_confidence )
                {
                    deleteBlockToplevel(labelA, level, sublabel);
                    addBlockToplevel(labelB, level, sublabel);
                    done = true;
                }
            }

            if( !done && (nr_partitions[labelB] > MINIMUM_NR_SUBLABELS) )
            {
                // try opposite direction
                sublabel = (y + 1) * step + x;
                int a31 = parent[level][(y + 1) * step + (x - 1)];
                int a32 = parent[level][(y + 1) * step + (x)];
                int a33 = parent[level][(y + 1) * step + (x + 1)];
                int a41 = parent[level][(y + 2) * step + (x - 1)];
                int a42 = parent[level][(y + 2) * step + (x)];
                int a43 = parent[level][(y + 2) * step + (x + 1)];
                if( nr_partitions[labelB] <= 2 // == 2
                        || (nr_partitions[labelB] > 2 && checkSplit_vb(a31, a32, a33, a41, a42, a43)) )
                {
                    // run algorithm as usual
                    float conf = intersectConf(seeds_top_level, labelA, labelB, level, sublabel);
                    if( conf > req_confidence )
                    {
                        deleteBlockToplevel(labelB, level, sublabel);
                        addBlockToplevel(labelA, level, sublabel);
                        y++;
                    }
                }
            }
        }
    }
}

int SuperpixelSEEDSImpl::goDownOneLevel()
{
    int old_level = seeds_current_level;
    int new_level = seeds_current_level - 1;

    if( new_level < 0 )
        return -1;

    // reset nr_partitions
    memset(nr_partitions, 0, sizeof(LABELINT) * nrLabels(seeds_top_level));

    // go through labels of new_level
    int labels_new_level = nrLabels(new_level);
    //the lowest level (0) has 1 partition, all higher levels are
    //initially partitioned into 4
    int partitions = new_level ? 4 : 1;

    for (int label = 0; label < labels_new_level; ++label)
    {
        // assign parent = parent of old_label
        LABELINT& cur_parent = parent[new_level][label];
        int p = parent[old_level][cur_parent];
        cur_parent = p;

        nr_partitions[p] += partitions;
    }

    return new_level;
}

void SuperpixelSEEDSImpl::updatePixels()
{
    int labelA;
    int labelB;
    int priorA = 0;
    int priorB = 0;

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 2; x++)
        {

            labelA = labels[(y) * width + (x)];
            labelB = labels[(y) * width + (x + 1)];

            if( labelA != labelB )
            {
                int a22 = labelA;
                int a23 = labelB;
                if( forwardbackward )
                {
                    // horizontal bidirectional
                    int a11 = labels[(y - 1) * width + (x - 1)];
                    int a12 = labels[(y - 1) * width + (x)];
                    int a21 = labels[(y) * width + (x - 1)];
                    int a31 = labels[(y + 1) * width + (x - 1)];
                    int a32 = labels[(y + 1) * width + (x)];
                    if( checkSplit_hf(a11, a12, a21, a22, a31, a32) )
                    {
#ifdef PRIOR
                        priorA = threebyfour(x, y, labelA);
                        priorB = threebyfour(x, y, labelB);
#endif

                        if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                        {
                            update(labelB, y * width + x, labelA);
                        }
                        else
                        {
                            int a13 = labels[(y - 1) * width + (x + 1)];
                            int a14 = labels[(y - 1) * width + (x + 2)];
                            int a24 = labels[(y) * width + (x + 2)];
                            int a33 = labels[(y + 1) * width + (x + 1)];
                            int a34 = labels[(y + 1) * width + (x + 2)];
                            if( checkSplit_hb(a13, a14, a23, a24, a33, a34) )
                            {
                                if( probability(y * width + x + 1, labelB, labelA, priorB, priorA) )
                                {
                                    update(labelA, y * width + x + 1, labelB);
                                    x++;
                                }
                            }
                        }
                    }
                }
                else
                { // forward backward
                    // horizontal bidirectional
                    int a13 = labels[(y - 1) * width + (x + 1)];
                    int a14 = labels[(y - 1) * width + (x + 2)];
                    int a24 = labels[(y) * width + (x + 2)];
                    int a33 = labels[(y + 1) * width + (x + 1)];
                    int a34 = labels[(y + 1) * width + (x + 2)];
                    if( checkSplit_hb(a13, a14, a23, a24, a33, a34) )
                    {
#ifdef PRIOR
                        priorA = threebyfour(x, y, labelA);
                        priorB = threebyfour(x, y, labelB);
#endif

                        if( probability(y * width + x + 1, labelB, labelA, priorB, priorA) )
                        {
                            update(labelA, y * width + x + 1, labelB);
                            x++;
                        }
                        else
                        {
                            int a11 = labels[(y - 1) * width + (x - 1)];
                            int a12 = labels[(y - 1) * width + (x)];
                            int a21 = labels[(y) * width + (x - 1)];
                            int a31 = labels[(y + 1) * width + (x - 1)];
                            int a32 = labels[(y + 1) * width + (x)];
                            if( checkSplit_hf(a11, a12, a21, a22, a31, a32) )
                            {
                                if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                                {
                                    update(labelB, y * width + x, labelA);
                                }
                            }
                        }
                    }
                }
            } // labelA != labelB
        } // for x
    } // for y

    for (int x = 1; x < width - 1; x++)
    {
        for (int y = 1; y < height - 2; y++)
        {

            labelA = labels[(y) * width + (x)];
            labelB = labels[(y + 1) * width + (x)];
            if( labelA != labelB )
            {
                int a22 = labelA;
                int a32 = labelB;

                if( forwardbackward )
                {
                    // vertical bidirectional
                    int a11 = labels[(y - 1) * width + (x - 1)];
                    int a12 = labels[(y - 1) * width + (x)];
                    int a13 = labels[(y - 1) * width + (x + 1)];
                    int a21 = labels[(y) * width + (x - 1)];
                    int a23 = labels[(y) * width + (x + 1)];
                    if( checkSplit_vf(a11, a12, a13, a21, a22, a23) )
                    {
#ifdef PRIOR
                        priorA = fourbythree(x, y, labelA);
                        priorB = fourbythree(x, y, labelB);
#endif

                        if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                        {
                            update(labelB, y * width + x, labelA);
                        }
                        else
                        {
                            int a31 = labels[(y + 1) * width + (x - 1)];
                            int a33 = labels[(y + 1) * width + (x + 1)];
                            int a41 = labels[(y + 2) * width + (x - 1)];
                            int a42 = labels[(y + 2) * width + (x)];
                            int a43 = labels[(y + 2) * width + (x + 1)];
                            if( checkSplit_vb(a31, a32, a33, a41, a42, a43) )
                            {
                                if( probability((y + 1) * width + x, labelB, labelA, priorB, priorA) )
                                {
                                    update(labelA, (y + 1) * width + x, labelB);
                                    y++;
                                }
                            }
                        }
                    }
                }
                else
                { // forwardbackward
                    // vertical bidirectional
                    int a31 = labels[(y + 1) * width + (x - 1)];
                    int a33 = labels[(y + 1) * width + (x + 1)];
                    int a41 = labels[(y + 2) * width + (x - 1)];
                    int a42 = labels[(y + 2) * width + (x)];
                    int a43 = labels[(y + 2) * width + (x + 1)];
                    if( checkSplit_vb(a31, a32, a33, a41, a42, a43) )
                    {
#ifdef PRIOR
                        priorA = fourbythree(x, y, labelA);
                        priorB = fourbythree(x, y, labelB);
#endif

                        if( probability((y + 1) * width + x, labelB, labelA, priorB, priorA) )
                        {
                            update(labelA, (y + 1) * width + x, labelB);
                            y++;
                        }
                        else
                        {
                            int a11 = labels[(y - 1) * width + (x - 1)];
                            int a12 = labels[(y - 1) * width + (x)];
                            int a13 = labels[(y - 1) * width + (x + 1)];
                            int a21 = labels[(y) * width + (x - 1)];
                            int a23 = labels[(y) * width + (x + 1)];
                            if( checkSplit_vf(a11, a12, a13, a21, a22, a23) )
                            {
                                if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                                {
                                    update(labelB, y * width + x, labelA);
                                }
                            }
                        }
                    }
                }
            } // labelA != labelB
        } // for y
    } // for x
    forwardbackward = !forwardbackward;

    // update border pixels
    for (int x = 0; x < width; x++)
    {
        labelA = labels[x];
        labelB = labels[width + x];
        if( labelA != labelB )
            update(labelB, x, labelA);
        labelA = labels[(height - 1) * width + x];
        labelB = labels[(height - 2) * width + x];
        if( labelA != labelB )
            update(labelB, (height - 1) * width + x, labelA);
    }
    for (int y = 0; y < height; y++)
    {
        labelA = labels[y * width];
        labelB = labels[y * width + 1];
        if( labelA != labelB )
            update(labelB, y * width, labelA);
        labelA = labels[y * width + width - 1];
        labelB = labels[y * width + width - 2];
        if( labelA != labelB )
            update(labelB, y * width + width - 1, labelA);
    }
}

void SuperpixelSEEDSImpl::update(int label_new, int image_idx, int label_old)
{
    //change the label of a single pixel
    deletePixel(seeds_top_level, label_old, image_idx);
    addPixel(seeds_top_level, label_new, image_idx);
    labels[image_idx] = label_new;
}

void SuperpixelSEEDSImpl::addPixel(int level, int label, int image_idx)
{
    histogram[level][label * histogram_size_aligned + image_bins[image_idx]]++;
    T[level][label]++;
}

void SuperpixelSEEDSImpl::deletePixel(int level, int label, int image_idx)
{
    histogram[level][label * histogram_size_aligned + image_bins[image_idx]]--;
    T[level][label]--;
}

void SuperpixelSEEDSImpl::addBlock(int level, int label, int sublevel,
        int sublabel)
{
    parent[sublevel][sublabel] = label;

    HISTN* h_label = &histogram[level][label * histogram_size_aligned];
    HISTN* h_sublabel = &histogram[sublevel][sublabel * histogram_size_aligned];

    //add the (sublevel, sublabel) block to the block (level, label)
    int n = 0;
#if CV_SSSE3
    const int loop_end = histogram_size - 3;
    for (; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time
        __m128 h_labelp = _mm_load_ps(h_label + n);
        __m128 h_sublabelp = _mm_load_ps(h_sublabel + n);
        h_labelp = _mm_add_ps(h_labelp, h_sublabelp);
        _mm_store_ps(h_label + n, h_labelp);
    }
#endif

    //loop peeling
    for (; n < histogram_size; n++)
        h_label[n] += h_sublabel[n];

    T[level][label] += T[sublevel][sublabel];
}

void SuperpixelSEEDSImpl::addBlockToplevel(int label, int sublevel, int sublabel)
{
    addBlock(seeds_top_level, label, sublevel, sublabel);
    nr_partitions[label]++;
}

void SuperpixelSEEDSImpl::deleteBlockToplevel(int label, int sublevel, int sublabel)
{
    HISTN* h_label = &histogram[seeds_top_level][label * histogram_size_aligned];
    HISTN* h_sublabel = &histogram[sublevel][sublabel * histogram_size_aligned];

    //do the reverse operation of add_block_toplevel
    int n = 0;
#if CV_SSSE3
    const int loop_end = histogram_size - 3;
    for (; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time
        __m128 h_labelp = _mm_load_ps(h_label + n);
        __m128 h_sublabelp = _mm_load_ps(h_sublabel + n);
        h_labelp = _mm_sub_ps(h_labelp, h_sublabelp);
        _mm_store_ps(h_label + n, h_labelp);
    }
#endif

    //loop peeling
    for (; n < histogram_size; ++n)
        h_label[n] -= h_sublabel[n];

    T[seeds_top_level][label] -= T[sublevel][sublabel];

    nr_partitions[label]--;
}

void SuperpixelSEEDSImpl::updateLabels()
{
    for (int i = 0; i < width * height; ++i)
        labels[i] = parent[0][labels_bottom[i]];
}

bool SuperpixelSEEDSImpl::probability(int image_idx, int label1, int label2,
        int prior1, int prior2)
{
    unsigned int color = image_bins[image_idx];
    float P_label1 = histogram[seeds_top_level][label1 * histogram_size_aligned + color]
                                                * T[seeds_top_level][label2];
    float P_label2 = histogram[seeds_top_level][label2 * histogram_size_aligned + color]
                                                * T[seeds_top_level][label1];

#ifdef PRIOR
    P_label1 *= (float) prior1;
    P_label2 *= (float) prior2;
#endif

    return (P_label2 > P_label1);
}

int SuperpixelSEEDSImpl::threebyfour(int x, int y, unsigned int label)
{
    /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
     * neighborhood (x=counted, o,O=ignored, O=(x,y)):
     * x x x x
     * x O o x
     * x x x x
     */

#if CV_SSSE3
    __m128i addp = _mm_set1_epi32(1);
    __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
    __m128i labelp = _mm_set1_epi32(label);
    /* 1. row */
    __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
    __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
    __m128i countp = _mm_and_si128(mask1, addp);
    /* 2. row */
    __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
    __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
    __m128i count1 = _mm_and_si128(mask2, addp_middle);
    countp = _mm_add_epi32(countp, count1);
    /* 3. row */
    __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
    __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
    __m128i count3 = _mm_and_si128(mask3, addp);
    countp = _mm_add_epi32(count3, countp);

    countp = _mm_hadd_epi32(countp, countp);
    countp = _mm_hadd_epi32(countp, countp);
    return _mm_cvtsi128_si32(countp);
#else
    int count = 0;
    count += (labels[(y - 1) * width + x - 1] == label);
    count += (labels[(y - 1) * width + x] == label);
    count += (labels[(y - 1) * width + x + 1] == label);
    count += (labels[(y - 1) * width + x + 2] == label);

    count += (labels[y * width + x - 1] == label);
    count += (labels[y * width + x + 2] == label);

    count += (labels[(y + 1) * width + x - 1] == label);
    count += (labels[(y + 1) * width + x] == label);
    count += (labels[(y + 1) * width + x + 1] == label);
    count += (labels[(y + 1) * width + x + 2] == label);

    return count;
#endif
}

int SuperpixelSEEDSImpl::fourbythree(int x, int y, unsigned int label)
{
    /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
     * neighborhood (x=counted, o,O=ignored, O=(x,y)):
     * x x x o
     * x O o x
     * x o o x
     * x x x o
     */

#if CV_SSSE3
    __m128i addp_border = _mm_set_epi32(0, 1, 1, 1);
    __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
    __m128i labelp = _mm_set1_epi32(label);
    /* 1. row */
    __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
    __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
    __m128i countp = _mm_and_si128(mask1, addp_border);
    /* 2. row */
    __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
    __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
    __m128i count1 = _mm_and_si128(mask2, addp_middle);
    countp = _mm_add_epi32(countp, count1);
    /* 3. row */
    __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
    __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
    __m128i count3 = _mm_and_si128(mask3, addp_middle);
    countp = _mm_add_epi32(count3, countp);
    /* 4. row */
    __m128i data4 = _mm_loadu_si128((__m128i*) (labels + (y+2)*width + x -1));
    __m128i mask4 = _mm_cmpeq_epi32(data4, labelp);
    __m128i count4 = _mm_and_si128(mask4, addp_border);
    countp = _mm_add_epi32(countp, count4);

    countp = _mm_hadd_epi32(countp, countp);
    countp = _mm_hadd_epi32(countp, countp);
    return _mm_cvtsi128_si32(countp);
#else
    int count = 0;
    count += (labels[(y - 1) * width + x - 1] == label);
    count += (labels[(y - 1) * width + x] == label);
    count += (labels[(y - 1) * width + x + 1] == label);

    count += (labels[y * width + x - 1] == label);
    count += (labels[y * width + x + 2] == label);

    count += (labels[(y + 1) * width + x - 1] == label);
    count += (labels[(y + 1) * width + x + 2] == label);

    count += (labels[(y + 2) * width + x - 1] == label);
    count += (labels[(y + 2) * width + x] == label);
    count += (labels[(y + 2) * width + x + 1] == label);

    return count;
#endif
}

float SuperpixelSEEDSImpl::intersectConf(int level1, int label1A, int label1B,
        int level2, int label2)
{
    float sumA = 0, sumB = 0;
    float* h1A = &histogram[level1][label1A * histogram_size_aligned];
    float* h1B = &histogram[level1][label1B * histogram_size_aligned];
    float* h2 = &histogram[level2][label2 * histogram_size_aligned];
    const float count1A = T[level1][label1A];
    const float count2 = T[level2][label2];
    const float count1B = T[level1][label1B] - count2;

    /* this calculates several things:
     * - normalized intersection of a histogram. which is equal to:
     *   sum i over bins ( min(histogram1_i / T1_i, histogram2_i / T2_i) )
     * - intersection A = intersection of (level1, label1A) and (level2, label2)
     * - intersection B =
     *     intersection of (level1, label1B) - (level2, label2) and (level2, label2)
     *   where (level1, label1B) - (level2, label2)
     *     is the substraction of 2 histograms (-> delete_block method)
     * - returns the difference between the 2 intersections: intA - intB
     */

    int n = 0;
#if CV_SSSE3
    __m128 count1Ap = _mm_set1_ps(count1A);
    __m128 count2p = _mm_set1_ps(count2);
    __m128 count1Bp = _mm_set1_ps(count1B);
    __m128 sumAp = _mm_set1_ps(0.0f);
    __m128 sumBp = _mm_set1_ps(0.0f);

    const int loop_end = histogram_size - 3;
    for(; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time

        // normal
        __m128 h1Ap = _mm_load_ps(h1A + n);
        __m128 h1Bp = _mm_load_ps(h1B + n);
        __m128 h2p = _mm_load_ps(h2 + n);

        __m128 h1ApC2 = _mm_mul_ps(h1Ap, count2p);
        __m128 h2pC1A = _mm_mul_ps(h2p, count1Ap);
        __m128 maskA = _mm_cmple_ps(h1ApC2, h2pC1A);
        __m128 sum1AddA = _mm_and_ps(maskA, h1ApC2);
        __m128 sum2AddA = _mm_andnot_ps(maskA, h2pC1A);
        sumAp = _mm_add_ps(sumAp, sum1AddA);
        sumAp = _mm_add_ps(sumAp, sum2AddA);

        // del
        __m128 diffp = _mm_sub_ps(h1Bp, h2p);
        __m128 h1BpC2 = _mm_mul_ps(diffp, count2p);
        __m128 h2pC1B = _mm_mul_ps(h2p, count1Bp);
        __m128 maskB = _mm_cmple_ps(h1BpC2, h2pC1B);
        __m128 sum1AddB = _mm_and_ps(maskB, h1BpC2);
        __m128 sum2AddB = _mm_andnot_ps(maskB, h2pC1B);
        sumBp = _mm_add_ps(sumBp, sum1AddB);
        sumBp = _mm_add_ps(sumBp, sum2AddB);
    }
    // merge results (quite expensive)
    float sum1Asse;
    sumAp = _mm_hadd_ps(sumAp, sumAp);
    sumAp = _mm_hadd_ps(sumAp, sumAp);
    _mm_store_ss(&sum1Asse, sumAp);

    float sum1Bsse;
    sumBp = _mm_hadd_ps(sumBp, sumBp);
    sumBp = _mm_hadd_ps(sumBp, sumBp);
    _mm_store_ss(&sum1Bsse, sumBp);

    sumA += sum1Asse;
    sumB += sum1Bsse;
#endif

    //loop peeling
    for (; n < histogram_size; ++n)
    {
        // normal intersect
        if( h1A[n] * count2 < h2[n] * count1A ) sumA += h1A[n] * count2;
        else sumA += h2[n] * count1A;

        // intersect_del
        float diff = h1B[n] - h2[n];
        if( diff * count2 < h2[n] * count1B ) sumB += diff * count2;
        else sumB += h2[n] * count1B;
    }

    float intA = sumA / (count1A * count2);
    float intB = sumB / (count1B * count2);
    return intA - intB;
}

bool SuperpixelSEEDSImpl::checkSplit_hf(int a11, int a12, int a21, int a22, int a31, int a32)
{
    if( (a22 != a21) && (a22 == a12) && (a22 == a32) ) return false;
    if( (a22 != a11) && (a22 == a12) && (a22 == a21) ) return false;
    if( (a22 != a31) && (a22 == a32) && (a22 == a21) ) return false;
    return true;
}
bool SuperpixelSEEDSImpl::checkSplit_hb(int a12, int a13, int a22, int a23, int a32, int a33)
{
    if( (a22 != a23) && (a22 == a12) && (a22 == a32) ) return false;
    if( (a22 != a13) && (a22 == a12) && (a22 == a23) ) return false;
    if( (a22 != a33) && (a22 == a32) && (a22 == a23) ) return false;
    return true;

}
bool SuperpixelSEEDSImpl::checkSplit_vf(int a11, int a12, int a13, int a21, int a22, int a23)
{
    if( (a22 != a12) && (a22 == a21) && (a22 == a23) ) return false;
    if( (a22 != a11) && (a22 == a21) && (a22 == a12) ) return false;
    if( (a22 != a13) && (a22 == a23) && (a22 == a12) ) return false;
    return true;
}
bool SuperpixelSEEDSImpl::checkSplit_vb(int a21, int a22, int a23, int a31, int a32, int a33)
{
    if( (a22 != a32) && (a22 == a21) && (a22 == a23) ) return false;
    if( (a22 != a31) && (a22 == a21) && (a22 == a32) ) return false;
    if( (a22 != a33) && (a22 == a23) && (a22 == a32) ) return false;
    return true;
}

int SuperpixelSEEDSImpl::countSuperpixels()
{
    int nr_labels_top_level = nrLabels(seeds_top_level);
    int* count_labels = new int[nr_labels_top_level];
    for (unsigned int i = 0; i < nr_labels_top_level; i++)
        count_labels[i] = 0;
    for (int i = 0; i < width * height; i++)
        count_labels[labels[i]] = 1;
    int count = 0;
    for (unsigned int i = 0; i < nr_labels_top_level; i++)
        count += count_labels[i];

    delete count_labels;
    return count;
}

void SuperpixelSEEDSImpl::saveLabelsToTextFile(string filename)
{
    ofstream outfile;
    outfile.open(filename.c_str());
    int i = 0;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width - 1; w++)
        {
            outfile << labels[i] << " ";
            i++;
        }
        outfile << labels[i] << endl;
        i++;
    }
    outfile.close();
}

void SuperpixelSEEDSImpl::drawContoursAroundLabels(InputOutputArray image,
        int color, bool thick_line)
{
    const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int width = image.size().width;
    int height = image.size().height;
    vector<bool> istaken(width * height, false);
    Mat img = image.getMat();

    int mainindex = 0;
    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int np = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                {
                    int index = y * width + x;
                    if( labels[mainindex] != labels[index] )
                    {
                        if( thick_line || !istaken[index] )
                            np++;
                    }
                }
            }
            if( np > 1 )
            {
                switch (img.depth())
                {
                case CV_8U:
                    img.ptr<uchar>(j, k)[0] = (color) & 0xff;
                    img.ptr<uchar>(j, k)[1] = (color >> 8) & 0xff;
                    img.ptr<uchar>(j, k)[2] = (color >> 16) & 0xff;
                    break;
                case CV_16U:
                    img.ptr<ushort>(j, k)[0] = ((color) & 0xff) << 8;
                    img.ptr<ushort>(j, k)[1] = ((color >> 8) & 0xff) << 8;
                    img.ptr<ushort>(j, k)[2] = ((color >> 16) & 0xff) << 8;
                    break;
                case CV_32F:
                    img.ptr<float>(j, k)[0] = (float) ((color) & 0xff) / 255;
                    img.ptr<float>(j, k)[1] = (float) ((color >> 8) & 0xff) / 255;
                    img.ptr<float>(j, k)[2] = (float) ((color >> 16) & 0xff) / 255;
                    break;
                }
                istaken[mainindex] = true;
            }
            mainindex++;
        }
    }
}

} // namespace cv
