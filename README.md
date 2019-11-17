# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

This is a midterm project in the second section - "Camera" of Udacity's Sensor Fusion nanodegree. The overall goal of the "Camera" course is to build a collision detection system. As preparation for this, a feature tracking between two consecutive 2D camera images from the "KITTI" dataset is done. Various detector / descriptor combinations are tested to see which one is able to detect and match more keypoints in the least time. 


This mid-term project consists of four parts:

* First, create a ring buffer of a specific size to only keep a specified amount of images at a time and optimize memory load. 
* Then, integrate several keypoint detectors such as SHI-TOMASI, HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, several descriptor extraction and matching using brute force and also the FLANN Nearest neighbour (KNN) approach. 
* In the last part, test the various algorithms using all permutations and combinations of detector - descriptor in different combinations and compare performance measures. 


## Performance analysis - Data Description

### Generate descriptive statistics that summarize the central tendency, dispersion and  shape of a dataset’s distribution, excluding NaN values.


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Num of matched keypoints</th>
      <th>Time detectors (ms)</th>
      <th>Time descriptors (ms)</th>
      <th>Total time (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>121.436508</td>
      <td>86.860413</td>
      <td>0.710273</td>
      <td>87.570701</td>
    </tr>
    <tr>
      <th>std</th>
      <td>81.087233</td>
      <td>125.965220</td>
      <td>0.678844</td>
      <td>126.044320</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.000000</td>
      <td>1.969850</td>
      <td>0.075670</td>
      <td>3.927800</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>65.000000</td>
      <td>7.820625</td>
      <td>0.314046</td>
      <td>8.097220</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>102.000000</td>
      <td>18.934900</td>
      <td>0.433927</td>
      <td>19.232000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>165.000000</td>
      <td>91.301400</td>
      <td>0.993790</td>
      <td>91.849850</td>
    </tr>
    <tr>
      <th>max</th>
      <td>332.000000</td>
      <td>387.053000</td>
      <td>3.947520</td>
      <td>388.096000</td>
    </tr>
  </tbody>
</table>
</div>



Max Num of KeyPoints Detected




    Keypoint Detector              FAST
    Keypoint Descriptor           BRIEF
    Num of matched keypoints        332
    Time detectors (ms)         2.05355
    Time descriptors (ms)       2.08523
    Total time (ms)             4.13878
    Name: 82, dtype: object



Least total runtime for KeyPoints detection and matching




    Keypoint Detector              FAST
    Keypoint Descriptor           BRIEF
    Num of matched keypoints        331
    Time detectors (ms)         1.96985
    Time descriptors (ms)       1.95795
    Total time (ms)              3.9278
    Name: 84, dtype: object



Min Num of KeyPoints Detected




    Keypoint Detector             HARRIS
    Keypoint Descriptor            BRISK
    Num of matched keypoints          10
    Time detectors (ms)          18.7123
    Time descriptors (ms)       0.103116
    Total time (ms)              18.8154
    Name: 37, dtype: object



Longest total runtime for KeyPoints detection and matching




    Keypoint Detector             BRISK
    Keypoint Descriptor           BRIEF
    Num of matched keypoints        193
    Time detectors (ms)         387.053
    Time descriptors (ms)       1.04348
    Total time (ms)             388.096
    Name: 122, dtype: object



##  Analysis:

As can be seen from the above, the best detector - descriptor for keypoint calculation and matching is FAST detector and BRIEF descriptor based on both the most num of keypoints detector as well as least time for descriptor matching.


    <matplotlib.axes._subplots.AxesSubplot at 0x12626eb00>




![png](output_15_1.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1263f85f8>




![png](output_16_1.png)



## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.