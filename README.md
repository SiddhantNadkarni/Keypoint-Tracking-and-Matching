# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

https://github.com/SiddhantNadkarni/Keypoint-Tracking-and-Matching/blob/master/images/KeyPointGIF.mov

This is a midterm project in the second section - "Camera" of Udacity's Sensor Fusion nanodegree. The overall goal of the "Camera" course is to build a collision detection system. As preparation for this, a feature tracking between two consecutive 2D camera images from the "KITTI" dataset is done. Various detector / descriptor combinations are tested to see which one is able to detect and match more keypoints in the least time. 


This mid-term project consists of four parts:

* First, create a ring buffer of a specific size to only keep a specified amount of images at a time and optimize memory load. 
* Then, integrate several keypoint detectors such as SHI-TOMASI, HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, several descriptor extraction and matching using brute force and also the FLANN Nearest neighbour (KNN) approach. 
* In the last part, test the various algorithms using all permutations and combinations of detector - descriptor in different combinations and compare performance measures. 


Task MP.1:
* In this task, I used a ```std::vector``` as a buffer. As soon as the buffer exceeded the specified size, I erased the first (oldest) image in the buffer. The complexity of this operation is O(n). We can instead use std::deque to improve insertion and deletion complexity to O(1). However, I went ahead with std::vector because in our case the data buffer size is only 2.
```
dataBuffer.push_back(frame);
if(dataBuffer.size() > dataBufferSize)
{
    dataBuffer.erase(dataBuffer.begin());
}
```

Task MP.2 and Task MP.4:
* In this task, I used two ```std::vector<std::string>``` to store the detector and descriptor types as strings. 
* Used OpenCV to implement various detector types such as Shi-Tomasi, Harris, FAST, BRIEF, BRISK, ORB, AKAZE, and SIFT. A detector type of selected by the string entered in the ```std::vector<std::string> detectors_vec``` string-vector.
* Used OpenCV to implement various descriptor types such as BRIEF, BRISK, FREAK, ORB, AKAZE, and SIFT. A descriptor type of selected by the string entered in the ```std::vector<std::string> descriptors_vec``` string-vector.

```
std::vector<std::string> detectors_vec{"SIFT"};
std::vector<std::string> descriptors_vec{"BRISK", "BRIEF", "FREAK", "SIFT"};
```

Task MP.3:
* A bounding box (rectangle) is created using ``` cv::Rect ``` data structure from OpenCV around cars which are directly infront in the same lane. 
* All keypoints outside this box are erased.
* In this project, the bounding box co-ordinates are assumed. However in the next project, a bounding box will be created using Deep Learning (YOLO - Conv Neural Net).

Task MP.5:
* Keypoints are detected and their descriptors are calculated between two successive camera images. The next task is to match these keypoints and to decrease the false positives by selecting the correct matches. The function ```matchDescriptors``` takes in a ``` stringmatcherType``` which can be a ```"MAT_BF"``` i.e. Brute Force matching between descriptors in both the images by using the Hamming distance as a measure of similarity. 
* The Hamming distance computes the difference between both vectors by using an XOR function, which returns zero if two bits are identical and one if two bits are different. This is better and faster than other types such as sum of absolute differences (L1 Norm) or sum of squared difference (L2 Norm). * Brute force matching works well for small keypoint numbers but can be computationally expensive as the number of keypoints increases. Thus to improve on this, we use a FLANN matcher ```string matcherType = "MAT_FLANN"``` which uses KD-Tree data structure to search for matching pairs and avoid the exhaustive search of the brute force approach. *
Both BFMatching and FLANN accept a descriptor distance threshold T which is used to limit the number of matches to the good ones and discard matches where the respective pairs are no correspondences. 
* The next step is the reject or minimize the number of false positives by using a ```string selectorType``` which can be set to ```"SEL_NN"``` (nearest neighbour search) or ```"SEL_KNN"``` (K nearest neighbour search). 

Task MP.6:
* In this task I compute a nearest neighbor distance ratio for each keypoint which is a very efficient way of lowering the number of false positives.
* The main idea is to not apply the threshold on the SSD directly. Instead, for each keypoint in the source image, the two best matches are located in the reference image and the ratio between the descriptor distances is computed. Then, a threshold is applied to the ratio to sort out ambiguous matches.
* By computing the SSD ratio between best and second-best match, weak candidates can be filtered out.
* I have used a threshold of 0.8


Task MP.7:
* In this I ran all the detectors on all the images and analysed the performance of each detector in terms number of Keypoints detected.
* The final results are stored in excel sheets named [Task_7.xlsx](https://github.com/SiddhantNadkarni/Keypoint-Tracking-and-Matching/blob/master/Excel%20Sheets/Task_7.xlsx).
* The data analysis is performed in [Task_7_Data_Analysis](https://github.com/SiddhantNadkarni/Keypoint-Tracking-and-Matching/blob/master/Excel%20Sheets/iPython%20Notebooks/Task_7_Data_Analysis.ipynb) notebook.

Data Description:

Generate descriptive statistics that summarize the central tendency, dispersion and  shape of a dataset’s distribution, excluding NaN values.


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image Number</th>
      <th>Num of Keypoints</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>70.00000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.50000</td>
      <td>177.285714</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.89302</td>
      <td>117.853648</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.00000</td>
      <td>113.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.50000</td>
      <td>135.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.00000</td>
      <td>257.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.00000</td>
      <td>427.000000</td>
    </tr>
  </tbody>
</table>
</div>



Max Num of KeyPoints Detected



    Image Number            1
    Keypoint Detector    FAST
    Num of Keypoints      427
    Name: 31, dtype: object



Min Num of KeyPoints Detected




    Image Number              1
    Keypoint Detector    HARRIS
    Num of Keypoints         14
    Name: 11, dtype: object



Analysis:

As can be seen from the above, the best detector for keypoint calculation is FAST detector  based on both the most num of keypoints detected and worst is HARRIS which computes the lowest num of Keypoints.


![Screen Shot 2019-11-17 at 11 50 52 AM](https://user-images.githubusercontent.com/19183728/69014340-21540c00-093e-11ea-941e-51d46c08e60c.png)

Task MP.8 and MP.9:

* In these two tasks, I ran all permutation-combination of detector and descriptor types. I used two ```std::vector<std::string>``` to store the detector and descriptor types as strings. I then looped over all detectors and descriptors and wrote the number of keypoints detected as well as total time (ms) for keypoint detection and matching in a csv file called [results.csv](https://github.com/SiddhantNadkarni/Keypoint-Tracking-and-Matching/blob/master/src/results.csv) which can be found in src folder.
* However, not all detector type were compatible with the descriptor types. Thus, I had to do restrict to only a single detector at a time with the specific descriptor types it was compatible with.
* The final results are stored in excel sheets named [Task_8_and_9.xlsx](https://github.com/SiddhantNadkarni/Keypoint-Tracking-and-Matching/blob/master/Excel%20Sheets/Task_8_and_9.xlsx).
* The data analysis is performed in [Task_8_and_9_Analysis](https://github.com/SiddhantNadkarni/Keypoint-Tracking-and-Matching/blob/master/Excel%20Sheets/iPython%20Notebooks/Task_8_and_9_Analysis.ipynb) notebook.

Performance analysis - Data Description:

Generate descriptive statistics that summarize the central tendency, dispersion and  shape of a dataset’s distribution, excluding NaN values.


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



Analysis:

As can be seen from the above, the best detector - descriptor for keypoint calculation and matching is FAST detector and BRIEF descriptor based on both the most num of keypoints detector as well as least time for descriptor matching.

Num Keypoint matched - Histogram

![Screen Shot 2019-11-17 at 11 50 41 AM](https://user-images.githubusercontent.com/19183728/69013043-9ddfee00-0930-11ea-94d9-0e0e8b98dfdd.png)



Total time (ms) - Histogram 

![Screen Shot 2019-11-17 at 11 50 52 AM](https://user-images.githubusercontent.com/19183728/69013053-b18b5480-0930-11ea-8c84-0f45ff27abcb.png)




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
