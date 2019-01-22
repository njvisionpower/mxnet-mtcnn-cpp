# MTCNN-mxnet-cpp
--------------------
This project aims to implement face detection algorithm with mxnet c++ version. You can deploy the face detect part if use insightface(https://github.com/deepinsight/insightface) or other face recognition pipeline.

## Why start this project?
There are many bugs or implement error for original mtcnn cpp version (such as https://github.com/deepinsight/mxnet-mtcnn), here I list some typical issue:

### 1. Memory leak
Forget to free pNet memory, especially pNet will run many times for different scale.
    
### 2. Computing overhead optimization
Frequently load pNet for every scale original image, loading params is a very expensive operation in mxnet. Two strategy for this:  

    (1)If your image shape is fixed, you can loading different input shape models to feed different scale shape, 
    pnet is very small so will not occupy too much memory. Aslo this step can be easyly implement with multi-thread.
    
    (2)If shape may change, you can set a relatively larger shape PredictorHandle and padding 0 for smalle input.
    
Anyway, currently mxnet C/C++ predict api can't automatically adjust predictor shape when network input batch or shape changes. You can use MXPredReshape to change shape but this operation will also spent some time and some case may fail.

### 3. Numerical calculation
#### (1) One case is round function get different result:
In c++ and python:

    round(11.5) = 12, round(12.5) = 13  
    np.round(11.5) = 12, np.round(12.5) = 12  
That is numpy consider the number is even or odd before "5".
#### (2) Forget to +1 when calculate the bottom position of face bounding box:  
    float bottom_x = (int)((x*stride + cellSize) / scale);  -> float bottom_x = (int)((x*stride + cellSize +1 ) / scale); 
    float bottom_y = (int)((y*stride + cellSize) / scale);  -> float bottom_y = (int)((y*stride + cellSize +1 ) / scale);

This will make a big deviation.

This project will solve the issue above and make the detect close to python version, thus can deploy the trained model with c++ mxnet.
