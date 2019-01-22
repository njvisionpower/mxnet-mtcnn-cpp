# MTCNN-mxnet-cpp
--------------------
This project aims to implement face detection algorithm with mxnet c++ version. You can deploy the face detect part if use insightface(https://github.com/deepinsight/insightface) or other face recognition pipeline.

## Why start this project?
There are many bugs or implement error for original mtcnn cpp version (such as https://github.com/deepinsight/mxnet-mtcnn), I list some typical issue:

### 1.Memory leak
Forget to free pNet memory, especially pNet will run many times for different scale.
    
### 2.Computing overhead optimization
Frequently load pNet as the shape of different scale original input image, loading params is a very expensive operation in mxnet. Two strategy for this:  
(1)If your image shape is fixed, you can loading k(k is the number of scales) model for different scale, pnet is very small so will    not occupy too many memory. Aslo this step can be easyly implement with multi-thread.

(2)If image shape may change, you can set a relatively larger PredictorHandle for different scale.
    
Anyway, mxnet C/C++ api can't automatically 

### 3.Numerical calculation
#### (1)One case is round function get different result:
In c++:  
    `round(11.5) = 12, round(12.5) = 13`        
but if use python with numpy:  
    `np.round(11.5) = 12, np.round(12.5) = 12`  
that is numpy consider the number is even or odd before "5".
#### (2)Forget to +1 when calculate the bottom position of face bounding box:  
    float bottom_x = (int)((x*stride + cellSize) / scale);  -> float bottom_x = (int)((x*stride + cellSize +1 ) / scale); 
    float bottom_y = (int)((y*stride + cellSize) / scale);  -> float bottom_y = (int)((y*stride + cellSize +1 ) / scale);

This will make a big deviation.

This project will make a clean environment when deploy the trained model with c++ mxnet.
