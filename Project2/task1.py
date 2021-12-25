"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random

ratioFilter = 0.07

#function will return the key points and features of the image
def getKeyPointsNFeatures(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    siftDescriptor = cv2.xfeatures2d.SIFT_create()
    (keyPts, features) = siftDescriptor.detectAndCompute(image, None)
    return (keyPts, features)


#function will return 4 random matches 
def getRandomMatches(goodMatches):
    randomMatches = []
    random.seed(4)
    n=len(goodMatches)
    vx1 = [random.randint(0,n-1) for i in range(4)]
    i=0
    while i<4:
        randomMatch = goodMatches[vx1[i]]
        if randomMatch not in randomMatches:
            randomMatches.append(randomMatch)
        i=i+1
    return randomMatches


#function will return the homography matrix for matches
def getHomographyMatrix(randomMatches, keyPtsLeft, keyPtsRight):
    A = []
    for match in randomMatches:
        x, y = keyPtsLeft[match["leftIndex"]].pt
        u, v = keyPtsRight[match["rightIndex"]].pt
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    H = np.ndarray.reshape(Vh[8], (3, 3))
    H = (1/H.item(8)) * H
    return H


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """
    #find the key points and features
    (keyPtsLeft, featuresLeft) = getKeyPointsNFeatures(left_img)
    (keyPtsRight, featuresRight) = getKeyPointsNFeatures(right_img)
    
    #store the good matches
    goodMatches = []
    for i in range(len(featuresLeft)):
        distances = []
        for j in range(len(featuresRight)):
            distanceInfo = {"leftIndex": i, 
                            "rightIndex": j,
                            "normDist": np.sum(np.square(np.subtract(featuresLeft[i],featuresRight[j])))}
            distances.append(distanceInfo)
        sortedDistances = []
        sortedDistances = sorted(distances, key = lambda x:x["normDist"])
        #best two matches of a feature
        firstMatch = sortedDistances[0]["normDist"]
        secondMatch = sortedDistances[1]["normDist"]
        #ratio check
        if firstMatch < ratioFilter*secondMatch:
            goodMatches.append(sortedDistances[0])
              
    
    #finding homography using RANSAC algorithm
    max_inliers = 0
    maxIterations = 5000
    homographyMatrix = []
    for i in range(maxIterations):
        randomMatches = getRandomMatches(goodMatches)
        tempHomography = getHomographyMatrix(randomMatches, keyPtsLeft, keyPtsRight)
        inliers = 0
        for match in goodMatches:
            #getting the coordinates of the key point
            x1,y1 = keyPtsLeft[match["leftIndex"]].pt
            x2,y2 = keyPtsRight[match["rightIndex"]].pt
            
            #using point1 of left image will try to find point2 using homography matrix
            point1 = np.matrix([[x1], [y1], [1]])
            point2 = np.matrix([[x2], [y2], [1]])
            estimatedPoint = np.dot(tempHomography, point1)
            if estimatedPoint[2] != 0:
                 estimatedPoint = estimatedPoint/estimatedPoint[2]
            
            #checking the difference between the estimatedPoint and point2
            difference = np.linalg.norm(estimatedPoint - point2)
            if difference < 1:
                inliers += 1
        
        if inliers > max_inliers:
            max_inliers = inliers
            homographyMatrix = tempHomography
        
        if max_inliers >= 4:
            break;
            
            
    #wrapping image using homography matrix
    left_height = left_img.shape[1]
    left_width = left_img.shape[0]
    right_height = right_img.shape[1]
    right_width = right_img.shape[0]
    
    right_frame = np.float32([[0, 0], 
                               [0, right_width], 
                               [right_height, right_width], 
                               [right_height, 0]]).reshape(-1, 1, 2)
    left_frame = np.float32([[0, 0], 
                               [0, left_width], 
                               [left_height, left_width], 
                               [left_height, 0]]).reshape(-1, 1, 2)
    right_transformed = cv2.perspectiveTransform(left_frame, homographyMatrix)
    
    final_image_frame = np.vstack((right_frame, right_transformed))
    [minx, miny] = np.int32(final_image_frame.min(axis=0).flatten())
    [maxx, maxy] = np.int32(final_image_frame.max(axis=0).flatten())
    
    translation_dist = [-minx, -miny]
    h_translation = np.array([[1, 0, translation_dist[0]], 
                              [0, 1, translation_dist[1]], 
                              [0, 0, 1]])
    product = h_translation.dot(homographyMatrix)
    wrapped_img = cv2.warpPerspective(left_img, 
                                      product, 
                                      (maxx - minx, maxy - miny))
    wrapped_img[translation_dist[1]:right_width + translation_dist[1], 
                translation_dist[0]:right_height + translation_dist[0]] = right_img
    return wrapped_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)