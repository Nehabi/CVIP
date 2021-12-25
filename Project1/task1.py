###############
##Design the function "findRotMat" to  return
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz
# It is ok to add other functions if you need
###############

import numpy as np
import cv2


def findRotMat(alpha, beta, gamma):
    # converting degree to radians
    alphaRadian = alpha * np.pi / 180
    betaRadian = beta * np.pi / 180
    gammaRadian = gamma * np.pi / 180

    # rotate around z axis with alpha i.e. 45
    z1Rotation = ([np.cos(alphaRadian), -np.sin(alphaRadian), 0],
                  [np.sin(alphaRadian), np.cos(alphaRadian), 0],
                  [0, 0, 1])
    # rotate around x axis with beta i.e. 30
    xRotation = ([1, 0, 0],
                 [0, np.cos(betaRadian), -np.sin(betaRadian)],
                 [0, np.sin(betaRadian), np.cos(betaRadian)])
    # rotate around x axis with gamma i.e. 60
    z2Rotation = ([np.cos(gammaRadian), -np.sin(gammaRadian), 0],
                  [np.sin(gammaRadian), np.cos(gammaRadian), 0],
                  [0, 0, 1])
    # rotation matrix 1
    rotMat1 = np.dot(np.dot(z2Rotation, xRotation), z1Rotation)

    # rotate around z axis with -alpha i.e. -45
    z1_Rotation = ([np.cos(gammaRadian), np.sin(gammaRadian), 0],
                   [-np.sin(gammaRadian), np.cos(gammaRadian), 0],
                   [0, 0, 1])
    # rotate around x axis with -beta i.e. -30
    x_Rotation = ([1, 0, 0],
                  [0, np.cos(betaRadian), np.sin(betaRadian)],
                  [0, -np.sin(betaRadian), np.cos(betaRadian)])
    # rotate around z axis with -gamma i.e. -60
    z2_Rotation = ([np.cos(alphaRadian), np.sin(alphaRadian), 0],
                   [-np.sin(alphaRadian), np.cos(alphaRadian), 0],
                   [0, 0, 1])
    # rotation matrix 2
    rotMat2 = np.dot(np.dot(z2_Rotation, x_Rotation), z1_Rotation)
    # return rotMat1, rotMat2
    return rotMat1, rotMat2


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)
