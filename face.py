'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


from re import L

import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    # A list of tuples of found face locations in css (top, right, bottom, left) order
    bb = face_recognition.face_locations(img.permute(1, 2, 0).contiguous().numpy(), model="hog")
    # print(bb)
    for top, right, bottom, left in bb:
        detection_results.append([float(left), float(top), float(right - left), float(bottom - top)])
    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    #https://en.wikipedia.org/wiki/K-means_clustering
    # https://en.wikipedia.org/wiki/K-means%2B%2B
    centroids = []
    firstidk = torch.randint(0, len(imgs), (1,))
    start = imgs[list(imgs.keys())[firstidk]]
    # print(start)
    # return []
    centroids.append(imgs[list(imgs.keys())[firstidk]])
    converged = False
    while not converged:
        clusters = [[] for _ in range(K)]
        for i in imgs:
            point = imgs[i]
            closestIndex = 0
            print(centroids[0].shape, point.shape)
            minDistance = torch.dist(point, centroids[0])
            for j in range(1, K):
                d = torch.dist(point, centroids[j])
                if d < minDistance:
                    minDistance = d
                    closestIndex = j
            clusters[closestIndex].append(i)
        newCentroids = []
        for i in range(K):
            newCentroid = torch.mean(torch.stack([imgs[j] for j in clusters[i]]), dim=0)
            newCentroids.append(newCentroid)
        if all(torch.equal(newCentroids[i], centroids[i]) for i in range(K)):
            converged = True
        else:
            centroids = newCentroids
    cluster_results = clusters  
    
            
    #https://en.wikipedia.org/wiki/K-means_clustering:
    #function kmeans(k, points) is
    # // Initialize centroids
    # centroids ← list of k starting centroids
    # converged ← false

    # while converged == false do
    #     // Create empty clusters
    #     clusters ← list of k empty lists

    #     // Assign each point to the nearest centroid
    #     for i ← 0 to length(points) - 1 do
    #         point ← points[i]
    #         closestIndex ← 0
    #         minDistance ← distance(point, centroids[0])
    #         for j ← 1 to k - 1 do
    #             d ← distance(point, centroids[j])
    #             if d < minDistance THEN
    #                 minDistance ← d
    #                 closestIndex ← j
    #         clusters[closestIndex].append(point)

    #     // Recalculate centroids as the mean of each cluster
    #     newCentroids ← empty list
    #     for i ← 0 to k - 1 do
    #         newCentroid ← calculateCentroid(clusters[i])
    #         newCentroids.append(newCentroid)

    #     // Check for convergence
    #     if newCentroids == centroids THEN
    #         converged ← true
    #     else
    #         centroids ← newCentroids

    # return clusters
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
