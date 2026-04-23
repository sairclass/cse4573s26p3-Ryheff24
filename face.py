'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


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
    bb = face_recognition.face_locations(img.permute(1, 2, 0).contiguous().numpy(), model="hog", number_of_times_to_upsample=3)
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
    # torch.manual_seed(1)
    if imgs is None or len(imgs) == 0:
        return cluster_results
    
    iters = 100
    runs = 100
    cluster_results = runKmeans(imgs, K, iters, runs)
        
    return cluster_results if len(cluster_results) > 0 else [[] for _ in range(K)]

'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
def get_embeddings(imgs):
    keys = list(imgs.keys())
    vkeys = []
    embeddings = torch.zeros(len(imgs), 128) 
    x = 0
    for i in range(len(imgs)):
        img = imgs[keys[i]]
        bb = face_recognition.face_locations(img.permute(1, 2, 0).contiguous().numpy(), model="hog", number_of_times_to_upsample=2)
        if len(bb) == 0:
            continue

        embedding = face_recognition.face_encodings(img.permute(1, 2, 0).contiguous().numpy(), known_face_locations=bb)[0]
        embeddings[x] = torch.from_numpy(embedding)
        vkeys.append(keys[i])
        x += 1
    embeddings = embeddings[:x]
    return embeddings, vkeys

def runKmeans(imgs, K, max_iters=100, runs=100):
    ret: List[List[str]] = [[] for _ in range(K)]
    embeddings, keys = get_embeddings(imgs)
    # Introduction to information retrieval Textbook by Christopher D. Manning, Hinrich Schütze, and Prabhakar Raghavan:
    # trying out multiple starting points and choosing the cluster-ing with lowest cost, section 16.4: K-means, page 364

    bestdistortion = float('inf')
    best = None
    for _ in range(runs):
        closest, distortion = kmeans(embeddings, K, max_iters)
        if distortion < bestdistortion:
            bestdistortion = distortion
            best = closest
    for i, c in enumerate(best):
        ret[c].append(keys[i])
    return ret

def kmeans(embeddings, K, max_iters=100):
    # https://en.wikipedia.org/wiki/K-means_clustering
    
    centroids = kmpp(embeddings, K)
    prev = None
    # print(K, centroids.shape) 
    # print(torch.cdist(centroids, centroids))
    for _ in range(max_iters):
        dists = torch.cdist(embeddings, centroids) # just a dist matrix
        # Machine Learning: A Probabilistic Perspective Textbook by Kevin P. Murphy:
        # Assign each data point to its closest cluster center: z_i = argmin_k||xi−μk||^2_2;
        # section 11.4.2.6 Vector quantization, cost funciton/distortion
        # print(dists)
        # print(dists.argmin(dim=1))
        closest = dists.argmin(dim=1)
        mindists, _ = dists.min(dim=1) 
        distortion = (mindists**2).sum() # equation 11.38 ML textbook, page 354
        if prev is not None and torch.equal(closest, prev):
            break
        newCentroids = torch.zeros(K, embeddings.shape[1], dtype=embeddings.dtype)
        idx = closest.unsqueeze(1).expand_as(embeddings)
        newCentroids.scatter_reduce_(0, idx, embeddings, "mean", include_self=False)
        counts = torch.bincount(closest, minlength=K)
        none = counts == 0
        if none.any():
            newCentroids[none] = centroids[none]
        centroids = newCentroids
        prev = closest
        # # print(F"K: {K},\nCentroids shape: {centroids.shape},\nDists: {dists},\nDists shape: {dists.shape},\nEmbeddings shape: {embeddings.shape}, \nClosest Values: {closest} \nClosest shape: {closest.shape}")
        # # the values inside of closest can be used as a row index inside of dists 
        # # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_
    return closest, distortion

def kmpp(embeddings, K):
    # https://en.wikipedia.org/wiki/K-means%2B%2B
    # embeddings = torch.nn.functional.normalize(embeddings, dim=1) if norm required, face_Rec pre norms 
    centroids = torch.empty(K, embeddings.shape[1], dtype=embeddings.dtype)
    firstidx = torch.randint(0, embeddings.shape[0], (1,)).item()
    centroids[0] = embeddings[firstidx]
    c_d_2 = ((embeddings - centroids[0])** 2).sum(dim=1)
    for i in range(1, K):
        probs = c_d_2 / c_d_2.sum()
        nextidx = torch.multinomial(probs, 1).item()
        centroids[i] = embeddings[nextidx]
        n_d_2 = ((embeddings - centroids[i])** 2).sum(dim=1)
        c_d_2 = torch.minimum(c_d_2, n_d_2)
    
    return centroids
