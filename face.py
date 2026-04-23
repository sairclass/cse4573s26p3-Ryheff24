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
    bb = face_recognition.face_locations(img.permute(1, 2, 0).contiguous().numpy(), model="hog", number_of_times_to_upsample=2)
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
    cluster_type = "KMEANS" # KMEANS | SHAC
    torch.manual_seed(1)
    if imgs is None or len(imgs) == 0:
        return cluster_results
    
    if cluster_type == "KMEANS":
        embeddings, keys = get_embeddings(imgs)
        cluster_results = kmeans(embeddings, K, keys, cluster_results)
        
    elif cluster_type == "SHAC":
        embeddings, keys = get_embeddings(imgs)
        cluster_results = SHAC(embeddings, K, keys)

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
        bb = face_recognition.face_locations(img.permute(1, 2, 0).contiguous().numpy(), model="hog")
        if len(bb) == 0:
            continue

        embedding = face_recognition.face_encodings(img.permute(1, 2, 0).contiguous().numpy(), known_face_locations=bb)[0]
        embeddings[x] = torch.from_numpy(embedding)
        vkeys.append(keys[i])
        x += 1
    embeddings = embeddings[:x]
    return embeddings, vkeys

def kmeans(embeddings, K, keys, cluster_results, max_iters=100):
    #https://en.wikipedia.org/wiki/K-means_clustering
    centroids = kmpp(embeddings, K)
    prev = None
    # print(K, centroids.shape) 
    # print(torch.cdist(centroids, centroids))
    for _ in range(max_iters):

        dists = torch.cdist(embeddings, centroids) # just a dist matrix
        # Machine Learning: A Probabilistic Perspective Textbook by Kevin P. Murphy
        # Assign each data point to its closest cluster center: z_i = argmin_k||xi−μk||^2_2;
        # print(dists)
        # print(dists.argmin(dim=1))
        closest = dists.argmin(dim=1)
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
        
        
        # NORMAL KMEANS
        # print(F"K: {K},\nCentroids shape: {centroids.shape},\nDists: {dists},\nDists shape: {dists.shape},\nEmbeddings shape: {embeddings.shape}, \nClosest Values: {closest} \nClosest shape: {closest.shape}")
        # the values inside of closest can be used as a row index inside of dists 
        # print(closest, torch.max(closest))
        # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_
        # clusters = [[] for _ in range(K)]
        # return []
        # for i in range(embeddings.shape[0]):
        #     point = embeddings[i]
        #     closestIndex = 0
        #     # print(centroids[0].shape, point.shape)
        #     minDistance = torch.dist(embeddings[i], centroids[0]) # 
        #     for j in range(1, K):
        #         d = torch.dist(point, centroids[j])
        #         if d < minDistance:
        #             minDistance = d
        #             closestIndex = j
        #     clusters[closestIndex].append(i)
        # newCentroids = torch.mean
        
        # # centroid update 
        # for i in range(K):
        #     newCentroid = embeddings[clusters[i]].mean(dim=0) if len(clusters[i]) > 0 else centroids[i]
        #     newCentroids[i] = newCentroid
        # if all(torch.allclose(newCentroids[i], centroids[i]) for i in range(K)):
        #     converged = True
        # else:
        #     centroids = newCentroids
    for i, c in enumerate(closest):
        cluster_results[c].append(keys[i])
    return cluster_results

def bandaidfix(embeddings, K):
    # run kmeans++ multiple times on different seeds
    for i in range(50):
        pass
        

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

def nonVectkmpp(embeddings, K):
    centroids = []
    firstidx = torch.randint(0, embeddings.shape[0], (1,)).item()
    centroids.append(embeddings[firstidx])
    while len(centroids) < K:
        d_2 = []
        for i in range(embeddings.shape[0]):
            point = embeddings[i]
            minDist = torch.dist(point, centroids[0]) ** 2 # vectorize the dist 
            for j in range(1, len(centroids)):
                d = torch.dist(point, centroids[j]) ** 2
                if d < minDist:
                    minDist = d
            d_2.append(minDist)
        total = sum(d_2)
        threshold = torch.rand(1).item() * total
        cumulative = 0
        for i in range(len(embeddings)):
            cumulative += d_2[i]
            if cumulative >= threshold:
                centroids.append(embeddings[i])
                break
    return centroids

def SHAC(embeddings, K, keys):
    N = embeddings.shape[0]
    cluster_results: List[List[str]] = [[] for _ in range(K)] # temp while algo is done
    
    C = torch.cdist(embeddings, embeddings) 
    I = torch.ones((N,), dtype=torch.float32)

            
    # print(C, C.dtype, C.shape)
    A = []
    for k in range(N - K):
        i, m = torch.randint(0, N, (1,)).item(), torch.randint(0, N, (1,)).item() # placeholder
        # (i,m): i != m AND I[i]=1 AND I[m]=1
        # print(I)
        # t = torch.where(I==1, 1, 0)
        # print(t, t.sum())
        
        # return cluster_results
        A.append((i,m))
        for j in range(N):
            # textbook:
            # "choosing the cluster pair whose merge has the smallest diameter"
            # minimuze it then for complete link
            sim = torch.max(
                torch.cdist(embeddings[i].unsqueeze(0), 
                            embeddings[j].unsqueeze(0)), 
                torch.cdist(embeddings[m].unsqueeze(0), 
                            embeddings[j].unsqueeze(0))
                )
            
            C[i, j] = sim
            C[j, i] = sim
        I[m] = 0
    return cluster_results