import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from config import config #lo cambie pq me tiraba error y puse el config dentro de kmeans para probar
import glob
import os
import logging

config_file = '../../config'
config_data = None

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def load_txt_files():

    LOGGER.info("Loading txt files ...")

    summary_file_path = config_data["paths"]["train"]["labels"] + '\summary.txt'

    if os.path.exists(summary_file_path):

        LOGGER.info("Removing existing summary file ...")

        try:
            os.remove(summary_file_path)
        except:
            LOGGER.error(f"Not able to remove file: {summary_file_path}")

    files = glob.glob(config_data["paths"]["train"]["labels"] + '\*.txt')

    for file in files:

        content = None

        with open(file, 'r') as f:
            content = f.read()
        
        with open(summary_file_path, 'a+') as summary_f:
            summary_f.write(content)

    LOGGER.info("Creating summary file ...")

   
def load_bboxes():

    LOGGER.info("Loading bounding boxes ...")

    bboxes_file_path = config_data['paths']['train']['labels'] + "\summary.txt"

    bboxes = torch.tensor(np.loadtxt(bboxes_file_path, dtype=float, delimiter=" ", ndmin=2, usecols=(1, 2, 3, 4))) * 640 # Take only x, y, w, h 

    # Convert from (x, y, w, h ) format, to (x1,y1,x2,y2 format)

    bboxes[:, 2] = (bboxes[:, 2] + bboxes[:, 0])
    bboxes[:, 3] = (bboxes[:, 3] + bboxes[:, 1])

    return bboxes

def IoU(clusters: torch.tensor, bboxes: torch.tensor):
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.

    Parameters:
        clusters (Tensor[9, 4]): first set of boxes
        bboxes (Tensor[M, 4]) : second set of boxes - M = Amount of bboxes read from file.

    Returns:
        Tensor[9, M]: the 9xM matrix containing the pairwise IoU values for every element in clusters and bboxes.
    """

    # Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.

    iou_values = box_iou(clusters, bboxes)
    return iou_values


def KMeans(bboxes:torch.tensor, k:int, dist=torch.mean, stop_iter=200):

    LOGGER.info("Running Kmeans ...")

    rows = bboxes.shape[0]
    distances = torch.empty((rows, k))
    last_clusters = torch.zeros((rows, ))

    cluster_indxs = np.random.choice(rows, k, replace=False) # choose unique indexs in rows
    clusters = bboxes[cluster_indxs].clone()

    iteration = 0

    while True:
        # calculate the distances

        distances = IoU(bboxes, clusters)

        # Take the one with higher IoU value.
        nearest_clusters = torch.argmax(distances, dim=1) # 0, 1, 2 ... K

        if (last_clusters == nearest_clusters).all(): # break if nothing changes
            LOGGER.info(f"Iteration number {iteration}")
            iteration += 1
            if iteration == stop_iter:
                break
        else:
            iteration = 0

        # Take the mean and step for cluster coordiantes
        for cluster in range(k):
            clusters[cluster] = torch.mean(bboxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters.clone()

    return clusters, distances

def plot(widths:np.array, heights:np.array):

    plt.scatter(widths, heights, c=[i for i in range(9)])
    plt.show()

    
if __name__ == "__main__":

    k = 9

    config_data = config()

    load_txt_files()

    bboxes = load_bboxes()

    # Run k-means
    clusters_centers, distances = KMeans(bboxes=bboxes, k=k, stop_iter=300)

    # Convert to relative widths and heights
    widths = abs(clusters_centers[:, 0] - clusters_centers[:, 2]) 
    heights = abs(clusters_centers[:, 1] - clusters_centers[:, 3]) 

    LOGGER.info(f"Widths: {np.sort(widths,axis=0)}")
    LOGGER.info(f"Heights: {np.sort(heights, axis=0)}")

    np.savetxt('ouput.txt', distances)

    plot(widths, heights)

