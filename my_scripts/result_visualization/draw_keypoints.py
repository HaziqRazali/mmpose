
import cv2
import json

import numpy as np

if __name__ == "__main__":
    
    result_filename    = "/home/haziq/mmpose/vis_results/results_000000196141.json"
    image_filename      = "/home/haziq/mmpose/tests/data/mpii/004645041.jpg"
    
    #################### load image
    
    image = cv2.imread(image_filename)
    
    #################### load and draw annotations
    
    result = json.load(open(result_filename,"r"))
    skeleton_links = result["meta_info"]["skeleton_links"]
    
    for i,instance in enumerate(result["instance_info"]):
        
        #if i != 1:
        #    continue
        
        keypoints = np.array(instance["keypoints"])             # [num_keypoints, 2]
        keypoint_scores = np.array(instance["keypoint_scores"]) # [num_keypoints]
        
        # draw keypoints and their scores
        for idx, (x, y) in enumerate(keypoints):
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
            
            score = keypoint_scores[idx]
            label = f"{score:.2f}"
            cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        for j1, j2 in skeleton_links:
            pt1 = tuple(map(int, keypoints[j1]))
            pt2 = tuple(map(int, keypoints[j2]))
            cv2.line(image, pt1, pt2, color=(0, 0, 255), thickness=2)

            
    #################### show or save the result
    cv2.imshow("Skeleton", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()