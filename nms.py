#!/usr/bin/env python
# encoding: utf-8
import numpy as np

class NMS:

    def __init__(self, boxes, scores, thresholdScore=0.5, thresholdIoU=0.5):
        """
        prarmeter boxes:list [[y0, x0, y1, x1],...] or [[x0, y0, x1, y1],...]
        prarmeter scores:list [score1,score2,...]
        """
        self.boxes = np.array(boxes)
        self.scores = np.array(scores)
        self.thresholdScore = thresholdScore
        self.thresholdIoU = thresholdIoU
        self.filter_boxes()
        
    def filter_boxes(self):
        # filter boxes score lower then thresholdScore
        index = np.where(self.scores>self.thresholdScore)
        self.scores = self.scores[index]
        self.boxes = self.boxes[index]

    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        param rec2: (y0, x0, y1, x1)
        return: scala value of IoU
        addition: Infact it's no influence to the result if useing (x0, y0, x1, y1)
                  Beacuse this equate to that you exchange X_axis and Y_axis
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1]) 
        # computing the sum_area
        sum_area = S_rec1 + S_rec2 
        # find each edge of the intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0

    def run(self):
        # delete the repreat box use NMS algorithm
        result_box = []
        result_score = []
        print(self.scores)
        print(self.boxes)
        while len(self.scores):
            # get max score index
            index = np.argmax(self.scores)
            box_ = self.boxes[index]
            result_score.append(self.scores[index])
            result_box.append(box_)
            self.scores = np.delete(self.scores, index)         # delete score from source scores
            self.boxes = np.delete(self.boxes, index, axis=0)   # delete box from source boxes
            index_del = [] # record the box to be deleted
            for i,box in enumerate(self.boxes):
                IoU  = self.compute_iou(box, box_)
                if IoU > self.thresholdIoU:
                    index_del.append(i) 
            self.scores = np.delete(self.scores, index_del)
            self.boxes = np.delete(self.boxes, index_del, axis=0)
        return result_box,result_score

if __name__=='__main__':
    boxes = np.array([[0,0,10,10],[1,1,10,10],[3,3,5,5]])
    scores = np.array([0.6,0.7,0.3])
    nms = NMS(boxes, scores, thresholdScore=0.5, thresholdIoU=0.5)
    box_nms,score_nms = nms.run()
    print(box_nms)
    print(score_nms)


        