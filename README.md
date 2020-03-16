##NMS with python

parameter boxes: list [[y0, x0, y1, x1],...] or [[x0, y0, x1, y1],...]

parameter scores: list [score1,score2,...]

parameter thresholdScore: threshold of box score

parameter thresholdIoU: threshold of box IoU

parameter labels: list [label1,label2,...]

return:

boxes_nms: list box after nms

scores_nms: list score after nms

labels_nms: list labels after nms