import cv2
import numpy as np

if __name__ == "__main__":
    np.random.seed(0)
    color = np.random.randint(0, 255, size=[150, 3])

    imgpath = 'labelme_xthqmdozlpcfeah.jpg'
    frame = cv2.imread(imgpath)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]


    # 加载模型
    net = cv2.dnn.readNet("frozen_inference_graph_opt.pb")
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (513, 513), (127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    score = net.forward()
    numClasses = score.shape[1]
    height = score.shape[2]
    width = score.shape[3]

    classIds = np.argmax(score[0], axis=0)  # 在列上求最大的值的索引
    segm = np.stack([color[idx] for idx in classIds.flatten()])
    segm = segm.reshape(height, width, 3)

    segm = cv2.resize(segm, (frameWidth, frameHeight), interpolation=cv2.INTER_NEAREST)
    frame = (0.3*frame + 0.8*segm).astype(np.uint8)

    #showLegend(classes)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()