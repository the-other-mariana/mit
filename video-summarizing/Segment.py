import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt

def lookup_legend():
    # create an image of size (n classes * 25, 300, 3) 
    legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype='uint8')

    for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
        color = [int(c) for c in color]
        cv2.putText(legend, className, (5, (i*25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(legend, (100, (i*25)), (300, (i*25) + 25), tuple(color), -1)
    plt.imshow(legend)
    plt.savefig('legend.png', dpi=500)
    plt.show()


def init_info():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,
        help="path to deep learning segmentation model")
    ap.add_argument("-c", "--classes", required=True,
        help="path to .txt file containing class labels")
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-l", "--colors", type=str,
        help="path to .txt file containing colors for labels")
    ap.add_argument("-w", "--width", type=int, default=500,
        help="desired width (in pixels) of input image")
    args = vars(ap.parse_args())

    # load class label names
    global CLASSES
    CLASSES = open(args["classes"]).read().strip().split('\n')
    print(CLASSES, len(CLASSES))
    # if a colors file was supplied, load it
    global COLORS
    if args['colors']:
        COLORS = open(args["colors"]).read().strip().split('\n')
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype='uint8')
    else:
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3), dtype='uint8')
        # stack 0,0,0 in the first position of colors array
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype('uint8')
    print(COLORS, len(COLORS))
    lookup_legend()


def main():
    init_info()

if __name__ == '__main__':
    main()