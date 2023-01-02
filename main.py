from utils import utils
import canny_edge_detector as ced

# load images
imgs = utils.load_data()
# utils.visualize(imgs, 'gray')

# filtering images with canny edge detector
detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5,
                                 lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
# the final result
imgs_final = detector.detect()

# show result
utils.visualize(imgs_final, 'gray')