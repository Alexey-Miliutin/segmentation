import os
import cv2
import argparse
import numpy as np
import torch

from models.unet import UNet


def select_device(device: str = ''):

    if device.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            if device != '':
                device = torch.device('cuda:' + device)
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    return device


def preprocess_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    # Normalize image
    image = image / 255.
    
    # HWC to BCHW
    image = np.moveaxis(image, -1, 0)
    image = np.expand_dims(image, 0)
    # Move to device
    image = torch.from_numpy(image).to(device, non_blocking = True).float()
    return image


def postrocess_image(image: torch.Tensor, threshold = 0.5) -> np.ndarray:
    # Sigmoid 
    image = torch.nn.Sigmoid()(image[0])
    # Treshold 
    # image = (image > threshold).float()
    # Move to CPU
    image = image.data.cpu().numpy()
    # Normalize
    image -= image.min()
    image /= image.max()
    image *= 255.
    # Reshape image
    image = np.transpose(image.astype(np.uint8), (1,2,0))
    return image


def visualize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:

    mask = cv2.threshold(mask, 140, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.merge((np.zeros_like(mask), np.zeros_like(mask), mask))
    alpha = mask.astype(float)/255
    
    image = image.astype(float)
    mask = mask.astype(float)

    image = cv2.multiply(1.0 - alpha, image)
    mask = cv2.multiply(alpha, mask)
    
    merged = cv2.add(mask, image)
    return merged

    # Calculate defect height
    check_defect(merged)


def find_number_of_clusters(mask, image):
    thresh_count = 0
    # Make 4 parts and check every
    for half in range(4):
        half_mask = mask[:, half*128:(half+1) * 128]

        # Threshhold to binary
        thresh = cv2.threshold(half_mask, 100, 255, cv2.THRESH_BINARY)[1]
        erosion_kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(thresh,erosion_kernel, iterations = 1)

        contours = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        index = 1
        isolated_count = 0
        cluster_count = 0
        for cntr in contours:
            area = cv2.contourArea(cntr)
            convex_hull = cv2.convexHull(cntr)
            convex_hull_area = cv2.contourArea(convex_hull)

            if(convex_hull_area == 0):
                convex_hull_area = 1
            if(area == 0):
                area = 1

            # Find a box around the area and compare to area
            ratio = area / convex_hull_area
            # If area less then box it is not a noize
            if ratio < 0.90:
                cluster_count = cluster_count + 1
            # And if area close to box area it mostly noize
            else:
                isolated_count = isolated_count + 1

            index = index + 1

        # If there is one bottom increase thresh_counter
        if(cluster_count > 1):
            thresh_count += 1
        # If there is more then one (mostly 2) decrease thresh_counter
        else:
            thresh_count -= 1

    return 1 if thresh_count <= 0 else 2


@torch.no_grad()
def inference(opt: dict) -> None:
    # Init device
    device = select_device(opt.device)
    # Init model class
    model = UNet(opt.inp_ch, opt.out_ch)
    # Move model to device(cuda if available)
    model.to(device)
    # Load weights
    path = os.path.join(os.getcwd(), opt.weights)
    print("Weights took from: ", path)
    # Load weights
    if os.path.exists(path):
        model.load_state_dict(torch.load(path)['model'])
    # Enable eval mode
    model.eval()  
    # Make image list
    input_files = os.listdir(os.path.join(os.getcwd(), opt.input_path))
    
    if (opt.way == 'photo'):
    # Run inference
        for file in input_files:
            image = cv2.imread(os.path.join(os.getcwd(), opt.input_path, file))
            image = cv2.resize(image, (512, 512))
            pimage = preprocess_image(image, device)
            out = model(pimage)
            mask = postrocess_image(out)

            number_of_clusters = find_number_of_clusters(mask, image)

            print("====================")
            print("File: ", file)
            print("Number of bottoms: ", 1)
            cv2.putText(image, str(1), (100,450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
            cv2.imwrite(os.path.join(os.getcwd(), "mask", file + "_Clust_" + str(number_of_clusters),), image)

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common
    parser.add_argument('--way', default='photo', help='photo/video')
    parser.add_argument('--weights', default='./weights/best.pt', help='Path to weights')
    parser.add_argument('--device', type=str, default='', help='cuda device id or cpu')
    # Model parameters
    parser.add_argument('--inp_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)
    # Paths to files
    parser.add_argument('--input_path', type=str, default='val')
    parser.add_argument('--out_path', type=str, default='out')

    opt = parser.parse_args()
    print(opt)

    # run inference  
    inference(opt)
