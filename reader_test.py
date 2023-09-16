import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageEnhance
import flow_library.flow_IO as flow_IO

class Augmentor:
    def __init__(
        self,
        image_height=384,
        image_width=512,
        max_disp=256,
        scale_min=0.6,
        scale_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def __call__(self, left_img, right_img, left_disp):
        # 1. chromatic augmentation
        left_img = self.chromatic_augmentation(left_img)
        right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        # 2.2) random resize
        resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

        left_img = cv2.resize(
            left_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        right_img = cv2.resize(
            right_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        disp_mask = disp_mask.astype("float32")
        disp_mask = cv2.resize(
            disp_mask,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        left_disp = (
            cv2.resize(
                left_disp,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            * resize_scale
        )

        # 2.3) random crop
        h, w, c = left_img.shape
        dx = w - self.image_width
        dy = h - self.image_height
        dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
        dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        left_img = cv2.warpAffine(
            left_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        right_img = cv2.warpAffine(
            right_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        left_disp = cv2.warpAffine(
            left_disp,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        disp_mask = cv2.warpAffine(
            disp_mask,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        # 3. add random occlusion to right image
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        return left_img, right_img, left_disp, disp_mask

class CREStereoDataset():
    def __init__(self, root):
        super().__init__()
        self.imgs = glob.glob(os.path.join(root, "**/*_left.jpg"), recursive=True)
        self.augmentor = Augmentor(
            image_height=384,
            image_width=512,
            max_disp=256,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)

    def get_disp(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return disp.astype(np.float32) / 32

    def get_item(self, index):
        # find path
        left_path = self.imgs[index]
        prefix = left_path[: left_path.rfind("_")]
        right_path = prefix + "_right.jpg"
        left_disp_path = prefix + "_left.disp.png"
        right_disp_path = prefix + "_right.disp.png"

        # read img, disp
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        left_disp = self.get_disp(left_disp_path)
        right_disp = self.get_disp(right_disp_path)

        if self.rng.binomial(1, 0.5):
            left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
            left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
        left_disp[left_disp == np.inf] = 0

        # augmentaion
        left_img, right_img, left_disp, disp_mask = self.augmentor(
            left_img, right_img, left_disp
        )

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        return {
            "left": left_img,
            "right": right_img,
            "disparity": left_disp,
            "mask": disp_mask,
        }

    def __len__(self):
        return len(self.imgs)

class SpringStereoDataset():
    def __init__(self, root, split='train', subsample_groundtruth=True):
        super(SpringStereoDataset, self).__init__()

        self.augmentor = Augmentor(
            image_height=540,
            image_width=960,
            max_disp=256,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )

        assert split in ["train", "test"]
        seq_root = os.path.join(root, split)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []

        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob.glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                for frame in range(1, len(images)+1):
                    self.data_list.append((frame, scene, cam))
        
    def get_item(self, index):
        frame_data = self.data_list[index]
        frame, scene, cam = frame_data

        img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")
        if cam == "left":
            othercam = "right"
        else:
            othercam = "left"
        img2_path = os.path.join(self.seq_root, scene, f"frame_{othercam}", f"frame_{othercam}_{frame:04d}.png")
        
        #img1 = np.asarray(Image.open(img1_path)) / 255.0
        #img2 = np.asarray(Image.open(img2_path)) / 255.0
        
        disp_path = os.path.join(self.seq_root, scene, f"disp1_{cam}", f"disp1_{cam}_{frame:04d}.dsp5")
        disp = flow_IO.readDispFile(disp_path)
        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> disp will have same dimensions as images
            disp = disp[::2,::2]

        #if cam == "left":
            # left-to-right disparity should be negative, which is not encoded in the data
          #  disp *= -1

        disp = disp.astype(np.float32)

        left_img = np.asarray(Image.open(img1_path))
        right_img = np.asarray(Image.open(img2_path))

        left_img, right_img, disp, frame_data = self.augmentor(left_img, right_img, disp)

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        if self.split == "test":
            return left_img, right_img, frame_data
        
        return left_img, right_img, disp, frame_data

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    dataset_path = "F:\\CREStereo Dataset\\stereo_trainset"
    dataset = CREStereoDataset(dataset_path)
    left1, right1 ,disp1, mask1 = dataset.get_item(0)
    
    dataset2_path = "F:\\Datasets\\spring\\spring"
    dataset2 = SpringStereoDataset(dataset2_path)
    left2, right2 ,disp2, mask2 = dataset2.get_item(0)

    disp2 = disp2 *5
    disp2 = disp2.astype(np.uint8)
    disp2 = Image.fromarray(disp2)
    disp2.save("disp2.png")

    disp1 = disp1 *5
    disp1 = disp1.astype(np.uint8)
    disp1 = Image.fromarray(disp1)
    disp1.save("disp1.png")
    
    print("placeholder")