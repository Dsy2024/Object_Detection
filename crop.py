import cv2
from detect import refine_crop


if __name__ == "__main__":
    for i in range(121, 146):
        img_path = f"temp/{i}.jpg"
        img = cv2.imread(img_path)
        cropped_img = refine_crop(img)
        cv2.imwrite(f"temp/cropped_{i}.jpg", cropped_img)