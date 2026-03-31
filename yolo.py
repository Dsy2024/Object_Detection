from ultralytics import YOLO

model = YOLO("models/yolo11n_detect_OX/weights/best.pt")

def yolo_infer(model_path=None, img="data/images/val/54.jpg", save_name="0.jpg"):
    # Load the model if a path is provided, otherwise use the pre-loaded model
    if model_path:
        model = YOLO(model_path)

    
    # results = model(img, save=True, show=True, project="outputs/", name="image_OX", exist_ok=True)
    results = model(img)

    # Save the results with bounding boxes drawn on the image
    for r in results:
        r.save(f"outputs/image_OX/{save_name}")

    return results


def yolo_train():
    model = YOLO("yolo11n.pt")
    # Train the model on the dataset for 100 epochs
    model.train(data="config.yaml", epochs=100, imgsz=640, project="models/", name="yolo11n_detect") # The model output path will be outputs/yolo11n_detect/weights/best.pt

    # Run inference on a single image to verify the trained model
    results = model("input_image_path", save=True, show=True, project="outputs/", name="image", exist_ok=True) # The inference output path will be outputs/image/
    

if __name__ == "__main__":
    yolo_train()