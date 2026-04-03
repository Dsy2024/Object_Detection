from ultralytics import YOLO


def yolo_infer(model_path=None, img="images/65.jpg", save_name="0.jpg"):
    model = YOLO("models/yolo11n_detect_OX/weights/best.pt")
    # Load the model if a path is provided, otherwise use the pre-loaded model
    if model_path:
        model = YOLO(model_path)

    
    # results = model(img, save=True, show=True, project="outputs/", name="image_OX", exist_ok=True)
    results = model(img)

    # Save the results with bounding boxes drawn on the image
    for r in results:
        r.save(f"outputs/images/{save_name}")

    return results


def yolo_train():
    model = YOLO("yolo26l.pt")
    # Train the model on the dataset for 100 epochs
    model.train(data="config.yaml", epochs=100, imgsz=640, save=True, project="models/", name="yolo26l_detect", exist_ok=True) # The model output path will be models/yolo26l_detect/weights/best.pt

    # Run inference on a single image to verify the trained model
    results = model("data/images/val/101.jpg", save=True, show=True, project="outputs/", name="image", exist_ok=True) # The inference output path will be outputs/image/
    

if __name__ == "__main__":
    yolo_train()