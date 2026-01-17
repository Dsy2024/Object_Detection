from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("data/5.jpg", save=True, show=True, project="outputs/", name="image")
# results = model("data/traffic-mini.mp4", save=True, show=True, project="outputs/", name="video")