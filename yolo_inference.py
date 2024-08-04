from ultralytics import YOLO 

model = YOLO('models/best.pt')

results = model.predict('input_videos/Test_Video.mp4.mp4',save=True) 
#results = model.predict('input_videos/08fd33_4.mp4',save=True) # Changed video file path
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)