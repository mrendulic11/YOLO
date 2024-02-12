from ultralytics import YOLO
import cv2
import numpy as np


class Colors:
    def __init__(self, num_colors=80):
        self.num_colors = num_colors
        self.color_palette = self.generate_color_palette()

    def generate_color_palette(self):
        hsv_palette = np.zeros((self.num_colors, 1, 3), dtype=np.uint8)
        hsv_palette[:, 0, 0] = np.linspace(0, 180, self.num_colors, endpoint=False)
        hsv_palette[:, :, 1:] = 255
        bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)
        return bgr_palette.reshape(-1, 3)

    def __call__(self, class_id):
        color = tuple(map(int, self.color_palette[class_id]))
        return color
    
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
orange = (0, 165, 255)
purple = (155, 102, 178)
white = (255,255,255)
black = (0,0,0)

def draw_tracks(frame, tracks):
    current_count = 0
    for track in tracks:   
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
        conf = track[5]
        class_id = int(track[6])
        class_name = classes[class_id]
        id = int(track[4])

        relevant_point = (x1+x2)//2, y2

        if is_in_main_area(main_area, relevant_point):
            current_count += 1
            position = is_in_subarea(sub_areas, relevant_point)
            if position != None:
                frame = draw_relevant_point(frame, relevant_point)
                update_dict(tracker_dict, position, id)
                if tracker_dict[id]["direction"] != None:
                    label = f'{tracker_dict[id]["direction"]}'
                else:
                    label = f'ID={id}'
                #drawing
                #cv2.rectangle(frame, (x1,y1), (x2, y2), colors(class_id), 2)
                #label_direction = f'{tracker_dict[id]["dir"]}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                #cv2.rectangle(frame, (x1, y1-h-15), (x1+w, y1), colors(class_id), -1)
                cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red , 1)

    current_label = f"Current: {current_count}"
    cv2.putText(frame, current_label, (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 1)
    return frame

# additional draws
def draw_relevant_point(frame, relevant_point):
    cv2.circle(frame, (relevant_point), 5, (255,0,0), 2)
    return frame

def draw_main_area(frame, main_area, show):
    if show:
        cv2.polylines(frame, [main_area], isClosed=True, color=blue, thickness=2)
    return frame

def draw_sub_areas(frame, sub_areas, show):
    if show:
        for area in sub_areas.values():
            cv2.polylines(frame, [area], isClosed=True, color=yellow, thickness=1)
    return frame

# check if is in the main_area
def is_in_main_area(main_area, relevant_point):
    point_inside = cv2.pointPolygonTest(main_area, relevant_point, False)
    if point_inside >= 0:
        return True
    else:
        return False

def is_in_subarea(sub_areas, relevant_point):
    for position, area in sub_areas.items():
        # print(f'Position {position}')
        point_inside = cv2.pointPolygonTest(area, relevant_point, False)
        if point_inside >= 0:
            return position
    return None

    
# Update tracker dictionary
def update_dict(tracker_dict, position, id):
    if id not in tracker_dict:
        # Add a new entry for the ID
        tracker_dict[id] = {
            "s_position": position,
            "f_position": None,
            "direction": None
        }
    else:
        tracker_dict[id]["f_position"] = position
        tracker_dict[id]["direction"] = tracker_dict[id]["f_position"] if tracker_dict[id]["f_position"] != tracker_dict[id]["s_position"] else None
    
def check_and_count_dict(frame, tracker_dict):
    total_counts = len(tracker_dict)
    top_count = 0
    bottom_count = 0

    for track in tracker_dict.values():
        if track['direction'] == "top":
            top_count+=1
        if track['direction'] == "bottom":
            bottom_count+=1

    total_label = f"Total: {total_counts}"
    top_label = f"Top: {top_count}"
    bottom_label = f"Bottom: {bottom_count}"

    cv2.putText(frame, total_label, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 1)
    cv2.putText(frame, top_label, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 1)
    cv2.putText(frame, bottom_label, (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 1)

    return frame

# Load the video
video_source = r"C:\Users\Luka\Desktop\test videos\pedestrians\split_riva_day_2mins.mp4"
cap = cv2.VideoCapture(video_source)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(r'polaznici\Marijana\test\riva_day_2mins.avi', fourcc, fps, (width, height))

model = YOLO(r"weights\crowd_human\person_only_X.pt")
classes = model.names
colors = Colors(len(classes))

# Define the main and sub area
main_area = np.array([(0, 420), (600, 290), (780, 350), (450,720), (0, 720)], dtype=np.int32)
sub_areas = {"top": np.array([(250, 375), (600, 290), (780, 350), (520,510)]), 
             "bottom": np.array([(0, 420), (250, 375), (520,510), (450,720), (0, 720)])}

# Define the tracker dictionary
tracker_dict = {}

# Loop through the frames of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Draw the main_area on frame
        cv2.rectangle(frame, (0, 0), (200,150), white, -1)
        frame = draw_main_area(frame, main_area, show=True)
        frame = draw_sub_areas(frame, sub_areas, show=True)

        # Process the frame here
        tracks = model.track(frame, persist=True, stream=True, verbose=False)
        for track in tracks:
            track = track.boxes.data.cpu().numpy()
            if not(track.shape[1]!=7):
                frame = draw_tracks(frame, track)
        frame = check_and_count_dict(frame, tracker_dict)
        out.write(frame)
        
        # Show the frame
        # cv2.imshow("video", frame)

        # Wait for the user to press a key (optional)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()