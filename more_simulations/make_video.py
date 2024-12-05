import cv2
import os


def images_to_video(location_name, wake, bed, start_number, end_number, folder_path, output_video, fps=30):
    frame_array = []
    img_size = None

    for i in range(start_number, end_number + 1):

        img_filename = f"{location_name}_{wake}_{bed}_{i:03d}.png"
        img_path = os.path.join(folder_path, img_filename)

        if os.path.isfile(img_path):
            img = cv2.imread(img_path)

            if img_size is None:
                img_size = (img.shape[1], img.shape[0])  # width, height

            img_resized = cv2.resize(img, img_size)
            frame_array.append(img_resized)
        else:
            print(f"Image {img_filename} not found. Skipping.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for .mp4 output
    out = cv2.VideoWriter(output_video, fourcc, fps, img_size)

    for frame in frame_array:
        out.write(frame)

    out.release()
    print("Video created successfully!")


location_name = "Miami"
wake = 6
bed = 22
images_to_video(
    location_name=location_name,
    wake=wake,
    bed=bed,
    start_number=33,
    end_number=241,
    folder_path="output/",
    output_video=f"{location_name}.mp4",
    fps=10
)
