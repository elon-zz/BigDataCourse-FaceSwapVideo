import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def swap_faces(source_img_path, target_video_path, output_video_path):
    # 初始化FaceAnalysis应用
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 加载源图像并提取人脸
    source_img = cv2.imread(source_img_path)
    source_faces = app.get(source_img)
    if len(source_faces) == 0:
        print("未在源图像中检测到人脸!")
        return
    source_face = source_faces[0]

    # 打开目标视频
    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened():
        print("无法打开目标视频!")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 初始化人脸交换模型
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=False)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测目标视频帧中的人脸
        target_faces = app.get(frame)
        if len(target_faces) > 0:
            # 只处理第一个人脸
            target_face = target_faces[0]

            # 执行人脸交换
            result = swapper.get(frame, target_face, source_face, paste_back=True)
            frame = result

        # 写入输出视频
        out.write(frame)
        frame_count += 1
        print(f"已处理帧: {frame_count}", end='\r')

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n视频处理完成，已保存到: {output_video_path}")


if __name__ == "__main__":
    face_photo_name = "musk"
    # 使用示例
    source_image = f"face_photos\\{face_photo_name}.jpeg"  # 源人脸图像路径
    target_video = "input_videos\\test3.mp4"  # 目标视频路径
    output_video = f"output_videos\\output_{face_photo_name}.mp4"  # 输出视频路径

    swap_faces(source_image, target_video, output_video)