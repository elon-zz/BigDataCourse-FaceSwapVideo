
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import os

class RealtimeFaceSwapper:
    def __init__(self, target_image_path=None):
        """初始化人脸交换器"""
        # 初始化FaceAnalysis应用
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # 加载换脸模型
        model_path = 'models/inswapper_128.onnx'
        if not os.path.exists(model_path):
            print(f"请下载换脸模型并放置在: {os.path.abspath(model_path)}")
            print("下载地址: https://huggingface.co/datasets/Gourieff/ReActor/tree/main/models")
            exit(1)
        self.swapper = insightface.model_zoo.get_model(model_path)

        # 目标人脸相关变量
        self.source_face = None
        self.source_img = None
        if target_image_path:
            self.load_source_face(target_image_path)

    def load_source_face(self, image_path):
        """加载源人脸图像"""
        if not os.path.exists(image_path):
            print(f"源图像不存在: {image_path}")
            return False

        self.source_img = cv2.imread(image_path)
        if self.source_img is None:
            print(f"无法读取源图像: {image_path}")
            return False

        # 检测人脸
        source_faces = self.app.get(self.source_img)
        if len(source_faces) == 0:
            print("源图像中未检测到人脸")
            return False

        # 使用第一个检测到的人脸
        self.source_face = source_faces[0]
        print("源人脸加载成功")
        return True

    def process_frame(self, frame):
        """处理单个视频帧"""
        if self.source_face is None:
            return frame

        # 检测当前帧中的人脸
        target_faces = self.app.get(frame)
        if len(target_faces) == 0:
            return frame

        result = frame.copy()
        for target_face in target_faces:
            # 执行人脸交换
            result = self.swapper.get(result, target_face, self.source_face, paste_back=True)

        return result

    def start_webcam(self, camera_id=0):
        """启动实时换脸处理"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        # 设置缓冲区大小
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("按 'q' 退出")
        print("按 's' 保存当前帧")

        frame_count = 0
        start_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # 调整帧大小以提高性能
            frame = cv2.resize(frame, (640, 480))

            # 处理帧
            result_frame = self.process_frame(frame)

            # 显示FPS
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow("Realtime Face Swap", result_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                save_path = f"output_photos/face_swap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_path, result_frame)
                print(f"已保存: {save_path}")

        cap.release()
        cv2.destroyAllWindows()

def main():
    # 设置源人脸图像路径
    source_image = "face_photos/cxk.jpeg"

    # 检查默认图像是否存在
    if not os.path.exists(source_image):
        print(f"默认图像不存在: {source_image}")
        print("请提供一张包含清晰人脸的图像")
        return

    # 创建实时人脸交换器实例
    face_swapper = RealtimeFaceSwapper(source_image)
    
    # 启动网络摄像头
    face_swapper.start_webcam()

if __name__ == "__main__":
    main()