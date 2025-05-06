import cv2
import numpy as np
import dlib
import face_recognition
import os
import time

PHOTO_PATH = "face_photos/musk.jpeg"

class FaceSwapperRealTime:
    def __init__(self, target_image_path=None):
        # 初始化人脸检测器和关键点预测器
        self.detector = dlib.get_frontal_face_detector()
        # 下载并加载面部关键点预测模型
        model_path = "models/shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            print(f"请下载面部关键点预测模型并放置在 {os.path.abspath(model_path)}")
            print("下载链接: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            exit(1)
        self.predictor = dlib.shape_predictor(model_path)
        
        # 加载目标图像（要替换到视频中的脸）
        self.target_face_img = None
        self.target_face_landmarks = None
        if target_image_path:
            self.load_target_face(target_image_path)
    
    def load_target_face(self, image_path):
        """加载目标人脸图像并提取关键点"""
        if not os.path.exists(image_path):
            print(f"目标图像不存在: {image_path}")
            return False
        
        self.target_face_img = cv2.imread(image_path)
        if self.target_face_img is None:
            print(f"无法读取目标图像: {image_path}")
            return False
        
        # 转换为RGB格式用于face_recognition库
        rgb_img = cv2.cvtColor(self.target_face_img, cv2.COLOR_BGR2RGB)
        
        # 检测人脸位置
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            print("目标图像中未检测到人脸")
            return False
        
        # 获取人脸关键点
        face_landmarks = face_recognition.face_landmarks(rgb_img, face_locations)
        if not face_landmarks:
            print("无法在目标图像中检测到人脸关键点")
            return False
        
        self.target_face_landmarks = face_landmarks[0]
        print("目标人脸已加载")
        return True
    
    def extract_landmarks(self, image, face_rect):
        """从图像中提取面部关键点"""
        shape = self.predictor(image, face_rect)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        return landmarks
    
    def get_face_mask(self, image, landmarks):
        """根据关键点创建面部蒙版"""
        mask = np.zeros_like(image)
        
        # 获取面部轮廓点
        face_points = []
        for group in ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 
                      'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']:
            if group in landmarks:
                face_points.extend(landmarks[group])
        
        # 创建凸包
        hull = cv2.convexHull(np.array(face_points))
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        
        # 添加模糊处理以平滑边缘
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def warp_image(self, source_img, source_landmarks, target_landmarks, target_shape):
        """使用Delaunay三角剖分进行图像变形"""
        source_img = source_img.copy()
        output_img = np.zeros(target_shape, dtype=np.uint8)
        
        # 将关键点转换为numpy数组 - 修复数组形状不一致的问题
        src_points = []
        for group in source_landmarks.values():
            src_points.extend(group)
        src_points = np.array(src_points, dtype=np.int32)
        
        dst_points = []
        for group in target_landmarks.values():
            dst_points.extend(group)
        dst_points = np.array(dst_points, dtype=np.int32)
        
        # 计算Delaunay三角剖分
        rect = (0, 0, target_shape[1], target_shape[0])
        subdiv = cv2.Subdiv2D(rect)
        
        # 确保所有点都在矩形范围内
        valid_points = []
        for point in dst_points:
            x, y = point
            if 0 <= x < target_shape[1] and 0 <= y < target_shape[0]:
                valid_points.append(point)
                subdiv.insert((int(x), int(y)))
        
        # 如果有效点太少，无法进行三角剖分，则返回空图像
        if len(valid_points) < 3:
            return output_img
            
        triangles = subdiv.getTriangleList()
        
        # 其余代码保持不变
        for triangle in triangles:
            x1, y1, x2, y2, x3, y3 = triangle
            
            # 找到目标图像中三角形顶点的索引
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            pt3 = (int(x3), int(y3))
            
            # 检查点是否在目标图像边界内
            if (0 <= x1 < target_shape[1] and 0 <= y1 < target_shape[0] and
                0 <= x2 < target_shape[1] and 0 <= y2 < target_shape[0] and
                0 <= x3 < target_shape[1] and 0 <= y3 < target_shape[0]):
                
                # 找到源图像中对应的三角形顶点
                idx1 = self.find_closest_point(dst_points, pt1)
                idx2 = self.find_closest_point(dst_points, pt2)
                idx3 = self.find_closest_point(dst_points, pt3)
                
                if idx1 is not None and idx2 is not None and idx3 is not None:
                    src_tri = [src_points[idx1], src_points[idx2], src_points[idx3]]
                    dst_tri = [dst_points[idx1], dst_points[idx2], dst_points[idx3]]
                    
                    # 对三角形进行仿射变换
                    self.warp_triangle(source_img, output_img, src_tri, dst_tri)
        
        return output_img
    
    def find_closest_point(self, points, point):
        """找到最接近给定点的点的索引"""
        min_dist = float('inf')
        min_idx = None
        
        for i, p in enumerate(points):
            dist = np.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # 如果距离太大，认为没有找到匹配点
        if min_dist > 30:
            return None
            
        return min_idx
    
    def warp_triangle(self, src_img, dst_img, src_tri, dst_tri):
        """对三角形区域进行仿射变换"""
        # 转换为numpy数组
        src_tri = np.array(src_tri, dtype=np.float32)
        dst_tri = np.array(dst_tri, dtype=np.float32)
        
        # 计算源三角形的边界矩形
        r1 = cv2.boundingRect(np.array([src_tri], dtype=np.int32))
        r2 = cv2.boundingRect(np.array([dst_tri], dtype=np.int32))
        
        # 偏移三角形顶点
        src_tri_cropped = []
        for i in range(3):
            src_tri_cropped.append(((src_tri[i][0] - r1[0]), (src_tri[i][1] - r1[1])))
        
        dst_tri_cropped = []
        for i in range(3):
            dst_tri_cropped.append(((dst_tri[i][0] - r2[0]), (dst_tri[i][1] - r2[1])))
        
        # 裁剪源图像
        src_cropped = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        
        # 创建目标图像的裁剪区域
        dst_cropped_size = (r2[2], r2[3])
        dst_cropped = np.zeros((dst_cropped_size[1], dst_cropped_size[0], 3), dtype=np.uint8)
        
        # 获取仿射变换矩阵
        warp_mat = cv2.getAffineTransform(np.array(src_tri_cropped, dtype=np.float32),
                                          np.array(dst_tri_cropped, dtype=np.float32))
        
        # 应用仿射变换
        cv2.warpAffine(src_cropped, warp_mat, dst_cropped_size, dst_cropped,
                       borderMode=cv2.BORDER_REFLECT_101)
        
        # 创建蒙版
        mask = np.zeros((dst_cropped_size[1], dst_cropped_size[0]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(dst_tri_cropped, dtype=np.int32), 255)
        
        # 应用蒙版
        dst_cropped_masked = cv2.bitwise_and(dst_cropped, dst_cropped, mask=mask)
        
        # 复制变换后的三角形到目标图像
        dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask[:, :, np.newaxis] / 255.0) + dst_cropped_masked
    
    def color_correct(self, source, target, mask):
        """对源图像进行颜色校正，使其与目标图像颜色一致"""
        # 确保掩码是单通道的
        if len(mask.shape) == 3:
            mask_single = mask[:, :, 0]  # 只取第一个通道
        else:
            mask_single = mask
            
        # 转换为LAB颜色空间
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # 计算掩码区域的平均值和标准差
        source_mean, source_std = cv2.meanStdDev(source_lab, mask=mask_single)
        target_mean, target_std = cv2.meanStdDev(target_lab, mask=mask_single)
        
        # 应用颜色校正
        result_lab = source_lab.copy()
        for i in range(3):
            result_lab[:, :, i] = (result_lab[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]
        
        # 转换回BGR颜色空间
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def seamless_clone(self, source, target, mask):
        """使用泊松融合进行无缝克隆"""
        # 确保掩码是单通道的
        if len(mask.shape) == 3:
            mask_single = mask[:, :, 0]
        else:
            mask_single = mask
            
        # 二值化掩码以确保值为0或255
        _, mask_binary = cv2.threshold(mask_single, 127, 255, cv2.THRESH_BINARY)
        
        # 检查掩码是否有效
        if cv2.countNonZero(mask_binary) == 0:
            return target
            
        # 找到掩码的中心点
        try:
            moments = cv2.moments(mask_binary)
            if moments['m00'] == 0:
                return target
                
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            center = (center_x, center_y)
            
            # 应用泊松融合
            result = cv2.seamlessClone(source, target, mask_binary, center, cv2.NORMAL_CLONE)
            return result
        except Exception as e:
            print(f"融合错误: {e}")
            return target
    
    def swap_face(self, frame):
        """在视频帧中进行人脸交换"""
        if self.target_face_landmarks is None:
            return frame
        
        # 转换为灰度图用于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.detector(gray)
        
        # 处理每个检测到的人脸
        for face in faces:
            try:
                # 获取人脸关键点
                landmarks = self.extract_landmarks(gray, face)
                
                # 将关键点转换为与target_face_landmarks相同的格式
                frame_landmarks = {}
                parts = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 
                         'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
                
                # 根据dlib的68个关键点映射到face_recognition库的格式
                indices = {
                    'chin': list(range(0, 17)),
                    'left_eyebrow': list(range(17, 22)),
                    'right_eyebrow': list(range(22, 27)),
                    'nose_bridge': list(range(27, 31)),
                    'nose_tip': list(range(31, 36)),
                    'left_eye': list(range(36, 42)),
                    'right_eye': list(range(42, 48)),
                    'top_lip': list(range(48, 55)) + [64, 63, 62, 61, 60],
                    'bottom_lip': list(range(54, 60)) + [48, 60, 67, 66, 65, 64]
                }
                
                for part in parts:
                    frame_landmarks[part] = [landmarks[i] for i in indices[part]]
                
                # 创建面部蒙版
                face_mask = self.get_face_mask(frame, frame_landmarks)
                
                # 变形目标人脸以匹配视频中的人脸
                warped_target = self.warp_image(self.target_face_img, self.target_face_landmarks, 
                                               frame_landmarks, frame.shape)
                
                # 检查变形后的图像是否有效
                if np.sum(warped_target) == 0:
                    continue
                
                # 颜色校正
                color_corrected = self.color_correct(warped_target, frame, face_mask)
                
                # 无缝融合
                result = self.seamless_clone(color_corrected, frame, face_mask)
                
                # 更新帧
                frame = result
            except Exception as e:
                print(f"处理人脸时出错: {e}")
                continue
        
        return frame
    
    def start_webcam(self, camera_id=0):
        """启动网络摄像头并进行实时人脸交换"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("按 'q' 退出")
        print("按 's' 保存当前帧")
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("无法获取视频帧")
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
            
            # 进行人脸交换
            result_frame = self.swap_face(frame)
            
            # 显示FPS
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow("Face Swap", result_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                save_path = f"output_photos/face_swap_result_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_path, result_frame)
                print(f"已保存: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # 设置目标人脸图像路径
    target_image_path = PHOTO_PATH
    print(f"使用默认图像: {target_image_path}")
        
    # 检查默认图像是否存在，如果不存在则提示用户
    if not os.path.exists(target_image_path):
        print(f"默认图像不存在: {target_image_path}")
        print("请提供一张包含清晰人脸的图像")
        return
    
    # 创建FaceSwapper实例
    face_swapper = FaceSwapperRealTime(target_image_path)
    
    # 启动网络摄像头
    face_swapper.start_webcam()

if __name__ == "__main__":
    main()