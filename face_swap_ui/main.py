import sys
import os
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QGridLayout, QMessageBox, QFrame, QSplitter,
                            QProgressBar, QStatusBar, QToolTip, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor

# 导入insightface库
import insightface
from insightface.app import FaceAnalysis

# 导入全屏所需的模块
from PyQt5.QtWidgets import QDialog

# 人脸交换器类
class RealtimeFaceSwapper:
    def __init__(self, target_image_path=None):
        """初始化人脸交换器"""
        # 初始化FaceAnalysis应用
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # 加载换脸模型
        model_path = '../models/inswapper_128.onnx'
        if not os.path.exists(model_path):
            print(f"请下载换脸模型并放置在: {os.path.abspath(model_path)}")
            print("下载地址: https://huggingface.co/datasets/Gourieff/ReActor/tree/main/models")
            os.makedirs('models', exist_ok=True)
            sys.exit(1)
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

# 摄像头线程类
class CameraThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    
    def __init__(self, face_swapper, camera_id=0):
        super().__init__()
        self.face_swapper = face_swapper
        self.camera_id = camera_id
        self.running = False
        self.recording = False
        self.video_writer = None
        self.output_path = ""
        
    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
            
        # 设置缓冲区大小
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 调整帧大小以提高性能
            frame = cv2.resize(frame, (640, 480))
            
            # 处理帧
            result_frame = self.face_swapper.process_frame(frame)
            
            # 发送处理后的帧
            self.update_frame.emit(result_frame)
            
            # 如果正在录制，写入视频
            if self.recording and self.video_writer is not None:
                self.video_writer.write(result_frame)
                
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            
    def stop(self):
        self.running = False
        self.wait()
        
    def start_recording(self, output_path):
        if self.recording:
            return
            
        self.output_path = output_path
        # 使用MP4V编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        self.recording = True
        
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            return self.output_path
        return None

# 主窗口类
class FaceSwapUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("人脸交换应用 - FaceSwap")
        self.setMinimumSize(1400, 800)  # 放大窗体大小
        
        # 设置应用样式
        self.set_application_style()
        
        # 初始化UI
        self.init_ui()
        
        # 初始化人脸交换器
        self.face_swapper = RealtimeFaceSwapper()
        
        # 预设人脸图片路径
        self.preset_faces = [
            "face_photos/mabaoguo.jpg",
            "face_photos/musk.jpeg",
            "face_photos/zhaoliying.jpeg"
        ]
        
        # 确保目录存在
        os.makedirs("output_videos", exist_ok=True)
        os.makedirs("output_photos", exist_ok=True)
        
        # 加载预设人脸图片
        self.load_preset_faces()
        
        # 初始化摄像头线程
        self.camera_thread = None
        self.is_recording = False
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("欢迎使用人脸交换应用")
        
        # 使用QTimer延迟显示欢迎消息，确保主界面已经显示
        QTimer.singleShot(500, self.show_welcome_message)
    
    def set_application_style(self):
        """设置应用程序样式"""
        # 设置全局字体
        app_font = QFont("Microsoft YaHei", 12)  # 增大字体大小
        QApplication.setFont(app_font)
        
        # 设置工具提示样式
        QToolTip.setFont(QFont("Microsoft YaHei", 11))  # 增大提示字体
        
        # 设置应用样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
                font-size: 12pt;  /* 增大标签字体 */
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 10px 18px;  /* 增大按钮内边距 */
                border-radius: 5px;
                font-weight: bold;
                font-size: 12pt;  /* 增大按钮字体 */
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QFrame {
                border-radius: 8px;
                background-color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 12px;
                font-size: 13pt;  /* 增大分组框标题字体 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QStatusBar {
                font-size: 12pt;  /* 增大状态栏字体 */
            }
            QProgressBar {
                font-size: 11pt;  /* 增大进度条字体 */
            }
        """)
    
    def show_welcome_message(self):
        """显示欢迎信息和使用提示"""
        welcome_msg = QMessageBox(self)
        welcome_msg.setWindowTitle("欢迎使用人脸交换应用")
        welcome_msg.setIcon(QMessageBox.Information)
        welcome_msg.setText("欢迎使用人脸交换应用！")
        
        # 设置欢迎消息的字体
        font = QFont("Microsoft YaHei", 12)
        welcome_msg.setFont(font)
        
        welcome_msg.setInformativeText(
            "使用步骤：\n"
            "1. 从右侧选择一个预设人脸或上传自定义人脸\n"
            "2. 点击\"启动摄像头\"按钮开始实时人脸交换\n"
            "3. 可以点击\"开始录制\"按钮录制视频\n\n"
            "提示：确保已下载模型文件并放置在正确位置"
        )
        welcome_msg.setStandardButtons(QMessageBox.Ok)
        
        # 设置按钮字体
        for button in welcome_msg.buttons():
            button.setFont(font)
        
        welcome_msg.show()  # 使用show()而不是exec_()，这样不会阻塞主界面
        
    def init_ui(self):
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)  # 增大边距
        main_layout.setSpacing(20)  # 增大间距
        
        # 创建分割器，允许用户调整左右两侧的宽度
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)  # 增大分割条宽度
        
        # 左侧摄像头显示区域
        camera_container = QGroupBox("实时预览")
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setContentsMargins(15, 20, 15, 15)  # 增大内边距
        
        self.camera_view = QLabel()
        self.camera_view.setMinimumSize(720, 540)  # 增大摄像头视图大小
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background-color: #222222; color: white; border-radius: 10px; font-size: 14pt;")  # 增大字体
        self.camera_view.setText("摄像头未启动\n\n请先选择一个人脸图片，然后点击\"启动摄像头\"按钮")
        
        # 添加FPS显示标签和全屏按钮的水平布局
        bottom_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #4a86e8; font-weight: bold; font-size: 13pt;")  # 增大字体
        self.fps_label.setAlignment(Qt.AlignLeft)
        
        # 添加全屏按钮
        self.fullscreen_button = QPushButton("全屏预览")
        self.fullscreen_button.setIcon(QIcon.fromTheme("view-fullscreen"))
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setToolTip("切换到全屏预览模式")
        self.fullscreen_button.setEnabled(False)  # 初始时禁用
        
        bottom_layout.addWidget(self.fps_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.fullscreen_button)
        
        camera_layout.addWidget(self.camera_view)
        camera_layout.addLayout(bottom_layout)
        
        # 右侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        
        # 预设人脸选择区域
        face_selection_group = QGroupBox("选择预设人脸")
        face_selection_layout = QVBoxLayout(face_selection_group)
        
        # 预设人脸网格布局
        preset_faces_grid = QGridLayout()
        preset_faces_grid.setSpacing(10)
        self.preset_face_buttons = []
        
        for i in range(3):
            face_frame = QFrame()
            face_frame.setFrameShape(QFrame.StyledPanel)
            face_frame.setMinimumSize(150, 180)
            face_frame.setMaximumSize(150, 180)
            face_frame.setStyleSheet("border: 1px solid #dddddd;")
            
            face_layout = QVBoxLayout(face_frame)
            
            # 添加人脸名称标签
            face_name_label = QLabel(f"预设人脸 {i+1}")
            face_name_label.setAlignment(Qt.AlignCenter)
            face_name_label.setStyleSheet("font-weight: bold; color: #4a86e8;")
            
            face_label = QLabel()
            face_label.setAlignment(Qt.AlignCenter)
            face_label.setMinimumSize(120, 120)
            face_label.setMaximumSize(120, 120)
            face_label.setStyleSheet("background-color: #f0f0f0; border-radius: 60px;")
            face_label.setText(f"人脸 {i+1}")
            
            select_button = QPushButton("选择")
            select_button.setProperty("face_index", i)
            select_button.clicked.connect(self.select_preset_face)
            select_button.setToolTip(f"点击选择预设人脸 {i+1} 作为换脸源")
            
            face_layout.addWidget(face_name_label)
            face_layout.addWidget(face_label)
            face_layout.addWidget(select_button)
            
            preset_faces_grid.addWidget(face_frame, 0, i)
            self.preset_face_buttons.append((face_label, select_button, face_name_label))
        
        face_selection_layout.addLayout(preset_faces_grid)
        
        # 上传自定义人脸
        upload_button = QPushButton("上传自定义人脸")
        upload_button.setIcon(QIcon.fromTheme("document-open"))
        upload_button.setToolTip("从本地选择一张包含人脸的图片作为换脸源")
        upload_button.clicked.connect(self.upload_custom_face)
        upload_button.setMinimumHeight(40)
        face_selection_layout.addWidget(upload_button)
        
        control_layout.addWidget(face_selection_group)
        
        # 摄像头控制组
        camera_control_group = QGroupBox("摄像头控制")
        camera_control_layout = QVBoxLayout(camera_control_group)
        
        # 摄像头按钮
        camera_buttons_layout = QHBoxLayout()
        
        self.start_camera_button = QPushButton("启动摄像头")
        self.start_camera_button.setIcon(QIcon.fromTheme("camera-video"))
        self.start_camera_button.clicked.connect(self.toggle_camera)
        self.start_camera_button.setToolTip("启动/停止摄像头进行实时人脸交换")
        self.start_camera_button.setMinimumHeight(40)
        camera_buttons_layout.addWidget(self.start_camera_button)
        
        self.record_button = QPushButton("开始录制")
        self.record_button.setIcon(QIcon.fromTheme("media-record"))
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        self.record_button.setToolTip("开始/停止录制当前视频")
        self.record_button.setMinimumHeight(40)
        camera_buttons_layout.addWidget(self.record_button)
        
        camera_control_layout.addLayout(camera_buttons_layout)
        
        # 添加截图按钮
        self.snapshot_button = QPushButton("拍摄截图")
        self.snapshot_button.setIcon(QIcon.fromTheme("camera-photo"))
        self.snapshot_button.clicked.connect(self.take_snapshot)
        self.snapshot_button.setEnabled(False)
        self.snapshot_button.setToolTip("保存当前画面为图片")
        self.snapshot_button.setMinimumHeight(40)
        camera_control_layout.addWidget(self.snapshot_button)
        
        # 录制进度条
        self.recording_progress = QProgressBar()
        self.recording_progress.setRange(0, 100)
        self.recording_progress.setValue(0)
        self.recording_progress.setTextVisible(True)
        self.recording_progress.setFormat("录制中: %p%")
        self.recording_progress.setVisible(False)
        camera_control_layout.addWidget(self.recording_progress)
        
        control_layout.addWidget(camera_control_group)
        
        # 状态信息
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("请选择一个人脸图片并启动摄像头")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 10px;")
        status_layout.addWidget(self.status_label)
        
        control_layout.addWidget(status_group)
        
        # 添加使用说明
        help_group = QGroupBox("使用说明")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QLabel(
            "1. 从上方选择一个预设人脸或上传自定义人脸\n"
            "2. 点击\"启动摄像头\"按钮开始实时人脸交换\n"
            "3. 可以点击\"开始录制\"按钮录制视频\n"
            "4. 可以点击\"拍摄截图\"按钮保存当前画面\n"
            "5. 录制的视频保存在output_videos目录\n"
            "6. 截图保存在output_photos目录"
        )
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        
        control_layout.addWidget(help_group)
        
        # 添加弹性空间
        control_layout.addStretch()
        
        # 将左右两部分添加到分割器
        splitter.addWidget(camera_container)
        splitter.addWidget(control_panel)
        
        # 设置分割器的初始大小比例
        splitter.setSizes([700, 500])
        
        # 将分割器添加到主布局
        main_layout.addWidget(splitter)
    
    def load_preset_faces(self):
        face_names = ["马保国", "马斯克", "赵丽颖"]
        
        for i, face_path in enumerate(self.preset_faces):
            if os.path.exists(face_path):
                # 读取原始图片
                pixmap = QPixmap(face_path)
                
                # 创建正方形画布，确保图像居中显示
                size = 140  # 头像大小
                rounded_pixmap = QPixmap(size, size)
                rounded_pixmap.fill(Qt.transparent)
                
                # 使用QPainter绘制圆形头像
                from PyQt5.QtGui import QPainter, QBrush, QPainterPath
                painter = QPainter(rounded_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                
                # 创建圆形裁剪路径
                path = QPainterPath()
                path.addEllipse(0, 0, size, size)
                painter.setClipPath(path)
                
                # 计算缩放和居中位置
                scaled_pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x_offset = (size - scaled_pixmap.width()) // 2
                y_offset = (size - scaled_pixmap.height()) // 2
                
                # 在圆形区域内居中绘制图像
                painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
                painter.end()
                
                self.preset_face_buttons[i][0].setPixmap(rounded_pixmap)
                self.preset_face_buttons[i][0].setText("")
                self.preset_face_buttons[i][2].setText(face_names[i])
            else:
                self.preset_face_buttons[i][0].setText(f"人脸 {i+1}\n(未找到)")
                self.preset_face_buttons[i][2].setText(f"预设人脸 {i+1} (未找到)")
    
    def select_preset_face(self):
        sender = self.sender()
        face_index = sender.property("face_index")
        face_path = self.preset_faces[face_index]
        face_names = ["马保国", "马斯克", "赵丽颖"]
        
        if not os.path.exists(face_path):
            QMessageBox.warning(self, "错误", f"人脸图片不存在: {face_path}")
            return
        
        # 显示加载中提示
        self.statusBar.showMessage("正在加载人脸模型...")
        QApplication.processEvents()
        
        if self.face_swapper.load_source_face(face_path):
            self.status_label.setText(f"已选择人脸: {face_names[face_index]}")
            self.statusBar.showMessage(f"已成功加载人脸: {face_names[face_index]}", 3000)
            
            # 高亮显示选中的人脸
            for i, (label, button, name_label) in enumerate(self.preset_face_buttons):
                if i == face_index:
                    label.setStyleSheet("background-color: #e6f2ff; border: 2px solid #4a86e8; border-radius: 60px;")
                    button.setStyleSheet("background-color: #2a66c8;")
                else:
                    label.setStyleSheet("background-color: #f0f0f0; border-radius: 60px;")
                    button.setStyleSheet("")
        else:
            QMessageBox.warning(self, "错误", "无法加载人脸图片或未检测到人脸")
            self.statusBar.showMessage("人脸加载失败", 3000)
    
    def upload_custom_face(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择人脸图片", "", "图片文件 (*.png *.jpg *.jpeg)")
        
        if file_path:
            # 显示加载中提示
            self.statusBar.showMessage("正在加载人脸模型...")
            QApplication.processEvents()
            
            if self.face_swapper.load_source_face(file_path):
                self.status_label.setText(f"已加载自定义人脸: {os.path.basename(file_path)}")
                self.statusBar.showMessage(f"已成功加载自定义人脸", 3000)
                
                # 重置所有预设人脸的样式
                for label, button, _ in self.preset_face_buttons:
                    label.setStyleSheet("background-color: #f0f0f0; border-radius: 60px;")
                    button.setStyleSheet("")
            else:
                QMessageBox.warning(self, "错误", "无法加载人脸图片或未检测到人脸")
                self.statusBar.showMessage("人脸加载失败", 3000)
    
    def toggle_camera(self):
        if self.camera_thread is None or not self.camera_thread.running:
            # 启动摄像头
            if self.face_swapper.source_face is None:
                QMessageBox.warning(self, "警告", "请先选择一个人脸图片")
                return
            
            self.statusBar.showMessage("正在启动摄像头...")
            QApplication.processEvents()
            
            self.camera_thread = CameraThread(self.face_swapper)
            self.camera_thread.update_frame.connect(self.update_camera_view)
            self.camera_thread.start()
            
            self.start_camera_button.setText("停止摄像头")
            self.start_camera_button.setStyleSheet("background-color: #e74c3c;")
            self.record_button.setEnabled(True)
            self.snapshot_button.setEnabled(True)
            self.fullscreen_button.setEnabled(True)  # 启用全屏按钮
            self.status_label.setText("摄像头已启动，正在进行实时人脸交换")
            self.statusBar.showMessage("摄像头已启动", 3000)
            
            # 启动FPS计时器
            self.fps_timer = QTimer()
            self.fps_timer.timeout.connect(self.update_fps)
            self.fps_timer.start(1000)  # 每秒更新一次
            self.frame_count = 0
            self.last_fps_time = time.time()
        else:
            # 停止摄像头
            if self.is_recording:
                self.toggle_recording()
            
            self.statusBar.showMessage("正在停止摄像头...")
            QApplication.processEvents()
            
            self.camera_thread.stop()
            self.camera_thread = None
            
            # 停止FPS计时器
            if hasattr(self, 'fps_timer'):
                self.fps_timer.stop()
            
            self.start_camera_button.setText("启动摄像头")
            self.start_camera_button.setStyleSheet("")
            self.record_button.setEnabled(False)
            self.snapshot_button.setEnabled(False)
            self.fullscreen_button.setEnabled(False)  # 禁用全屏按钮
            self.camera_view.setText("摄像头未启动\n\n请先选择一个人脸图片，然后点击\"启动摄像头\"按钮")
            self.status_label.setText("摄像头已停止")
            self.statusBar.showMessage("摄像头已停止", 3000)
    
    def update_fps(self):
        """更新FPS显示"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def toggle_recording(self):
        if not self.is_recording:
            # 开始录制
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            # 将文件扩展名从.avi改为.mp4
            output_path = os.path.join(output_dir, f"face_swap_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            
            self.camera_thread.start_recording(output_path)
            self.is_recording = True
            self.record_button.setText("停止录制")
            self.record_button.setStyleSheet("background-color: #e74c3c;")
            self.status_label.setText(f"正在录制视频: {os.path.basename(output_path)}")
            self.statusBar.showMessage("视频录制中...", 3000)
            
            # 显示录制进度条
            self.recording_progress.setValue(0)
            self.recording_progress.setVisible(True)
            
            # 启动录制计时器
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.update_recording_progress)
            self.recording_timer.start(1000)  # 每秒更新一次
            self.recording_start_time = time.time()
        else:
            # 停止录制
            output_path = self.camera_thread.stop_recording()
            self.is_recording = False
            self.record_button.setText("开始录制")
            self.record_button.setStyleSheet("")
            
            # 停止录制计时器
            if hasattr(self, 'recording_timer'):
                self.recording_timer.stop()
            
            # 隐藏录制进度条
            self.recording_progress.setVisible(False)
            
            if output_path:
                self.status_label.setText(f"视频已保存: {os.path.basename(output_path)}")
                self.statusBar.showMessage(f"视频已保存: {os.path.basename(output_path)}", 3000)
                QMessageBox.information(self, "录制完成", f"视频已保存到: {output_path}")
    
    def update_recording_progress(self):
        """更新录制进度条"""
        elapsed = time.time() - self.recording_start_time
        # 假设最长录制时间为5分钟
        max_recording_time = 5 * 60
        progress = min(100, (elapsed / max_recording_time) * 100)
        self.recording_progress.setValue(int(progress))
        
        # 更新录制时间显示
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.recording_progress.setFormat(f"录制中: {minutes:02d}:{seconds:02d} (%p%)")
    
    def take_snapshot(self):
        """拍摄当前画面的截图"""
        if self.camera_thread is None or not self.camera_thread.running:
            return
        
        output_dir = "output_photos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"face_swap_photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        
        # 获取当前帧
        current_pixmap = self.camera_view.pixmap()
        if current_pixmap:
            # 保存图片
            current_pixmap.save(output_path, "JPG")
            self.status_label.setText(f"截图已保存: {os.path.basename(output_path)}")
            self.statusBar.showMessage(f"截图已保存: {os.path.basename(output_path)}", 3000)
            
            # 显示成功消息
            QMessageBox.information(self, "截图成功", f"截图已保存到: {output_path}")
    
    def update_camera_view(self, frame):
        # 更新帧计数
        self.frame_count += 1
        
        # 将OpenCV的BGR格式转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 创建QPixmap并缩放以适应容器
        pixmap = QPixmap.fromImage(q_img)
        
        # 获取容器大小
        container_width = self.camera_view.width()
        container_height = self.camera_view.height()
        
        # 缩放图像以适应容器，保持原始比例
        scaled_pixmap = pixmap.scaled(container_width, container_height, 
                                       Qt.KeepAspectRatio, 
                                       Qt.SmoothTransformation)
        
        # 显示图像
        self.camera_view.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        
        # 如果当前有图像显示，则重新调整图像大小
        if hasattr(self, 'camera_view') and self.camera_view.pixmap() is not None:
            current_pixmap = self.camera_view.pixmap()
            container_width = self.camera_view.width()
            container_height = self.camera_view.height()
            
            # 重新缩放图像
            scaled_pixmap = current_pixmap.scaled(container_width, container_height, 
                                                 Qt.KeepAspectRatio, 
                                                 Qt.SmoothTransformation)
            self.camera_view.setPixmap(scaled_pixmap)
    
    # 在FaceSwapUI类中添加以下方法
    
    def toggle_fullscreen(self):
        """切换全屏预览模式"""
        if not hasattr(self, 'fullscreen_dialog'):
            # 创建全屏对话框
            self.fullscreen_dialog = QDialog(self)
            self.fullscreen_dialog.setWindowTitle("全屏预览")
            self.fullscreen_dialog.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            
            # 创建全屏布局
            fullscreen_layout = QVBoxLayout(self.fullscreen_dialog)
            fullscreen_layout.setContentsMargins(0, 0, 0, 0)
            
            # 创建全屏预览标签
            self.fullscreen_view = QLabel()
            self.fullscreen_view.setAlignment(Qt.AlignCenter)
            self.fullscreen_view.setStyleSheet("background-color: black;")
            
            # 创建退出全屏按钮
            exit_button = QPushButton("退出全屏")
            exit_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 120);
                    color: white;
                    border: 1px solid white;
                    border-radius: 5px;
                    padding: 8px 15px;
                    position: absolute;
                    bottom: 20px;
                    right: 20px;
                }
                QPushButton:hover {
                    background-color: rgba(50, 50, 50, 180);
                }
            """)
            exit_button.clicked.connect(self.exit_fullscreen)
            
            # 添加到布局
            fullscreen_layout.addWidget(self.fullscreen_view)
            
            # 创建一个水平布局来放置退出按钮在右下角
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(exit_button)
            fullscreen_layout.addLayout(button_layout)
            
            # 设置键盘事件处理
            self.fullscreen_dialog.keyPressEvent = self.fullscreen_key_press
            
            # 连接摄像头帧更新信号到全屏视图
            if self.camera_thread:
                self.camera_thread.update_frame.connect(self.update_fullscreen_view)
        
        # 显示全屏对话框
        self.fullscreen_dialog.showFullScreen()
        self.statusBar.showMessage("已进入全屏预览模式，按ESC或点击退出全屏按钮退出", 3000)
    
    def exit_fullscreen(self):
        """退出全屏模式"""
        if hasattr(self, 'fullscreen_dialog'):
            self.fullscreen_dialog.hide()
            self.statusBar.showMessage("已退出全屏预览模式", 3000)
    
    def fullscreen_key_press(self, event):
        """处理全屏模式下的键盘事件"""
        if event.key() == Qt.Key_Escape:
            self.exit_fullscreen()
        else:
            # 将其他键盘事件传递给默认处理程序
            QDialog.keyPressEvent(self.fullscreen_dialog, event)
    
    def update_fullscreen_view(self, frame):
        """更新全屏视图"""
        if hasattr(self, 'fullscreen_dialog') and self.fullscreen_dialog.isVisible():
            # 将OpenCV的BGR格式转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 创建QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 创建QPixmap
            pixmap = QPixmap.fromImage(q_img)
            
            # 获取全屏容器大小
            container_width = self.fullscreen_view.width()
            container_height = self.fullscreen_view.height()
            
            # 缩放图像以适应容器，保持原始比例
            scaled_pixmap = pixmap.scaled(container_width, container_height, 
                                         Qt.KeepAspectRatio, 
                                         Qt.SmoothTransformation)
            
            # 显示图像
            self.fullscreen_view.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        # 关闭窗口时停止摄像头线程
        if hasattr(self, 'fullscreen_dialog') and self.fullscreen_dialog.isVisible():
            self.fullscreen_dialog.hide()
            
        if self.camera_thread is not None and self.camera_thread.running:
            if self.is_recording:
                self.toggle_recording()
            self.camera_thread.stop()
        event.accept()

# 添加一个简单的异常处理函数
def handle_exception(exc_type, exc_value, exc_traceback):
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"发生错误:\n{error_msg}")
    QMessageBox.critical(None, "应用错误", f"发生错误:\n{str(exc_value)}")
    
if __name__ == "__main__":
    # 导入traceback用于异常处理
    import traceback
    
    # 设置异常处理器
    sys.excepthook = handle_exception
    
    # 创建应用
    app = QApplication(sys.argv)
    
    try:
        # 创建并显示主窗口
        window = FaceSwapUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "启动错误", f"应用启动失败:\n{str(e)}")
        sys.exit(1)