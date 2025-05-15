# -*- coding: utf-8 -*-



import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, ttk, Scale, messagebox
from PIL import Image, ImageTk
import threading
import os
import numpy as np
import time
import pygame  # 用於音頻播放
from collections import defaultdict  # 用於噪聲過濾


# 基本設置類：負責初始化和配置
class AppConfig:
    """負責應用程序的基本配置和設置"""
    
    def __init__(self):
        """初始化應用程序配置"""
        # 初始化 MediaPipe 解決方案
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # 初始化檢測模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3)
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3)
        
        # 繪圖規格
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # 性能追蹤
        self.last_frame_time = 0
        self.fps = 0
        
        # 設置顏色映射
        self.color_map = {
            'right_elbow': (255, 0, 0),     # 藍色
            'left_elbow': (0, 255, 0),      # 綠色
            'right_wrist': (255, 0, 255),   # 洋紅色
            'left_wrist': (0, 255, 255),    # 黃色
            'right_shoulder': (255, 255, 0), # 青色
            'left_shoulder': (0, 0, 255),   # 紅色
            'right_knee': (128, 0, 128),    # 紫色
            'left_knee': (128, 128, 0),     # 橄欖色
            'right_ankle': (0, 128, 128),   # 藍綠色
            'left_ankle': (100, 100, 100)   # 灰色
        }


# 視頻處理類：負責視頻捕獲和幀處理
class VideoProcessor:
    """處理視頻捕獲和幀處理的類"""
    
    def __init__(self, config, tracking_manager):
        """初始化視頻處理器
        
        Args:
            config: AppConfig 實例
            tracking_manager: TrackingManager 實例
        """
        self.config = config
        self.tracking_manager = tracking_manager
        
        # 視頻捕獲變量
        self.cap = None
        self.video_source = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_paused = False
        self.last_processed_frame = None
        
        # 音頻播放變量
        pygame.init()  # 初始化 pygame 用於音頻
        pygame.mixer.init()
        self.sound = None
        self.audio_enabled = False
        self.audio_thread = None
        self.audio_stop_event = threading.Event()
        self.temp_audio_file = None
    
    def open_video(self, source, is_webcam=False):
        """打開視頻源
        
        Args:
            source: 視頻源（文件路徑或攝像頭索引）
            is_webcam: 是否為網絡攝像頭
        
        Returns:
            bool: 是否成功打開視頻
        """
        self.video_source = source
        
        # 釋放任何現有的捕獲
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.1)  # 短暫延遲以確保資源被釋放
        
        # 打開視頻捕獲
        self.cap = cv2.VideoCapture(source)
        
        # 優化視頻捕獲設置
        if is_webcam:
            # 設置攝像頭屬性以獲得更好的性能
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 設置緩衝區大小以減少延遲
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            return False
            
        # 獲取視頻屬性
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # 處理無效的 fps
            fps = 30
        
        self.current_frame = 0
        self.video_paused = False
        
        return True
    
    def process_frame(self, frame, detect_hands, detect_pose, tracking_enabled, enable_pose_checking):
        """處理視頻幀
        
        Args:
            frame: 要處理的幀
            detect_hands: 是否檢測手
            detect_pose: 是否檢測姿勢
            tracking_enabled: 是否啟用軌跡跟踪
            enable_pose_checking: 是否啟用姿勢檢查
            
        Returns:
            處理後的幀
        """
        # 檢查幀是否為空
        if frame is None or frame.size == 0:
            # 返回空白幀而不是失敗
            height, width = 480, 640
            blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
            return blank_frame
            
        # 如果太大，調整幀的大小以獲得更好的性能
        height, width = frame.shape[:2]
        if width > 1280 or height > 720:
            scale_factor = min(1280 / width, 720 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
            
        # 將 BGR 圖像轉換為 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 水平翻轉圖像以產生鏡像效果
        rgb_frame = cv2.flip(rgb_frame, 1)
        
        # 基於選定的選項處理圖像
        hand_results = None
        pose_results = None
        
        # 使用 try-except 來處理潛在的 MediaPipe 故障
        try:
            if detect_hands:
                hand_results = self.config.hands.process(rgb_frame)
                
            if detect_pose:
                pose_results = self.config.pose.process(rgb_frame)
        except Exception as e:
            print(f"MediaPipe 錯誤: {e}")
            # 繼續而不進行檢測
        
        # 轉換回 BGR 以顯示
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # MediaPipe 地標與我們的追蹤關鍵點之間的映射
        landmark_mapping = {
            'right_elbow': self.config.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_elbow': self.config.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_wrist': self.config.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_wrist': self.config.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_shoulder': self.config.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_shoulder': self.config.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_knee': self.config.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_knee': self.config.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_ankle': self.config.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_ankle': self.config.mp_pose.PoseLandmark.LEFT_ANKLE
        }
        
        # 如果檢測到並啟用了選項，則繪製姿勢地標
        if pose_results and pose_results.pose_landmarks and detect_pose:
            # 獲取圖像尺寸以計算實際的像素坐標
            h, w, _ = frame.shape
            
            # 繪製所有姿勢地標
            self.config.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.config.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.config.mp_drawing_styles.get_default_pose_landmarks_style())
            
            # 如果啟用，則更新軌跡跟踪點
            if tracking_enabled:
                self.tracking_manager.update_tracking_points(pose_results, landmark_mapping, h, w)
        
        # 如果檢測到並啟用了選項，則繪製手部地標
        if hand_results and hand_results.multi_hand_landmarks and detect_hands:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.config.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.config.mp_hands.HAND_CONNECTIONS,
                    self.config.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.config.mp_drawing_styles.get_default_hand_connections_style())
        
        # 儲存處理過的幀以供軌跡顯示使用
        self.last_processed_frame = frame.copy()
        
        # 如果啟用了跟踪和過濾，則更新過濾後的跟踪點
        if tracking_enabled and self.tracking_manager.filtering_enabled:
            self.tracking_manager.update_filtered_trajectories()
            
        # 如果啟用，則更新姿勢比較
        if tracking_enabled and enable_pose_checking:
            self.tracking_manager.update_pose_comparison()
            
            # 如果啟用了姿勢檢查，在幀上繪製姿勢比較結果
            if enable_pose_checking and any(x is not None for x in self.tracking_manager.standard_trajectory_images.values()):
                # 計算所有活動部分的平均分數
                active_scores = []
                for landmark_name in self.tracking_manager.pose_similarity_scores:
                    is_enabled, has_data, has_standard = self.tracking_manager.check_tracking_status(landmark_name)
                    if is_enabled and has_data and has_standard:
                        active_scores.append(self.tracking_manager.pose_similarity_scores[landmark_name])
                
                if active_scores:
                    avg_score = sum(active_scores) / len(active_scores)
                    
                    # 繪製總體分數
                    score_text = f"姿勢: {avg_score:.1f}%"
                    
                    # 基於分數的顏色
                    if avg_score >= 80:
                        result_text = "正確"
                        score_color = (0, 255, 0)
                    elif avg_score >= 50:
                        result_text = "可接受"
                        score_color = (0, 255, 255)
                    else:
                        result_text = "不正確"
                        score_color = (0, 0, 255)
                        
                    # 在幀上繪製信息
                    cv2.putText(frame, score_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)
                    cv2.putText(frame, result_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)
            
        return frame
    
    def setup_audio(self, video_source, enable_audio):
        """為視頻文件設置音頻播放
        
        Args:
            video_source: 視頻源文件路徑
            enable_audio: 是否啟用音頻
        """
        # 只有當它是視頻文件並且啟用了音頻時才繼續
        if not enable_audio:
            self.audio_enabled = False
            return
        
        try:
            # 使用 ffmpeg 將音頻提取到臨時 wav 文件
            import subprocess
            import tempfile
            
            # 創建臨時文件
            self.temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self.temp_audio_file.close()
            
            # 使用 ffmpeg 提取音頻
            try:
                subprocess.call([
                    "ffmpeg", "-i", video_source, "-q:a", "0", "-map", "a", 
                    self.temp_audio_file.name, "-y"
                ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                
                # 加載音頻文件
                pygame.mixer.music.load(self.temp_audio_file.name)
                self.audio_enabled = True
                
                # 在單獨的線程中啟動音頻以避免阻塞
                self.audio_thread = threading.Thread(target=self.play_audio)
                self.audio_thread.daemon = True
                self.audio_thread.start()
                
            except Exception as e:
                print(f"提取音頻時出錯: {e}")
                self.audio_enabled = False
                
        except (ImportError, Exception) as e:
            print(f"設置音頻時出錯: {e}")
            self.audio_enabled = False
    
    def play_audio(self):
        """與視頻同步播放音頻"""
        pygame.mixer.music.play()
        
        # 監控 audio_stop_event 以在需要時停止播放
        while pygame.mixer.music.get_busy() and not self.audio_stop_event.is_set():
            time.sleep(0.1)
            
            # 如果視頻暫停，也暫停音頻
            if self.video_paused and pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
            elif not self.video_paused and not pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
        
        # 完成後停止音頻播放
        pygame.mixer.music.stop()
    
    def format_time(self, seconds):
        """將秒轉換為 MM:SS 格式
        
        Args:
            seconds: 總秒數
            
        Returns:
            格式化的時間字符串
        """
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    def cleanup(self):
        """清理視頻和音頻資源"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        # 停止音頻播放
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        
        # 清理音頻臨時文件（如果存在）
        if self.temp_audio_file and os.path.exists(self.temp_audio_file.name):
            try:
                os.unlink(self.temp_audio_file.name)
            except:
                pass


# 軌跡管理類：負責處理和顯示運動軌跡
class TrackingManager:
    """管理運動軌跡的類"""
    
    def __init__(self, config):
        """初始化軌跡管理器
        
        Args:
            config: AppConfig 實例
        """
        self.config = config
        
        # 軌跡追蹤變量
        self.trajectories = {}
        self.trajectory_threads = {}
        self.trajectory_tracks = {}
        self.trajectory_stop_event = threading.Event()
        self.trajectory_thread = None
        self.trajectory_canvas = None
        
        # 初始化追蹤點數組
        self.tracking_points = {
            'right_elbow': [],
            'left_elbow': [],
            'right_wrist': [],
            'left_wrist': [],
            'right_shoulder': [],
            'left_shoulder': [],
            'right_knee': [],
            'left_knee': [],
            'right_ankle': [],
            'left_ankle': []
        }
        
        # 初始化過濾後的追蹤點
        self.filtered_tracking_points = {
            'right_elbow': [],
            'left_elbow': [],
            'right_wrist': [],
            'left_wrist': [],
            'right_shoulder': [],
            'left_shoulder': [],
            'right_knee': [],
            'left_knee': [],
            'right_ankle': [],
            'left_ankle': []
        }
        
        # 初始化用於比較的標準姿勢軌跡圖像
        self.standard_trajectory_images = {
            'right_elbow': None,
            'left_elbow': None,
            'right_wrist': None,
            'left_wrist': None,
            'right_shoulder': None,
            'left_shoulder': None,
            'right_knee': None,
            'left_knee': None,
            'right_ankle': None,
            'left_ankle': None
        }
        
        # 初始化姿勢比較結果
        self.pose_similarity_scores = {
            'right_elbow': 0,
            'left_elbow': 0,
            'right_wrist': 0,
            'left_wrist': 0,
            'right_shoulder': 0,
            'left_shoulder': 0, 
            'right_knee': 0,
            'left_knee': 0,
            'right_ankle': 0,
            'left_ankle': 0
        }
        
        # 噪聲過濾參數
        self.filtering_area = 10
        self.filtering_time = 4
        self.filtering_enabled = True
        
        # 追蹤旗標
        self.tracking_enabled = False
        self.track_elbows = True
        self.track_wrists = False
        self.track_shoulders = False
        self.track_knees = False
        self.track_ankles = False
        
        # 顯示模式
        self.current_display_mode = "combined"
        
    def show_combined_trajectories(self):
        """在單個窗口中顯示所有軌跡"""
        window_name = "動作軌跡"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)
        cv2.resizeWindow(window_name, 640, 480)
        
        last_display_mode = None
        
        while not self.trajectory_stop_event.is_set():
            # 為此幀創建一個新畫布
            if self.current_display_mode == "combined":  # 使用線程安全標志
                # 使用持久畫布顯示所有軌跡歷史
                display_canvas = self.trajectory_canvas.copy()
                
                # 在畫布上繪製所有活動軌跡
                if self.filtering_enabled:  # 使用線程安全標志
                    # 繪製原始和過濾的軌跡 
                    # 顏色較淺的原始軌跡
                    self.draw_all_trajectories(display_canvas, use_filtered=False, alpha=0.3)
                    
                    # 顏色較強的過濾軌跡
                    self.draw_all_trajectories(display_canvas, use_filtered=True, alpha=1.0)
                else:
                    # 只繪製原始軌跡
                    self.draw_all_trajectories(display_canvas, use_filtered=False)
                
                # 向圖像添加圖例
                self.add_trajectory_legend(display_canvas)
                
            elif self.current_display_mode == "filtered-only":  # 使用線程安全標志
                # 僅顯示過濾軌跡
                display_canvas = self.trajectory_canvas.copy()
                
                # 只繪製過濾軌跡
                self.draw_all_trajectories(display_canvas, use_filtered=True)
                
                # 添加圖例
                self.add_trajectory_legend(display_canvas)
                
                # 添加過濾信息
                filter_info = f"過濾: 區域={self.filtering_area}x{self.filtering_area}, 閾值={self.filtering_time}"
                cv2.putText(display_canvas, filter_info, (10, display_canvas.shape[0]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
            elif self.current_display_mode == "color-coded":  # 使用線程安全標志
                # 創建黑色畫布
                h, w = display_canvas.shape[:2] if 'display_canvas' in locals() else (480, 640)
                
                display_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                
                # 在視頻幀上繪製軌跡
                if self.filtering_enabled:  # 使用線程安全標志
                    # 繪製原始和過濾的軌跡
                    # 顏色較淺的原始軌跡
                    self.draw_all_trajectories(display_canvas, use_filtered=False, alpha=0.3)
                    
                    # 顏色較強的過濾軌跡
                    self.draw_all_trajectories(display_canvas, use_filtered=True, alpha=1.0)
                else:
                    # 只繪製原始軌跡
                    self.draw_all_trajectories(display_canvas, use_filtered=False)
                
                # 向圖像添加圖例
                self.add_trajectory_legend(display_canvas)
                
            elif self.current_display_mode == "individual":  # 使用線程安全標志
                # 創建個別軌跡視圖的網格
                h, w = display_canvas.shape[:2] if 'display_canvas' in locals() else (480, 640)
                    
                # 創建網格佈局（2x5 用於 10 個潛在軌跡）
                grid_h, grid_w = 2, 5
                cell_h, cell_w = h // grid_h, w // grid_w
                # 創建更大的網格畫布
                display_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                
                # 根據過濾設置選擇要顯示的點
                points_dict = self.filtered_tracking_points if self.filtering_enabled else self.tracking_points
                
                # 在其自己的網格單元中繪製每個軌跡
                i = 0
                for landmark_name, color in self.config.color_map.items():
                    if len(points_dict[landmark_name]) > 0:
                        # 計算網格位置
                        row, col = i // grid_w, i % grid_w
                        x0, y0 = col * cell_w, row * cell_h
                        
                        # 在此單元格中繪製軌跡
                        for point in points_dict[landmark_name]:
                            # 縮放點以適合單元格
                            scaled_x = x0 + (point[0] * cell_w) // w
                            scaled_y = y0 + (point[1] * cell_h) // h
                            
                            # 確保點在單元格範圍內
                            scaled_x = min(max(scaled_x, x0), x0 + cell_w - 1)
                            scaled_y = min(max(scaled_y, y0), y0 + cell_h - 1)
                            
                            cv2.circle(display_canvas, (scaled_x, scaled_y), 1, color, -1)
                        
                        # 為此軌跡添加標籤
                        label = landmark_name.replace('_', ' ').title()
                        cv2.putText(display_canvas, label, (x0 + 5, y0 + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # 繪製單元格邊框
                        cv2.rectangle(display_canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), 
                                     (100, 100, 100), 1)
                        
                        i += 1
                        
            elif self.current_display_mode == "pose-check":  # 姿勢檢查模式
                # 創建網格佈局顯示姿勢分析結果
                h, w = display_canvas.shape[:2] if 'display_canvas' in locals() else (480, 640)
                
                # 創建姿勢檢查視圖的畫布
                display_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                
                # 添加標題
                cv2.putText(display_canvas, "姿勢分析結果", (w//2 - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 添加姿勢檢查狀態
                if not self.tracking_enabled:
                    cv2.putText(display_canvas, "姿勢檢查已禁用", (w//2 - 120, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_canvas, "啟用姿勢檢查以查看結果", (w//2 - 180, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                elif not any(x is not None for x in self.standard_trajectory_images.values()):
                    cv2.putText(display_canvas, "未加載標準姿勢", (w//2 - 150, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_canvas, "使用'載入標準姿勢'設置參考", (w//2 - 200, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                else:
                    # 為分數創建表格佈局 - 確保列大小正確
                    table_x = 50
                    table_y = 80
                    row_height = 40
                    col_width = 150  # 減小以更好地適合
                    
                    # 繪製表格標題，間距更好
                    cv2.putText(display_canvas, "身體部位", (table_x + 20, table_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(display_canvas, "狀態", (table_x + col_width + 20, table_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(display_canvas, "相似度", (table_x + 2*col_width + 20, table_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(display_canvas, "結果", (table_x + 3*col_width + 20, table_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    
                    table_y += 25
                    
                    # 在標題下繪製水平線
                    cv2.line(display_canvas, (table_x, table_y), 
                            (table_x + 4*col_width, table_y), (120, 120, 120), 1)
                    
                    table_y += 5
                    
                    # 按類型分組地標
                    landmark_groups = {
                        "肘部": ["right_elbow", "left_elbow"],
                        "手腕": ["right_wrist", "left_wrist"],
                        "肩部": ["right_shoulder", "left_shoulder"],
                        "膝蓋": ["right_knee", "left_knee"],
                        "腳踝": ["right_ankle", "left_ankle"]
                    }
                    
                    # 繪製每個組的結果
                    for group_name, landmarks in landmark_groups.items():
                        # 如果此組中沒有地標被跟踪，則跳過
                        if not any(self.check_tracking_status(lm)[0] for lm in landmarks):
                            continue
                            
                        # 繪製組標題
                        cv2.putText(display_canvas, group_name, (table_x, table_y + row_height//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                        table_y += row_height
                        
                        # 繪製此組中每個地標的結果
                        for landmark_name in landmarks:
                            # 獲取此地標的跟踪狀態
                            is_enabled, has_data, has_standard = self.check_tracking_status(landmark_name)
                            
                            # 如果未啟用則跳過
                            if not is_enabled:
                                continue
                                
                            # 獲取顯示名稱
                            display_name = landmark_name.replace('_', ' ').title()
                            if "Right" in display_name:
                                display_name = display_name.replace("Right ", "右")
                            if "Left" in display_name:
                                display_name = display_name.replace("Left ", "左")
                            
                            # 繪製地標名稱
                            cv2.putText(display_canvas, display_name, (table_x + 20, table_y + row_height//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            # 繪製狀態
                            status_text = "未跟踪"
                            status_color = (100, 100, 100)  # 灰色
                            
                            if has_data and has_standard:
                                status_text = "活動"
                                status_color = (0, 255, 0)  # 綠色
                            elif has_data and not has_standard:
                                status_text = "無標準"
                                status_color = (0, 200, 255)  # 黃色
                            elif not has_data and has_standard:
                                status_text = "無數據"
                                status_color = (0, 0, 255)  # 紅色
                                
                            cv2.putText(display_canvas, status_text, 
                                       (table_x + col_width + 20, table_y + row_height//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
                            
                            # 繪製相似度分數
                            if has_data and has_standard:
                                score = self.pose_similarity_scores[landmark_name]
                                score_text = f"{score:.1f}%"
                                
                                # 基於分數的顏色
                                if score >= 80:
                                    score_color = (0, 255, 0)  # 綠色
                                elif score >= 50:
                                    score_color = (0, 255, 255)  # 黃色
                                else:
                                    score_color = (0, 0, 255)  # 紅色
                                    
                                cv2.putText(display_canvas, score_text, 
                                           (table_x + 2*col_width + 20, table_y + row_height//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1, cv2.LINE_AA)
                                
                                # 繪製結果
                                if score >= 80:
                                    result_text = "正確"
                                    result_color = (0, 255, 0)
                                elif score >= 50:
                                    result_text = "可接受"
                                    result_color = (0, 255, 255)
                                else:
                                    result_text = "不正確"
                                    result_color = (0, 0, 255)
                                    
                                cv2.putText(display_canvas, result_text, 
                                           (table_x + 3*col_width + 20, table_y + row_height//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, result_color, 1, cv2.LINE_AA)
                            else:
                                cv2.putText(display_canvas, "N/A", 
                                           (table_x + 2*col_width + 20, table_y + row_height//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
                                           
                                cv2.putText(display_canvas, "N/A", 
                                           (table_x + 3*col_width + 20, table_y + row_height//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
                            
                            # 在行之間繪製水平線
                            cv2.line(display_canvas, (table_x, table_y + row_height), 
                                    (table_x + 4*col_width, table_y + row_height), (50, 50, 50), 1)
                            
                            table_y += row_height
                        
                        # 在組之間添加一些空間
                        table_y += 10
                        
                    # 在底部添加總體分數
                    if self.tracking_enabled and any(x is not None for x in self.standard_trajectory_images.values()):
                        # 計算所有活動部分的平均分數
                        active_scores = []
                        for landmark_name in self.pose_similarity_scores:
                            is_enabled, has_data, has_standard = self.check_tracking_status(landmark_name)
                            if is_enabled and has_data and has_standard:
                                active_scores.append(self.pose_similarity_scores[landmark_name])
                        
                        if active_scores:
                            avg_score = sum(active_scores) / len(active_scores)
                            
                            # 繪製總體分數
                            score_text = f"總體姿勢分數: {avg_score:.1f}%"
                            
                            # 基於分數的顏色
                            if avg_score >= 80:
                                result_text = "正確姿勢"
                                score_color = (0, 255, 0)
                            elif avg_score >= 50:
                                result_text = "可接受姿勢"
                                score_color = (0, 255, 255)
                            else:
                                result_text = "不正確姿勢"
                                score_color = (0, 0, 255)
                                
                            cv2.putText(display_canvas, score_text, (w//2 - 120, h - 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)
                                       
                            cv2.putText(display_canvas, result_text, (w//2 - 100, h - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2, cv2.LINE_AA)
            
            # 如果我們切換了顯示模式，清除主軌跡畫布
            if last_display_mode != self.current_display_mode:
                # 如果從 individual 模式切換到或從 individual 模式切換出去，則清除軌跡畫布
                if last_display_mode == "individual" or self.current_display_mode == "individual":
                    self.trajectory_canvas = np.zeros_like(self.trajectory_canvas)
                
                last_display_mode = self.current_display_mode
            
            # 顯示結果畫布
            cv2.imshow(window_name, display_canvas)
            
            # 檢查 Esc 鍵以關閉
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
            # 減慢更新率以節省 CPU
            time.sleep(0.03)
        
        # 清理
        cv2.destroyWindow(window_name)
    
    def track_landmark_thread(self, landmark_name, track_image, color=(255, 255, 255)):
        """跟踪特定地標軌跡的線程函數"""
        window_name = f"{landmark_name.replace('_', ' ').title()} 軌跡"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)  # 定位窗口
        
        while not self.trajectory_stop_event.is_set():
            # 創建圖像的副本以繪製
            display_img = track_image.copy()
            
            # 根據過濾設置選擇要使用的點
            if self.filtering_enabled:  # 使用線程安全標志而不是 .get()
                # 用強烈的顏色繪製過濾後的點
                if len(self.filtered_tracking_points[landmark_name]) >= 1:
                    for point in self.filtered_tracking_points[landmark_name]:
                        cv2.circle(display_img, point, 4, color, -1)
                        # 同時添加到持久圖像
                        cv2.circle(track_image, point, 4, color, -1)
                        
                # 用較淺的顏色繪製原始點
                if len(self.tracking_points[landmark_name]) >= 1:
                    lighter_color = tuple(min(c + 100, 255) for c in color)  # 使顏色更淺
                    for point in self.tracking_points[landmark_name]:
                        cv2.circle(display_img, point, 2, lighter_color, -1)
                        # 不要將這些添加到持久圖像
            else:
                # 只繪製原始點
                if len(self.tracking_points[landmark_name]) >= 1:
                    for point in self.tracking_points[landmark_name]:
                        cv2.circle(display_img, point, 3, color, -1)
                        # 添加到持久圖像
                        cv2.circle(track_image, point, 3, color, -1)
            
            # 添加過濾狀態
            if self.filtering_enabled:  # 使用線程安全標志而不是 .get()
                filter_info = f"過濾: {self.filtering_area}x{self.filtering_area}, 閾值={self.filtering_time}"
                raw_count = len(self.tracking_points[landmark_name])
                filtered_count = len(self.filtered_tracking_points[landmark_name])
                count_info = f"點數: {filtered_count}/{raw_count} ({filtered_count/max(1, raw_count)*100:.1f}%)"
                
                cv2.putText(display_img, filter_info, (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_img, count_info, (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 如果啟用，添加姿勢檢查信息
            if self.tracking_enabled:
                is_enabled, has_data, has_standard = self.check_tracking_status(landmark_name)
                if has_data and has_standard:
                    score = self.pose_similarity_scores[landmark_name]
                    score_text = f"相似度: {score:.1f}%"
                    
                    # 基於分數的顏色
                    if score >= 80:
                        result_text = "正確"
                        score_color = (0, 255, 0)
                    elif score >= 50:
                        result_text = "可接受"
                        score_color = (0, 255, 255)
                    else:
                        result_text = "不正確"
                        score_color = (0, 0, 255)
                        
                    # 在幀上繪製信息
                    cv2.putText(display_img, score_text, (10, display_img.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 1)
                    cv2.putText(display_img, result_text, (10, display_img.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 1)
            
            # 顯示軌跡圖像
            cv2.imshow(window_name, display_img)
            
            # 檢查 Esc 鍵以關閉
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 鍵
                break
                
            # 減慢更新率以節省 CPU
            time.sleep(0.03)
                
        # 清理
        cv2.destroyWindow(window_name)
    
    def update_tracking_points(self, pose_results, landmark_mapping, height, width):
        """根據檢測到的姿勢更新軌跡跟踪點
        
        Args:
            pose_results: MediaPipe 姿勢檢測結果
            landmark_mapping: 地標映射字典
            height: 幀高度
            width: 幀寬度
        """
        for track_name, landmark_idx in landmark_mapping.items():
            landmark = pose_results.pose_landmarks.landmark[landmark_idx]
            # 只有當跟踪此點並且有足夠的可見度時才跟踪
            if (
                (track_name.startswith('right_elbow') or track_name.startswith('left_elbow')) and self.track_elbows or
                (track_name.startswith('right_wrist') or track_name.startswith('left_wrist')) and self.track_wrists or
                (track_name.startswith('right_shoulder') or track_name.startswith('left_shoulder')) and self.track_shoulders or
                (track_name.startswith('right_knee') or track_name.startswith('left_knee')) and self.track_knees or
                (track_name.startswith('right_ankle') or track_name.startswith('left_ankle')) and self.track_ankles
            ) and landmark.visibility > 0.5:
                # 添加軌跡點
                pt = (int(landmark.x * width), int(landmark.y * height))
                self.tracking_points[track_name].append(pt)
                
                # 限制點數以防性能問題
                max_points = 5000  # 根據需要調整
                if len(self.tracking_points[track_name]) > max_points:
                    self.tracking_points[track_name] = self.tracking_points[track_name][-max_points:]
    
    def track_filter(self, track, filtering_area, filtering_time):
        """通過查找具有重複點的區域來過濾軌跡噪聲。
        
        Args:
            track: 點列表 [(x,y), ...]
            filtering_area: 檢查點集群的區域大小（例如，10 = 10x10 像素區域）
            filtering_time: 一個區域中保留所需的點數
            
        Returns:
            過濾後的點列表
        """
        area_points = defaultdict(list)
        area_count = defaultdict(int)

        # 按各自的區域對所有點進行分組
        for pt in track:
            if pt is None:
                continue
            area_x = pt[0] // filtering_area
            area_y = pt[1] // filtering_area
            area_key = (area_x, area_y)
            area_points[area_key].append(pt)
            area_count[area_key] += 1

        # 只保留具有足夠重複次數的區域中的點
        filtered_track = []
        for key, count in area_count.items():
            if count >= filtering_time:
                filtered_track.extend(area_points[key])

        return filtered_track
    
    def update_filtered_trajectories(self):
        """根據當前原始數據和過濾設置更新所有過濾後的軌跡"""
        if not self.tracking_enabled:
            return
            
        # 對每個軌跡應用過濾
        for key in self.tracking_points:
            if len(self.tracking_points[key]) > 0:
                self.filtered_tracking_points[key] = self.track_filter(
                    self.tracking_points[key], 
                    self.filtering_area, 
                    self.filtering_time
                )
    
    def track_to_image(self, track, shape):
        """將軌跡轉換為填充形狀圖像
        
        Args:
            track: 點列表 [(x,y), ...]
            shape: (height, width, channels) 的元組
            
        Returns:
            填充形狀圖像
        """
        img = np.zeros(shape, dtype=np.uint8)
        if len(track) < 3:
            return img
        
        pts = np.array(track, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.drawContours(img, [hull], -1, (255, 255, 255), thickness=-1)
        return img
    
    def comparing(self, standardTrackImg, comparingTrackImg):
        """比較兩個軌跡圖像並計算相似度分數
        
        Args:
            standardTrackImg: 參考軌跡圖像
            comparingTrackImg: 要比較的當前軌跡圖像
            
        Returns:
            相似度分數 (0-100)
        """
        # 處理空圖像
        if standardTrackImg is None or comparingTrackImg is None:
            return 0.0
            
        # 轉換為灰度並二值化
        std_gray = cv2.cvtColor(standardTrackImg, cv2.COLOR_BGR2GRAY)
        cmp_gray = cv2.cvtColor(comparingTrackImg, cv2.COLOR_BGR2GRAY)
        
        _, std_bin = cv2.threshold(std_gray, 10, 255, cv2.THRESH_BINARY)
        _, cmp_bin = cv2.threshold(cmp_gray, 10, 255, cv2.THRESH_BINARY)
        
        # 計算輪廓
        contours1, _ = cv2.findContours(std_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(cmp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours1) == 0 or len(contours2) == 0:
            return 0.0
            
        # 獲取最大輪廓
        cnt1 = max(contours1, key=cv2.contourArea)
        cnt2 = max(contours2, key=cv2.contourArea)
        
        # 計算 Hu 矩
        hu1 = cv2.HuMoments(cv2.moments(cnt1)).flatten()
        hu2 = cv2.HuMoments(cv2.moments(cnt2)).flatten()
        
        # 使用對數尺度以避免極小值影響結果
        hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
        hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
        
        # 計算歐幾里得距離
        distance = np.linalg.norm(hu1 - hu2)
        
        # 將距離轉換為相似度分數 (0-100)
        similarity = max(0.0, 100 - distance * 10)
        similarity = min(similarity, 100.0)  # 上限為 100
        
        return similarity
    
    def update_pose_comparison(self):
        """更新所有追蹤身體部位的姿勢比較分數"""
        if not self.tracking_enabled:
            return
            
        # 獲取幀尺寸以用於軌跡圖像
        shape = (480, 640, 3)  # 預設值，應從視頻處理器獲取實際值
            
        # 比較每個追蹤的身體部位與其標準姿勢
        for key in self.filtered_tracking_points:
            # 如果沒有標準圖像或點不足，則跳過
            if self.standard_trajectory_images[key] is None or len(self.filtered_tracking_points[key]) < 3:
                self.pose_similarity_scores[key] = 0
                continue
                
            # 從當前軌跡創建圖像
            current_track_img = self.track_to_image(self.filtered_tracking_points[key], shape)
            
            # 與標準姿勢比較
            similarity = self.comparing(self.standard_trajectory_images[key], current_track_img)
            
            # 更新分數
            self.pose_similarity_scores[key] = similarity
    
    def check_tracking_status(self, landmark_name):
        """檢查地標是否正在被追蹤並且有足夠的數據進行分析
        
        Args:
            landmark_name: 要檢查的地標名稱
            
        Returns:
            (is_enabled, has_data, has_standard) 的元組
        """
        # 檢查此地標類型是否啟用追蹤
        is_enabled = False
        if 'elbow' in landmark_name and self.track_elbows:
            is_enabled = True
        elif 'wrist' in landmark_name and self.track_wrists:
            is_enabled = True
        elif 'shoulder' in landmark_name and self.track_shoulders:
            is_enabled = True
        elif 'knee' in landmark_name and self.track_knees:
            is_enabled = True
        elif 'ankle' in landmark_name and self.track_ankles:
            is_enabled = True
            
        # 檢查我們是否有足夠的軌跡數據
        has_data = len(self.filtered_tracking_points[landmark_name]) >= 3
        
        # 檢查我們是否有用於比較的標準姿勢數據
        has_standard = self.standard_trajectory_images[landmark_name] is not None
        
        return (is_enabled, has_data, has_standard)
    
    def clear_tracking(self):
        """清除所有軌跡追蹤數據"""
        # 清除追蹤點
        for key in self.tracking_points:
            self.tracking_points[key].clear()
            
        # 清除過濾後的追蹤點
        for key in self.filtered_tracking_points:
            self.filtered_tracking_points[key].clear()
        
        # 重置姿勢相似度分數
        for key in self.pose_similarity_scores:
            self.pose_similarity_scores[key] = 0
            
        # 清除合併軌跡畫布
        if hasattr(self, 'trajectory_canvas') and self.trajectory_canvas is not None:
            self.trajectory_canvas.fill(0)  # 用黑色填充
        
        # 清除個別軌跡畫布
        for key, track in self.trajectory_tracks.items():
            if track is not None:
                track.fill(0)  # 用黑色填充
    
    def draw_all_trajectories(self, canvas, use_filtered=False, alpha=1.0):
        """在提供的畫布上繪製所有活動軌跡
        
        Args:
            canvas: 要繪製的畫布
            use_filtered: 如果為 True，使用過濾後的點而不是原始點
            alpha: 點的不透明度值（0.0-1.0）- 可用於使原始點更透明
        """
        # 檢查哪些追蹤選項已啟用
        enabled_landmarks = []
        if self.track_elbows:
            enabled_landmarks.extend(['right_elbow', 'left_elbow'])
        if self.track_wrists:
            enabled_landmarks.extend(['right_wrist', 'left_wrist'])
        if self.track_shoulders:
            enabled_landmarks.extend(['right_shoulder', 'left_shoulder'])
        if self.track_knees:
            enabled_landmarks.extend(['right_knee', 'left_knee'])
        if self.track_ankles:
            enabled_landmarks.extend(['right_ankle', 'left_ankle'])
        
        # 選擇基於過濾設置要顯示的點
        points_dict = self.filtered_tracking_points if use_filtered else self.tracking_points
        
        # 繪製每個啟用的軌跡
        for landmark_name in enabled_landmarks:
            if landmark_name in points_dict and len(points_dict[landmark_name]) > 0:
                # 獲取此地標的顏色
                base_color = self.config.color_map.get(landmark_name, (255, 255, 255))
                
                # 如果需要，應用 alpha（根據 alpha 修改顏色）
                if alpha < 1.0:
                    color = tuple(int(c * alpha) for c in base_color)
                else:
                    color = base_color
                
                # 確定點大小（過濾後的點可以更大）
                point_size = 3 if use_filtered else 2
                
                # 繪製軌跡中的所有點
                for point in points_dict[landmark_name]:
                    # 在組合畫布上繪製點
                    cv2.circle(canvas, point, point_size, color, -1)
                    
                    # 如果我們在組合模式下，也更新持久畫布
                    if self.current_display_mode == "combined": 
                        cv2.circle(self.trajectory_canvas, point, point_size, color, -1)
    
    def add_trajectory_legend(self, canvas):
        """添加顏色圖例到軌跡顯示"""
        # 設置圖例區域
        legend_x = 10
        legend_y = 30
        spacing = 20
        
        # 添加標題
        cv2.putText(canvas, "軌跡圖例:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += spacing
        
        # 如果啟用，添加過濾信息
        if self.filtering_enabled:
            filter_info = f"噪聲過濾: {self.filtering_area}x{self.filtering_area} px, 閾值={self.filtering_time}"
            cv2.putText(canvas, filter_info, (legend_x, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            legend_y += spacing
        
        # 在圖例中添加每個被追蹤的地標
        for landmark_name, color in self.config.color_map.items():
            # 只有當此類型被追蹤時才顯示在圖例中
            if (('elbow' in landmark_name and self.track_elbows) or
                ('wrist' in landmark_name and self.track_wrists) or
                ('shoulder' in landmark_name and self.track_shoulders) or
                ('knee' in landmark_name and self.track_knees) or
                ('ankle' in landmark_name and self.track_ankles)):
                
                # 繪製顏色樣本
                cv2.rectangle(canvas, (legend_x, legend_y-10), (legend_x+10, legend_y), color, -1)
                
                # 繪製標籤
                label = landmark_name.replace('_', ' ').title()
                cv2.putText(canvas, label, (legend_x+15, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 如果啟用了過濾，顯示點計數
                if self.filtering_enabled:
                    raw_count = len(self.tracking_points[landmark_name])
                    filtered_count = len(self.filtered_tracking_points[landmark_name])
                    count_text = f" ({filtered_count}/{raw_count})"
                    cv2.putText(canvas, count_text, (legend_x+100, legend_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                legend_y += spacing
                
        # 如果啟用，添加姿勢檢查狀態
        if self.tracking_enabled and any(x is not None for x in self.standard_trajectory_images.values()):
            # 計算所有活動部分的平均分數
            active_scores = []
            for landmark_name in self.pose_similarity_scores:
                is_enabled, has_data, has_standard = self.check_tracking_status(landmark_name)
                if is_enabled and has_data and has_standard:
                    active_scores.append(self.pose_similarity_scores[landmark_name])
            
            if active_scores:
                legend_y += 5
                avg_score = sum(active_scores) / len(active_scores)
                
                # 繪製總體分數
                score_text = f"姿勢分數: {avg_score:.1f}%"
                
                # 基於分數的顏色
                if avg_score >= 80:
                    score_color = (0, 255, 0)  # 綠色
                elif avg_score >= 50:
                    score_color = (0, 255, 255)  # 黃色
                else:
                    score_color = (0, 0, 255)  # 紅色
                    
                cv2.putText(canvas, score_text, (legend_x, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)
    
    def load_standard_pose(self, file_path, mp_pose):
        """從文件加載標準姿勢軌跡
        
        Args:
            file_path: 包含標準姿勢的視頻文件路徑
            mp_pose: MediaPipe 姿勢解決方案實例
            
        Returns:
            bool: 是否成功加載標準姿勢
        """
        if not file_path or not os.path.exists(file_path):
            return False
            
        # 清除現有的標準姿勢數據
        for key in self.standard_trajectory_images:
            self.standard_trajectory_images[key] = None
            
        # 打開視頻
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
            
        # 清除任何現有的追蹤點
        for key in self.tracking_points:
            self.tracking_points[key] = []
            self.filtered_tracking_points[key] = []
        
        # 處理視頻幀以提取標準姿勢
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 為標準姿勢使用更高質量
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        # MediaPipe 地標與我們的追蹤關鍵點之間的映射
        landmark_mapping = {
            'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
            'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
            'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE
        }
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 處理所有幀
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 更新進度
            frame_count += 1
            
            # 處理幀
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)  # 鏡像以保持一致性
            
            pose_results = pose.process(frame_rgb)
            
            if pose_results.pose_landmarks:
                # 獲取圖像尺寸
                h, w, _ = frame.shape
                
                # 追蹤地標位置
                for track_name, landmark_idx in landmark_mapping.items():
                    landmark = pose_results.pose_landmarks.landmark[landmark_idx]
                    if landmark.visibility > 0.5:
                        pt = (int(landmark.x * w), int(landmark.y * h))
                        self.tracking_points[track_name].append(pt)
        
        # 關閉視頻
        cap.release()
        
        # 對標準姿勢軌跡應用過濾
        for key in self.tracking_points:
            if len(self.tracking_points[key]) > 0:
                self.filtered_tracking_points[key] = self.track_filter(
                    self.tracking_points[key], self.filtering_area, self.filtering_time)
        
        # 獲取視頻尺寸
        if cap.isOpened():
            h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            h, w = 480, 640
        shape = (h, w, 3)
        
        # 創建標準軌跡圖像
        for key in self.filtered_tracking_points:
            if len(self.filtered_tracking_points[key]) > 0:
                self.standard_trajectory_images[key] = self.track_to_image(
                    self.filtered_tracking_points[key], shape)
        
        # 清除追蹤點以避免混淆
        for key in self.tracking_points:
            self.tracking_points[key] = []
            self.filtered_tracking_points[key] = []
        
        return True


# 用戶界面類：負責創建和管理圖形界面
class MotionTrackingUI:
    """創建和管理應用程序的圖形用戶界面"""
    
    def __init__(self, root, config, video_processor, tracking_manager):
        """初始化用戶界面
        
        Args:
            root: Tkinter 根窗口
            config: AppConfig 實例
            video_processor: VideoProcessor 實例
            tracking_manager: TrackingManager 實例
        """
        self.root = root
        self.config = config
        self.video_processor = video_processor
        self.tracking_manager = tracking_manager
        
        # 設置主窗口
        self.root.title("動作追蹤應用程序")
        self.root.geometry("1920x1080")
        self.root.resizable(True, True)
        
        # 初始化變量
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.photo = None
        
        # 創建整合式軌跡顯示
        self.unified_display = UnifiedTrackingDisplay(self.tracking_manager, self.video_processor)
        
        # 創建 UI 元素
        self.create_widgets()
        
    def create_widgets(self):
        """創建所有 UI 小部件"""
        # 配置樣式以獲得更現代的外觀
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TRadiobutton", background="#f0f0f0")
        style.configure("TCheckbutton", background="#f0f0f0")
        
        # 創建具有適當權重以進行調整大小的框架
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 創建時間軸框架
        self.timeline_frame = ttk.Frame(self.root, padding="10")
        self.timeline_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 創建追蹤選項框架 - 在視頻框架之前打包以保持順序
        self.tracking_frame = ttk.Frame(self.root, padding="10")
        self.tracking_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 創建視頻和繪圖框架
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.drawing_frame = ttk.Frame(self.root)
        self.drawing_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 配置網格權重以進行適當的調整大小
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # 創建視頻畫布
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 創建 mp_drawing 畫布
        self.drawing_canvas = tk.Canvas(self.drawing_frame, bg="white")
        self.drawing_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 創建控制部件
        self.create_source_controls()
        self.create_detection_controls()
        self.create_tracking_controls()  # 這裡會調用修改後的方法
        self.create_timeline_controls()
        
        # 狀態欄
        self.status_var = tk.StringVar(value="就緒")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 初始隱藏時間軸（僅對視頻文件顯示）
        self.timeline_frame.pack_forget()
    
    def create_source_controls(self):
        """創建源選擇控件"""
        ttk.Label(self.control_frame, text="視頻源:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.source_var = tk.StringVar(value="webcam")
        ttk.Radiobutton(self.control_frame, text="網絡攝像頭", variable=self.source_var, 
                       value="webcam").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="視頻文件", variable=self.source_var, 
                       value="file").grid(row=0, column=2, padx=5, pady=5)
        
        # 文件選擇
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(self.control_frame, textvariable=self.file_path, width=40)
        self.file_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        
        self.browse_btn = ttk.Button(self.control_frame, text="瀏覽", command=self.browse_file)
        self.browse_btn.grid(row=1, column=3, padx=5, pady=5)
    
    def create_tracking_controls(self):
        """創建軌跡追蹤控件"""
        ttk.Label(self.tracking_frame, text="軌跡追蹤:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.enable_tracking = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tracking_frame, text="啟用追蹤", variable=self.enable_tracking,
                        command=self.toggle_tracking).grid(row=0, column=1, padx=5, pady=5)
        
        # 添加 追蹤全部 和 不追蹤 按鈕
        self.track_all_btn = ttk.Button(self.tracking_frame, text="追蹤全部", command=self.track_all)
        self.track_all_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.track_none_btn = ttk.Button(self.tracking_frame, text="不追蹤", command=self.track_none)
        self.track_none_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # 添加姿勢檢查選項
        self.enable_pose_checking = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tracking_frame, text="啟用姿勢檢查", 
                        variable=self.enable_pose_checking).grid(
            row=0, column=4, padx=5, pady=5)
            
        # 添加載入標準姿勢按鈕
        self.load_standard_btn = ttk.Button(self.tracking_frame, text="載入標準姿勢", 
                                           command=self.load_standard_pose)
        self.load_standard_btn.grid(row=0, column=5, padx=5, pady=5)
        
        # 追蹤點選項
        self.track_elbows = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tracking_frame, text="追蹤肘部", variable=self.track_elbows,
                       command=lambda: self.update_track_flag('elbows')).grid(
            row=1, column=0, padx=5, pady=5)
        
        self.track_wrists = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tracking_frame, text="追蹤手腕", variable=self.track_wrists,
                       command=lambda: self.update_track_flag('wrists')).grid(
            row=1, column=1, padx=5, pady=5)
        
        self.track_shoulders = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tracking_frame, text="追蹤肩部", variable=self.track_shoulders,
                       command=lambda: self.update_track_flag('shoulders')).grid(
            row=1, column=2, padx=5, pady=5)
        
        self.track_knees = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tracking_frame, text="追蹤膝蓋", variable=self.track_knees,
                       command=lambda: self.update_track_flag('knees')).grid(
            row=2, column=0, padx=5, pady=5)
        
        self.track_ankles = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tracking_frame, text="追蹤腳踝", variable=self.track_ankles,
                       command=lambda: self.update_track_flag('ankles')).grid(
            row=2, column=1, padx=5, pady=5)
        
        self.clear_tracking_btn = ttk.Button(self.tracking_frame, text="清除追蹤", command=self.clear_tracking)
        self.clear_tracking_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # 噪聲過濾選項
        ttk.Label(self.tracking_frame, text="噪聲過濾:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.enable_filtering = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tracking_frame, text="啟用過濾", variable=self.enable_filtering,
                       command=self.on_filtering_change).grid(
            row=3, column=1, padx=5, pady=5)
            
        # 過濾區域滑塊
        ttk.Label(self.tracking_frame, text="過濾區域:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.filter_area_slider = Scale(self.tracking_frame, from_=5, to=30, 
                                       orient=tk.HORIZONTAL, command=self.update_filter_area)
        self.filter_area_slider.set(self.tracking_manager.filtering_area)
        self.filter_area_slider.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # 過濾時間滑塊
        ttk.Label(self.tracking_frame, text="過濾閾值:").grid(row=4, column=2, padx=5, pady=5, sticky=tk.W)
        self.filter_time_slider = Scale(self.tracking_frame, from_=1, to=10, 
                                       orient=tk.HORIZONTAL, command=self.update_filter_time)
        self.filter_time_slider.set(self.tracking_manager.filtering_time)
        self.filter_time_slider.grid(row=4, column=3, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # 顯示模式選擇
        ttk.Label(self.tracking_frame, text="顯示模式:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.display_mode = tk.StringVar(value="combined")
        display_combo = ttk.Combobox(self.tracking_frame, textvariable=self.display_mode, 
                                      values=["combined", "individual", "color-coded", "3D", "filtered-only", "pose-check"])
        display_combo.grid(row=5, column=1, padx=5, pady=5)
        display_combo.bind("<<ComboboxSelected>>", self.on_display_mode_change)
        
        # 添加比較按鈕
        self.compare_btn = ttk.Button(self.tracking_frame, text="顯示比較", command=self.show_trajectory_comparison)
        self.compare_btn.grid(row=5, column=2, padx=5, pady=5)
        
        # ===== 新增加的顯示選項控件 =====
        ttk.Label(self.tracking_frame, text="顯示選項:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 引導顯示切換
        self.show_guide = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tracking_frame, text="顯示引導軌跡", variable=self.show_guide,
                       command=self.toggle_guide).grid(row=6, column=1, padx=5, pady=5)
        
        # 引導不透明度滑塊
        ttk.Label(self.tracking_frame, text="引導不透明度:").grid(row=6, column=2, padx=5, pady=5, sticky=tk.W)
        self.guide_opacity_slider = Scale(self.tracking_frame, from_=0.1, to=1.0, resolution=0.1,
                                        orient=tk.HORIZONTAL, command=self.update_guide_opacity)
        self.guide_opacity_slider.set(0.3)
        self.guide_opacity_slider.grid(row=6, column=3, padx=5, pady=5, sticky=tk.W+tk.E)

    def toggle_guide(self):
        """切換引導軌跡顯示"""
        self.unified_display.show_guide = self.show_guide.get()
        self.status_var.set(f"引導軌跡顯示: {'開啟' if self.show_guide.get() else '關閉'}")
    
    def update_guide_opacity(self, value):
        """更新引導軌跡的不透明度"""
        opacity = float(value)
        self.unified_display.set_guide_opacity(opacity)
        self.status_var.set(f"引導軌跡不透明度: {opacity:.1f}")
    
    def create_detection_controls(self):
        """創建檢測選項控件"""
        ttk.Label(self.control_frame, text="檢測選項:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.detect_hands = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="手部", variable=self.detect_hands).grid(
            row=2, column=1, padx=5, pady=5)
        
        self.detect_pose = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="身體/手臂", variable=self.detect_pose).grid(
            row=2, column=2, padx=5, pady=5)
        
        # 音頻選項
        self.enable_audio = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="啟用音頻", variable=self.enable_audio).grid(
            row=3, column=1, padx=5, pady=5)
        
        # 控制按鈕
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.grid(row=4, column=0, columnspan=4, pady=10)
        
        self.start_btn = ttk.Button(self.button_frame, text="開始", command=self.start_tracking)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(self.button_frame, text="停止", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(self.button_frame, text="暫停", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
    
    
    def create_timeline_controls(self):
        """創建時間軸控件"""
        self.timeline_label = ttk.Label(self.timeline_frame, text="時間軸:")
        self.timeline_label.pack(side=tk.LEFT, padx=5)
        
        self.timeline_slider = Scale(self.timeline_frame, from_=0, to=100, 
                                   orient=tk.HORIZONTAL, command=self.on_timeline_change)
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.time_label = ttk.Label(self.timeline_frame, text="0:00 / 0:00")
        self.time_label.pack(side=tk.LEFT, padx=5)
    
    def browse_file(self):
        """瀏覽文件對話框"""
        file_path = filedialog.askopenfilename(
            filetypes=[("視頻文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if file_path:
            self.file_path.set(file_path)
            self.source_var.set("file")  # 自動選擇文件單選按鈕
    
    def toggle_tracking(self):
        """啟用或禁用軌跡追蹤"""
        self.tracking_manager.tracking_enabled = self.enable_tracking.get()
        
        if self.tracking_manager.tracking_enabled:
            # 檢查視頻捕獲是否已經運行
            if self.video_processor.cap is not None and self.video_processor.cap.isOpened():
                self.status_var.set("軌跡追蹤已啟用")
                # 開始顯示軌跡窗口
                try:
                    self.start_trajectory_tracking()
                except Exception as e:
                    print(f"啟動軌跡追蹤時出錯: {e}")
                    self.status_var.set(f"啟動軌跡追蹤時出錯: {str(e)}")
            else:
                self.status_var.set("按下開始時將開始軌跡追蹤")
        else:
            self.status_var.set("軌跡追蹤已禁用")
            # 在禁用時完全停止軌跡追蹤
            try:
                self.stop_trajectory_tracking()
            except Exception as e:
                print(f"停止軌跡追蹤時出錯: {e}")
            
            # 禁用追蹤時重置姿勢比較分數
            for key in self.tracking_manager.pose_similarity_scores:
                self.tracking_manager.pose_similarity_scores[key] = 0
    
    def on_filtering_change(self):
        """過濾複選框更改時更新線程安全標志"""
        self.tracking_manager.filtering_enabled = self.enable_filtering.get()
    
    def on_display_mode_change(self, event=None):
        """顯示模式更改時更新線程安全標志"""
        self.tracking_manager.current_display_mode = self.display_mode.get()
        self.status_var.set(f"軌跡顯示模式: {self.tracking_manager.current_display_mode}")
    
    def update_track_flag(self, track_type):
        """更新追蹤選項的標志"""
        if track_type == 'elbows':
            self.tracking_manager.track_elbows = self.track_elbows.get()
        elif track_type == 'wrists':
            self.tracking_manager.track_wrists = self.track_wrists.get()
        elif track_type == 'shoulders':
            self.tracking_manager.track_shoulders = self.track_shoulders.get()
        elif track_type == 'knees':
            self.tracking_manager.track_knees = self.track_knees.get()
        elif track_type == 'ankles':
            self.tracking_manager.track_ankles = self.track_ankles.get()
    
    def update_filter_area(self, value):
        """更新過濾區域大小"""
        self.tracking_manager.filtering_area = int(value)
        self.status_var.set(f"過濾區域設置為: {self.tracking_manager.filtering_area}x{self.tracking_manager.filtering_area} 像素")
        
        # 更新過濾的軌跡
        self.tracking_manager.update_filtered_trajectories()
    
    def update_filter_time(self, value):
        """更新過濾時間閾值"""
        self.tracking_manager.filtering_time = int(value)
        self.status_var.set(f"過濾閾值設置為: {self.tracking_manager.filtering_time}")
        
        # 更新過濾的軌跡
        self.tracking_manager.update_filtered_trajectories()
    
    def track_all(self):
        """選擇所有追蹤選項"""
        self.track_elbows.set(True)
        self.track_wrists.set(True)
        self.track_shoulders.set(True)
        self.track_knees.set(True)
        self.track_ankles.set(True)
    
        # 更新追蹤管理器的標志
        self.tracking_manager.track_elbows = True
        self.tracking_manager.track_wrists = True
        self.tracking_manager.track_shoulders = True
        self.tracking_manager.track_knees = True
        self.tracking_manager.track_ankles = True
    
        # 如果追蹤已經啟用，更新活動的追蹤線程
        if self.tracking_manager.tracking_enabled and self.video_processor.cap is not None and self.video_processor.cap.isOpened():
            self.update_tracking_selection()
        else:
            self.status_var.set("已選擇所有追蹤點。按下開始以開始追蹤。")
    
    def track_none(self):
        """取消選擇所有追蹤選項"""
        self.track_elbows.set(False)
        self.track_wrists.set(False)
        self.track_shoulders.set(False)
        self.track_knees.set(False)
        self.track_ankles.set(False)
    
        # 更新追蹤管理器的標志
        self.tracking_manager.track_elbows = False
        self.tracking_manager.track_wrists = False
        self.tracking_manager.track_shoulders = False
        self.tracking_manager.track_knees = False
        self.tracking_manager.track_ankles = False
    
    def clear_tracking(self):
        """清除所有軌跡追蹤數據"""
        self.tracking_manager.clear_tracking()
        self.status_var.set("追蹤數據已清除")
    
    def on_timeline_change(self, value):
        """處理時間軸滑塊更改"""
        if self.video_processor.cap is not None and self.video_processor.cap.isOpened() and not self.stop_event.is_set():
            # 將滑塊值轉換為幀號
            target_frame = int((float(value) / 100) * self.video_processor.total_frames)
            
            # 尋找該幀
            self.video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.video_processor.current_frame = target_frame
            
            # 如果啟用，更新音頻位置
            if self.video_processor.sound and pygame.mixer.get_init() and self.video_processor.audio_enabled:
                # 以秒為單位計算位置
                fps = self.video_processor.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    pos_seconds = target_frame / fps
                    pygame.mixer.music.stop()
                    pygame.mixer.music.play(start=pos_seconds)
    
    def toggle_pause(self):
        """暫停或恢復視頻播放"""
        if self.video_processor.cap is not None and self.video_processor.cap.isOpened():
            self.video_processor.video_paused = not self.video_processor.video_paused
            
            if self.video_processor.video_paused:
                self.pause_btn.config(text="繼續")
                # 暫停音頻
                if pygame.mixer.music.get_busy() and self.video_processor.audio_enabled:
                    pygame.mixer.music.pause()
            else:
                self.pause_btn.config(text="暫停")
                # 恢復音頻
                if self.video_processor.audio_enabled and not pygame.mixer.music.get_busy():
                    pygame.mixer.music.unpause()
    
    def load_standard_pose(self):
        """從文件加載標準姿勢軌跡"""
        # 要求用戶選擇包含標準姿勢的視頻文件
        file_path = filedialog.askopenfilename(
            title="選擇標準姿勢視頻",
            filetypes=[("視頻文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
            
        if not file_path:
            return
        
        # 創建進度對話框
        progress_window = tk.Toplevel(self.root)
        progress_window.title("正在加載標準姿勢")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="正在分析標準姿勢視頻...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="determinate")
        progress_bar.pack(pady=10)
            
        # 加載標準姿勢
        success = self.tracking_manager.load_standard_pose(file_path, self.config.mp_pose)
        
        # 關閉進度窗口
        progress_window.destroy()
        
        if success:
            # 啟用姿勢檢查
            self.enable_pose_checking.set(True)
            
            # 顯示確認
            messagebox.showinfo("標準姿勢已加載", 
                              "標準姿勢已成功加載。\n"
                              "啟用姿勢檢查以與當前動作進行比較。")
            self.status_var.set("標準姿勢已加載")
        else:
            messagebox.showerror("錯誤", "無法加載標準姿勢視頻")
    
    def show_trajectory_comparison(self):
        """顯示當前軌跡與標準姿勢軌跡的並排比較"""
        # 只有在有標準姿勢數據時才繼續
        if not any(x is not None for x in self.tracking_manager.standard_trajectory_images.values()):
            messagebox.showinfo("沒有標準姿勢", "請首先加載標準姿勢")
            return
            
        # 對於每個具有標準姿勢的跟踪身體部位創建一個窗口
        for landmark_name, std_img in self.tracking_manager.standard_trajectory_images.items():
            # 如果沒有標準圖像或未被跟踪，則跳過
            is_enabled, _, has_standard = self.tracking_manager.check_tracking_status(landmark_name)
            if not is_enabled or not has_standard:
                continue
                
            # 創建窗口名稱
            window_name = f"{landmark_name.replace('_', ' ').title()} 比較"
            
            # 獲取幀尺寸
            if self.video_processor.cap is not None and self.video_processor.cap.isOpened():
                h, w = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            else:
                h, w = 480, 640
                
            # 創建窗口
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(window_name, 100, 100)
            cv2.resizeWindow(window_name, w * 2, h)
    
    def start_tracking(self):
        """開始視頻跟踪"""
        # 禁用開始按鈕並啟用停止按鈕
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # 重置視頻狀態
        self.video_processor.video_paused = False
        self.video_processor.current_frame = 0
        
        # 重置停止事件
        self.stop_event.clear()
        self.video_processor.audio_stop_event.clear()
        self.tracking_manager.trajectory_stop_event.clear()
        
        # 自動啟用所有追蹤點
        self.track_all()
        # 啟用追蹤
        self.enable_tracking.set(True)
        self.tracking_manager.tracking_enabled = True
        
        # 檢查是文件還是網絡攝像頭
        if self.source_var.get() == "webcam":
            self.video_processor.video_source = 0
            self.status_var.set("從網絡攝像頭跟踪")
            # 隱藏網絡攝像頭的時間軸
            self.timeline_frame.pack_forget()
            self.pause_btn.config(state=tk.DISABLED)
        else:
            if not self.file_path.get() or not os.path.exists(self.file_path.get()):
                self.status_var.set("請選擇有效的視頻文件")
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                return
            
            self.video_processor.video_source = self.file_path.get()
            self.status_var.set(f"從文件跟踪: {os.path.basename(self.video_processor.video_source)}")
            
            # 顯示視頻文件的時間軸控件
            self.timeline_frame.pack(side=tk.TOP, fill=tk.X, before=self.status_bar)
            self.pause_btn.config(state=tk.NORMAL, text="暫停")
        
        # 打開視頻捕獲
        is_webcam = (self.source_var.get() == "webcam")
        if not self.video_processor.open_video(self.video_processor.video_source, is_webcam):
            self.status_var.set("錯誤: 無法打開視頻源")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return
            
        # 設置時間軸
        if self.source_var.get() == "file":
            fps = self.video_processor.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                duration_seconds = self.video_processor.total_frames / fps
                self.time_label.config(text=f"0:00 / {self.video_processor.format_time(duration_seconds)}")
                
                # 配置時間軸滑塊
                self.timeline_slider.config(from_=0, to=100)
                self.timeline_slider.set(0)
                
                # 如果啟用，設置音頻
                if self.enable_audio.get():
                    self.video_processor.setup_audio(self.video_processor.video_source, self.enable_audio.get())
        
        # 獲取實際的幀尺寸
        frame_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 配置畫布大小以匹配視頻尺寸
        self.canvas.config(width=frame_width, height=frame_height)
        self.drawing_canvas.config(width=frame_width, height=frame_height)
        
        # 清除現有的追蹤數據
        self.clear_tracking()
        
        # 確保追蹤幀保持可見
        if not self.tracking_frame.winfo_ismapped():
            self.tracking_frame.pack(side=tk.TOP, fill=tk.X, before=self.status_bar)
            
        # 開始軌跡追蹤（已經啟用）
        try:
            self.start_trajectory_tracking()
        except Exception as e:
            print(f"啟動軌跡追蹤時出錯: {e}")
            self.status_var.set(f"啟動軌跡追蹤時出錯: {str(e)}")
        
        # 開始處理視頻幀
        try:
            self.update_frame()
        except Exception as e:
            print(f"更新幀時出錯: {e}")
            self.status_var.set(f"更新幀時出錯: {str(e)}")
    
    def stop_tracking(self):
        """停止視頻跟踪"""
        # 設置停止事件以終止捕獲線程
        self.stop_event.set()
        self.video_processor.audio_stop_event.set()
        
        # 同時停止軌跡線程
        self.tracking_manager.trajectory_stop_event.set()
        
        # 等待線程終止
        time.sleep(0.2)
        
        # 停止音頻
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        
        # 啟用開始按鈕並禁用停止/暫停按鈕
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        
        self.status_var.set("跟踪已停止")
        
        # 釋放視頻捕獲
        self.video_processor.cleanup()
        
        # 確保追蹤幀保持可見
        if not self.tracking_frame.winfo_ismapped():
            self.tracking_frame.pack(side=tk.TOP, fill=tk.X, before=self.status_bar)
    
    def update_frame(self):
        """處理並顯示下一個視頻幀"""
        # 檢查是否應該停止
        if self.stop_event.is_set() or not self.root.winfo_exists():
            return
        
        # 如果視頻暫停，則跳過幀更新
        if self.video_processor.video_paused:
            self.root.after(33, self.update_frame)
            return
        
        ret, frame = self.video_processor.cap.read()
        if not ret:
            # 視頻文件結束或攝像頭斷開連接
            self.status_var.set("視頻結束或攝像頭斷開連接")
            self.root.after(0, self.stop_tracking)
            return
    
        # 更新當前幀計數器
        self.video_processor.current_frame = int(self.video_processor.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        self.drawing_canvas.delete("all")
    
        # 更新視頻文件的時間軸
        if self.source_var.get() == "file" and self.video_processor.total_frames > 0:
            # 更新滑塊位置
            position_percent = (self.video_processor.current_frame / self.video_processor.total_frames) * 100
            self.timeline_slider.set(position_percent)
        
            # 更新時間顯示
            fps = self.video_processor.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                current_time = self.video_processor.current_frame / fps
                total_time = self.video_processor.total_frames / fps
                self.time_label.config(text=f"{self.video_processor.format_time(current_time)} / {self.video_processor.format_time(total_time)}")
        
        try:
            # 使用 MediaPipe 處理幀
            processed_frame = self.video_processor.process_frame(
                frame, 
                self.detect_hands.get(), 
                self.detect_pose.get(), 
                self.tracking_manager.tracking_enabled,
                self.enable_pose_checking.get()
            )
        
            # 轉換為適合 tkinter 的格式
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
            # 獲取當前畫布尺寸
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
        
            # 僅在必要時和尺寸有效時調整大小
            if (canvas_width > 1 and canvas_height > 1 and 
                (img.width != canvas_width or img.height != canvas_height)):
                # 使用 BILINEAR 而不是 LANCZOS 以獲得更好的性能
                img = img.resize((canvas_width, canvas_height), Image.BILINEAR)
                
            # 保持引用以避免垃圾收集問題
            self.photo = ImageTk.PhotoImage(image=img)
    
            # 檢查 root 窗口和 canvas 是否仍然存在
            if self.root.winfo_exists() and self.canvas.winfo_exists():
                # 清除上一個圖像以防止重影
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        except Exception as e:
            print(f"處理幀時出錯: {e}")
            # 如果出錯則繼續到下一幀
    
        # 安排下一個幀更新（約 30 fps）
        if self.root.winfo_exists():  # 確保 root 窗口仍然存在
            self.root.after(33, self.update_frame)
    
    def draw_landmarks_on_canvas(self, landmarks, connections, landmark_type="pose"):
        """在 mp_drawing 畫布上繪製地標和連接"""
        # 獲取畫布尺寸
        canvas_width = self.drawing_canvas.winfo_width()
        canvas_height = self.drawing_canvas.winfo_height()
        
        # 如果畫布尺寸尚未準備好，則預設為幀尺寸
        if canvas_width <= 1 or canvas_height <= 1:
            if self.video_processor.cap is not None:
                canvas_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                canvas_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                canvas_width = 640
                canvas_height = 480
        
        # 繪製連接
        for connection in connections:
            start_idx, end_idx = connection
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]
            
            # 將地標位置轉換為畫布坐標
            start_x = int(start_landmark.x * canvas_width)
            start_y = int(start_landmark.y * canvas_height)
            end_x = int(end_landmark.x * canvas_width)
            end_y = int(end_landmark.y * canvas_height)
            
            # 繪製帶有手部與姿勢不同顏色的連接線
            line_color = "red" if landmark_type == "hand" else "blue"
            self.drawing_canvas.create_line(start_x, start_y, end_x, end_y, fill=line_color, width=2)
        
        # 將地標繪製為圓圈
        for i, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * canvas_width)
            y = int(landmark.y * canvas_height)
            
            # 對手部與姿勢地標使用不同的顏色
            fill_color = "green" if landmark_type == "hand" else "red"
            self.drawing_canvas.create_oval(x-5, y-5, x+5, y+5, fill=fill_color, outline="black")
            
            # 標記關鍵手部地標
            if landmark_type == "hand" and i in [4, 8, 12, 16, 20]:  # 拇指尖，手指尖
                landmark_names = {4: "拇指", 8: "食指", 12: "中指", 16: "無名指", 20: "小指"}
                self.drawing_canvas.create_text(x, y-15, text=landmark_names[i], fill="black")
    
    def start_trajectory_tracking(self):
        """啟動軌跡追蹤顯示"""
        if not self.tracking_manager.tracking_enabled:
            return
        
        # 啟動整合式顯示
        success = self.unified_display.start_display()
        
        if success:
            self.status_var.set("軌跡追蹤已啟用")
            
            # 更新追蹤設置
            self.update_tracking_selection()
        else:
            self.status_var.set("無法啟動軌跡追蹤")
    
    def update_tracking_selection(self):
        """根據選定的身體部位更新追蹤"""
        
        self.tracking_manager.track_elbows = self.track_elbows.get()
        self.tracking_manager.track_wrists = self.track_wrists.get()
        self.tracking_manager.track_shoulders = self.track_shoulders.get()
        self.tracking_manager.track_knees = self.track_knees.get()
        self.tracking_manager.track_ankles = self.track_ankles.get()
        
        # 如果啟用了姿勢檢查，更新比較
        if self.enable_pose_checking.get():
            self.tracking_manager.update_pose_comparison()
    
    def _ensure_tracking_thread(self, landmark_name, shape, color):
        """確保為給定的地標運行跟踪線程"""
        # 更新追蹤狀態
        # 如果線程不存在或不活動，則創建它
        if landmark_name not in self.tracking_manager.trajectory_threads or not self.tracking_manager.trajectory_threads[landmark_name].is_alive():
            # 如果需要，初始化跟踪圖像
            if landmark_name not in self.tracking_manager.trajectory_tracks or self.tracking_manager.trajectory_tracks[landmark_name] is None:
                self.tracking_manager.trajectory_tracks[landmark_name] = np.zeros(shape, dtype=np.uint8)
            
            # 創建並啟動線程
            self.tracking_manager.trajectory_threads[landmark_name] = threading.Thread(
                target=self.tracking_manager.track_landmark_thread, 
                args=(landmark_name, self.tracking_manager.trajectory_tracks[landmark_name], color))
            self.tracking_manager.trajectory_threads[landmark_name].daemon = True
            self.tracking_manager.trajectory_threads[landmark_name].start()
    
    def stop_trajectory_tracking(self):
            """停止軌跡追蹤顯示"""
            # 停止整合式顯示
            self.unified_display.stop_display()
            
            # 關閉所有 OpenCV 窗口
            cv2.destroyAllWindows()
        
    
    def on_closing(self):
        """關閉應用程序時的處理"""
        # 如果活動，停止跟踪
        self.stop_event.set()
        self.video_processor.audio_stop_event.set()
        self.tracking_manager.trajectory_stop_event.set()
                
        # 留出時間讓線程清理
        self.root.after(100, self._finalize_closing)
            
    def _finalize_closing(self):
        """最終清理和關閉應用程序"""
        # 釋放資源
        self.video_processor.cleanup()
        
        # 關閉所有 OpenCV 窗口
        cv2.destroyAllWindows()
        
        # 關閉應用程序
        self.root.destroy()
        
        
class UnifiedTrackingDisplay:
    """整合所有軌跡顯示到單一界面的類"""
    
    def __init__(self, tracking_manager, video_processor):
        """初始化整合式軌跡顯示
        
        Args:
            tracking_manager: 軌跡管理器實例
            video_processor: 視頻處理器實例
        """
        self.tracking_manager = tracking_manager
        self.video_processor = video_processor
        self.window_name = "動作軌跡追蹤系統"
        self.display_active = False
        self.stop_event = threading.Event()
        self.display_thread = None
        
        # 用於引導用戶的標準軌跡參考
        self.guide_opacity = 0.3  # 引導軌跡的透明度
        self.show_guide = True    # 是否顯示引導
        
        # 視窗佈局設定
        self.layout = {
            'main': {'x': 0, 'y': 0, 'w': 0.6, 'h': 1.0},      # 主視頻區域占60%寬度
            'info': {'x': 0.75, 'y': 0, 'w': 0.5, 'h': 0.3},   # 信息區域
            'elbows': {'x': 0.75, 'y': 0.2, 'w': 0.25, 'h': 0.15},  # 肘部區域
            'wrists': {'x': 0.75, 'y': 0.35, 'w': 0.25, 'h': 0.15},  # 手腕區域
            'shoulders': {'x': 0.75, 'y': 0.5, 'w': 0.25, 'h': 0.15},  # 肩部區域
            'knees_ankles': {'x': 0.75, 'y': 0.65, 'w': 0.25, 'h': 0.15}  # 膝蓋和腳踝區域
        }
    
    def start_display(self):
        """啟動整合式軌跡顯示"""
        if self.display_active:
            return
            
        self.stop_event.clear()
        self.display_active = True
        
        # 創建並啟動顯示線程
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        return True
    
    def stop_display(self):
        """停止軌跡顯示"""
        self.stop_event.set()
        self.display_active = False
        
        # 等待線程終止
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        # 安全關閉窗口 - 使用 try/except 來捕獲窗口不存在的錯誤
        try:
            # 檢查窗口是否存在
            window_exists = False
            all_windows = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if all_windows >= 0:
                window_exists = True
                
            if window_exists:
                cv2.destroyWindow(self.window_name)
        except:
            # 窗口可能不存在，忽略錯誤
            pass
    
    def toggle_guide(self):
        """切換是否顯示引導軌跡"""
        self.show_guide = not self.show_guide
        
    def set_guide_opacity(self, opacity):
        """設置引導軌跡的透明度
        
        Args:
            opacity: 透明度值 (0.0-1.0)
        """
        self.guide_opacity = max(0.0, min(1.0, opacity))
    
    def display_loop(self):
        """顯示循環 - 在單一窗口中顯示所有軌跡"""
        # 創建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 根據視頻尺寸設置窗口大小
        if self.video_processor.cap is not None and self.video_processor.cap.isOpened():
            width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            width, height = 1920, 1080
        
        # 計算顯示窗口尺寸 (16:9 比例)
        display_width = 1920
        display_height = 1080
        
        cv2.resizeWindow(self.window_name, display_width, display_height)
        
        # 確保我們有黑色畫布
        if not hasattr(self.tracking_manager, 'trajectory_canvas') or self.tracking_manager.trajectory_canvas is None:
            self.tracking_manager.trajectory_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 主循環
        while not self.stop_event.is_set():
            # 創建畫布
            canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            
            # 獲取當前視頻幀 (如果可用)
            current_frame = None
            if (self.video_processor.cap is not None and 
                self.video_processor.cap.isOpened() and 
                self.video_processor.last_processed_frame is not None):
                current_frame = self.video_processor.last_processed_frame.copy()
                # 調整大小以適應顯示區域
                current_frame = cv2.resize(current_frame, 
                                          (int(display_width * self.layout['main']['w']), 
                                           int(display_height * self.layout['main']['h'])))
            else:
                # 如果沒有可用的幀，創建空白區域
                current_frame = np.zeros((int(display_height * self.layout['main']['h']), 
                                         int(display_width * self.layout['main']['w']), 3), 
                                         dtype=np.uint8)
            
            # 繪製主視頻區域
            x_offset = int(display_width * self.layout['main']['x'])
            y_offset = int(display_height * self.layout['main']['y'])
            canvas[y_offset:y_offset+current_frame.shape[0], 
                   x_offset:x_offset+current_frame.shape[1]] = current_frame
            
            # 繪製信息區域
            info_x = int(display_width * self.layout['info']['x'])
            info_y = int(display_height * self.layout['info']['y'])
            info_w = int(display_width * self.layout['info']['w'])
            info_h = int(display_height * self.layout['info']['h'])
            
            # 繪製信息面板背景
            cv2.rectangle(canvas, (info_x, info_y), (info_x+info_w, info_y+info_h), (30, 30, 30), -1)
            
            # 添加標題和信息
            self._draw_text(canvas, "軌跡追蹤系統", (info_x+10, info_y+30), 0.7, (255, 255, 255), 2)
            
            # 顯示過濾設置
            filter_text = f"過濾設置: 區域={self.tracking_manager.filtering_area}, 閾值={self.tracking_manager.filtering_time}"
            self._draw_text(canvas, filter_text, (info_x+10, info_y+60), 0.5, (200, 200, 200), 1)
            
            # 顯示跟踪狀態
            if self.tracking_manager.tracking_enabled:
                status_text = "追蹤狀態: 已啟用"
                status_color = (0, 255, 0)
            else:
                status_text = "追蹤狀態: 已禁用"
                status_color = (0, 0, 255)
            self._draw_text(canvas, status_text, (info_x+10, info_y+90), 0.5, status_color, 1)
            
            # 顯示姿勢比較信息 (如果啟用)
            if self.tracking_manager.tracking_enabled and any(x is not None for x in self.tracking_manager.standard_trajectory_images.values()):
                # 計算平均分數
                active_scores = []
                for landmark_name in self.tracking_manager.pose_similarity_scores:
                    is_enabled, has_data, has_standard = self.tracking_manager.check_tracking_status(landmark_name)
                    if is_enabled and has_data and has_standard:
                        active_scores.append(self.tracking_manager.pose_similarity_scores[landmark_name])
                
                if active_scores:
                    avg_score = sum(active_scores) / len(active_scores)
                    score_text = f"姿勢分數: {avg_score:.1f}%"
                    
                    # 基於分數的顏色
                    if avg_score >= 80:
                        result_text = "正確姿勢"
                        score_color = (0, 255, 0)
                    elif avg_score >= 50:
                        result_text = "可接受姿勢"
                        score_color = (0, 255, 255)
                    else:
                        result_text = "不正確姿勢"
                        score_color = (0, 0, 255)
                    
                    self._draw_text(canvas, score_text, (info_x+10, info_y+120), 0.6, score_color, 1)
                    self._draw_text(canvas, result_text, (info_x+10, info_y+150), 0.6, score_color, 1)
            
            # 繪製身體部位區域
            self._draw_body_part_area(canvas, 'elbows', "肘部追蹤", ['right_elbow', 'left_elbow'], 
                                     self.tracking_manager.track_elbows, display_width, display_height)
            self._draw_body_part_area(canvas, 'wrists', "手腕追蹤", ['right_wrist', 'left_wrist'], 
                                     self.tracking_manager.track_wrists, display_width, display_height)
            self._draw_body_part_area(canvas, 'shoulders', "肩部追蹤", ['right_shoulder', 'left_shoulder'], 
                                     self.tracking_manager.track_shoulders, display_width, display_height)
            
            # 膝蓋和腳踝共享一個區域
            self._draw_combined_body_parts(canvas, 'knees_ankles', "下肢追蹤", 
                                         ['right_knee', 'left_knee', 'right_ankle', 'left_ankle'],
                                         self.tracking_manager.track_knees or self.tracking_manager.track_ankles, 
                                         display_width, display_height)
            
            # 在主視頻區域上繪製引導軌跡 (如果啟用)
            if self.show_guide and self.tracking_manager.tracking_enabled:
                self._draw_guide_trajectories(canvas, x_offset, y_offset, current_frame.shape[1], current_frame.shape[0])
            
            # 在主視頻區域上疊加當前軌跡
            self._draw_current_trajectories(canvas, x_offset, y_offset, current_frame.shape[1], current_frame.shape[0])
            
            # 顯示整合式畫布
            cv2.imshow(self.window_name, canvas)
            
            # 檢查按鍵
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('g'):  # 'g' 鍵切換引導顯示
                self.show_guide = not self.show_guide
            
            # 短暫睡眠以減少 CPU 使用率
            time.sleep(0.03)
        
        # 關閉窗口
        cv2.destroyWindow(self.window_name)
    
    def _draw_text(self, img, text, pos, scale, color, thickness=1):
        """在圖像上繪製文字，支持中文
        
        Args:
            img: 目標圖像
            text: 要顯示的文字
            pos: 位置 (x, y)
            scale: 字體尺寸
            color: 顏色 (B, G, R)
            thickness: 線條粗細
        """
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    
    def _draw_body_part_area(self, canvas, area_key, title, landmarks, is_tracking, display_width, display_height):
        """繪製身體部位區域
        
        Args:
            canvas: 目標畫布
            area_key: 區域鍵名 (來自 self.layout)
            title: 區域標題
            landmarks: 相關地標名稱列表
            is_tracking: 是否正在追蹤
            display_width: 顯示寬度
            display_height: 顯示高度
        """
        area = self.layout[area_key]
        x = int(display_width * area['x'])
        y = int(display_height * area['y'])
        w = int(display_width * area['w'])
        h = int(display_height * area['h'])
        
        # 繪製背景
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (70, 70, 70), 1)
        
        # 繪製標題
        self._draw_text(canvas, title, (x+10, y+20), 0.5, (255, 255, 255), 1)
        
        if not is_tracking:
            # 如果未跟踪，顯示未跟踪信息
            self._draw_text(canvas, "未跟踪", (x+w//2-30, y+h//2), 0.7, (100, 100, 255), 1)
            return
        
        # 繪製具體地標信息
        y_offset = y + 40
        for landmark_name in landmarks:
            display_name = landmark_name.replace('_', ' ').title()
            if "Right" in display_name:
                display_name = "右" + display_name.replace("Right ", "")
            if "Left" in display_name:
                display_name = "左" + display_name.replace("Left ", "")
            
            # 獲取此地標的顏色
            color = self.tracking_manager.config.color_map.get(landmark_name, (255, 255, 255))
            
            # 繪製顏色樣本
            cv2.rectangle(canvas, (x+10, y_offset-10), (x+20, y_offset), color, -1)
            
            # 繪製地標名稱
            self._draw_text(canvas, display_name, (x+25, y_offset), 0.4, (255, 255, 255), 1)
            
            # 獲取點數
            raw_count = len(self.tracking_manager.tracking_points[landmark_name])
            filtered_count = len(self.tracking_manager.filtered_tracking_points[landmark_name])
            
            # 繪製點數信息
            point_info = f"點數: {filtered_count}/{raw_count}"
            self._draw_text(canvas, point_info, (x+w-100, y_offset), 0.4, (200, 200, 200), 1)
            
            # 如果有姿勢比較分數，顯示
            is_enabled, has_data, has_standard = self.tracking_manager.check_tracking_status(landmark_name)
            if has_data and has_standard:
                score = self.tracking_manager.pose_similarity_scores[landmark_name]
                score_text = f"{score:.1f}%"
                
                # 基於分數的顏色
                if score >= 80:
                    score_color = (0, 255, 0)  # 綠色
                elif score >= 50:
                    score_color = (0, 255, 255)  # 黃色
                else:
                    score_color = (0, 0, 255)  # 紅色
                
                # 繪製分數
                self._draw_text(canvas, score_text, (x+w-50, y_offset), 0.4, score_color, 1)
            
            y_offset += 20
            
            # 在區域下部繪製小型軌跡預覽
            preview_height = 50
            if y_offset + preview_height < y + h:
                # 獲取軌跡點
                points = self.tracking_manager.filtered_tracking_points[landmark_name] if self.tracking_manager.filtering_enabled else self.tracking_manager.tracking_points[landmark_name]
                
                if len(points) > 1:
                    # 創建預覽區
                    preview_x = x + 10
                    preview_y = y_offset
                    preview_width = w - 20
                    
                    # 繪製預覽區域邊框
                    cv2.rectangle(canvas, (preview_x, preview_y), 
                                 (preview_x + preview_width, preview_y + preview_height), 
                                 (50, 50, 50), 1)
                    
                    # 縮放點以適應預覽區域
                    for point in points:
                        # 獲取原始視頻尺寸進行縮放計算
                        orig_width = self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        orig_height = self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        
                        # 縮放點到預覽區域
                        scaled_x = preview_x + int(point[0] / orig_width * preview_width)
                        scaled_y = preview_y + int(point[1] / orig_height * preview_height)
                        
                        # 確保點在預覽區域內
                        scaled_x = min(max(scaled_x, preview_x), preview_x + preview_width)
                        scaled_y = min(max(scaled_y, preview_y), preview_y + preview_height)
                        
                        # 繪製點
                        cv2.circle(canvas, (scaled_x, scaled_y), 1, color, -1)
    
    def _draw_combined_body_parts(self, canvas, area_key, title, landmarks, is_tracking, display_width, display_height):
        """繪製組合的身體部位區域 (用於膝蓋和腳踝)
        
        Args:
            canvas: 目標畫布
            area_key: 區域鍵名 (來自 self.layout)
            title: 區域標題
            landmarks: 相關地標名稱列表
            is_tracking: 是否正在追蹤
            display_width: 顯示寬度
            display_height: 顯示高度
        """
        area = self.layout[area_key]
        x = int(display_width * area['x'])
        y = int(display_height * area['y'])
        w = int(display_width * area['w'])
        h = int(display_height * area['h'])
        
        # 繪製背景
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (70, 70, 70), 1)
        
        # 繪製標題
        self._draw_text(canvas, title, (x+10, y+20), 0.5, (255, 255, 255), 1)
        
        if not is_tracking:
            # 如果未跟踪，顯示未跟踪信息
            self._draw_text(canvas, "未跟踪", (x+w//2-30, y+h//2), 0.7, (100, 100, 255), 1)
            return
        
        # 繪製肢體部分信息 - 將膝蓋和腳踝分為兩行
        knee_y = y + 40
        ankle_y = y + 80
        
        # 繪製膝蓋標籤
        if self.tracking_manager.track_knees:
            self._draw_text(canvas, "膝蓋:", (x+10, knee_y), 0.5, (255, 255, 255), 1)
            
            # 繪製右膝和左膝
            right_knee = "right_knee"
            left_knee = "left_knee"
            
            # 獲取顏色
            right_color = self.tracking_manager.config.color_map.get(right_knee, (255, 255, 255))
            left_color = self.tracking_manager.config.color_map.get(left_knee, (255, 255, 255))
            
            # 繪製右膝
            cv2.rectangle(canvas, (x+70, knee_y-10), (x+80, knee_y), right_color, -1)
            self._draw_text(canvas, "右膝", (x+85, knee_y), 0.4, (255, 255, 255), 1)
            
            # 繪製左膝
            cv2.rectangle(canvas, (x+150, knee_y-10), (x+160, knee_y), left_color, -1)
            self._draw_text(canvas, "左膝", (x+165, knee_y), 0.4, (255, 255, 255), 1)
        
        # 繪製腳踝標籤
        if self.tracking_manager.track_ankles:
            self._draw_text(canvas, "腳踝:", (x+10, ankle_y), 0.5, (255, 255, 255), 1)
            
            # 繪製右踝和左踝
            right_ankle = "right_ankle"
            left_ankle = "left_ankle"
            
            # 獲取顏色
            right_color = self.tracking_manager.config.color_map.get(right_ankle, (255, 255, 255))
            left_color = self.tracking_manager.config.color_map.get(left_ankle, (255, 255, 255))
            
            # 繪製右踝
            cv2.rectangle(canvas, (x+70, ankle_y-10), (x+80, ankle_y), right_color, -1)
            self._draw_text(canvas, "右踝", (x+85, ankle_y), 0.4, (255, 255, 255), 1)
            
            # 繪製左踝
            cv2.rectangle(canvas, (x+150, ankle_y-10), (x+160, ankle_y), left_color, -1)
            self._draw_text(canvas, "左踝", (x+165, ankle_y), 0.4, (255, 255, 255), 1)
        
        # 在區域下部繪製小型軌跡預覽
        preview_y = ankle_y + 20
        preview_height = 40
        if preview_y + preview_height < y + h:
            preview_x = x + 10
            preview_width = w - 20
            
            # 繪製預覽區域邊框
            cv2.rectangle(canvas, (preview_x, preview_y), 
                         (preview_x + preview_width, preview_y + preview_height), 
                         (50, 50, 50), 1)
            
            # 繪製所有啟用的軌跡
            for landmark_name in landmarks:
                if (('knee' in landmark_name and self.tracking_manager.track_knees) or
                    ('ankle' in landmark_name and self.tracking_manager.track_ankles)):
                    
                    # 獲取此地標的顏色
                    color = self.tracking_manager.config.color_map.get(landmark_name, (255, 255, 255))
                    
                    # 獲取軌跡點
                    points = self.tracking_manager.filtered_tracking_points[landmark_name] if self.tracking_manager.filtering_enabled else self.tracking_manager.tracking_points[landmark_name]
                    
                    if len(points) > 1:
                        # 縮放點以適應預覽區域
                        for point in points:
                            # 獲取原始視頻尺寸進行縮放計算
                            orig_width = self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            orig_height = self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            
                            # 縮放點到預覽區域
                            scaled_x = preview_x + int(point[0] / orig_width * preview_width)
                            scaled_y = preview_y + int(point[1] / orig_height * preview_height)
                            
                            # 確保點在預覽區域內
                            scaled_x = min(max(scaled_x, preview_x), preview_x + preview_width)
                            scaled_y = min(max(scaled_y, preview_y), preview_y + preview_height)
                            
                            # 繪製點
                            cv2.circle(canvas, (scaled_x, scaled_y), 1, color, -1)
    
    def _draw_guide_trajectories(self, canvas, x_offset, y_offset, width, height):
        """在主視頻區域繪製引導軌跡
        
        Args:
            canvas: 目標畫布
            x_offset: X 偏移量
            y_offset: Y 偏移量
            width: 區域寬度
            height: 區域高度
        """
        # 只有當標準姿勢已加載時才繪製引導
        if not any(x is not None for x in self.tracking_manager.standard_trajectory_images.values()):
            return
        
        # 遍歷所有身體部位
        for landmark_name, std_img in self.tracking_manager.standard_trajectory_images.items():
            # 檢查是否正在跟踪此部位
            is_enabled, _, has_standard = self.tracking_manager.check_tracking_status(landmark_name)
            if not is_enabled or not has_standard or std_img is None:
                continue
            
            # 獲取此地標的顏色
            color = self.tracking_manager.config.color_map.get(landmark_name, (255, 255, 255))
            
            # 調整顏色的不透明度以顯示為引導
            guide_color = tuple(int(c * self.guide_opacity) for c in color)
            
            # 將標準姿勢圖像調整為視頻區域大小
            resized_std = cv2.resize(std_img, (width, height))
            
            # 創建掩碼
            mask = cv2.cvtColor(resized_std, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            
            # 創建引導軌跡的彩色版本
            guide_img = np.zeros((height, width, 3), dtype=np.uint8)
            guide_img[:] = guide_color
            guide_img = cv2.bitwise_and(guide_img, guide_img, mask=mask)
            
            # 將引導軌跡添加到主畫布
            guide_region = canvas[y_offset:y_offset+height, x_offset:x_offset+width]
            canvas[y_offset:y_offset+height, x_offset:x_offset+width] = cv2.addWeighted(
                guide_region, 1.0, guide_img, 1.0, 0)
    
    def _draw_current_trajectories(self, canvas, x_offset, y_offset, width, height):
        """在主視頻區域繪製當前軌跡
        
        Args:
            canvas: 目標畫布
            x_offset: X 偏移量
            y_offset: Y 偏移量
            width: 區域寬度
            height: 區域高度
        """
        if not self.tracking_manager.tracking_enabled:
            return
        
        # 獲取要為哪些身體部位繪製軌跡
        active_landmarks = []
        if self.tracking_manager.track_elbows:
            active_landmarks.extend(['right_elbow', 'left_elbow'])
        if self.tracking_manager.track_wrists:
            active_landmarks.extend(['right_wrist', 'left_wrist'])
        if self.tracking_manager.track_shoulders:
            active_landmarks.extend(['right_shoulder', 'left_shoulder'])
        if self.tracking_manager.track_knees:
            active_landmarks.extend(['right_knee', 'left_knee'])
        if self.tracking_manager.track_ankles:
            active_landmarks.extend(['right_ankle', 'left_ankle'])
        
        # 選擇要繪製的點 (過濾或原始)
        points_dict = self.tracking_manager.filtered_tracking_points if self.tracking_manager.filtering_enabled else self.tracking_manager.tracking_points
        
        # 為每個活動地標繪製軌跡
        for landmark_name in active_landmarks:
            if len(points_dict[landmark_name]) > 0:
                # 獲取此地標的顏色
                color = self.tracking_manager.config.color_map.get(landmark_name, (255, 255, 255))
                
                # 確定點大小
                point_size = 3 if self.tracking_manager.filtering_enabled else 2
                
                # 獲取原始尺寸進行縮放計算
                if self.video_processor.cap is not None and self.video_processor.cap.isOpened():
                    orig_width = self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    orig_height = self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                else:
                    orig_width, orig_height = 640, 480
                
                # 繪製軌跡中的所有點
                for point in points_dict[landmark_name]:
                    # 縮放點以適應顯示區域
                    scaled_x = x_offset + int(point[0] / orig_width * width)
                    scaled_y = y_offset + int(point[1] / orig_height * height)
                    
                    # 確保點在顯示區域內
                    scaled_x = min(max(scaled_x, x_offset), x_offset + width - 1)
                    scaled_y = min(max(scaled_y, y_offset), y_offset + height - 1)
                    
                    # 繪製點
                    cv2.circle(canvas, (scaled_x, scaled_y), point_size, color, -1)



# 主應用程序類：整合所有組件
class MotionTrackingApp:
    """動作追蹤應用程序的主類"""
    
    def __init__(self, root):
        """初始化應用程序
        
        Args:
            root: Tkinter 根窗口
        """
        # 創建組件
        self.config = AppConfig()
        self.tracking_manager = TrackingManager(self.config)
        self.video_processor = VideoProcessor(self.config, self.tracking_manager)
        self.ui = MotionTrackingUI(root, self.config, self.video_processor, self.tracking_manager)
        
        # 設置適當的窗口關閉處理
        root.protocol("WM_DELETE_WINDOW", self.ui.on_closing)
        

# 主入口點
if __name__ == "__main__":
    root = tk.Tk()
    app = MotionTrackingApp(root)
    root.mainloop()