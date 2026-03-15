import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pyautogui
import keyboard
import time
import cv2
import numpy as np
import math  # 新增 math 库用于计算两点间距离
from PIL import Image, ImageTk


class AutoSketchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("杀戮尖塔2 自动素描机器人")
        self.root.geometry("500x720")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        self.is_running = False
        self.is_paused = False       # ★ 新增：暂停状态标志
        self.stop_requested = False
        self.image_path = None
        self.contours = []
        self.image_size = (0, 0)

        self.setup_ui()

        keyboard.add_hotkey('F9', self.on_hotkey_start)
        keyboard.add_hotkey('F8', self.on_hotkey_pause)  # ★ 新增：F8暂停快捷键
        keyboard.add_hotkey('F10', self.on_hotkey_stop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Microsoft YaHei", 9))
        style.configure("TButton", font=("Microsoft YaHei", 10))

        # --- 第一部分：图片加载与预览 ---
        img_frame = ttk.LabelFrame(self.root, text=" 图片设置 ")
        img_frame.pack(fill="x", padx=15, pady=5)

        ttk.Button(img_frame, text="📁 选择并上传图片", command=self.load_image).pack(pady=5)

        self.canvas = tk.Canvas(img_frame, width=300, height=200, bg="black")
        self.canvas.pack(pady=5)
        self.preview_label = ttk.Label(img_frame, text="等待上传图片...", foreground="gray")
        self.preview_label.pack()

        detail_frame = ttk.Frame(img_frame)
        detail_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(detail_frame, text="线条细节(阈值):").pack(side="left")
        self.threshold_var = tk.IntVar(value=100)

        # ★ 直接输入数值的微调框（支持跳跃式修改）
        thresh_spin = ttk.Spinbox(detail_frame, from_=10, to=200, textvariable=self.threshold_var, width=5,
                                  command=self.update_preview)
        thresh_spin.pack(side="right", padx=(0, 5))
        thresh_spin.bind('<Return>', self.update_preview)  # 绑定回车键确认修改

        # ★ 优化：移除 command 实时触发，改为绑定鼠标松开事件，彻底解决拖动卡顿
        thresh_scale = ttk.Scale(detail_frame, from_=10, to=200, variable=self.threshold_var, orient="horizontal")
        thresh_scale.pack(side="left", fill="x", expand=True, padx=5)
        thresh_scale.bind("<ButtonRelease-1>", self.update_preview)

        # --- 第二部分：绘图区域设置 (红框参数) ---
        area_frame = ttk.LabelFrame(self.root, text=" 绘制区域 (红框范围 %) ")
        area_frame.pack(fill="x", padx=15, pady=5)

        ttk.Label(area_frame, text="左侧避开%:").grid(row=0, column=0, padx=5, pady=5)
        self.left_margin = tk.IntVar(value=16)
        ttk.Spinbox(area_frame, from_=0, to=50, textvariable=self.left_margin, width=5).grid(row=0, column=1)

        ttk.Label(area_frame, text="右侧避开%:").grid(row=0, column=2, padx=5, pady=5)
        self.right_margin = tk.IntVar(value=19)
        ttk.Spinbox(area_frame, from_=0, to=50, textvariable=self.right_margin, width=5).grid(row=0, column=3)

        ttk.Label(area_frame, text="顶部避开%:").grid(row=1, column=0, padx=5, pady=5)
        self.top_margin = tk.IntVar(value=9)
        ttk.Spinbox(area_frame, from_=0, to=50, textvariable=self.top_margin, width=5).grid(row=1, column=1)

        ttk.Label(area_frame, text="底部避开%:").grid(row=1, column=2, padx=5, pady=5)
        self.bottom_margin = tk.IntVar(value=7)
        ttk.Spinbox(area_frame, from_=0, to=50, textvariable=self.bottom_margin, width=5).grid(row=1, column=3)

        # --- 第三部分：绘制参数 ---
        draw_frame = ttk.LabelFrame(self.root, text=" 绘制参数 (防乱线设置) ")
        draw_frame.pack(fill="x", padx=15, pady=5)

        ttk.Label(draw_frame, text="拖拽步长(像素):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.drag_step_var = tk.IntVar(value=5)  # 默认每5个像素发送一次坐标，非常平滑
        ttk.Spinbox(draw_frame, from_=1, to=50, textvariable=self.drag_step_var, width=8).grid(row=0, column=1)

        # ★ 彻底解除限制，允许设为 0.00，追求极致速度
        ttk.Label(draw_frame, text="起落笔延迟 (秒):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.delay_var = tk.DoubleVar(value=0.00)
        ttk.Spinbox(draw_frame, from_=0.00, to=0.2, increment=0.01, textvariable=self.delay_var, width=8).grid(row=1,
                                                                                                               column=1)

        ttk.Label(draw_frame, text="使用按键:").grid(row=0, column=2, padx=10, pady=5, sticky="e")
        self.btn_var = tk.StringVar(value="right")
        ttk.Combobox(draw_frame, textvariable=self.btn_var, values=["left", "right"], width=5, state="readonly").grid(
            row=0, column=3)

        # --- 状态与控制 ---
        ttk.Label(self.root, text="F9: 开始 | F8: 暂停/继续 | F10: 停止", foreground="red",
                  font=("Microsoft YaHei", 10, "bold")).pack(pady=5)
        self.status_label = ttk.Label(self.root, text="当前状态: 请先上传图片", font=("Microsoft YaHei", 11, "bold"),
                                      foreground="blue")
        self.status_label.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image_path = file_path
            self.update_preview()
            self.status_label.config(text="当前状态: 准备就绪", foreground="green")

    def update_preview(self, *args):
        if not self.image_path:
            return

        img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return

        self.image_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh1 = self.threshold_var.get()
        thresh2 = thresh1 * 2
        edges = cv2.Canny(gray, thresh1, thresh2)

        # ★★★ 核心修复：将 RETR_EXTERNAL 改为了 RETR_LIST ★★★
        # 之前的 EXTERNAL 只会获取最外层轮廓，导致文字内部或包裹在里面的线条全部丢失！
        # LIST 会提取所有的线条（包括内部细节）
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        raw_contours = []
        for c in contours:
            epsilon = 0.002 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, False)
            if len(approx) > 1:
                raw_contours.append(approx)

        # 路径优化算法 (最近邻启发式算法)，减少乱飞的横线
        self.contours = []
        if raw_contours:
            unvisited = list(raw_contours)
            current_point = (0, 0)

            while unvisited:
                best_idx = 0
                min_dist = float('inf')
                reverse_best = False

                for i, contour in enumerate(unvisited):
                    start_pt = contour[0][0]
                    end_pt = contour[-1][0]

                    dist_to_start = (start_pt[0] - current_point[0]) ** 2 + (start_pt[1] - current_point[1]) ** 2
                    dist_to_end = (end_pt[0] - current_point[0]) ** 2 + (end_pt[1] - current_point[1]) ** 2

                    if dist_to_start < min_dist:
                        min_dist = dist_to_start
                        best_idx = i
                        reverse_best = False
                    if dist_to_end < min_dist:
                        min_dist = dist_to_end
                        best_idx = i
                        reverse_best = True

                best_contour = unvisited.pop(best_idx)
                if reverse_best:
                    best_contour = best_contour[::-1]

                self.contours.append(best_contour)
                current_point = best_contour[-1][0]

        preview_img = Image.fromarray(edges)
        preview_img.thumbnail((300, 200), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(preview_img)

        self.canvas.delete("all")
        self.canvas.create_image(150, 100, anchor="center", image=self.tk_img)
        self.preview_label.config(text=f"解析完成：优化并提取 {len(self.contours)} 条路径段")

    def on_hotkey_start(self):
        if not self.is_running and self.contours:
            self.start_drawing()

    # ★ 新增：处理暂停指令
    def on_hotkey_pause(self):
        if self.is_running and not self.stop_requested:
            self.is_paused = not self.is_paused

    def on_hotkey_stop(self):
        if self.is_running:
            self.stop_requested = True
            self.is_paused = False  # 解除暂停阻塞，让线程安全退出
            self.status_label.config(text="正在强制停止...", foreground="orange")

    def start_drawing(self):
        self.is_running = True
        self.is_paused = False
        self.stop_requested = False
        self.status_label.config(text="正在作画中! (F8暂停 | F10停止)", foreground="red")
        threading.Thread(target=self.draw_task, daemon=True).start()

    def draw_task(self):
        pyautogui.PAUSE = 0
        screen_w, screen_h = pyautogui.size()

        box_x = screen_w * (self.left_margin.get() / 100.0)
        box_y = screen_h * (self.top_margin.get() / 100.0)
        box_w = screen_w * (1 - self.left_margin.get() / 100.0 - self.right_margin.get() / 100.0)
        box_h = screen_h * (1 - self.top_margin.get() / 100.0 - self.bottom_margin.get() / 100.0)

        img_w, img_h = self.image_size
        if img_w == 0 or img_h == 0:
            self.root.after(0, self.reset_ui)
            return

        scale = min(box_w / img_w, box_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        offset_x = box_x + (box_w - draw_w) / 2
        offset_y = box_y + (box_h - draw_h) / 2

        step_px = max(1, self.drag_step_var.get())
        # ★ 移除强制最低延迟，完全听从用户设置
        delay = max(0.00, self.delay_var.get())
        mouse_btn = self.btn_var.get()

        # ★ 新增核心逻辑：检查是否触发了暂停
        def check_pause(target_x, target_y, should_be_down):
            if self.is_paused:
                # 暂停时：抬起鼠标释放控制权
                pyautogui.mouseUp(button=mouse_btn)
                self.root.after(0, lambda: self.status_label.config(text="当前状态: 已暂停 (按F8继续)", foreground="purple"))
                
                # 阻塞循环直到取消暂停或按下停止
                while self.is_paused and not self.stop_requested:
                    time.sleep(0.1)
                    
                if self.stop_requested:
                    return False
                
                # 继续时：恢复UI，鼠标移回刚才的点，重新按下按键
                self.root.after(0, lambda: self.status_label.config(text="正在作画中! (F8暂停 | F10停止)", foreground="red"))
                pyautogui.moveTo(target_x, target_y)
                time.sleep(delay)
                if should_be_down:
                    pyautogui.mouseDown(button=mouse_btn)
                    time.sleep(delay)
            return True

        try:
            pyautogui.mouseUp(button=mouse_btn)
            time.sleep(delay)

            for contour in self.contours:
                if self.stop_requested: break

                first_point = True
                prev_x, prev_y = 0, 0
                for point in contour:
                    if self.stop_requested: break

                    img_x, img_y = point[0]
                    screen_x = int(offset_x + img_x * scale)
                    screen_y = int(offset_y + img_y * scale)

                    if first_point:
                        # 检查暂停状态 (移动到下一笔前)
                        if not check_pause(screen_x, screen_y, False): break

                        pyautogui.mouseUp(button=mouse_btn)
                        pyautogui.moveTo(screen_x, screen_y)
                        time.sleep(delay)

                        pyautogui.mouseDown(button=mouse_btn)
                        time.sleep(delay)
                        first_point = False
                        prev_x, prev_y = screen_x, screen_y
                    else:
                        dist = math.hypot(screen_x - prev_x, screen_y - prev_y)
                        steps = int(dist / step_px)
                        if steps < 1: steps = 1

                        for i in range(1, steps + 1):
                            if self.stop_requested: break
                            nx = prev_x + (screen_x - prev_x) * (i / steps)
                            ny = prev_y + (screen_y - prev_y) * (i / steps)
                            
                            # 检查暂停状态 (拖拽中途)
                            if not check_pause(int(nx), int(ny), True): break

                            pyautogui.moveTo(int(nx), int(ny))

                            # 仅在需要时进行微小休眠，保证游戏引擎捕捉拖拽轨迹
                            if self.drag_step_var.get() < 20:
                                time.sleep(0.001)

                        prev_x, prev_y = screen_x, screen_y

                pyautogui.mouseUp(button=mouse_btn)
                time.sleep(delay)

        finally:
            pyautogui.mouseUp(button=mouse_btn)
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.is_running = False
        self.is_paused = False
        self.status_label.config(text="当前状态: 待机中...", foreground="blue")

    def on_closing(self):
        self.stop_requested = True
        keyboard.unhook_all()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AutoSketchApp(root)
    root.mainloop()