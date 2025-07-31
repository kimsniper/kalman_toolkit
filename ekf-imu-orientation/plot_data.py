import serial
import queue
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Serial settings
PORT = '/dev/ttyUSB0'
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=1)

# Shared queues
out_queue = queue.Queue()

# Data storage
cf_roll_data = []
cf_pitch_data = []
cf_yaw_data = []
ekf_roll_data = []
ekf_pitch_data = []
ekf_yaw_data = []
max_points = 100

def rotation_matrix(roll, pitch, yaw):
    """Generate rotation matrix from roll, pitch, yaw in degrees"""
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx


class OrientationVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("THEMIS: Orientation Estimation Comparison")

        # Main frames
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Complementary filter alpha
        self.alpha_var = tk.DoubleVar(value=0.98)

        # Alpha slider
        self.add_alpha_slider("Complementary Filter α (Gyro Weight)", 0.0, 1.0, self.alpha_var, self.set_alpha)

        # Terminal
        ttk.Label(self.control_frame, text="Terminal Output").pack(pady=(20, 0))
        self.terminal = ScrolledText(self.control_frame, width=40, height=20, state=tk.DISABLED, font=("Courier", 8))
        self.terminal.pack(fill=tk.BOTH, expand=True, pady=5)

        # Plot setup - 2D & 3D
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_cf = self.fig.add_subplot(221)
        self.ax_ekf = self.fig.add_subplot(222)
        self.ax_cf_3d = self.fig.add_subplot(223, projection='3d')
        self.ax_ekf_3d = self.fig.add_subplot(224, projection='3d')
        self.fig.tight_layout(pad=3.0)

        # Setup lines
        self.setup_plots()

        # Setup 3D boxes
        self.cf_box = self.create_box(self.ax_cf_3d, "Complimentary Filter")
        self.ekf_box = self.create_box(self.ax_ekf_3d, "Extended Kalman Filter")

        # Canvas and animation
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)

        # Request alpha value from MCU after startup
        self.root.after(2000, self.query_mcu_gains)

    def create_box(self, ax, title):
        """Initialize a 3D rectangular box"""
        # Rectangular box coordinates
        x_range = [-1.0, 1.0]   # Wider X-axis
        y_range = [-0.5, 0.5]   # Same Y-axis
        z_range = [-0.25, 0.25] # Half height on Z-axis

        X, Y = np.meshgrid(x_range, y_range)
        ones_XY = np.ones_like(X)
        Xz, Z = np.meshgrid(x_range, z_range)
        ones_XZ = np.ones_like(Xz)
        Yz, Z2 = np.meshgrid(y_range, z_range)
        ones_YZ = np.ones_like(Yz)

        faces = [
            (X, Y, z_range[1] * ones_XY),   # Top
            (X, Y, z_range[0] * ones_XY),   # Bottom
            (X, y_range[1] * ones_XY, Z2),  # Front
            (X, y_range[0] * ones_XY, Z2),  # Back
            (x_range[1] * ones_YZ, Yz, Z2), # Right
            (x_range[0] * ones_YZ, Yz, Z2)  # Left
        ]

        # Set plot limits
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title(title)
        ax.view_init(30, 45)

        polys = []
        for Xf, Yf, Zf in faces:
            polys.append(ax.plot_surface(Xf, Yf, Zf, color='cyan', alpha=0.5))

        return faces

    def update_box(self, ax, box_faces, roll, pitch, yaw):
        R = rotation_matrix(-pitch, roll, yaw)
        ax.collections.clear()  # Clear previous surfaces

        for X, Y, Z in box_faces:
            coords = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
            rotated = R @ coords
            Xr = rotated[0, :].reshape(X.shape)
            Yr = rotated[1, :].reshape(Y.shape)
            Zr = rotated[2, :].reshape(Z.shape)
            ax.plot_surface(Xr, Yr, Zr, color='cyan', alpha=0.5)

    def setup_plots(self):
        # Complimentary Filter lines
        self.cf_roll_line, = self.ax_cf.plot([], [], color='blue', label='Roll (°)')
        self.cf_pitch_line, = self.ax_cf.plot([], [], color='green', label='Pitch (°)')
        self.cf_yaw_line, = self.ax_cf.plot([], [], color='red', label='Yaw (°)')
        self.ax_cf.set_ylim(-180, 180)
        self.ax_cf.set_ylabel("Angle (°)")
        self.ax_cf.set_title("Complementary Filter")
        self.ax_cf.grid(True)
        self.ax_cf.legend()

        # Extended Kalman Filter lines
        self.ekf_roll_line, = self.ax_ekf.plot([], [], color='blue', linestyle='--', label='Roll (°)')
        self.ekf_pitch_line, = self.ax_ekf.plot([], [], color='green', linestyle='--', label='Pitch (°)')
        self.ekf_yaw_line, = self.ax_ekf.plot([], [], color='red', linestyle='--', label='Yaw (°)')
        self.ax_ekf.set_ylim(-180, 180)
        self.ax_ekf.set_ylabel("Angle (°)")
        self.ax_ekf.set_title("Extended Kalman Filter")
        self.ax_ekf.grid(True)
        self.ax_ekf.legend()

    def add_alpha_slider(self, label, frm, to, var, command):
        frame = tk.Frame(self.control_frame)
        frame.pack(pady=(10, 0), anchor="w")

        ttk.Label(frame, text=label).pack(anchor="w")

        slider = ttk.Scale(frame, from_=frm, to=to, orient=tk.HORIZONTAL, length=120, variable=var, command=lambda v: self.update_alpha_labels(v, command))
        slider.pack(anchor="w")

        self.alpha_gyro_label = ttk.Label(frame, text=f"Gyro (α): {var.get():.3f}")
        self.alpha_gyro_label.pack(anchor="w")
        self.alpha_accel_label = ttk.Label(frame, text=f"Accel (1−α): {1 - var.get():.3f}")
        self.alpha_accel_label.pack(anchor="w")

    def update_alpha_labels(self, val, command):
        alpha = float(val)
        self.alpha_gyro_label.config(text=f"Gyro (α): {alpha:.3f}")
        self.alpha_accel_label.config(text=f"Accel (1−α): {1 - alpha:.3f}")
        command(alpha)

    def set_alpha(self, val):
        out_queue.put(f"a {val:.3f}\n")

    def query_mcu_gains(self):
        out_queue.put("get\n")

    def log_terminal(self, message):
        self.terminal.config(state=tk.NORMAL)
        self.terminal.insert(tk.END, message + '\n')
        if int(self.terminal.index('end-1c').split('.')[0]) > 200:
            self.terminal.delete('1.0', '2.0')
        self.terminal.see(tk.END)
        self.terminal.config(state=tk.DISABLED)

    def update_plot(self, _):
        while not out_queue.empty():
            cmd = out_queue.get()
            ser.write(cmd.encode())
            self.log_terminal(f"→ {cmd.strip()}")

        if ser.in_waiting:
            try:
                line = ser.readline().decode().strip()
                if not line:
                    return

                self.log_terminal(line)

                if line.startswith("Gains"):
                    _, alpha = line.split(',')
                    self.alpha_var.set(float(alpha))
                    self.alpha_gyro_label.config(text=f"Gyro (α): {float(alpha):.3f}")
                    self.alpha_accel_label.config(text=f"Accel (1−α): {1 - float(alpha):.3f}")
                    return

                if line.startswith("CF,"):
                    parts = line.split(',')
                    if len(parts) == 8:
                        cf_roll_data.append(float(parts[1]))
                        cf_pitch_data.append(float(parts[2]))
                        cf_yaw_data.append(float(parts[3]))
                        ekf_roll_data.append(float(parts[5]))
                        ekf_pitch_data.append(float(parts[6]))
                        ekf_yaw_data.append(float(parts[7]))

                        for lst in [cf_roll_data, cf_pitch_data, cf_yaw_data,
                                    ekf_roll_data, ekf_pitch_data, ekf_yaw_data]:
                            if len(lst) > max_points:
                                lst.pop(0)

                        # Update 2D plots
                        self.update_plot_lines()

                        # Update 3D boxes
                        if cf_roll_data:
                            self.update_box(self.ax_cf_3d, self.cf_box,
                                            cf_roll_data[-1], cf_pitch_data[-1], cf_yaw_data[-1])

                        if ekf_roll_data:
                            self.update_box(self.ax_ekf_3d, self.ekf_box,
                                            ekf_roll_data[-1], ekf_pitch_data[-1], ekf_yaw_data[-1])

                        self.canvas.draw()
            except Exception as e:
                self.log_terminal(f"Parse error: {e}")

    def update_plot_lines(self):
        # Complimentary Filter lines
        xdata = range(len(cf_roll_data))
        self.cf_roll_line.set_data(xdata, cf_roll_data)
        self.cf_pitch_line.set_data(xdata, cf_pitch_data)
        self.cf_yaw_line.set_data(xdata, cf_yaw_data)
        self.ax_cf.set_xlim(0, max(len(cf_roll_data), 10))

        # Extended Kalman Filter lines
        xdata2 = range(len(ekf_roll_data))
        self.ekf_roll_line.set_data(xdata2, ekf_roll_data)
        self.ekf_pitch_line.set_data(xdata2, ekf_pitch_data)
        self.ekf_yaw_line.set_data(xdata2, ekf_yaw_data)
        self.ax_ekf.set_xlim(0, max(len(ekf_roll_data), 10))


# Run
if __name__ == "__main__":
    print(f"Connected to {PORT} at {BAUD} baud")
    root = tk.Tk()
    app = OrientationVisualizer(root)
    root.mainloop()
