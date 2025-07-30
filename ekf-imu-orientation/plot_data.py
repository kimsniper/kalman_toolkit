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

        # Plot setup - 2 subplots
        self.fig, (self.ax_cf, self.ax_ekf) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
        self.fig.tight_layout(pad=3.0)

        # Setup lines once (no clearing)
        self.setup_plots()

        # Canvas and animation
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)

        # Request alpha value from MCU after startup
        self.root.after(2000, self.query_mcu_gains)

    def setup_plots(self):
        # CF lines
        self.cf_roll_line, = self.ax_cf.plot([], [], color='blue', label='Roll (°)')
        self.cf_pitch_line, = self.ax_cf.plot([], [], color='green', label='Pitch (°)')
        self.cf_yaw_line, = self.ax_cf.plot([], [], color='red', label='Yaw (°)')
        self.cf_labels = {
            'roll': self.ax_cf.text(0, 0, '', color='blue', fontsize=9),
            'pitch': self.ax_cf.text(0, 0, '', color='green', fontsize=9),
            'yaw': self.ax_cf.text(0, 0, '', color='red', fontsize=9)
        }
        self.ax_cf.axhline(y=0, color='gray', linestyle='--')
        self.ax_cf.set_ylim(-180, 180)
        self.ax_cf.set_ylabel("Angle (°)")
        self.ax_cf.set_title("Complementary Filter Orientation")
        self.ax_cf.grid(True)
        self.ax_cf.legend()

        # EKF lines
        self.ekf_roll_line, = self.ax_ekf.plot([], [], color='blue', linestyle='--', label='Roll (°)')
        self.ekf_pitch_line, = self.ax_ekf.plot([], [], color='green', linestyle='--', label='Pitch (°)')
        self.ekf_yaw_line, = self.ax_ekf.plot([], [], color='red', linestyle='--', label='Yaw (°)')
        self.ekf_labels = {
            'roll': self.ax_ekf.text(0, 0, '', color='blue', fontsize=9),
            'pitch': self.ax_ekf.text(0, 0, '', color='green', fontsize=9),
            'yaw': self.ax_ekf.text(0, 0, '', color='red', fontsize=9)
        }
        self.ax_ekf.axhline(y=0, color='gray', linestyle='--')
        self.ax_ekf.set_ylim(-180, 180)
        self.ax_ekf.set_ylabel("Angle (°)")
        self.ax_ekf.set_xlabel("Time (frames)")
        self.ax_ekf.set_title("EKF Orientation")
        self.ax_ekf.grid(True)
        self.ax_ekf.legend()

    def add_alpha_slider(self, label, frm, to, var, command):
        frame = tk.Frame(self.control_frame)
        frame.pack(pady=(10, 0), anchor="w")

        ttk.Label(frame, text=label).pack(anchor="w")

        control_frame = tk.Frame(frame)
        control_frame.pack(anchor="w")

        def decrement():
            value = var.get() - 0.01
            var.set(max(frm, round(value, 3)))
            on_slide(var.get())
            command(var.get())

        btn_dec = ttk.Button(control_frame, text="−", width=2, command=decrement)
        btn_dec.pack(side=tk.LEFT)

        slider = ttk.Scale(control_frame, from_=frm, to=to, orient=tk.HORIZONTAL, length=120, variable=var)
        slider.pack(side=tk.LEFT)

        def increment():
            value = var.get() + 0.01
            var.set(min(to, round(value, 3)))
            on_slide(var.get())
            command(var.get())

        btn_inc = ttk.Button(control_frame, text="+", width=2, command=increment)
        btn_inc.pack(side=tk.LEFT)

        self.alpha_gyro_label = ttk.Label(frame, text=f"Gyro (α): {var.get():.3f}")
        self.alpha_gyro_label.pack(anchor="w")

        self.alpha_accel_label = ttk.Label(frame, text=f"Accel (1−α): {1 - var.get():.3f}")
        self.alpha_accel_label.pack(anchor="w")

        def on_slide(val):
            alpha = float(val)
            self.alpha_gyro_label.config(text=f"Gyro (α): {alpha:.3f}")
            self.alpha_accel_label.config(text=f"Accel (1−α): {1 - alpha:.3f}")

        def on_release(event):
            command(var.get())

        slider.config(command=on_slide)
        slider.bind("<ButtonRelease-1>", on_release)

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
                    try:
                        _, alpha = line.split(',')
                        self.alpha_var.set(float(alpha))
                        self.alpha_gyro_label.config(text=f"Gyro (α): {float(alpha):.3f}")
                        self.alpha_accel_label.config(text=f"Accel (1−α): {1 - float(alpha):.3f}")
                    except Exception as e:
                        self.log_terminal(f"Gain parse error: {e}")
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

                        self.update_plot_lines()
                        self.canvas.draw()
            except Exception as e:
                self.log_terminal(f"Parse error: {e}")

    def update_plot_lines(self):
        # CF lines and labels
        xdata = range(len(cf_roll_data))
        self.cf_roll_line.set_data(xdata, cf_roll_data)
        self.cf_pitch_line.set_data(xdata, cf_pitch_data)
        self.cf_yaw_line.set_data(xdata, cf_yaw_data)

        if cf_roll_data:
            x = len(cf_roll_data) - 1
            self.cf_labels['roll'].set_position((x, cf_roll_data[-1]))
            self.cf_labels['roll'].set_text(f"R:{cf_roll_data[-1]:.1f}°")
            self.cf_labels['pitch'].set_position((x, cf_pitch_data[-1]))
            self.cf_labels['pitch'].set_text(f"P:{cf_pitch_data[-1]:.1f}°")
            self.cf_labels['yaw'].set_position((x, cf_yaw_data[-1]))
            self.cf_labels['yaw'].set_text(f"Y:{cf_yaw_data[-1]:.1f}°")

        # Rescale x-axis
        self.ax_cf.set_xlim(0, max(len(cf_roll_data), 10))

        # EKF lines and labels
        xdata2 = range(len(ekf_roll_data))
        self.ekf_roll_line.set_data(xdata2, ekf_roll_data)
        self.ekf_pitch_line.set_data(xdata2, ekf_pitch_data)
        self.ekf_yaw_line.set_data(xdata2, ekf_yaw_data)

        if ekf_roll_data:
            x = len(ekf_roll_data) - 1
            self.ekf_labels['roll'].set_position((x, ekf_roll_data[-1]))
            self.ekf_labels['roll'].set_text(f"R:{ekf_roll_data[-1]:.1f}°")
            self.ekf_labels['pitch'].set_position((x, ekf_pitch_data[-1]))
            self.ekf_labels['pitch'].set_text(f"P:{ekf_pitch_data[-1]:.1f}°")
            self.ekf_labels['yaw'].set_position((x, ekf_yaw_data[-1]))
            self.ekf_labels['yaw'].set_text(f"Y:{ekf_yaw_data[-1]:.1f}°")

        # Rescale x-axis
        self.ax_ekf.set_xlim(0, max(len(ekf_roll_data), 10))


# Run
if __name__ == "__main__":
    print(f"Connected to {PORT} at {BAUD} baud")
    root = tk.Tk()
    app = OrientationVisualizer(root)
    root.mainloop()
