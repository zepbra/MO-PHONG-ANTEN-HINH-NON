import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import pandas as pd
from tabulate import tabulate

class ConeAntennaAnalyzer:
    def __init__(self, height=1.5, base_radius=1.5, feed_length=0.8, frequency=900e6):
        self.height = height
        self.base_radius = base_radius
        self.feed_length = feed_length
        self.frequency = frequency
        self.c = 3e8
        
    def calculate_parameters(self):
        wavelength = self.c / self.frequency
        
        slant_height = np.sqrt(self.height**2 + self.base_radius**2)
        opening_angle = 2 * np.degrees(np.arcsin(self.base_radius / slant_height))
        
        surface_area = np.pi * self.base_radius * slant_height
        
        volume = (1/3) * np.pi * self.base_radius**2 * self.height
        
        aspect_ratio = self.height / (2 * self.base_radius)
        
        estimated_gain = 3.5
        
        beamwidth_e = 110.0
        beamwidth_h = 360.0
        
        input_impedance = 50
        
        bandwidth_ratio = 2.5
        
        vswr = 1.8
        
        radiation_efficiency = 0.92
        
        directivity = 4.5
        
        front_to_back_ratio = 12.0
        
        return {
            'Bước sóng (m)': wavelength,
            'Góc mở nón (°)': opening_angle,
            'Chiều cao nghiêng (m)': slant_height,
            'Diện tích bề mặt (m²)': surface_area,
            'Thể tích (m³)': volume,
            'Tỉ số H/D': aspect_ratio,
            'Độ lợi (dBi)': estimated_gain,
            'Beamwidth E-plane (°)': beamwidth_e,
            'Beamwidth H-plane (°)': beamwidth_h,
            'Trở kháng (Ω)': input_impedance,
            'Tỉ số băng thông': bandwidth_ratio,
            'VSWR (tỷ số sóng đứng)': vswr,
            'Hiệu suất bức xạ (%)': radiation_efficiency * 100,
            'Độ định hướng (dBi)': directivity,
            'Tỉ số F/B (dB)': front_to_back_ratio
        }

def create_cone_antenna(height=1.5, base_radius=1.5, segments=50):
    z = np.linspace(0, height, segments)
    theta = np.linspace(0, 2 * np.pi, segments)
    Z, Theta = np.meshgrid(z, theta)
    
    R = base_radius * (Z / height)
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    Z = height - Z
    
    return X, Y, Z

class SmoothAntennaVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.params = analyzer.calculate_parameters()
        
        self.fig = plt.figure(figsize=(20, 16))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.X_cone, self.Y_cone, self.Z_cone = create_cone_antenna(
            analyzer.height, analyzer.base_radius
        )
        
        self.X_feed, self.Y_feed, self.Z_feed = self._create_feed_line()
        self.X_ground, self.Y_ground, self.Z_ground = self._create_ground_plane()
        
        self.cone_surface = None
        self.feed_surface = None
        self.ground_surface = None
        self.annotations = []
        
        self.animation_running = True
        self.ani = None
        self.animation_interval = 16
        
        self.setup_display()
        
    def setup_display(self):
        max_range = max(self.analyzer.base_radius, self.analyzer.height) * 1.2
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        
        min_z = -0.5
        max_z = self.analyzer.height + self.analyzer.feed_length + 0.5
        self.ax.set_zlim([min_z, max_z])
        
        self.ax.xaxis.label.set_size(12)
        self.ax.yaxis.label.set_size(12)
        self.ax.zaxis.label.set_size(12)
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        
    def _create_feed_line(self):
        z_feed = np.linspace(self.analyzer.height, self.analyzer.height + self.analyzer.feed_length, 20)
        r_feed = 0.08
        theta_feed = np.linspace(0, 2 * np.pi, 20)
        Z_feed, Theta_feed = np.meshgrid(z_feed, theta_feed)
        X_feed = r_feed * np.cos(Theta_feed)
        Y_feed = r_feed * np.sin(Theta_feed)
        return X_feed, Y_feed, Z_feed
    
    def _create_ground_plane(self):
        ground_radius = self.analyzer.base_radius * 1.5
        theta_ground = np.linspace(0, 2 * np.pi, 30)
        r_ground = np.linspace(0, ground_radius, 15)
        R_ground, Theta_ground = np.meshgrid(r_ground, theta_ground)
        X_ground = R_ground * np.cos(Theta_ground)
        Y_ground = R_ground * np.sin(Theta_ground)
        
        Z_ground = np.ones_like(X_ground) * self.analyzer.height
        
        return X_ground, Y_ground, Z_ground
    
    def rotate_points(self, X, Y, Z, angle_degrees):
        angle_rad = np.radians(angle_degrees)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        rotated_points = rot_matrix @ points
        
        return (rotated_points[0].reshape(X.shape),
                rotated_points[1].reshape(Y.shape),
                rotated_points[2].reshape(Z.shape))
    
    def add_annotations(self, rotation_angle):
        for ann in self.annotations:
            ann.remove()
        self.annotations.clear()
        
        annotations_data = [
            (0, 0, self.analyzer.height/2, "ANTEN NÓN", 'white', 11, 'darkblue'),
            (0, 0, self.analyzer.height + self.analyzer.feed_length/2, "CÁP TIẾP ĐIỆN", 'white', 10, 'red'),
            (self.analyzer.base_radius*1.2, 0, self.analyzer.height, "MẶT ĐẤT", 'white', 10, 'black'),
            (0, 0, 0.1, "ĐẦU NHỌN", 'white', 9, 'purple'),
            (self.analyzer.base_radius*0.7, 0, self.analyzer.height*0.8, "ĐÁY RỘNG", 'white', 9, 'green'),
        ]
        
        for x, y, z, text, color, fontsize, bg_color in annotations_data:
            angle_rad = np.radians(rotation_angle)
            x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            
            ann = self.ax.text(x_rot, y_rot, z, text, 
                             color=color, fontsize=fontsize, fontweight='bold',
                             ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.4", facecolor=bg_color, 
                                     alpha=0.9, edgecolor='white', linewidth=2))
            self.annotations.append(ann)
    
    def add_parameter_display(self, rotation_angle):
        param_text = (
            f"THÔNG SỐ KỸ THUẬT:\n"
            f"• Chiều cao: {self.analyzer.height:.1f} m\n"
            f"• Bán kính: {self.analyzer.base_radius:.1f} m\n"
            f"• Góc mở: {self.params['Góc mở nón (°)']:.0f}°\n"
            f"• Tần số: {self.analyzer.frequency/1e6:.0f} MHz\n"
            f"• VSWR: {self.params['VSWR (tỷ số sóng đứng)']:.1f}:1"
        )
        
        param_ann = self.ax.text2D(
            0.02, 0.98, param_text, transform=self.ax.transAxes,
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="blue", alpha=0.95),
            verticalalignment='top'
        )
        self.annotations.append(param_ann)
        
        radiation_text = (
            f"ĐẶC TÍNH BỨC XẠ:\n"
            f"• Độ lợi: {self.params['Độ lợi (dBi)']:.1f} dBi\n"
            f"• Beamwidth: {self.params['Beamwidth E-plane (°)']:.0f}°\n"
            f"• Hiệu suất: {self.params['Hiệu suất bức xạ (%)']:.0f}%\n"
            f"• Phân cực: Dọc"
        )
        
        radiation_ann = self.ax.text2D(
            0.02, 0.75, radiation_text, transform=self.ax.transAxes,
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="green", alpha=0.95),
            verticalalignment='top'
        )
        self.annotations.append(radiation_ann)
        
        angle_text = f"GÓC XOAY: {rotation_angle}°"
        angle_ann = self.ax.text2D(
            0.5, 0.02, angle_text, transform=self.ax.transAxes,
            fontsize=14, fontweight='bold', ha='center', color='white',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="red", alpha=0.95)
        )
        self.annotations.append(angle_ann)
        
        vswr = self.params['VSWR (tỷ số sóng đứng)']
        if vswr <= 1.5:
            vswr_status = "RẤT TỐT"
            vswr_color = "green"
        elif vswr <= 2.0:
            vswr_status = "TỐT"
            vswr_color = "orange"
        else:
            vswr_status = "CHẤP NHẬN ĐƯỢC"
            vswr_color = "red"
            
        vswr_text = f"CHẤT LƯỢNG: {vswr_status}"
        vswr_ann = self.ax.text2D(
            0.85, 0.98, vswr_text, transform=self.ax.transAxes,
            fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=vswr_color, alpha=0.9),
            verticalalignment='top'
        )
        self.annotations.append(vswr_ann)
    
    def add_dimension_annotations(self):
        self.ax.plot([0.5, 0.5], [0, 0], [0, self.analyzer.height], 'k-', linewidth=2, alpha=0.8)
        height_ann = self.ax.text(0.7, 0, self.analyzer.height/2, 
                                f'{self.analyzer.height}m', 
                                fontsize=9, fontweight='bold', color='black')
        self.annotations.append(height_ann)
        
        self.ax.plot([0, self.analyzer.base_radius], [0.5, 0.5], 
                    [self.analyzer.height, self.analyzer.height], 'k-', linewidth=2, alpha=0.8)
        radius_ann = self.ax.text(self.analyzer.base_radius/2, 0.7, self.analyzer.height + 0.1,
                                f'{self.analyzer.base_radius}m',
                                fontsize=9, fontweight='bold', color='black')
        self.annotations.append(radius_ann)
    
    def plot_antenna(self, rotation_angle=0):
        self.ax.clear()
        
        X_cone_rot, Y_cone_rot, Z_cone_rot = self.rotate_points(
            self.X_cone, self.Y_cone, self.Z_cone, rotation_angle
        )
        X_feed_rot, Y_feed_rot, Z_feed_rot = self.rotate_points(
            self.X_feed, self.Y_feed, self.Z_feed, rotation_angle
        )
        X_ground_rot, Y_ground_rot, Z_ground_rot = self.rotate_points(
            self.X_ground, self.Y_ground, self.Z_ground, rotation_angle
        )
        
        self.cone_surface = self.ax.plot_surface(
            X_cone_rot, Y_cone_rot, Z_cone_rot, 
            color='lightblue', alpha=0.8, edgecolor='blue',
            linewidth=0.5, antialiased=True, rstride=1, cstride=1
        )
        
        self.feed_surface = self.ax.plot_surface(
            X_feed_rot, Y_feed_rot, Z_feed_rot, 
            color='red', alpha=0.9, edgecolor='darkred',
            linewidth=0.3, antialiased=True, rstride=1, cstride=1
        )
        
        self.ground_surface = self.ax.plot_surface(
            X_ground_rot, Y_ground_rot, Z_ground_rot, 
            color='gray', alpha=0.7, edgecolor='black',
            linewidth=0.2, antialiased=True, rstride=1, cstride=1
        )
        
        self.setup_display()
        
        self.add_annotations(rotation_angle)
        self.add_parameter_display(rotation_angle)
        self.add_dimension_annotations()
        
        self.ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        
        self.ax.set_title('ANTEN HÌNH NÓN', 
                         fontsize=16, fontweight='bold', pad=20)
        
        self.ax.view_init(elev=25, azim=45)
        self.ax.grid(True, alpha=0.2)
        self.ax.set_box_aspect([1, 1, 1])
    
    def create_smooth_animation(self):
        def animate(frame):
            rotation_angle = (frame * 2) % 360
            self.plot_antenna(rotation_angle)
            return [self.cone_surface, self.feed_surface, self.ground_surface] + self.annotations
        
        self.ani = animation.FuncAnimation(
            self.fig, animate, frames=180,
            interval=self.animation_interval, blit=False, repeat=True
        )
        return self.ani
    
    def start_animation(self):
        if not self.animation_running:
            self.ani = self.create_smooth_animation()
            self.animation_running = True
    
    def stop_animation(self):
        if self.animation_running and self.ani:
            self.ani.event_source.stop()
            self.animation_running = False

def create_interactive_control(visualizer):
    plt.subplots_adjust(bottom=0.1, top=0.9)
    
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_rotation = Slider(ax_slider, 'Điều chỉnh góc xoay (°)', 0, 360, valinit=0)
    
    def update_slider(val):
        visualizer.stop_animation()
        visualizer.plot_antenna(slider_rotation.val)
        visualizer.fig.canvas.draw_idle()
    
    slider_rotation.on_changed(update_slider)
    
    return slider_rotation

if __name__ == "__main__":
    analyzer = ConeAntennaAnalyzer(
        height=1.5,
        base_radius=1.5,
        feed_length=0.8,
        frequency=900e6
    )
    
    print("ĐANG TẢI MÔ PHỎNG ANTEN...")
    
    params = analyzer.calculate_parameters()
    
    print("\n" + "="*60)
    print("THÔNG SỐ ANTEN")
    print("="*60)
    
    for key, value in params.items():
        print(f"• {key}: {value}")
    
    visualizer = SmoothAntennaVisualizer(analyzer)
    
    visualizer.plot_antenna(0)
    
    visualizer.start_animation()
    
    slider = create_interactive_control(visualizer)
    
    info_text = """
    ANTEN HÌNH NÓN
    • Góc mở ~90°
    • Vùng phủ rộng
    • Độ lợi ~3.5 dBi
    • VSWR ~1.8:1
    """
    
    plt.figtext(0.02, 0.02, info_text, fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", 
                        edgecolor="blue", linewidth=2))
    
    print("\n✓ Mô phỏng đã sẵn sàng!")
    print("✓ Anten HÌNH NÓN")
    print("✓ Tự động xoay 360°")
    
    plt.show()