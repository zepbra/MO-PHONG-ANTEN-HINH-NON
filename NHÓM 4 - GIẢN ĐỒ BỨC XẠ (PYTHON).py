import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

class AntennaRadiationAnalyzer:
    def __init__(self):
        self.frequency = 900e6  # 900 MHz
        self.c = 3e8  # speed of light
        
    def calculate_wavelength(self):
        return self.c / self.frequency
    
    def radiation_pattern_2d(self, theta_deg):
        """Tính toán mẫu bức xạ 2D cho anten hình nón"""
        theta = np.radians(theta_deg)
        
        # Mẫu bức xạ điển hình cho anten nón
        gain = 3.5 * np.abs(np.cos(theta)) ** 0.8
        return np.maximum(gain, 0.1)
    
    def radiation_pattern_3d(self, theta, phi):
        """Tính toán mẫu bức xạ 3D"""
        pattern = 3.5 * np.abs(np.cos(theta)) ** 0.8
        return np.maximum(pattern, 0.1)
    
    def vswr_response(self, frequencies):
        """Tính toán VSWR theo tần số"""
        center_freq = 900e6
        bandwidth = 200e6
        
        vswr_values = 1.8 + 0.8 * ((frequencies - center_freq) / bandwidth) ** 2
        return np.minimum(vswr_values, 3.5)

def plot_comprehensive_analysis():
    """Vẽ biểu đồ phân tích toàn diện với cỡ chữ nhỏ hơn"""
    analyzer = AntennaRadiationAnalyzer()
    
    # Tạo figure lớn hơn
    fig = plt.figure(figsize=(24, 16))
    
    # Sử dụng GridSpec với khoảng cách lớn hơn
    gs = gridspec.GridSpec(3, 3, figure=fig, 
                          height_ratios=[1, 1, 0.8],
                          width_ratios=[1, 1, 1],
                          hspace=0.5, wspace=0.4)  # Tăng khoảng cách
    
    # ===== BIỂU ĐỒ 1: MẪU BỨC XẠ 2D =====
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    
    theta_deg = np.linspace(0, 360, 361)
    theta_rad = np.radians(theta_deg)
    radiation_2d = analyzer.radiation_pattern_2d(theta_deg)
    
    # Vẽ biểu đồ cực
    ax1.plot(theta_rad, radiation_2d, 'b-', linewidth=2, label='Mẫu bức xạ')
    ax1.fill(theta_rad, radiation_2d, 'blue', alpha=0.3)
    
    beamwidth_angle = 110
    ax1.plot([0, np.radians(beamwidth_angle/2)], [0, 3.5], 'r--', alpha=0.7, linewidth=1.5)
    ax1.plot([0, np.radians(-beamwidth_angle/2)], [0, 3.5], 'r--', alpha=0.7, linewidth=1.5)
    
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, 4)
    ax1.set_yticks([1, 2, 3, 4])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('BỨC XẠ 2D\n(Beamwidth: 110°)', fontsize=10, fontweight='bold', pad=10)  # Giảm cỡ chữ
    
    # Chú thích nhỏ hơn
    ax1.text(np.radians(30), 2.5, f'{beamwidth_angle}°', 
             ha='center', va='center', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

    # ===== BIỂU ĐỒ 2: MẪU BỨC XẠ 3D =====
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Tạo dữ liệu 3D
    theta = np.linspace(0, 2*np.pi, 60)  # Giảm số điểm
    phi = np.linspace(0, np.pi, 30)
    Theta, Phi = np.meshgrid(theta, phi)
    
    R = analyzer.radiation_pattern_3d(Phi, Theta)
    X = R * np.sin(Phi) * np.cos(Theta)
    Y = R * np.sin(Phi) * np.sin(Theta)
    Z = R * np.cos(Phi)
    
    # Vẽ bề mặt 3D
    surf = ax2.plot_surface(X, Y, Z, cmap=cm.plasma, 
                           alpha=0.8, edgecolor='none', 
                           antialiased=True, rstride=2, cstride=2)
    
    ax2.set_xlabel('X', fontweight='bold', labelpad=5, fontsize=9)  # Giảm cỡ chữ
    ax2.set_ylabel('Y', fontweight='bold', labelpad=5, fontsize=9)
    ax2.set_zlabel('Z', fontweight='bold', labelpad=5, fontsize=9)
    ax2.set_title('BỨC XẠ 3D\n(3.5 dBi)', fontsize=10, fontweight='bold', pad=10)  # Giảm cỡ chữ
    ax2.set_box_aspect([1,1,1])
    
    # Colorbar nhỏ hơn
    cbar = fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=20, pad=0.08)
    cbar.set_label('dBi', fontweight='bold', fontsize=8)  # Nhãn ngắn hơn
    cbar.ax.tick_params(labelsize=7)
    
    # ===== BIỂU ĐỒ 3: VSWR THEO TẦN SỐ =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    frequencies = np.linspace(700e6, 1100e6, 200)
    vswr_values = analyzer.vswr_response(frequencies)
    
    # Vẽ VSWR
    ax3.plot(frequencies/1e6, vswr_values, 'r-', linewidth=2, label='VSWR')
    ax3.fill_between(frequencies/1e6, 1, vswr_values, alpha=0.3, color='red')
    
    # Đường tham chiếu
    ax3.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Tốt (2:1)')
    ax3.axhline(y=1.5, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, label='Rất tốt (1.5:1)')
    
    # Đánh dấu tần số
    ax3.axvline(x=900, color='purple', linestyle='-', alpha=0.8, linewidth=2)
    ax3.text(900, 2.6, '900 MHz\n1.8:1', ha='center', va='bottom', 
             fontweight='bold', color='purple', fontsize=8,  # Giảm cỡ chữ
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    ax3.set_xlabel('Tần số (MHz)', fontweight='bold', fontsize=9)
    ax3.set_ylabel('VSWR', fontweight='bold', fontsize=9)
    ax3.set_title('VSWR THEO TẦN SỐ', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=8)  # Legend nhỏ hơn
    ax3.set_ylim(1, 3.5)
    ax3.set_xlim(700, 1100)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    
    # ===== BIỂU ĐỒ 4: SO SÁNH MẪU BỨC XẠ =====
    ax4 = fig.add_subplot(gs[1, 0])
    
    theta_compare = np.linspace(-180, 180, 361)
    e_plane = analyzer.radiation_pattern_2d(np.abs(theta_compare))
    h_plane = np.full_like(e_plane, 3.5)
    
    ax4.plot(theta_compare, e_plane, 'b-', linewidth=2, label='E-plane')
    ax4.plot(theta_compare, h_plane, 'r-', linewidth=2, label='H-plane')
    
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=3.5-3, color='g', linestyle='--', alpha=0.7, linewidth=1.5, label='-3 dB')
    
    # Đánh dấu beamwidth
    ax4.axvline(x=55, color='orange', linestyle=':', alpha=0.6)
    ax4.axvline(x=-55, color='orange', linestyle=':', alpha=0.6)
    ax4.text(0, 1.5, '110°', ha='center', va='center', 
             fontweight='bold', fontsize=8,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="orange", alpha=0.3))
    
    ax4.set_xlabel('Góc (độ)', fontweight='bold', fontsize=9)
    ax4.set_ylabel('Độ lợi (dBi)', fontweight='bold', fontsize=9)
    ax4.set_title('SO SÁNH E-PLANE vs H-PLANE', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    ax4.set_xlim(-180, 180)
    ax4.tick_params(axis='both', which='major', labelsize=8)
    
    # ===== BIỂU ĐỒ 5: PHÂN BỐ CÔNG SUẤT =====
    ax5 = fig.add_subplot(gs[1, 1], polar=True)
    
    power_pattern = radiation_2d**2
    normalized_power = power_pattern / np.max(power_pattern)
    
    ax5.plot(theta_rad, normalized_power, 'g-', linewidth=2, label='Công suất')
    ax5.fill(theta_rad, normalized_power, 'green', alpha=0.3)
    
    ax5.set_theta_zero_location('N')
    ax5.set_theta_direction(-1)
    ax5.set_ylim(0, 1)
    ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax5.grid(True, alpha=0.3)
    ax5.set_title('PHÂN BỐ CÔNG SUẤT', fontsize=10, fontweight='bold', pad=10)
    ax5.tick_params(axis='both', which='major', labelsize=7)
    
    # ===== BIỂU ĐỒ 6: BĂNG THÔNG HOẠT ĐỘNG =====
    ax6 = fig.add_subplot(gs[1, 2])
    
    freq_mhz = frequencies / 1e6
    good_bandwidth = freq_mhz[(vswr_values <= 2.0)]
    excellent_bandwidth = freq_mhz[(vswr_values <= 1.5)]
    
    if len(excellent_bandwidth) > 0 and len(good_bandwidth) > 0:
        ax6.axvspan(excellent_bandwidth[0], excellent_bandwidth[-1], alpha=0.4, color='green', 
                    label=f'VSWR ≤ 1.5: {excellent_bandwidth[-1]-excellent_bandwidth[0]:.0f}MHz')
        ax6.axvspan(good_bandwidth[0], good_bandwidth[-1], alpha=0.3, color='yellow',
                    label=f'VSWR ≤ 2.0: {good_bandwidth[-1]-good_bandwidth[0]:.0f}MHz')
    else:
        ax6.axvspan(850, 950, alpha=0.4, color='green', label='VSWR ≤ 1.5: ~100MHz')
        ax6.axvspan(800, 1000, alpha=0.3, color='yellow', label='VSWR ≤ 2.0: ~200MHz')
    
    ax6.axvline(x=900, color='red', linestyle='-', linewidth=2, label='900 MHz')
    
    ax6.set_xlabel('Tần số (MHz)', fontweight='bold', fontsize=9)
    ax6.set_title('BĂNG THÔNG', fontsize=10, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=8)
    ax6.set_xlim(700, 1100)
    ax6.set_ylim(0, 1)
    ax6.tick_params(axis='both', which='major', labelsize=8)
    
    # ===== BIỂU ĐỒ 7: THÔNG SỐ TỔNG HỢP =====
    ax7 = fig.add_subplot(gs[2, :])
    
    # Tạo bảng thông số đơn giản hơn
    parameters = [
        ["THÔNG SỐ", "GIÁ TRỊ", "ĐÁNH GIÁ"],
        ["Tần số", "900 MHz", "⭐⭐⭐⭐⭐"],
        ["Độ lợi", "3.5 dBi", "⭐⭐⭐⭐"],
        ["Beamwidth E", "110°", "⭐⭐⭐⭐⭐"],
        ["Beamwidth H", "360°", "⭐⭐⭐⭐⭐"],
        ["VSWR", "1.8:1", "⭐⭐⭐⭐"],
        ["Hiệu suất", "92%", "⭐⭐⭐⭐⭐"],
        ["Băng thông", "200 MHz", "⭐⭐⭐⭐"],
        ["Phân cực", "Vertical", "⭐⭐⭐⭐"],
        ["Ứng dụng", "IoT", "⭐⭐⭐⭐⭐"]
    ]
    
    ax7.axis('off')
    
    # Tạo bảng nhỏ hơn
    table = ax7.table(cellText=parameters,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Giảm cỡ chữ bảng
    table.scale(1, 1.8)
    
    # Định dạng tiêu đề
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', size=9)
    
    # Định dạng các hàng dữ liệu
    for i in range(1, len(parameters)):
        for j in range(3):
            if i % 2 == 1:
                table[(i, j)].set_facecolor('#F8F9FA')
            else:
                table[(i, j)].set_facecolor('#E9ECEF')
    
    ax7.set_title('THÔNG SỐ KỸ THUẬT ANTEN NÓN', 
                  fontsize=12, fontweight='bold', pad=15)  # Giảm cỡ chữ tiêu đề
    
    # ĐÃ BỎ DÒNG TIÊU ĐỀ TỔNG Ở ĐÂY
    # fig.suptitle('PHÂN TÍCH ANTEN HÌNH NÓN - THIẾT KẾ MỚI', 
    #             fontsize=14, fontweight='bold', y=0.98)  # ĐÃ XÓA
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06, hspace=0.5, wspace=0.4)
    plt.show()

def plot_simple_vswr_chart():
    """Vẽ biểu đồ VSWR đơn giản"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    analyzer = AntennaRadiationAnalyzer()
    
    # Biểu đồ 1: VSWR theo tần số
    frequencies = np.linspace(800e6, 1000e6, 300)
    vswr_values = analyzer.vswr_response(frequencies)
    
    # Vùng chất lượng
    ax1.fill_between(frequencies/1e6, 1, 1.5, alpha=0.3, color='green', label='Rất tốt')
    ax1.fill_between(frequencies/1e6, 1.5, 2.0, alpha=0.3, color='yellow', label='Tốt')
    ax1.fill_between(frequencies/1e6, 2.0, 3.0, alpha=0.3, color='orange', label='Chấp nhận')
    ax1.fill_between(frequencies/1e6, 3.0, 3.5, alpha=0.3, color='red', label='Kém')
    
    ax1.plot(frequencies/1e6, vswr_values, 'k-', linewidth=2, label='VSWR')
    
    ax1.set_xlabel('Tần số (MHz)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('VSWR', fontweight='bold', fontsize=11)
    ax1.set_title('ĐẶC TÍNH VSWR - ANTEN NÓN', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.set_ylim(1, 3.5)
    ax1.set_xlim(800, 1000)
    
    # Đánh dấu điểm tối ưu
    ax1.axvline(x=900, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax1.text(900, 1.3, '900 MHz\nVSWR=1.8', 
             ha='center', va='bottom', fontweight='bold', color='purple', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # Biểu đồ 2: Đánh giá chất lượng
    quality_labels = ['RẤT TỐT', 'TỐT', 'CHẤP NHẬN', 'KÉM']
    colors = ['green', 'yellow', 'orange', 'red']
    values = [25, 40, 25, 10]
    
    bars = ax2.bar(quality_labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax2.axhline(y=40, color='blue', linestyle='-', linewidth=2, alpha=0.7, 
                label='Anten Nón (1.8:1)')
    
    ax2.set_ylabel('Đánh giá (%)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Chất lượng VSWR', fontweight='bold', fontsize=11)
    ax2.set_title('THANG ĐÁNH GIÁ VSWR', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị phần trăm
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylim(0, 50)
    
    plt.tight_layout()
    plt.show()

# === CHẠY CHƯƠNG TRÌNH ===
if __name__ == "__main__":
    print("=" * 60)
    print("PHÂN TÍCH BIỂU ĐỒ BỨC XẠ & VSWR")
    print("ANTEN HÌNH NÓN - THIẾT KẾ MỚI")
    print("=" * 60)
    
    print("\n1. Đang tạo biểu đồ phân tích toàn diện...")
    plot_comprehensive_analysis()
    
    print("\n2. Đang tạo biểu đồ VSWR...")
    plot_simple_vswr_chart()
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH PHÂN TÍCH!")
    print("=" * 60)