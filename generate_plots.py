import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp

# =================================================================
# FUNGSI GENERASI DATA DAN PLOT
# =================================================================

def generate_and_save_plots():
    """
    Menghasilkan 4 grafik kunci untuk validasi Symbolic Regression 
    dan menyimpannya sebagai file PNG.
    """
    
    np.random.seed(42)

    # --- 1. Data 1D untuk Kecocokan dan Residual ---
    X_1D = np.linspace(0.1, 5.0, 50)
    C_TRUE = 1.5
    
    # Persamaan Analitik Sejati (Target SR untuk ditemukan)
    Z_TRUE = np.exp(-C_TRUE / X_1D)
    
    # Data Numerik (Hasil Simulasi/Eksperimen dengan Noise)
    Z_NUMERIK = Z_TRUE + np.random.normal(0, 0.05, Z_TRUE.shape) 

    # Solusi yang Ditemukan oleh SR (Anggap berhasil menemukan C = 1.52)
    C_FOUND = 1.52 
    Z_ANALITIK_SR = np.exp(-C_FOUND / X_1D)
    
    # Rentang data halus untuk kurva
    X_smooth = np.linspace(min(X_1D), max(X_1D), 100)
    Z_ANALITIK_SMOOTH = np.exp(-C_FOUND / X_smooth)


    # --- 2. Data 3D untuk Permukaan Solusi ---
    X_3D = np.linspace(-3, 3, 10)
    Y_3D = np.linspace(-3, 3, 10)
    X_mesh, Y_mesh = np.meshgrid(X_3D, Y_3D)
    
    # Persamaan Analitik 3D Sederhana (misalnya sin(sqrt(x^2 + y^2)))
    Z_SURFACE = np.sin(np.sqrt(X_mesh**2 + Y_mesh**2)) 
    
    # --- 3. Data Fiktif untuk Pareto Front ---
    Complexity = np.array([1, 2, 3, 4, 5, 6, 7, 8]) 
    RMSE = np.array([0.5, 0.2, 0.08, 0.05, 0.04, 0.041, 0.045, 0.05])
    Pareto_indices = [0, 1, 2, 3, 4] 


    # =================================================================
    # PLOTTING DAN SAVING GRAFIK
    # =================================================================

    # GRAFIK 1: KECOCOKAN KURVA
    plt.figure(figsize=(10, 6))
    plt.scatter(X_1D, Z_NUMERIK, color='red', marker='o', label='Data Numerik')
    plt.plot(X_smooth, Z_ANALITIK_SMOOTH, color='blue', linestyle='-', linewidth=2, label='Solusi Analitik SR')
    plt.title('Grafik 1: Kecocokan Data vs. Solusi Analitik')
    plt.xlabel('Variabel Input X')
    plt.ylabel('Nilai Output Z')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig('plot_1_fit.png')
    plt.close()


    # GRAFIK 2: RESIDUAL
    Residuals = Z_NUMERIK - sr_solution(X_1D)
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.scatter(X_1D, Residuals, color='green', marker='x')
    plt.title('Grafik 2: Analisis Residual (Kesalahan Model SR)')
    plt.xlabel('Variabel Input X')
    plt.ylabel('Residual')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('plot_2_residuals.png')
    plt.close()


    # GRAFIK 3: PERMUKAAN SOLUSI (3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_mesh, Y_mesh, Z_SURFACE, cmap='viridis', alpha=0.8)
    ax.set_title('Grafik 3: Permukaan Solusi Analitik 3D')
    ax.set_xlabel('Variabel Input X')
    ax.set_ylabel('Variabel Input Y')
    ax.set_zlabel('Nilai Output Z')
    fig.tight_layout()
    plt.savefig('plot_3_surface.png')
    plt.close()


    # GRAFIK 4: PARETO FRONT
    plt.figure(figsize=(10, 6))
    plt.plot(Complexity[Pareto_indices], RMSE[Pareto_indices], 
             color='purple', linestyle='-', marker='o', label='Pareto Front Optimal')
    plt.scatter(Complexity, RMSE, color='gray', alpha=0.6)
    plt.title('Grafik 4: Pareto Front (Kompleksitas vs. Akurasi)')
    plt.xlabel('Kompleksitas Model (Jumlah Suku)')
    plt.ylabel('RMSE (Akurasi/Kesalahan)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig('plot_4_pareto.png')
    plt.close()
    
    print("Empat file grafik telah berhasil disimpan di direktori Anda: plot_1_fit.png, plot_2_residuals.png, plot_3_surface.png, plot_4_pareto.png")


if __name__ == '__main__':
    generate_and_save_plots()
