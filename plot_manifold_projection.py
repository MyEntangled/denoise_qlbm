import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, to_rgb

# --- 1. Mathematical Definitions ---
c_s = 1.0 / np.sqrt(3)

def h_u(u_phys):
    return 1 + 1.5 * u_phys + 0.375 * u_phys**2

def h_prime_u(u_phys):
    return 1.5 + 0.75 * u_phys

# --- 2. Domain (High Res for smooth lighting) ---
# Lighting calculations look better with slightly higher resolution
xi = np.linspace(0, 1.5, 80)
u_norm = np.linspace(-1, 1, 80)
Xi, U_norm = np.meshgrid(xi, u_norm)
U_phys = U_norm * c_s

# Manifold Equation
G_surface = Xi * h_u(U_phys)
z_max = np.max(G_surface)
z_max = np.max(G_surface) * 1.2 ## Increase z limit slightly for P_tilde

# --- 3. Tangent Plane Derivation ---
xi_0 = 1.0
u_norm_0 = 0.5
u_phys_0 = u_norm_0 * c_s
g_0 = xi_0 * h_u(u_phys_0)

dG_dxi_0 = h_u(u_phys_0)
dG_du_norm_0 = (xi_0 * h_prime_u(u_phys_0)) * c_s

G_plane = dG_dxi_0 * Xi + dG_du_norm_0 * (U_norm - u_norm_0)

# --- 4. New Points Calculation ---
# Define P_tilde directly above P0
delta_xi = 0.
delta_u_norm = 0.05
delta_g = -0.8  # How far above the manifold

tilde_xi = 1.2 #xi_0 + delta_xi
tilde_u_norm = u_norm_0 + delta_u_norm
tilde_u_phys = tilde_u_norm * c_s
P_tilde = np.array([tilde_xi, tilde_u_norm, tilde_xi * h_u(tilde_u_phys) + delta_g])

# Calculate Projection P_proj onto the tangent plane
# The line is P(t) = P_tilde + t * n
# t = - (n . (P_tilde - P0)) / |n|^2. Since P_tilde lies vertically above P0:
# P_tilde - P0 = (0, 0, delta_g).
# n . (0, 0, delta_g) = nz * delta_g = -delta_g
nx = h_u(u_phys_0)                 # dG/dxi
ny = (xi_0 * h_prime_u(u_phys_0)) * c_s # dG/du_norm
nz = -1.0
norm_sq = nx**2 + ny**2 + nz**2
t_proj = delta_g / norm_sq

P_proj = P_tilde + t_proj * np.array([nx, ny, nz])

# --- 4. Manual Clipping ---
G_plane_clipped = G_plane.copy()
G_plane_clipped[G_plane_clipped < 0] = np.nan
G_plane_clipped[G_plane_clipped > z_max] = np.nan

# --- 5. Lighting Calculation ---
# Create a LightSource coming from the "North West" (azimuth 315)
# altdeg=45 gives a standard studio lighting look
#ls = LightSource(azdeg=315, altdeg=45)
ls = LightSource(azdeg=315, altdeg=45)

# Create a solid RGB color array for the surface (Midnight Blue)
base_color = np.array(to_rgb('#191970'))
rgb_array = np.tile(base_color, (Xi.shape[0], Xi.shape[1], 1))

# Apply shading: simulates light hitting the geometry of G_surface
# vert_exag accentuates the peaks/valleys for the shadow calculation
illuminated_surface = ls.shade_rgb(rgb_array, elevation=G_surface, vert_exag=0.5)

# --- 6. Visualization Setup ---
fig = plt.figure(figsize=(6, 5))
plt.subplots_adjust(bottom=0.25)

ax = fig.add_subplot(111, projection='3d')

# --- PLOTTING ---

# A. Manifold Surface (Lit)
# We use 'facecolors' to apply our custom lighting.
# We remove 'color' and 'shade' arguments as we are handling them manually.
surf = ax.plot_surface(Xi, U_norm, G_surface, facecolors=illuminated_surface,
                       rstride=1, cstride=1, linewidth=0, antialiased=False,
                       alpha=0.6, shade=False)

# B. Tangent Plane
plane = ax.plot_wireframe(Xi, U_norm, G_plane_clipped, color='orangered',
                          rstride=8, cstride=8, linewidth=1.5, alpha=0.4)

# ----------------

# Visual Elements
#ax.scatter([0], [0], [0], color='#333333', s=100, linewidths=2, label='Origin', zorder=15, depthshade=False)
#ax.scatter([xi_0], [u_norm_0], [g_0], color='#E67E22', s=100, linewidths=2, label='$P_0$', zorder=100, depthshade=False)
#ax.plot([0], [0], [0], marker='o', markersize=10, color='#333333', linestyle='None', zorder=100, label='Origin')
ax.plot([xi_0], [u_norm_0], [g_0], marker='o', markersize=10, color='#e09f3e', linestyle='None', zorder=100, label='$P_0$')

# NEW: P_tilde (Off-manifold point) - Magenta
# ax.scatter([P_tilde[0]], [P_tilde[1]], [P_tilde[2]],
#            color='#ef233c', edgecolor='white', s=150, marker='D',
#            label=r'$\tilde{P}$ (Off-Manifold)', zorder=150, depthshade=False)

ax.plot([P_tilde[0]], [P_tilde[1]], [P_tilde[2]], marker='D', markersize=10, color='#ef233c', linestyle='None', zorder=100, label=r'$\tilde{P}$ (Off-Manifold)')



# NEW: P_proj (Projection) - Lime Green
# ax.scatter([P_proj[0]], [P_proj[1]], [P_proj[2]],
#            color='#32CD32', edgecolor='white', s=150, marker='X', linewidth=2,
#            label=r'$P_{proj}$ (Projection)', zorder=200, depthshade=False)
ax.plot([P_proj[0]],
        [P_proj[1]],
        [P_proj[2]],
        color='#32CD32', linestyle='None', markersize=10, marker='X', alpha=0.6, zorder=105, label=r'$P_{proj}$ (Projection)')

# NEW: Projection Line connecting P_tilde and P_proj
ax.plot([P_tilde[0], P_proj[0]],
        [P_tilde[1], P_proj[1]],
        [P_tilde[2], P_proj[2]],
        color='#32CD32', linestyle=':', linewidth=3, zorder=105)

# Ray
xi_line = np.linspace(0, 1.5, 20)
u_line_norm = np.full_like(xi_line, u_norm_0)
g_line = xi_line * h_u(u_phys_0)
ax.plot(xi_line, u_line_norm, g_line, color='#e09f3e', linestyle='--', linewidth=2, label=r'Tangent Ray', zorder=15)



# Ticks
ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.6])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_zticks([0, 1.0, 2.0])

# Labels
ax.set_xlabel(r'$\sqrt{\rho}$', fontsize=11)
ax.set_ylabel(r'$u / c_s$', fontsize=11)
ax.set_zlabel(r'$g(\rho, u)$', fontsize=11)
#ax.set_title(r'Illuminated Manifold: Tangent Space Geometry', fontsize=14)

# Axis Limits
ax.set_zlim(0., z_max)

# Legend
ax.legend()

# Initial View
initial_elev = 30
initial_azim = -100 # -100/-150
ax.view_init(elev=initial_elev, azim=initial_azim)

# # --- 7. Interactive Sliders ---
# ax_elev = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
# ax_azim = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgray')
#
# s_elev = Slider(ax_elev, 'Elevation', 0, 90, valinit=initial_elev)
# s_azim = Slider(ax_azim, 'Azimuth', -180, 180, valinit=initial_azim)
#
# def update(val):
#     ax.view_init(elev=s_elev.val, azim=s_azim.val)
#     fig.canvas.draw_idle()
#
# s_elev.on_changed(update)
# s_azim.on_changed(update)

plt.tight_layout()
plt.savefig("manifold_proj.pdf", format="pdf", bbox_inches="tight")
plt.show()