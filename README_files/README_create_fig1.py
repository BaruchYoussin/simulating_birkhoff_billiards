import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(6.4, 4.8))
plt.rcParams.update({'figure.autolayout': True})
ax.axis('off')  # Turn off the coordinate axes
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# The lines need to be filled by points (rather than segments defined by their endpoints) to use fill_between(..):
x = np.linspace(-1, 1, 100)
slope_left_right = 1
slope_separatrix = -1

# Line 1: Left and Right separatrix directions
y1 = slope_left_right * x
ax.plot(x, y1, color='black')
ax.text(1.05, 1.05, 'R', fontsize=12)
ax.text(-1.1, -1.05, 'L', fontsize=12)

# Line 2: the separatrix direction
y2 = slope_separatrix * x
ax.plot(x[x < 0], y2[x < 0], color='black', dashes=[5, 5])
ax.plot(x[x >= 0], y2[x >= 0], color='black')
ax.text(1.05, -1.05, 'S', fontsize=12)

# The hyperbolic point P
hyperbolic_point_P = (0, 0)
ax.plot(hyperbolic_point_P[0], hyperbolic_point_P[1], 'ko', markersize=5)
ax.text(hyperbolic_point_P[0], hyperbolic_point_P[1], 'P', fontsize=12, ha='right', va='bottom')

# Angle boundaries:
slope1 = slope_left_right + 0.2
slope2 = slope_left_right - 0.2
y_slope1 = slope1 * x
y_slope2 = slope2 * x
ax.plot(x, y_slope1, 'k-')
ax.plot(x, y_slope2, 'k-')

# Filling angles
ax.fill_between(x, y_slope1, y_slope2, where=(x > 0), color='cyan', alpha=0.25, label=r"$A_R$" + ": right angle")
ax.fill_between(x, y_slope2, y_slope1, where=(x < 0), color='blue', alpha=0.25, label=r"$A_L$" + ": left angle")

# Gray region: upper left obtuse angle
lower_boundary_gray = np.maximum(y_slope1, y_slope2)
upper_boundary_gray = np.full_like(lower_boundary_gray, np.linspace(y2.max(), lower_boundary_gray.max(), 100))
ax.fill_between(x, lower_boundary_gray, upper_boundary_gray, color='gray', alpha=0.25,
                label=r"$A_B$" + ": back angle")

# Circle at P
circle_radius = 0.1
circle = plt.Circle(hyperbolic_point_P, circle_radius, color='red', alpha=0.25, label=r"$C$" + ": neighborhood of " + r"$P$")
ax.add_patch(circle)

# Initial points:
initial_point_1 = (0.7, -0.9)
initial_point_2 = (1, -0.56)
initial_point_3 = (0.85, -0.73)
initial_point_final = (0.82 , -0.764)
points = (initial_point_1, initial_point_2, initial_point_3, initial_point_final)
ax.plot(*zip(*points), 'ko', markersize=3)

for point, text in zip(points, (r"$I_1$", r"$I_2$", r"$I_3$", r"$I$")):
    ax.text(point[0], point[1] - 0.03, text, fontsize=12, ha='left', va='top')

ax.add_patch(patches.FancyArrowPatch(posA=initial_point_1, posB=(-0.2, -0.19), arrowstyle="->",  mutation_scale=20))
ax.add_patch(patches.FancyArrowPatch(posA=initial_point_2, posB=(0.3, 0.3), arrowstyle="->",  mutation_scale=20))
ax.add_patch(patches.FancyArrowPatch(posA=initial_point_3, posB=(0.13, 0.145), arrowstyle="->",  mutation_scale=20))
ax.add_patch(patches.FancyArrowPatch(posA=initial_point_final, posB=(0.07, 0), arrowstyle="->",  mutation_scale=20))

fig.subplots_adjust(bottom=0.3)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2,
           title=r"$I_1$ and $I_2$ are the initial points such that the orbits starting from them, hit the left" "\n"
                 r"and the right angles; $I_3$ is the first step in bisection, and $I$ lies on the separatrix")

plt.text(0.5, -0.46, "Figure 1", ha='center', va='center', transform=plt.gca().transAxes, fontsize=15)

# plt.tight_layout()

fig.savefig("README_fig1.png")
