import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
dipole_length = 150    # total dipole length (mm)
dipole_diameter = 1    # diameter of the “wire” (mm)
# your dipole parameters
half_len = dipole_length / 2.0
radius   = dipole_diameter / 2.0

# define the 8 corners of the box
corners = np.array([
    [-half_len, -radius, -radius],
    [ half_len, -radius, -radius],
    [ half_len,  radius, -radius],
    [-half_len,  radius, -radius],
    [-half_len, -radius,  radius],
    [ half_len, -radius,  radius],
    [ half_len,  radius,  radius],
    [-half_len,  radius,  radius],
])

# list of edges as pairs of corner indices
edges = [
    (0,1),(1,2),(2,3),(3,0),  # bottom face
    (4,5),(5,6),(6,7),(7,4),  # top face
    (0,4),(1,5),(2,6),(3,7)   # vertical edges
]

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
for i,j in edges:
    p0 = corners[i]
    p1 = corners[j]
    ax.plot(*zip(p0,p1), 'k-', linewidth=2)

# axis limits
ax.set_xlim(-half_len*1.2, half_len*1.2)
ax.set_ylim(-radius*5, radius*5)
ax.set_zlim(-radius*5, radius*5)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
ax.set_title('Dipole Geometry')
plt.tight_layout()
plt.show()
