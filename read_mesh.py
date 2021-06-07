import farthest_point as fp
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import deformation_field as df

def scale_numpy_array(arr, min_v, max_v):
    # https://stackoverflow.com/questions/19299155/normalize-a-vector-of-3d-coordinates-to-be-in-between-0-and-1/19301193
    new_range = (min_v, max_v)
    max_range = max(new_range)
    min_range = min(new_range)
    scaled_unit = (max_range - min_range) / (np.max(arr) - np.min(arr))
    return arr*scaled_unit - np.min(arr)+scaled_unit + min_range

mesh = trimesh.load("data/topkids/kid01.off").process(validate=True)
mesh.apply_scale(0.004)
mesh.rezero()
mesh.apply_translation((0.4,0.4,0.3))
indices = fp.farthest_point_sampling(mesh.vertices, 3000)
vertices = mesh.vertices[indices]
print(vertices.shape)
ax = plt.axes(projection='3d')
ax.autoscale(False)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.scatter3D(*zip(*vertices))
plt.show()

js = df.generateJs()
cs = np.array([0,0,0,])
dField = df.DField(cs,js,vertices)
fn = df.rungeKutta(dField, T=100)
print(fn.shape)
ax = plt.axes(projection='3d')
ax.autoscale(False)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.scatter3D(*zip(*fn))
plt.show()
""" mask = np.zeros(len(mesh.vertices), dtype=bool)
print(mask.shape)
mask[indices] = True
mesh.update_vertices(mask)
mesh.show() """