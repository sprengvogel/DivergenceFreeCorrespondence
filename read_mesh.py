import farthest_point as fp
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import deformation_field as df
import pyshot
import pyvista as pv

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
mesh.apply_translation((0.4,0.3,0.3))
"""shot_descriptors = pyshot.get_descriptors(
	mesh.vertices,
	mesh.faces,
	radius=0.1,
	local_rf_radius=0.1,
	# The following parameters are optional
	min_neighbors=3,
	n_bins=20,
	double_volumes_sectors=True,
	use_interpolation=True,
	use_normalization=True,
)
print(shot_descriptors)"""
indices = fp.farthest_point_sampling(mesh.vertices, 3000)
vertices = mesh.vertices[indices]

"""cloud_x = pv.PolyData(vertices)
volume_x = cloud_x.delaunay_3d(alpha=0.033)
shell_x = volume_x.extract_geometry()
shell_x.plot()"""

print(vertices.shape)
ax = plt.axes(projection='3d')
ax.autoscale(False)
ax.set_xlim(0.2,0.8)
ax.set_ylim(0.2,0.8)
ax.set_zlim(0.2,0.8)
ax.scatter3D(*zip(*vertices))
plt.show()

js = df.generateJs()
cs = np.array([.2])
dField = df.DField(cs,js,vertices)
fn = df.rungeKutta(dField, T=100)

W = df.EStep(fn, vertices, 0.01)
r = df.calc_r(W,fn,vertices)
WSnake = df.calc_WSnake(W)
#print(W.shape)
#plt.matshow(W, cmap=plt.cm.Blues)
#plt.show()
print(fn.shape)
ax = plt.axes(projection='3d')
ax.autoscale(False)
ax.set_xlim(0.2,0.8)
ax.set_ylim(0.2,0.8)
ax.set_zlim(0.2,0.8)
ax.scatter3D(*zip(*fn))
plt.show()
""" mask = np.zeros(len(mesh.vertices), dtype=bool)
print(mask.shape)
mask[indices] = True
mesh.update_vertices(mask)
mesh.show() """