import farthest_point as fp
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import deformation_field as df
import pyshot
import pyvista as pv
from fp_kim import farthest_points_sampler

def scale_numpy_array(arr, min_v, max_v):
    # https://stackoverflow.com/questions/19299155/normalize-a-vector-of-3d-coordinates-to-be-in-between-0-and-1/19301193
    new_range = (min_v, max_v)
    max_range = max(new_range)
    min_range = min(new_range)
    scaled_unit = (max_range - min_range) / (np.max(arr) - np.min(arr))
    scaled_arr = arr*scaled_unit - np.min(arr)*scaled_unit + min_range
    return scaled_arr - np.average(scaled_arr, axis=0)+0.5

def load_points_and_mesh(model_file, num_points):
	x, x_ind = farthest_points_sampler(num_points, model_file)
	x = scale_numpy_array(x, 0, 1)
	return x

"""mesh = trimesh.load("data/topkids/kid01.off").process(validate=True)
mesh.apply_scale(0.004)
mesh.rezero()
mesh.apply_translation((0.4,0.3,0.3))
shot_descriptors = pyshot.get_descriptors(
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
print(shot_descriptors)
indices = fp.farthest_point_sampling(mesh.vertices, 30)
vertices = mesh.vertices[indices]

mesh2 = trimesh.load("data/topkids/kid02.off").process(validate=True)
mesh2.apply_scale(0.004)
mesh2.rezero()
mesh2.apply_translation((0.4,0.3,0.3))
indices2 = fp.farthest_point_sampling(mesh2.vertices, 30)
ym = mesh2.vertices[indices2]"""

"""cloud_x = pv.PolyData(vertices)
volume_x = cloud_x.delaunay_3d(alpha=0.033)
shell_x = volume_x.extract_geometry()
shell_x.plot()"""

xn = load_points_and_mesh("tosca/cat1.vert", 30)
ym = load_points_and_mesh("tosca/cat4.vert", 30)
print(xn)
print(ym)

ax = plt.axes(projection='3d')
ax.autoscale(False)
ax.set_xlim(0.,1.)
ax.set_ylim(0.,1.)
ax.set_zlim(0.,1.)
ax.scatter3D(*zip(*xn))
plt.show()

ax = plt.axes(projection='3d')
ax.autoscale(False)
ax.set_xlim(0.,1.)
ax.set_ylim(0.,1.)
ax.set_zlim(0.,1.)
ax.scatter3D(*zip(*ym))
plt.show()

js = df.generateJs()
cs = np.zeros(10)
dField = df.DField(cs,js)
fn = xn
for i in range(1000):
	W = df.eStep(fn, ym)
	dField, fn = df.mStep(dField, xn, ym, W)
print(dField.cs)

#W = df.eStep(fn, vertices, 0.01)
#r = df.calc_r(W,fn,vertices)
#WSnake = df.calc_WSnake(W)
#print(W.shape)
#plt.matshow(W, cmap=plt.cm.Blues)
#plt.show()
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