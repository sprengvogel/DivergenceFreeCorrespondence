#import farthest_point as fp
#import trimesh
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

def load_mesh(model_file):
	pts = []
	with open(model_file, 'r') as file_vert:
		for row in file_vert:
			pts.append([float(r) for r in row.split()])
	pts = np.array(pts)
	pts = scale_numpy_array(pts, 0, 1)
	return pts

def load_faces(model_file):
	faces = []
	with open(model_file, 'r') as file_tri:
		for row in file_tri:
			faces.append([int(r)-1 for r in row.split()])
	faces = np.array(faces)
	return faces

def load_mesh_with_shot(vert_file, tri_file, n):
	pts = load_mesh(vert_file)
	faces = load_faces(tri_file)
	shot_descriptors = pyshot.get_descriptors(
	pts,
	faces,
	radius=0.1,
	local_rf_radius=0.1,
	# The following parameters are optional
	min_neighbors=3,
	n_bins=20,
	double_volumes_sectors=True,
	use_interpolation=True,
	use_normalization=True,
)
	pts = pts[0::n]
	shot_descriptors = shot_descriptors[0::n]
	return pts, shot_descriptors

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

#xn = load_points_and_mesh("tosca/cat1.vert", 30)
#ym = load_points_and_mesh("tosca/cat4.vert", 30)
xn, shot_x = load_mesh_with_shot("tosca/cat1.vert", "tosca/cat1.tri", 200)
ym, shot_y = load_mesh_with_shot("tosca/cat4.vert", "tosca/cat4.tri", 200)
#ym += 0.01
print(xn.shape)
print(ym.shape)

fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.autoscale(False)
ax.set_xlim(0.,1.)
ax.set_ylim(0.,1.)
ax.set_zlim(0.,1.)
ax.scatter3D(*zip(*xn),c=np.arange(xn.shape[0]))

ax = fig.add_subplot(1,2,2,projection='3d')
ax.autoscale(False)
ax.set_xlim(0.,1.)
ax.set_ylim(0.,1.)
ax.set_zlim(0.,1.)
ax.scatter3D(*zip(*ym),c=np.arange(xn.shape[0]))
plt.show()



js = df.generateJs(3000)
cs = np.zeros(100)
dField = df.DField(cs,js)
fn = xn
ds_euclid = np.linalg.norm(fn[:,None,:]-ym[None,:,:], axis=2)
ds_shot = np.linalg.norm(shot_x[:,None,:]-shot_y[None,:,:], axis=2)
ds_shot = (np.mean(ds_euclid)/np.mean(ds_shot))*ds_shot
for i in range(100):
	W = df.eStep(fn, ym, ds_shot, True)
	#W = np.identity(xn.shape[0])
	#dField.cs = dField.cs - 0.2
	dField, _ = df.mStep(dField, xn, ym, W, False)
	fn = df.rungeKutta(dField, xn)
print(dField.cs)

#W = df.eStep(fn, vertices, 0.01)
#r = df.calc_r(W,fn,vertices)
#WSnake = df.calc_WSnake(W)
#print(W.shape)
#plt.matshow(W, cmap=plt.cm.Blues)
#plt.show()

start = load_points_and_mesh("tosca/cat1.vert", 300)
fig = plt.figure()
ax = fig.add_subplot(1,3,1,projection='3d')
ax.autoscale(False)
ax.set_xlim(0.,1.)
ax.set_ylim(0.,1.)
ax.set_zlim(0.,1.)
ax.scatter3D(*zip(*start))

end = load_points_and_mesh("tosca/cat4.vert", 300)
ax = fig.add_subplot(1,3,2,projection='3d')
ax.autoscale(False)
ax.set_xlim(0.,1.)
ax.set_ylim(0.,1.)
ax.set_zlim(0.,1.)
ax.scatter3D(*zip(*end))

fn = df.rungeKutta(dField, start)
ax = fig.add_subplot(1,3,3,projection='3d')
ax.autoscale(False)
ax.set_xlim(0.,1.)
ax.set_ylim(0.,1.)
ax.set_zlim(0.,1.)
ax.scatter3D(*zip(*fn))
plt.show()
""" mask = np.zeros(len(mesh.vertices), dtype=bool)
print(mask.shape)
mask[indices] = True
mesh.update_vertices(mask)
mesh.show() """