#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc as ArcPatch
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic as binstat
from scipy.integrate import trapz
from scipy.optimize import minimize
from time import time as wtime
import itertools
from os import cpu_count
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
import dill
set_loky_pickler("dill")



# TODO
# Parmeter perturbations
# Add methods to adjust center and orienation of elements like lenses and mirrors
# Make circle class funcional as an optical element
# Extend to 3D (need to update geometry methods for circles -> spheres, lines->planes, etc.)
# Unify fuzzy boundary checks
# Last partial step of GRIN ray trace can be made more accurate
# Write tests
# Write examples
# Document



# Global constants
n0 = 1 # background refracrive index
lmbda = 1064e-9 # laser wavelength
sound_speed = 343 # speed of sound

# Helper
def normalize(vector):
    return np.array(vector) / np.linalg.norm(vector)


# Class definitions

class Ray:
    def __init__(self, origin, direction, power=1, points=None):
        self.origin = np.array(origin)
        self.direction = normalize(direction)
        self.power = power
        self.points = np.array([origin])
        self.optical_path_length = 0

    def termination(self, length):
        return self.origin + self.direction * length

    def plot(self, ax, *args, **kwargs):
        ax.plot(self.points[:,0], self.points[:,1], *args, **kwargs)


class GaussianBundle:
    def __init__(self, waist, center, direction, number, total_power=1.0, dither=True, divergence=True):
        self.total_power = total_power
        self.direction = normalize(direction)
        tangent = np.array([-self.direction[1], self.direction[0]])
        self.waist = waist
        self.center = np.array(center)
        self.number = number
        # 4-sigma gaussian model
        if number == 1:
            origins = [self.center]
        else:
            origins = Detector(4*waist, center, np.arccos(tangent[0])).grid(number)
        rel = [(origin-center)@tangent for origin in origins]
        # Dither (randomize) ray origins by 1 percent of 4-sigma range to avoid artifacts
        if dither:
            flucts = np.random.random(number)
            flucts = (flucts - 0.5) * (np.max(rel) - np.min(rel)) / 100
            origins = [origin+fluct*tangent for origin, fluct in zip(origins, flucts)]
        if divergence:
            angle = np.arccos(self.direction[0])
            rads = np.random.random(number)
            rads = angle + (rads - 0.5) * lmbda / (np.pi * self.waist)
            directions = [[np.cos(rad), np.sin(rad)] for rad in rads]
        else:
            directions = [direction]*number
        powers = np.array([np.exp(-2*np.linalg.norm(origin-center)**2/waist**2) for origin in origins])
        self.powers = total_power / np.sum(powers) * powers
        self.rays = [Ray(origin, direction, power) for origin, direction, power in zip(origins, directions, self.powers)]
        self.relative_origins = np.array([(ray.origin-center)@tangent for ray in self.rays])

    @property
    def points(self):
        return [ray.points for ray in self.rays]

    @property
    def optical_path_lengths(self):
        return np.array([ray.optical_path_length for ray in self.rays])


    def plot(self, ax, *args, **kwargs):
        max_power= np.max([ray.power for ray in self.rays])
        for ray in self.rays:
            if max_power > 0:
                alpha = (ray.power/max_power)
            else:
                alpha = 0
            use_kw = {"alpha": alpha}
            use_kw = use_kw | kwargs
            ray.plot(ax, *args, **use_kw)

class LineSegment:
    def __init__(self, point1, point2):
        self.x1 = np.array(point1)
        self.x2 = np.array(point2)
        self.length = np.linalg.norm(point2 - point1)
        self.tangent = (point2 - point1) / self.length
        self.normal = np.array([self.tangent[1], -self.tangent[0]])
        self.center = point1 + self.tangent * self.length/2
        path = Path([point1, point2], [Path.MOVETO, Path.LINETO])
        self.patch = PathPatch(path, facecolor='none', edgecolor='k')

    def plot(self, ax):
        ax.add_collection(PatchCollection([self.patch], fc="none", ec="k"))

    def intersect_distance(self, ray):
        v1 = ray.origin - self.x1
        v2 = self.x2 - self.x1
        v3 = np.array([-ray.direction[1], ray.direction[0]])
        v2dotv3 = np.dot(v2, v3)
        if np.abs(v2dotv3) < 1e-16:
            return np.inf
        d = np.cross(v2, v1) / v2dotv3
        s = np.dot(v1, v3) / v2dotv3
        if np.abs(d) < 1e-14:
            return 0
        if 0 <= s <=1 and 0 <= d < np.inf:
            return d
        else:
            return np.inf

    def next_ray(self, ray, length):
        points = np.array([ray.origin, ray.termination(length)])
        optical_path_length = n0 * np.linalg.norm(np.diff(points.T))
        return ray, points[1:], optical_path_length


class Circle:
    def __init__(self, center, radius, color="k"):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.patch = plt.Circle(self.center, self.radius, ec=color, fc="none")

    def plot(self, ax):
        ax.add_collection(PatchCollection([self.patch], fc="none", ec="k"))

    def intersect_distance(self, ray):
        a = (ray.origin - self.center) @ ray.direction
        b = np.linalg.norm(ray.origin - self.center)**2 - self.radius**2
        if b <= a*a:
            d1 = -a + np.sqrt(a*a - b)
            d2 = -a - np.sqrt(a*a - b)
        else:
            d1, d2 = np.inf, np.inf
        ds = np.sort([d1, d2])
        return ds

    def normal(self, ray, length):
        surface_point = ray.termination(length)
        relative_position = (surface_point - self.center)
        return relative_position / np.linalg.norm(relative_position)


class Mirror(LineSegment):
    def __init__(self, length, center, angle):
        tangent = np.array([np.cos(angle), np.sin(angle)])
        point1 = np.array(center) - length/2 * tangent
        point2 = np.array(center) + length/2 * tangent
        super().__init__(point1, point2)

    def next_ray(self, ray, length):
        origin = ray.termination(length)
        c1 = - np.dot(ray.direction, self.normal)
        normal = self.normal
        if c1 < 0:
            normal = - self.normal
            c1 =  -c1
            self.normal = normal
        direction =  ray.direction + 2*c1*self.normal
        reflected_ray = Ray(origin, direction, ray.power)
        #transmitted_ray = Ray(origin, ray.direction, (1-self.reflectance)*ray.power)
        points = np.array([ray.origin, reflected_ray.origin])
        optical_path_length = n0 * np.linalg.norm(np.diff(points.T))
        return reflected_ray, points[1:], optical_path_length


class Detector(Mirror):
    def __init__(self, length, center, angle):
        self.reset()
        super().__init__(length, center, angle)

    def plot_result(self, axs, bundle_index=-1, **kwargs):
        self.plot_power(axs[0], bundle_index=bundle_index)
        self.plot_count(axs[1], bundle_index=bundle_index)
        self.plot_phase(axs[2], bundle_index=bundle_index)
        axs[0].set_xlabel("")
        axs[1].set_xlabel("")

    def plot_power(self, ax, bundle_index=-1, normalize=False, **kwargs):
        res = self.powers[bundle_index]
        if normalize:
            res = res / np.max(res)
        ax.plot(self.pixels, res, **kwargs)
        ax.set_ylabel("Power")
        ax.set_xlabel("Relative position")

    def plot_phase(self, ax, bundle_index=-1, **kwargs):
        ax.plot(self.pixels, self.phases[bundle_index], **kwargs)
        ax.set_ylabel("Phase")
        ax.set_xlabel("Relative position")

    def plot_count(self, ax, bundle_index=-1, **kwargs):
        ax.plot(self.pixels, self.counts[bundle_index], **kwargs)
        ax.set_ylabel("Counts")
        ax.set_xlabel("Relative position")

    @property
    def powers(self):
        return np.abs(self.amplitudes)**2

    def difference_signal(self, center=0, bundle_index=-1):
        power = self.powers[bundle_index]
        mask1 = np.logical_and(self.pixels<=center, ~np.isnan(power))
        mask2 = np.logical_and(self.pixels>center, ~np.isnan(power))
        P1 = trapz(power[mask1], x=self.pixels[mask1])
        P2 = trapz(power[mask2], x=self.pixels[mask2])
        return P2 - P1

    def total_signal(self, bundle_index=-1):
        power = self.powers[bundle_index]
        mask = ~np.isnan(power)
        return trapz(power[mask], x=self.pixels[mask])

    def set_bundles(self, bundles):
        self.bundles = bundles

    def reset(self):
        self.counts = []
        self.amplitudes = []
        self.phases = []
        self.optical_path_lengths = []

    def grid(self, N):
        steps = np.linspace(0, self.length, int(N), endpoint=True)
        grid = self.x1 + np.array(
            [self.tangent * step for step in steps])
        return grid

    def point_within(self, point):
        intervals = self.grid(2).T
        return np.all([np.logical_and(interval[0]-coordinate<=1e-14,
                                      coordinate-interval[1]<=1e-14 )
                       for coordinate, interval in zip(point, intervals)])

    def detect(self, bundle, Npixels, fix_count=False):
        final_points = np.array([points[-1] for points in bundle.points])
        hit_mask = np.array([self.point_within(point) for point in final_points])
        center_distances = np.array([(point - self.center)@self.tangent for point in final_points])
        bins = [(point - self.center)@self.tangent for point in self.grid(Npixels+1)]
        count, bins, inds = binstat(center_distances[hit_mask], hit_mask, statistic='count', bins=bins)

        if fix_count:
            if fix_count == True:
                if len(count) == 1:
                    set_count = count[0]
                else:
                    set_count = int(np.min(count[len(count)//4:3*len(count)//4]))
            else:
                set_count = fix_count
            set_count = max(1, set_count)
            inds_inds = np.argsort(inds)
            i = 0
            new_inds_list = []
            for k, v in itertools.groupby(inds[inds_inds]):
                size = len(list(v))
                old_inds = np.arange(i, i+size)
                new_inds = np.random.choice(old_inds, size=min(size, set_count))
                i += size
                new_inds_list.append(new_inds)
            new_inds = list(itertools.chain.from_iterable(new_inds_list))
            new_inds = inds_inds[new_inds]
        else:
            new_inds = ...

        count, bins, _ = binstat(center_distances[hit_mask][new_inds], hit_mask[hit_mask][new_inds], statistic='count', bins=bins)

        optical_path_lengths = bundle.optical_path_lengths[hit_mask][new_inds]

        opl, bins, _ = binstat(center_distances[hit_mask][new_inds], optical_path_lengths, statistic='mean', bins=bins)
        power, bins, _ = binstat(center_distances[hit_mask][new_inds], bundle.powers[hit_mask][new_inds], statistic='sum', bins=bins)
        #power, bins, _ = binstat(center_distances[hit_mask][new_inds], bundle.powers[hit_mask][new_inds], statistic='mean', bins=bins)
        #power = power * bundle.total_power * (np.sum(hit_mask)/len(hit_mask)) / np.nansum(power)

        phases = ((opl) * 2*np.pi/lmbda)
        amplitudes = np.sqrt(power) * np.exp(-1j * phases)
        pixels = bins[:-1]
        pixels += np.diff(bins)[0]/2

        self.pixels = pixels
        self.phases.append(phases)
        self.amplitudes.append(amplitudes)
        self.counts.append(count)
        self.optical_path_lengths.append(opl)
        return pixels, amplitudes, count, phases


    def detect_bundles(self, Npixels, bundles=None, relative_phase=0, fix_count=False):
        self.reset()
        if bundles is None:
            bundles = self.bundles
        for bundle in bundles:
            self.detect(bundle, Npixels, fix_count=fix_count)
        relative_phases = np.r_[0, relative_phase]
        amplitudes = 0
        counts = 0
        optical_path_lengths = self.optical_path_lengths - np.nanmin(self.optical_path_lengths)
        for power, count, opl, phase0 in zip(self.powers, self.counts, optical_path_lengths, relative_phases):
            phase = (opl * 2*np.pi/lmbda + phase0)
            amplitudes +=  np.sqrt(power) * np.exp(-1j * phase)
            counts += count
        self.amplitudes.append(amplitudes)
        self.counts.append(counts)
        return self.pixels, amplitudes, counts, self.phases


class BiconvexLens:
    def __init__(self, radius1, radius2, thickness, index, center=[0,0], angle=0):
        self.thickness = thickness
        self.focal_length = 1/((index-1) * (1/radius1+1/radius2 - (index-1)*thickness/(index*radius1*radius2)))
        self.index = index
        self.radius1 = radius1
        self.radius2 = radius2
        self.principal_plane1 = self.focal_length * (index - 1) * thickness / (radius2*index)
        self.principal_plane2 = self.focal_length * (index - 1) * thickness / (radius1*index)
        self.position(center, angle)

    def position(self, center, angle):
        self.center = center
        self.angle = angle
        self.direction = np.array([np.cos(angle), np.sin(angle)])
        self.tangent = np.array([-self.direction[1], self.direction[0]])
        center1 = center + self.direction * (-self.thickness/2 + self.radius1)
        center2 = center + self.direction * (self.thickness/2 - self.radius2)
        self.circle1 = Circle(center1, self.radius1)
        self.circle2 = Circle(center2, self.radius2)
        D = np.linalg.norm(center2 - center1)
        r, R = np.sort([self.radius1, self.radius2])
        if D == 0:
            self.half_height = r
        else:
            Q = R**2 - ((D**2 - r**2 + R**2) / (2*D))**2
            if Q >= 0:
                self.half_height = np.sqrt(Q)
            else:
                self.half_height = r
        th1 = np.arcsin(self.half_height/self.radius1)
        th2 = np.arcsin(self.half_height/self.radius2)
        patch1 = ArcPatch(xy=self.circle1.center,
                              width=2*self.radius1, height=2*self.radius1,
                              angle=angle*180/np.pi+180,
                              theta1=-th1*180/np.pi, theta2=th1*180/np.pi,
                              )
        patch2 = ArcPatch(xy=self.circle2.center,
                              width=2*self.radius2, height=2*self.radius2,
                              angle=angle*180/np.pi,
                              theta1=-th2*180/np.pi, theta2=th2*180/np.pi,
                              )
        self.patch = [patch1, patch2]

    def plot(self, ax):
        ax.add_collection(PatchCollection(self.patch, fc="none", ec="k"))

    def intersect_distance(self, ray):
        circles = np.array([self.circle1, self.circle2], dtype="object")
        if ray.direction @ self.direction > 0:
            ds = np.array([circles[0].intersect_distance(ray)[0], circles[1].intersect_distance(ray)[1]])
        else:
            ds = np.array([circles[1].intersect_distance(ray)[0], circles[0].intersect_distance(ray)[1]])
        mask = [False if d == np.inf else
                np.abs(self.tangent @ (ray.termination(d) - self.center)) < self.half_height
                for d in ds]
        ds = ds[mask]
        ds = ds[np.abs(ds)>1e-14]
        if len(ds) == 0:
            return np.inf
        else:
            return np.min(ds)

    def normal(self, ray):
        circles = np.array([self.circle1, self.circle2], dtype="object")
        if ray.direction @ self.direction > 0:
            ds = np.array([circles[0].intersect_distance(ray)[0], circles[1].intersect_distance(ray)[1]])
        else:
            ds = np.array([circles[1].intersect_distance(ray)[0], circles[0].intersect_distance(ray)[1]])
        mask = np.abs(ds)>1e-14
        ds = ds[mask]
        ind = np.argmin(ds)
        normal = circles[mask][ind].normal(ray, ds[ind])
        return normal

    def snells_law(self, ray, n1, n2, length):
        origin = ray.termination(length)
        normal = self.normal(ray)
        c1 = - np.dot(ray.direction, normal)
        if c1 < 0:
            normal = -normal
            c1 =  -c1
        c2 = np.sqrt(np.abs(1 - n1**2/n2**2 * (1-c1**2)))
        direction =  n1/n2*ray.direction + (n1/n2*c1 - c2)*normal
        ray = Ray(origin, direction, ray.power)
        return ray

    def next_ray(self, ray, length):
        internal_ray = self.snells_law(ray, n0, self.index, length)
        internal_ray_length = self.intersect_distance(internal_ray)
        final_ray = self.snells_law(internal_ray, self.index, n0, internal_ray_length)
        points = [internal_ray.origin, final_ray.origin]
        optical_path_length = n0*np.linalg.norm(internal_ray.origin - ray.origin)
        optical_path_length += self.index * np.linalg.norm(final_ray.origin - internal_ray.origin)
        return final_ray, points, optical_path_length


class Domain:
    def __init__(self, meshspec, spec="number"):
        self.meshspec = meshspec
        if spec == "number":
            axes = [np.linspace(*gs) for gs in meshspec]
        elif spec == "delta":
            for gs in meshspec:
                gs[1] = gs[1] + gs[2]
            axes = [np.arange(*gs) for gs in meshspec]
        self.axes = axes
        self.grid = np.meshgrid(*axes, indexing="ij")

    @property
    def extent(self):
        return [self.meshspec[0][0], self.meshspec[0][1], self.meshspec[1][0], self.meshspec[1][1]]

    @property
    def vertices(self):
        return np.array([
            [self.extent[0], self.extent[2]],
            [self.extent[1], self.extent[2]],
            [self.extent[1], self.extent[3]],
            [self.extent[0], self.extent[3]]])

    @property
    def shape(self):
        return self.grid[0].shape

    @property
    def deltas(self):
        return [x[1] - x[0] for x in self.axes]

    @property
    def points(self):
        return np.c_[[X.ravel() for X in self.grid]].T

    def point_within(self, point):
        return np.all([np.logical_and(interval[0]-coordinate<=1e-14,
                                      coordinate-interval[1]<=1e-14 )
                       for coordinate, interval in zip(point, self.meshspec)])

    def interpolation(self, V):
        assert tuple(len(ax) for ax in self.axes) == V.shape
        interp = RegularGridInterpolator(
            self.axes, V, method="linear", bounds_error=False, fill_value=0
        )
        return interp

    def vector_interpolation(self, VXYZ):
        return tuple([self.interpolation(V) for V in VXYZ])
        def interp(point):
            if type(point[0]) in (list, np.ndarray):
                return np.array([f(point) for f in interps])
            else:
                return np.array([f(point)[0] for f in interps])
        return interp


class GradientIndexRegion(Domain):
    def __init__(self, index_func, step_size, meshspec, spec="number", grad_index_func=None, **kwargs):
        super().__init__(meshspec, spec)
        self.step_size = step_size
        self.bbox = [LineSegment(self.vertices[i], self.vertices[(i+1)%len(self.vertices)])
                     for i in range(len(self.vertices))]
        self.index_func = index_func
        self.index_grid = np.array([index_func(point, **kwargs) for point in self.points]).reshape(self.shape)
        if grad_index_func is None:
            grad_index_grid = np.gradient(self.index_grid, *self.deltas)
            grad_index_interps = self.vector_interpolation(grad_index_grid)
            grad_index_func = lambda point, **kwargs: np.array([g(point)[0] for g in grad_index_interps])
        self.grad_index_func = grad_index_func
        self.patch = [edge.patch for edge in self.bbox]
        self.kwargs = kwargs

    def plot(self, ax, **kwargs):
        ax.imshow(self.index_grid.T, extent=self.extent, origin="lower", alpha=0.7, **kwargs)
        ax.add_collection(PatchCollection(self.patch, fc="none", ec="k"))

    def point_within(self, point):
        intervals = [[self.extent[0], self.extent[1]], [self.extent[2], self.extent[3]]]
        return np.all([np.logical_and(interval[0]-coordinate<=1e-14,
                                      coordinate-interval[1]<=1e-14 )
                       for coordinate, interval in zip(point, intervals)])


    def grad_index_interp(self, point):
        return np.array([g(point)[0] for g in self.gradn_interp])

    def _ngradn(self, point):
        return self.index_func(point,  **self.kwargs)*self.grad_index_func(point, **self.kwargs)

    # RK-3 method: https://opg.optica.org/ao/fulltext.cfm?uri=ao-21-6-984&id=25666
    def _step(self, point, ndirection):
        #if np.sum(np.abs(self._ngradn(point))) < 1e-4:
        #    return point+ self.step_size*ndirection, ndirection
        A = self.step_size * self._ngradn(point)
        B = self.step_size * self._ngradn(point + self.step_size/2 * ndirection + self.step_size/8 * A)
        C = self.step_size * self._ngradn(point + self.step_size*ndirection + self.step_size/2 * B)
        new_point = point + self.step_size*(ndirection + (A + 2*B) / 6)
        new_ndirection = ndirection + (A + 4*B + C) / 6
        return new_point, new_ndirection

    def _trace(self, R0, T0):
        Rs = [R0]
        Ts = [T0]
        i = 0
        while (self.point_within(Rs[i])):
            R, T = self._step(Rs[i], Ts[i])
            Rs.append(R)
            Ts.append(T)
            i += 1
        return np.array(Rs[:-1]), np.array(Ts[:-1])

    def intersect_distance(self, ray, all=False):
        ds = np.array([edge.intersect_distance(ray) for edge in self.bbox])
        if len(ds) > 0:
            return np.min(ds)
        else:
            return np.inf

    # opl calculation: https://opg.optica.org/ao/fulltext.cfm?uri=ao-24-24-4367&id=28953
    def next_ray(self, ray, length):
        # ray to region then trace region
        point0 = ray.termination(length)
        index = self.index_func(point0, **self.kwargs)
        ndirection0 = index * ray.direction
        points, ndirections = self._trace(point0, ndirection0)
        # optical pathlength of region, minus the last partial step
        Nm = np.array([self.index_func(R, **self.kwargs) for R in points])**2
        term1 = self.step_size*np.sum(Nm[1:-1])
        term2 = self.step_size**2/12 * (
            2 * self._ngradn(points[-1])@ndirections[-1] - 2 * self._ngradn(points[0])@ndirections[0])
        term3 = self.step_size/2 * (Nm[-1] - Nm[0])
        opl = term1 - term2 + term3
        # last partial step assumes constant index

        internal_ray = Ray(points[-1], normalize(ndirections[-1]), 1)
        internal_ray_length = self.intersect_distance(internal_ray)
        # Construct final ray
        origin = internal_ray.termination(internal_ray_length)
        direction = internal_ray.direction
        final_ray = Ray(origin, direction, ray.power)
        points = np.r_[points, [origin]]
        # correct opl for last partial step and initial propagation to region
        opl += (internal_ray_length + length)*n0
        return final_ray, points, opl


class SphericalSource:
    def __init__(self, temporal_profile, temporal_derivative, reference_amplitude, reference_distance, epicenter, sound_speed=343):
        self.temporal_profile = temporal_profile
        self.temporal_derivative = temporal_derivative
        self.reference_amplitude = reference_amplitude
        self.reference_distance = reference_distance
        self.epicenter = np.array(epicenter)
        self.sound_speed = sound_speed

    def index_func(self, point, time, **kwargs):
        distance = np.linalg.norm(point - self.epicenter)
        if distance < self.sound_speed * time:
            index = self.temporal_profile(time-distance/self.sound_speed, **kwargs)
            index *= self.reference_amplitude * self.reference_distance / distance
            return n0 + index
        else:
            return n0

    # Function for the gradient of a traveling spherical wave of known tmporal profile and derivative
    def grad_index_func(self, point, time, **kwargs):
        distance = np.linalg.norm(point - self.epicenter)
        if distance < self.sound_speed * time:
            ret_time  = time - distance/self.sound_speed
            grad_index = (-self.reference_distance*self.reference_amplitude / (self.sound_speed*distance**3))
            grad_index *= (self.temporal_profile(ret_time, **kwargs) + distance*self.temporal_derivative(ret_time, **kwargs))
            grad_index *= (point - self.epicenter)
            return grad_index
        else:
            return np.zeros_like(point)



class OpticalSystem:
    def __init__(self, elements=[], extent=[0, 1, 0, 1]):
        self.extent = extent
        vertices = np.array([
            [extent[0], extent[2]],
            [extent[1], extent[2]],
            [extent[1], extent[3]],
            [extent[0], extent[3]],
        ])
        bbox = [LineSegment(vertices[i], vertices[(i+1)%len(vertices)])
                     for i in range(len(vertices))]
        self.elements = np.array(elements, dtype="object")
        self.add_element(bbox)

    def plot(self, ax, **kwargs):
        for elem in self.elements:
            if type(elem) == GradientIndexRegion:
                elem.plot(ax, **kwargs)
            else:
                elem.plot(ax)

    def add_element(self, elem):
        self.patches = []
        self.elements = np.r_[self.elements, elem]
        for elem in self.elements:
            self.patches = np.r_[self.patches, elem.patch]

    def trace(self, ray, maxiter=100):
        optical_path_length = 0
        elem = None
        i = 0
        work_ray = ray
        while type(elem) not in (Detector, LineSegment) and i<=maxiter:
            lengths = np.array([elem.intersect_distance(work_ray) for elem in self.elements])
            mask = lengths>1e-14 # insure we dont get stuck at the same element
            ind = np.argmin(lengths[mask])
            length = lengths[mask][ind]
            elem = self.elements[mask][ind]
            # next_ray methods return a new ray, so we call the method on
            # a copy, but accumulate the points and optical path length in the
            # original
            work_ray, points, opl =  elem.next_ray(work_ray, length)
            ray.points = np.r_[ray.points, points]
            ray.optical_path_length += opl
            i+= 1
        return ray

    def trace_rays(self, rays):
        return [self.trace(ray) for ray in rays]


def trace_bundle(sys, bundle, n_jobs=-1, n_rays_per_job=None):
    number = bundle.number
    if n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1
    if n_rays_per_job is None:
        n_rays_per_job = max(1, number // n_jobs)
    t0 = wtime()
    rays_list = [bundle.rays[n_rays_per_job*i:n_rays_per_job*(i+1)] for i in range(1+number//n_rays_per_job)]
    new_rays_list = Parallel(n_jobs=n_jobs)(delayed(sys.trace_rays)(rays)
                         for rays in rays_list)
    rays = list(itertools.chain.from_iterable(new_rays_list))
    bundle.rays = rays
    wall_time = wtime() - t0
    print(f"Traced {number} rays in {round(wall_time, 3)} s using {n_jobs} cores")
    return number, wall_time



# Function to find lens radius and thickness of symmetric biconvex lens of a given focal length
def make_lens(target_focal_length, approximate_thickness=0.01, index=1.5):
    def error(params):
        radius, thickness = params
        lens = BiconvexLens(radius, radius, thickness, index, [0.0, 0.0], 0.0)
        return np.abs(target_focal_length - lens.focal_length)
    res = minimize(error, (target_focal_length, approximate_thickness), bounds=((1e-6, np.inf),(1e-6, np.inf)))
    radius, thickness = res.x
    lens = BiconvexLens(radius, radius, thickness, index, [0.0, 0.0], 0.0)
    return lens
