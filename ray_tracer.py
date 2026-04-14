
"""
Basic recursive ray tracer for the FIT ray-tracing assignment.

Implements:
1. Basic recursive ray tracing
2. Reflection
3. Refraction
4. Glossy reflection
5. Rendering from multiple viewpoints
6. Soft shadows

The code is self-contained and produces PNG images when run.

Usage:
    python ray_tracer_assignment.py

Optional:
    python ray_tracer_assignment.py --width 640 --height 480 --samples 8 --max-depth 4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


EPSILON = 1e-4
INF = 1e18


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return direction - 2.0 * np.dot(direction, normal) * normal


def refract(direction: np.ndarray, normal: np.ndarray, ior_from: float, ior_to: float) -> Optional[np.ndarray]:
    """
    Returns refracted direction using Snell's law.
    direction points *into* the surface from the ray origin.
    """
    d = normalize(direction)
    n = normalize(normal)
    cosi = float(np.clip(np.dot(d, n), -1.0, 1.0))

    etai = ior_from
    etat = ior_to
    nn = n

    # Ensure the normal opposes the incoming ray for the math.
    if cosi < 0:
        cosi = -cosi
    else:
        nn = -n
        etai, etat = etat, etai

    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0.0:
        return None  # total internal reflection
    return normalize(eta * d + (eta * cosi - math.sqrt(k)) * nn)


def fresnel(direction: np.ndarray, normal: np.ndarray, ior_from: float, ior_to: float) -> float:
    """
    Schlick-style Fresnel reflectance.
    """
    d = normalize(direction)
    n = normalize(normal)
    cosi = float(np.clip(np.dot(d, n), -1.0, 1.0))
    etai = ior_from
    etat = ior_to
    if cosi > 0:
        etai, etat = etat, etai
    r0 = ((etai - etat) / (etai + etat)) ** 2
    c = 1.0 - abs(cosi)
    return r0 + (1.0 - r0) * (c ** 5)


@dataclass
class Material:
    color: np.ndarray
    ambient: float = 0.08
    diffuse: float = 0.80
    specular: float = 0.45
    shininess: int = 64
    reflectivity: float = 0.0
    transparency: float = 0.0
    ior: float = 1.5
    glossy: float = 0.0  # angular jitter amount for glossy reflections
    checker: bool = False
    checker_scale: float = 100.0
    checker_color2: Optional[np.ndarray] = None


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def point_at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    material: Material
    entering: bool


class Sphere:
    def __init__(self, center: Tuple[float, float, float], radius: float, material: Material):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.material = material

    def intersect(self, ray: Ray) -> Optional[Hit]:
        oc = ray.origin - self.center
        a = float(np.dot(ray.direction, ray.direction))
        b = 2.0 * float(np.dot(oc, ray.direction))
        c = float(np.dot(oc, oc) - self.radius * self.radius)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        t = None
        if t1 > EPSILON and t2 > EPSILON:
            t = min(t1, t2)
        elif t1 > EPSILON:
            t = t1
        elif t2 > EPSILON:
            t = t2

        if t is None:
            return None

        p = ray.point_at(t)
        outward_normal = normalize(p - self.center)
        entering = np.dot(ray.direction, outward_normal) < 0.0
        normal = outward_normal if entering else -outward_normal
        return Hit(t=t, point=p, normal=normal, material=self.material, entering=entering)


class Plane:
    def __init__(self, point: Tuple[float, float, float], normal: Tuple[float, float, float], material: Material):
        self.point = np.array(point, dtype=float)
        self.normal = normalize(np.array(normal, dtype=float))
        self.material = material

    def intersect(self, ray: Ray) -> Optional[Hit]:
        denom = float(np.dot(ray.direction, self.normal))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(self.point - ray.origin, self.normal) / denom)
        if t <= EPSILON:
            return None
        p = ray.point_at(t)
        entering = np.dot(ray.direction, self.normal) < 0.0
        n = self.normal if entering else -self.normal
        return Hit(t=t, point=p, normal=n, material=self.material, entering=entering)


@dataclass
class Light:
    position: np.ndarray
    color: np.ndarray
    intensity: float = 1.0
    radius: float = 0.0  # > 0 creates an area light approximation for soft shadows


class Scene:
    def __init__(self, background: Tuple[float, float, float] = (0.04, 0.05, 0.08)):
        self.objects: List[object] = []
        self.lights: List[Light] = []
        self.background = np.array(background, dtype=float)

    def add_object(self, obj: object) -> None:
        self.objects.append(obj)

    def add_light(self, light: Light) -> None:
        self.lights.append(light)

    def nearest_hit(self, ray: Ray) -> Optional[Hit]:
        best = None
        best_t = INF
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit is not None and hit.t < best_t:
                best = hit
                best_t = hit.t
        return best

    def is_occluded(self, ray: Ray, max_distance: float) -> bool:
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit is not None and hit.t < max_distance - EPSILON:
                return True
        return False


class Camera:
    def __init__(
        self,
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float],
        fov_degrees: float,
        width: int,
        height: int,
    ):
        self.eye = np.array(eye, dtype=float)
        self.target = np.array(target, dtype=float)
        self.up = np.array(up, dtype=float)
        self.width = int(width)
        self.height = int(height)
        self.fov = math.radians(fov_degrees)

        self.forward = normalize(self.target - self.eye)
        self.right = normalize(np.cross(self.forward, self.up))
        self.true_up = normalize(np.cross(self.right, self.forward))

        self.aspect = self.width / self.height
        self.half_height = math.tan(self.fov / 2.0)
        self.half_width = self.aspect * self.half_height

    def make_ray(self, x: float, y: float) -> Ray:
        """
        x, y are pixel-space coordinates and can be fractional for anti-aliasing.
        """
        ndc_x = (2.0 * ((x + 0.5) / self.width) - 1.0) * self.half_width
        ndc_y = (1.0 - 2.0 * ((y + 0.5) / self.height)) * self.half_height
        direction = normalize(self.forward + ndc_x * self.right + ndc_y * self.true_up)
        return Ray(self.eye.copy(), direction)


def material_color(material: Material, point: np.ndarray) -> np.ndarray:
    if not material.checker:
        return material.color
    c2 = material.checker_color2 if material.checker_color2 is not None else np.array([35.0, 35.0, 35.0])
    sx = math.floor(point[0] / material.checker_scale)
    sz = math.floor(point[2] / material.checker_scale)
    if (sx + sz) % 2 == 0:
        return material.color
    return c2


def soft_shadow_factor(
    scene: Scene,
    point: np.ndarray,
    normal: np.ndarray,
    light: Light,
    shadow_samples: int,
    rng: np.random.Generator,
) -> float:
    if light.radius <= 0.0 or shadow_samples <= 1:
        to_light = light.position - point
        dist = np.linalg.norm(to_light)
        ldir = normalize(to_light)
        shadow_ray = Ray(point + normal * EPSILON, ldir)
        return 0.0 if scene.is_occluded(shadow_ray, dist) else 1.0

    visible = 0
    for _ in range(shadow_samples):
        # random sample in a sphere around the light position
        offset = rng.normal(size=3)
        offset = normalize(offset) * (rng.random() ** (1.0 / 3.0)) * light.radius
        sample_pos = light.position + offset
        to_light = sample_pos - point
        dist = np.linalg.norm(to_light)
        ldir = normalize(to_light)
        shadow_ray = Ray(point + normal * EPSILON, ldir)
        if not scene.is_occluded(shadow_ray, dist):
            visible += 1
    return visible / shadow_samples


def phong_shading(
    scene: Scene,
    hit: Hit,
    ray: Ray,
    shadow_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base_color = material_color(hit.material, hit.point)
    color = base_color * hit.material.ambient

    for light in scene.lights:
        shadow = soft_shadow_factor(scene, hit.point, hit.normal, light, shadow_samples, rng)
        if shadow <= 0.0:
            continue

        to_light = normalize(light.position - hit.point)
        lambert = max(0.0, float(np.dot(hit.normal, to_light)))

        view_dir = normalize(-ray.direction)
        reflect_dir = normalize(reflect(-to_light, hit.normal))
        spec = max(0.0, float(np.dot(view_dir, reflect_dir))) ** hit.material.shininess

        diffuse = base_color * hit.material.diffuse * lambert
        specular = np.array([255.0, 255.0, 255.0]) * hit.material.specular * spec

        atten_color = (diffuse + specular) * light.intensity * shadow
        color += atten_color * (light.color / 255.0)

    return np.clip(color, 0.0, 255.0)


def sample_glossy_direction(
    perfect_dir: np.ndarray,
    glossy_amount: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if glossy_amount <= 0.0:
        return perfect_dir

    w = normalize(perfect_dir)
    helper = np.array([0.0, 1.0, 0.0]) if abs(w[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(helper, w))
    v = normalize(np.cross(w, u))

    jitter = rng.normal(scale=glossy_amount, size=2)
    d = normalize(w + jitter[0] * u + jitter[1] * v)
    return d


def trace_ray(
    scene: Scene,
    ray: Ray,
    depth: int,
    max_depth: int,
    shadow_samples: int,
    glossy_samples: int,
    rng: np.random.Generator,
    current_ior: float = 1.0,
) -> np.ndarray:
    hit = scene.nearest_hit(ray)
    if hit is None:
        return scene.background * 255.0

    local_color = phong_shading(scene, hit, ray, shadow_samples, rng)
    color = local_color.copy()

    if depth >= max_depth:
        return np.clip(color, 0.0, 255.0)

    mat = hit.material

    # Reflection
    refl_weight = float(np.clip(mat.reflectivity, 0.0, 1.0))
    if refl_weight > 0.0:
        perfect_refl = normalize(reflect(ray.direction, hit.normal))

        if mat.glossy > 0.0 and glossy_samples > 1:
            refl_color = np.zeros(3, dtype=float)
            for _ in range(glossy_samples):
                glossy_dir = sample_glossy_direction(perfect_refl, mat.glossy, rng)
                refl_ray = Ray(hit.point + hit.normal * EPSILON, glossy_dir)
                refl_color += trace_ray(
                    scene,
                    refl_ray,
                    depth + 1,
                    max_depth,
                    shadow_samples,
                    glossy_samples,
                    rng,
                    current_ior=current_ior,
                )
            refl_color /= glossy_samples
        else:
            refl_ray = Ray(hit.point + hit.normal * EPSILON, perfect_refl)
            refl_color = trace_ray(
                scene,
                refl_ray,
                depth + 1,
                max_depth,
                shadow_samples,
                glossy_samples,
                rng,
                current_ior=current_ior,
            )

        color = (1.0 - refl_weight) * color + refl_weight * refl_color

    # Refraction
    trans_weight = float(np.clip(mat.transparency, 0.0, 1.0))
    if trans_weight > 0.0:
        next_ior = mat.ior if hit.entering else 1.0
        refr_dir = refract(ray.direction, hit.normal, current_ior, next_ior)
        kr = fresnel(ray.direction, hit.normal, current_ior, next_ior)

        if refr_dir is None:
            # total internal reflection fallback
            tir_dir = normalize(reflect(ray.direction, hit.normal))
            tir_ray = Ray(hit.point + hit.normal * EPSILON, tir_dir)
            refr_color = trace_ray(
                scene,
                tir_ray,
                depth + 1,
                max_depth,
                shadow_samples,
                glossy_samples,
                rng,
                current_ior=current_ior,
            )
            color = (1.0 - trans_weight) * color + trans_weight * refr_color
        else:
            refr_ray = Ray(hit.point - hit.normal * EPSILON, refr_dir)
            refr_color = trace_ray(
                scene,
                refr_ray,
                depth + 1,
                max_depth,
                shadow_samples,
                glossy_samples,
                rng,
                current_ior=next_ior,
            )
            color = (1.0 - trans_weight) * color + trans_weight * ((1.0 - kr) * refr_color + kr * color)

    return np.clip(color, 0.0, 255.0)


def render(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    aa_samples: int,
    shadow_samples: int,
    glossy_samples: int,
    max_depth: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)

    progress_step = max(1, height // 10)
    for y in range(height):
        if y % progress_step == 0 or y == height - 1:
            print(f"Rendering row {y + 1}/{height}...")
        for x in range(width):
            pixel = np.zeros(3, dtype=float)
            for _ in range(aa_samples):
                dx = rng.random() - 0.5
                dy = rng.random() - 0.5
                ray = camera.make_ray(x + dx, y + dy)
                pixel += trace_ray(
                    scene=scene,
                    ray=ray,
                    depth=0,
                    max_depth=max_depth,
                    shadow_samples=shadow_samples,
                    glossy_samples=glossy_samples,
                    rng=rng,
                )
            img[y, x] = pixel / aa_samples

    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def build_scene(mode: str = "all") -> Scene:
    scene = Scene(background=(0.03, 0.04, 0.06))

    plane_mat = Material(
        color=np.array([220.0, 220.0, 220.0]),
        ambient=0.06,
        diffuse=0.80,
        specular=0.15,
        shininess=24,
        checker=True,
        checker_scale=80.0,
        checker_color2=np.array([50.0, 60.0, 70.0]),
    )

    red_reflective = Material(
        color=np.array([220.0, 50.0, 50.0]),
        ambient=0.08,
        diffuse=0.70,
        specular=0.50,
        shininess=100,
        reflectivity=0.40 if mode in {"reflection", "all", "glossy"} else 0.0,
        glossy=0.025 if mode in {"glossy", "all"} else 0.0,
    )

    glass = Material(
        color=np.array([170.0, 210.0, 255.0]),
        ambient=0.03,
        diffuse=0.18,
        specular=0.75,
        shininess=180,
        reflectivity=0.10 if mode in {"refraction", "all"} else 0.0,
        transparency=0.80 if mode in {"refraction", "all"} else 0.0,
        ior=1.50,
    )

    blue = Material(
        color=np.array([70.0, 130.0, 230.0]),
        ambient=0.08,
        diffuse=0.72,
        specular=0.35,
        shininess=80,
        reflectivity=0.12 if mode in {"all"} else 0.0,
    )

    scene.add_object(Plane(point=(0.0, -120.0, 0.0), normal=(0.0, 1.0, 0.0), material=plane_mat))
    scene.add_object(Sphere(center=(-95.0, -20.0, -430.0), radius=100.0, material=red_reflective))
    scene.add_object(Sphere(center=(95.0, -30.0, -320.0), radius=90.0, material=glass))
    scene.add_object(Sphere(center=(10.0, 100.0, -580.0), radius=70.0, material=blue))

    light_radius = 35.0 if mode in {"softshadows", "all"} else 0.0
    scene.add_light(Light(position=np.array([220.0, 260.0, -80.0]), color=np.array([255.0, 245.0, 235.0]), intensity=1.15, radius=light_radius))
    scene.add_light(Light(position=np.array([-260.0, 180.0, -120.0]), color=np.array([200.0, 220.0, 255.0]), intensity=0.45, radius=0.0))

    return scene


def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


def render_multiple_views(
    output_dir: str,
    width: int,
    height: int,
    aa_samples: int,
    shadow_samples: int,
    glossy_samples: int,
    max_depth: int,
    seed: int,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    scene = build_scene(mode="all")

    views = [
        ("view_front.png", (0.0, 30.0, 160.0), (0.0, 0.0, -420.0)),
        ("view_left.png", (-180.0, 40.0, 120.0), (0.0, -10.0, -420.0)),
        ("view_right.png", (180.0, 35.0, 140.0), (0.0, -10.0, -420.0)),
    ]

    saved = []
    for index, (name, eye, target) in enumerate(views):
        cam = Camera(
            eye=eye,
            target=target,
            up=(0.0, 1.0, 0.0),
            fov_degrees=45.0,
            width=width,
            height=height,
        )
        img = render(
            scene=scene,
            camera=cam,
            width=width,
            height=height,
            aa_samples=aa_samples,
            shadow_samples=shadow_samples,
            glossy_samples=glossy_samples,
            max_depth=max_depth,
            seed=seed + index,
        )
        path = os.path.join(output_dir, name)
        save_image(img, path)
        saved.append(path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic recursive ray tracer assignment solution.")
    parser.add_argument("--width", type=int, default=200, help="Output image width")
    parser.add_argument("--height", type=int, default=150, help="Output image height")
    parser.add_argument("--aa", type=int, default=1, help="Anti-aliasing samples per pixel")
    parser.add_argument("--samples", type=int, default=2, help="Shadow and glossy sample budget")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum recursion depth")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["basic", "reflection", "refraction", "glossy", "softshadows", "all"],
                        help="Feature mode")
    parser.add_argument("--output", type=str, default="raytraced_scene.png", help="Main output filename")
    parser.add_argument("--output-dir", type=str, default="render_outputs", help="Directory for multiple views")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--no-multi-view", action="store_true", default=True, help="Skip rendering extra viewpoints")
    parser.add_argument("--multi-view", dest="no_multi_view", action="store_false", help="Render extra viewpoints too")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scene = build_scene(mode=args.mode)
    camera = Camera(
        eye=(0.0, 30.0, 160.0),
        target=(0.0, -5.0, -420.0),
        up=(0.0, 1.0, 0.0),
        fov_degrees=45.0,
        width=args.width,
        height=args.height,
    )

    shadow_samples = max(1, args.samples)
    glossy_samples = max(1, min(args.samples, 12))

    img = render(
        scene=scene,
        camera=camera,
        width=args.width,
        height=args.height,
        aa_samples=max(1, args.aa),
        shadow_samples=shadow_samples,
        glossy_samples=glossy_samples,
        max_depth=max(0, args.max_depth),
        seed=args.seed,
    )
    save_image(img, args.output)

    if not args.no_multi_view:
        render_multiple_views(
            output_dir=args.output_dir,
            width=args.width,
            height=args.height,
            aa_samples=max(1, args.aa),
            shadow_samples=shadow_samples,
            glossy_samples=glossy_samples,
            max_depth=max(0, args.max_depth),
            seed=args.seed + 100,
        )

    print(f"Saved main render to: {args.output}")
    if not args.no_multi_view:
        print(f"Saved additional viewpoints to: {args.output_dir}")


if __name__ == "__main__":
    main()
