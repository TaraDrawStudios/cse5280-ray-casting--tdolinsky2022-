# Ray Tracing Assignment Report  
**Student:** Tara Dolinsky  
**Course:** Computer Graphics / Ray Tracing  
**Date:** April 2026  

---

## 📌 Overview

In this assignment, I implemented a basic recursive ray tracer that renders a 3D scene consisting of spheres and a plane. The program simulates realistic lighting using the Phong reflection model and includes advanced features such as reflection, refraction, glossy surfaces, soft shadows, and multiple viewpoints.

The goal was to understand how light interacts with objects and how recursive ray tracing can be used to simulate realistic rendering.

---

## 🎯 Features Implemented

### 1. Basic Ray Tracing

The ray tracer works by shooting rays from the camera into the scene. Each ray checks for intersections with objects and calculates the color based on lighting.

The ray equation is:

\[
\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}
\]

Where:
- \( \mathbf{o} \) = ray origin  
- \( \mathbf{d} \) = ray direction  
- \( t \) = distance  

---

### 2. Object Intersections

#### Sphere Intersection

Spheres are intersected using the quadratic formula:

\[
a t^2 + b t + c = 0
\]

Where:
\[
a = \mathbf{d} \cdot \mathbf{d}
\]
\[
b = 2(\mathbf{o} - \mathbf{c}) \cdot \mathbf{d}
\]
\[
c = (\mathbf{o} - \mathbf{c}) \cdot (\mathbf{o} - \mathbf{c}) - r^2
\]

---

#### Plane Intersection

Plans use the formula:

\[
t = \frac{(\mathbf{p} - \mathbf{o}) \cdot \mathbf{n}}{\mathbf{d} \cdot \mathbf{n}}
\]

Where:
- \( \mathbf{n} \) = normal plane  
- \( \mathbf{p} \) = point on plane  

---

### 3. Phong Reflection Model

Lighting is calculated using the Phong model, which includes:

- Ambient lighting
- Diffuse reflection
- Specular highlights

#### Diffuse Lighting:

\[
I_d = k_d (\mathbf{n} \cdot \mathbf{l})
\]

#### Specular Lighting:

\[
I_s = k_s (\mathbf{v} \cdot \mathbf{r})^{\alpha}
\]

Where:
- \( \mathbf{n} \) = surface normal  
- \( \mathbf{l} \) = light direction  
- \( \mathbf{v} \) = view direction  
- \( \mathbf{r} \) = reflection direction  

---

### 4. Reflection

Reflection rays are computed using:

\[
\mathbf{r} = \mathbf{d} - 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n}
\]

The reflected ray is traced recursively to simulate mirror-like surfaces.

---

### 5. Refraction

Refraction is implemented using Snell's Law:

\[
n_1 \sin(\theta_1) = n_2 \sin(\theta_2)
\]

This allows light to bend when passing through transparent objects like glass.

Fresnel reflection is also used to mix reflection and refraction for realism.

---

### 6. Glossy Reflection

Glossy reflections are simulated by slightly randomizing reflection rays:

- Perfect reflection → sharp mirror
- Glossy reflection → blurred reflection

This is done by adding small random offsets to reflection directions.

---

### 7. Soft Shadows

Soft shadows are created by sampling multiple rays toward an area light.

Instead of a single shadow ray:
- Multiple rays are cast
- Some hit the object, some don't
- Final shadow = average visibility

This produces smoother and more realistic shadows.

---

### 8. Multiple Viewpoints

The scene is rendered from different camera positions to show:

- front view
- left view
- right view

This demonstrates that the scene works from multiple perspectives.

---

## 🖼️ Scene Description

The scene includes:
- A checkerboard plane
- Multiple spheres:
  - Reflective sphere
  - Glass (refractive) sphere
  - Colored diffuse sphere
- Two light sources:
  - One main light
  - One secondary light

---

## ⚙️ Program Controls

The program allows switching features using command-line arguments:
--modebasic
--mode reflection
--refraction mode
--mode glossy
--mode softshadows
--mode all


This allows testing each feature individually.

---

## 🧠Challenges

Some challenges I faced:
- Making recursion efficient without slowing down the program too much
- Handling edge cases like total internal reflection
- Balancing realism vs performance
- Debugging lighting calculations

---

## ✅ Conclusion

This project helped me understand how ray tracing works at a low level. I learned how light interacts with surfaces and how recursive algorithms can simulate realistic rendering effects.

Overall, the final program successfully implements all required features and produces realistically rendered images.

---
