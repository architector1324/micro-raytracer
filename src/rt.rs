use rand::Rng;
use std::f32::consts::PI;
use rand::distributions::Uniform;

use crate::lin::{Vec3f, Vec2f, Mat3f, Mat4f, Vec4f};

const E: f32 = 0.0001;

#[derive(Debug)]
pub struct Render {
    pub rt: RayTracer,
    pub frame: Frame,
    pub scene: Scene
}

#[derive(Debug, Clone)]
pub struct RayTracer {
    pub bounce: usize,
    pub sample: usize,
    pub loss: f32,
    pub sampler: Uniform<f32>,
}

#[derive(Debug, Clone)]
pub struct RaytraceIterator<'a> {
    rt: &'a RayTracer,
    scene: &'a Scene,
    next_ray: Ray
}

#[derive(Debug, Clone)]
pub struct Ray {
    pub orig: Vec3f,
    pub dir: Vec3f,
    pub t: f32,
    pub pwr: f32,
    pub bounce: usize
}

#[derive(Debug, Clone)]
pub struct RayHit<'a> {
    pub obj: &'a Renderer,
    pub inst: &'a RendererInstance,
    pub idx: Option<usize>,
    pub ray: Ray,
    pub norm: Vec3f
}

#[derive(Debug)]
pub struct Camera {
    pub pos: Vec3f,
    pub dir: Vec4f,
    pub fov: f32,
    pub gamma: f32,
    pub exp: f32,
    pub aprt: f32,
    pub foc: f32
}

#[derive(Debug)]
pub struct Frame {
    pub res: (u16, u16),
    pub ssaa: f32,
    pub cam: Camera
}

#[derive(Debug, Clone)]
pub struct Texture {
    pub w: usize,
    pub h: usize,
    pub dat: Option<Vec<Vec3f>>
}

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Vec3f,
    pub rough: f32,
    pub metal: f32,
    pub glass: f32,
    pub opacity: f32,
    pub emit: f32,

    pub tex: Option<Texture>,
    pub rmap: Option<Texture>, // rough map
    pub mmap: Option<Texture>, // metal map
    pub gmap: Option<Texture>, // glass map
    pub omap: Option<Texture>, // opacity map
    pub emap: Option<Texture>, // emit map
}

#[derive(Debug, Clone)]
pub struct BVH {
    aabb: Vec3f,
    rel_pos: Vec3f,
    content: Option<Vec<usize>>,
    childs: Option<Vec<Option<BVH>>>
}

#[derive(Debug)]
pub enum RendererKind {
    Sphere{r: f32},
    Plane{n: Vec3f},
    Box{sizes: Vec3f},
    Triangle{vtx: (Vec3f, Vec3f, Vec3f)},
    Mesh {
        mesh: Vec<(Vec3f, Vec3f, Vec3f)>,
        bvh: BVH
    }
}

#[derive(Debug)]
pub struct RendererInstance {
    pub pos: Vec3f,
    pub dir: Vec4f
}

#[derive(Debug)]
pub struct Renderer {
    pub kind: RendererKind,
    pub mat: Material,
    pub instance: Vec<RendererInstance>
}

#[derive(Debug)]
pub enum LightKind {
    Point {
        pos: Vec3f
    },
    Dir {
        dir: Vec3f
    }
}

#[derive(Debug)]
pub struct Light {
    pub kind: LightKind,
    pub pwr: f32,
    pub color: Vec3f
}

#[derive(Debug)]
pub struct Sky {
    pub color: Vec3f,
    pub pwr: f32
}

#[derive(Debug)]
pub struct Scene {
    pub renderer: Option<Vec<Renderer>>,
    pub light: Option<Vec<Light>>,
    pub sky: Sky
}

// data
impl <'a> From<&'a Ray> for Vec3f {
    fn from(ray: &'a Ray) -> Self {
        ray.orig + ray.dir * ray.t
    }
}

impl Default for Ray {
    fn default() -> Self {
        Ray {
            orig: Vec3f::zero(),
            dir: Vec3f::zero(),
            pwr: 1.0,
            bounce: 0,
            t: 0.0
        }
    }
}

// raytracing
impl Ray {
    pub fn cast(orig: Vec3f, dir: Vec3f, pwr: f32, bounce: usize) -> Ray {
        Ray{orig: orig + dir * E, dir: dir, pwr: pwr, bounce: bounce, t: 0.0}
    }

    pub fn cast_default(orig: Vec3f, dir: Vec3f) -> Ray {
        Ray{orig: orig + dir * E, dir: dir, ..Default::default()}
    }

    pub fn reflect(&self, rt: &RayTracer, hit: &RayHit) -> Ray {
        let mut rough = hit.get_rough();
        let opacity = hit.get_opacity();

        // 80% chance to diffuse for dielectric
        if hit.obj.mat.metal == 0.0 && opacity != 0.0 && rand::thread_rng().gen_bool(0.80) {
            rough = 1.0;
        }

        let norm = rt.rand(hit.norm, rough);
        let dir = self.dir.reflect(norm).norm();

        Ray::cast(self.into(), dir, self.pwr * (1.0 - rt.loss.min(1.0)), self.bounce + 1)
    }

    pub fn refract(&self, rt: &RayTracer, hit: &RayHit) -> Option<Ray> {
        let mut rough = hit.get_rough();
        let opacity = hit.get_opacity();

        // 80% chance to diffuse for dielectric
        if hit.obj.mat.metal == 0.0 && opacity != 0.0 && rand::thread_rng().gen_bool(0.80) {
            rough = 1.0;
        }

        let norm = rt.rand(hit.norm, rough);

        let eta = 1.0 + 0.5 * hit.get_glass();
        let dir = self.dir.refract(eta, norm)?.norm();

        Some(Ray::cast(self.into(), dir, self.pwr * (1.0 - rt.loss.min(1.0)), self.bounce + 1))
    }

    pub fn intersect_box(&self, aabb: Vec3f, pos: Vec3f) -> Option<(f32, f32)> {
        let mut m = self.dir.recip();

        // workaround for zero division
        if m.x.is_infinite() {
            m.x = E.recip();
        }

        if m.y.is_infinite() {
            m.y = E.recip();
        }

        if m.z.is_infinite() {
            m.z = E.recip();
        }

        let n = (self.orig - pos).hadam(m);
        let k = (0.5 * aabb).hadam(m.abs());

        let a = -n - k;
        let b = -n + k;

        let t0 = a.x.max(a.y).max(a.z);
        let t1 = b.x.min(b.y).min(b.z);

        if t0 > t1 || t1 < 0.0 {
            return None
        }

        Some((t0, t1))
    }

    pub fn intersect_plane(&self, n: Vec3f, pos: Vec3f) -> Option<(f32, f32)> {
        let d = -n.norm() * pos;
        let t = -(self.orig * n.norm() + d) / (self.dir * n.norm());

        if t <= 0.0 {
            return None;
        }
        Some((t, t))
    }

    pub fn intersect_sphere(&self, r: f32, pos: Vec3f) -> Option<(f32, f32)> {
        let o = self.orig - pos;

        let a = self.dir * self.dir;
        let b = 2.0 * (o * self.dir);
        let c = o * o - r.powi(2);

        let disc = b.powi(2) - 4.0 * a * c;

        if disc < 0.0 {
            return None
        }

        let t0 = (-b - disc.sqrt()) / (2.0 * a);
        let t1 = (-b + disc.sqrt()) / (2.0 * a);

        if t0 < 0.0 {
            return None
        }

        Some((t0, t1))
    }

    pub fn intersect_triangle(&self, tri: (Vec3f, Vec3f, Vec3f), pos: Vec3f) -> Option<(f32, f32)> {
        let e0 = tri.1 - tri.0;
        let e1 = tri.2 - tri.0;

        let p = self.dir.cross(e1);
        let d = e0 * p;

        if d < E && d > -E {
            return None;
        }

        let inv_d = d.recip();
        let t = self.orig - (tri.0 + pos);
        let u = (t * p) * inv_d;

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = t.cross(e0);
        let v = (self.dir * q) * inv_d;

        if v < 0.0 || (u + v) > 1.0 {
            return None;
        }

        let t = (e1 * q) * inv_d;

        if t < 0.0 {
            return None
        }

        Some((t, t))
    }
}

impl <'a> RayHit<'a> {
    pub fn get_color(&self) -> Vec3f {
        self.obj.get_color(self.inst, Some((&self.ray).into()))
    }

    pub fn get_rough(&self) -> f32 {
        self.obj.get_rough(self.inst, Some((&self.ray).into()))
    }

    pub fn get_metal(&self) -> f32 {
        self.obj.get_metal(self.inst, Some((&self.ray).into()))
    }

    pub fn get_glass(&self) -> f32 {
        self.obj.get_glass(self.inst, Some((&self.ray).into()))
    }

    pub fn get_opacity(&self) -> f32 {
        self.obj.get_opacity(self.inst, Some((&self.ray).into()))
    }

    pub fn get_emit(&self) -> f32 {
        self.obj.get_emit(self.inst, Some((&self.ray).into()))
    }
}

impl Texture {
    pub fn get_color(&self, uv: Vec2f) -> Vec3f {
        let x = (uv.x * self.w as f32) as usize;
        let y = (uv.y * self.h as f32) as usize;

        if let Some(dat) = &self.dat {
            return dat[(x + y * self.w) as usize];
        }
        return Vec3f::zero()
    }
}


impl Renderer {
    fn intersect_bvh(&self, inst: &RendererInstance, ray: &Ray, bvh: &BVH) -> Option<Vec<usize>> {
        if ray.intersect_box(bvh.aabb, inst.pos + bvh.rel_pos).is_none() {
            return None;
        }

        if bvh.content.is_some() {
            return bvh.content.clone();
        }

        Some(
            bvh.childs.as_ref().unwrap()
                .iter()
                .filter_map(|c| c.as_ref())
                .filter_map(|c| self.intersect_bvh(inst, ray, c))
                .flat_map(|v| v).collect()
        )
    }

    pub fn intersect(&self, inst: &RendererInstance, ray: &Ray) -> Option<((f32, Option<usize>), (f32, Option<usize>))> {
        let rot_y = Mat3f::rotate_y(-inst.dir);
        let look = Mat4f::lookat(-inst.dir, Vec3f::up());

        let n_ray = Ray {
            orig: inst.pos + rot_y * (look * (ray.orig - inst.pos)),
            dir: rot_y * (look * ray.dir),
            ..ray.clone()
        };

        match self.kind {
            RendererKind::Sphere{r} => {
                return n_ray.intersect_sphere(r, inst.pos).map(|(t0, t1)| ((t0, None), (t1, None)));
            },
            RendererKind::Plane{n} => {
                return n_ray.intersect_plane(n, inst.pos).map(|(t0, t1)| ((t0, None), (t1, None)));
            },
            RendererKind::Box{sizes} => {
                return n_ray.intersect_box(sizes, inst.pos).map(|(t0, t1)| ((t0, None), (t1, None)));
            },
            RendererKind::Triangle{vtx} => {
                return n_ray.intersect_triangle(vtx, inst.pos).map(|(t0, t1)| ((t0, None), (t1, None)));
            },
            RendererKind::Mesh{ref mesh, ref bvh} => {
                // get indexes
                let bvh_idx = self.intersect_bvh(inst, &n_ray, bvh);
                if bvh_idx.is_none() {
                    return None;
                }

                // check intersections
                let mut hits = Vec::new();

                let mut bvh_idx = bvh_idx.unwrap();
                bvh_idx.dedup();

                for idx in bvh_idx {
                    if let Some((t0, t1)) = n_ray.intersect_triangle(mesh[idx], inst.pos) {
                        hits.push(((t0, Some(idx)), (t1, Some(idx))));
                    }
                }

                let max = hits.iter().min_by(|((max, _), _), ((t0, _), _)| max.total_cmp(&t0)).copied();
                let min = hits.iter().max_by(|(_, (min, _)), (_, (t1, _))| min.total_cmp(&t1)).copied();

                if max.is_none() || min.is_none() {
                    return None;
                }

                Some((max.unwrap().0, min.unwrap().1))
            }
        }
    }

    pub fn gen_bvh(mesh: &Vec<(Vec3f, Vec3f, Vec3f)>, deep: usize) -> Option<BVH> {
        // helpers
        let gen_pos = || -> [Vec3f; 8] {
            [
                Vec3f::from([1.0, 1.0, 1.0]),
                Vec3f::from([-1.0, 1.0, 1.0]),
                Vec3f::from([-1.0, -1.0, 1.0]),
                Vec3f::from([1.0, -1.0, 1.0]),
                Vec3f::from([1.0, 1.0, -1.0]),
                Vec3f::from([-1.0, 1.0, -1.0]),
                Vec3f::from([-1.0, -1.0, -1.0]),
                Vec3f::from([1.0, -1.0, -1.0]),
            ]
        };

        let tri_in_aabb = |tri: &(Vec3f, Vec3f, Vec3f), aabb: Vec3f, rel_pos: Vec3f| -> bool {
            let v0 = rel_pos + 0.5 * aabb;
            let v1 = rel_pos - 0.5 * aabb;

            let vtx_in_aabb = |vtx: Vec3f| {
                if vtx.x > v0.x || vtx.y > v0.y || vtx.z > v0.z {
                    return false;
                }
    
                if vtx.x < v1.x || vtx.y < v1.y || vtx.z < v1.z {
                    return false;
                }

                true
            };

            if vtx_in_aabb(tri.0) || vtx_in_aabb(tri.1) || vtx_in_aabb(tri.2) {
                return true;
            }

            false
        };

        fn construct(aabb: Vec3f, rel_pos: Vec3f, mesh: &Vec<(Vec3f, Vec3f, Vec3f)>, d: usize, deep: usize, gen_pos: fn() -> [Vec3f; 8], check_tri: fn(tri: &(Vec3f, Vec3f, Vec3f), aabb: Vec3f, rel_pos: Vec3f) -> bool) -> Option<BVH> {
            let mut child = BVH {
                aabb: aabb,
                rel_pos,
                content: None,
                childs: None
            };

            // get content
            if d >= deep {
                let tmp = mesh.iter()
                    .enumerate()
                    .filter_map(|(idx, tri)| {
                        if check_tri(tri, child.aabb, child.rel_pos) {
                            return Some(idx)
                        }
                        None
                    })
                    .collect::<Vec<_>>();
    
                if tmp.len() != 0 {
                    child.content = Some(tmp);
                }

                return Some(child);
            }

            // get childs
            let tmp = gen_pos().iter()
                .copied()
                .map(|v| construct(0.5 * child.aabb, child.rel_pos + child.aabb.hadam(v * 0.25), mesh, d + 1, deep, gen_pos, check_tri))
                .filter(|c| c.is_some())
                .filter(|c| {
                    let c = c.as_ref().unwrap();
                    c.content.is_some() || c.childs.is_some()
                })
                .collect::<Vec<_>>();
            
            if tmp.len() != 0 {
                child.childs = Some(tmp);
            }

            Some(child)
        }

        // construct bvh
        let it = mesh.iter().copied().flat_map(|v| [v.0, v.1, v.2]);

        let root = construct(
            Vec3f::from([
                2.0 * it.clone().max_by(|max, v| max.x.abs().total_cmp(&v.x.abs()))?.x.abs(),
                2.0 * it.clone().max_by(|max, v| max.y.abs().total_cmp(&v.y.abs()))?.y.abs(),
                2.0 * it.max_by(|max, v| max.z.abs().total_cmp(&v.z.abs()))?.z.abs(),
            ]),
            Vec3f::zero(),
            mesh,
            0,
            deep,
            gen_pos,
            tri_in_aabb
        );

        root
    }

    pub fn normal<'a>(&self, inst: &RendererInstance, hit: &RayHit<'a>) -> Vec3f {
        let hit_p = Vec3f::from(&hit.ray);

        let rot_y = Mat3f::rotate_y(-inst.dir);
        let look = Mat4f::lookat(-inst.dir, Vec3f::up());

        let n_hit = inst.pos + rot_y * (look * (hit_p - inst.pos));

        let n = match self.kind {
            RendererKind::Sphere{..} => n_hit - inst.pos,
            RendererKind::Plane{n} => n,
            RendererKind::Box{sizes} => {
                let p = (n_hit - inst.pos).hadam(sizes.recip() * 2.0);

                let pos_r = 1.0-E..1.0+E;
                let neg_r = -1.0-E..-1.0+E;

                let mut n = Vec3f::zero();

                if pos_r.contains(&p.x) {
                    // right
                    n = Vec3f::right()
                } else if neg_r.contains(&p.x) {
                    // left
                    n = -Vec3f::right()
                } else if pos_r.contains(&p.y) {
                    // forward
                    n = Vec3f::forward()
                } else if neg_r.contains(&p.y) {
                    // backward
                    n = -Vec3f::forward()
                } if pos_r.contains(&p.z) {
                    // top
                    n = Vec3f::up()
                } else if neg_r.contains(&p.z) {
                    // bottom
                    n = -Vec3f::up()
                }

                n
            },
            RendererKind::Triangle{vtx} => {
                let e0 = vtx.1 - vtx.0;
                let e1 = vtx.2 - vtx.0;

                e0.cross(e1)
            },
            RendererKind::Mesh{ref mesh, ..} => {
                let tri = mesh[hit.idx.unwrap()];

                let e0 = tri.1 - tri.0;
                let e1 = tri.2 - tri.0;

                e0.cross(e1)
            }
        };

        (rot_y * (look * n)).norm()
    }

    pub fn to_uv(&self, inst: &RendererInstance, hit: Vec3f) -> Vec2f {
        let rot_y = Mat3f::rotate_y(-inst.dir);
        let look = Mat4f::lookat(-inst.dir, Vec3f::up());
        let n_hit = inst.pos + rot_y * (look * (hit - inst.pos));

        match self.kind {
            RendererKind::Sphere{..} => {
                let v = (n_hit - inst.pos).norm();
                Vec2f {
                    x: 0.5 + 0.5 * v.x.atan2(-v.y) / PI,
                    y: 0.5 - 0.5 * v.z
                }
            },
            RendererKind::Plane{..} => {
                let mut x = (n_hit.x + 0.5).fract();
                if x < 0.0 {
                    x = 1.0 + x;
                }

                let mut y = (n_hit.y + 0.5).fract();
                if y < 0.0 {
                    y = 1.0 + y;
                }

                Vec2f{x, y}
            },
            RendererKind::Box{sizes} => {
                let p = (n_hit - inst.pos).hadam(sizes.recip() * 2.0);

                let pos_r = 1.0-E..1.0+E;
                let neg_r = -1.0-E..-1.0+E;

                if pos_r.contains(&p.x) {
                    // right
                    return Vec2f {
                        x: (0.5 + 0.5 * p.y) / 4.0 + 2.0 / 4.0,
                        y: (0.5 - 0.5 * p.z) / 3.0 + 1.0 / 3.0
                    }
                } else if neg_r.contains(&p.x) {
                    // left
                    return Vec2f {
                        x: (0.5 - 0.5 * p.y) / 4.0,
                        y: (0.5 - 0.5 * p.z) / 3.0 + 1.0 / 3.0
                    }
                } else if pos_r.contains(&p.y) {
                    // forward
                    return Vec2f {
                        x: (0.5 - 0.5 * p.x) / 4.0 + 3.0 / 4.0,
                        y: (0.5 - 0.5 * p.z) / 3.0 + 1.0 / 3.0
                    };
                } else if neg_r.contains(&p.y) {
                    // backward
                    return Vec2f {
                        x: (0.5 + 0.5 * p.x) / 4.0 + 1.0 / 4.0,
                        y: (0.5 - 0.5 * p.z) / 3.0 + 1.0 / 3.0
                    };
                } if pos_r.contains(&p.z) {
                    // top
                    return Vec2f {
                        x: (0.5 + 0.5 * p.x) / 4.0 + 1.0 / 4.0,
                        y: (0.5 - 0.5 * p.y) / 3.0
                    }
                } else if neg_r.contains(&p.z) {
                    // bottom
                    return Vec2f {
                        x: (0.5 + 0.5 * p.x) / 4.0 + 1.0 / 4.0,
                        y: (0.5 + 0.5 * p.y) / 3.0 + 2.0 / 3.0
                    }
                } else {
                    // error
                    return Vec2f::zero();
                }
            },
            RendererKind::Triangle{vtx} => {
                todo!()
            },
            RendererKind::Mesh{ref mesh, ..} => {
                todo!()
            }
        }
    }

    pub fn get_color(&self, inst: &RendererInstance, v: Option<Vec3f>) -> Vec3f {
        if let Some(tex) = &self.mat.tex {
            if let Some(v) = v {
                return self.mat.albedo.hadam(tex.get_color(self.to_uv(inst, v)));
            }
        }
        self.mat.albedo
    }

    pub fn get_rough(&self, inst: &RendererInstance, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.rmap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(inst, v)).x
            }
        }
        self.mat.rough
    }

    pub fn get_metal(&self, inst: &RendererInstance, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.mmap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(inst, v)).x;
            }
        }
        self.mat.metal
    }

    pub fn get_glass(&self, inst: &RendererInstance, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.gmap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(inst, v)).x;
            }
        }
        self.mat.glass
    }

    pub fn get_opacity(&self, inst: &RendererInstance, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.omap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(inst, v)).x;
            }
        }
        self.mat.opacity
    }

    pub fn get_emit(&self, inst: &RendererInstance, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.emap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(inst, v)).x;
            }
        }
        self.mat.emit
    }
}

impl RayTracer {
    fn closest_hit<'a>(scene: &'a Scene, ray: &Ray) -> Option<(RayHit<'a>, RayHit<'a>)> {
        let hits = scene.renderer.as_deref()?.iter()
            .flat_map(|obj| std::iter::repeat(obj).zip(obj.instance.iter().map(|inst| (inst, obj.intersect(inst, &ray)))))
            .filter_map(|(obj, (inst, p))| Some((obj, inst, p?.0, p?.1)));

        hits.min_by(|(_, _, (max, _), _), (_, _, (p, _), _)| max.total_cmp(&p)).and_then(|v| {
            let r0 = Ray {t: v.2.0, ..ray.clone()};
            let r1 = Ray {t: v.3.0, ..ray.clone()};

            let mut hit0 = RayHit {
                obj: v.0,
                inst: v.1,
                idx: v.2.1,
                norm: Vec3f::zero(),
                ray: r0
            };

            hit0.norm = v.0.normal(&v.1, &hit0);

            let mut hit1 = RayHit {
                obj: v.0,
                inst: v.1,
                idx: v.3.1,
                norm: Vec3f::zero(),
                ray: r1
            };

            hit1.norm = v.0.normal(&v.1, &hit1);

            Some((hit0, hit1))
        })
    }

    fn cast(&self, uv: Vec2f, frame: &Frame) -> Ray {
        // get direction
        let tan_fov = (0.5 * frame.cam.fov).to_radians().tan();

        let dir = Vec3f{
            x: uv.x,
            y: 1.0 / (2.0 * tan_fov),
            z: -uv.y
        }.norm();

        // dof
        let mut ray = Ray::cast_default(frame.cam.pos, dir);
        ray.t = frame.cam.foc;

        let p = Vec3f::from(&ray);

        let pos = Vec3f {
            x: frame.cam.pos.x + (rand::thread_rng().sample(self.sampler) - 0.5) * frame.cam.aprt,
            y: frame.cam.pos.y,
            z: frame.cam.pos.z + (rand::thread_rng().sample(self.sampler) - 0.5) * frame.cam.aprt
        };

        let new_dir = (p - pos).norm();

        // rotation
        let cam_dir = frame.cam.dir;
        let look = Mat4f::lookat(cam_dir, Vec3f::up());
        let rot_y = Mat3f::rotate_y(cam_dir);

        // cast
        Ray::cast_default(pos, rot_y * (look * new_dir))
    }

    pub fn raytrace<'a, I>(&self, scene: &'a Scene, it: I) -> Vec3f where I: Iterator<Item = (RayHit<'a>, Option<Vec<&'a Light>>)> + Clone {
        (0..self.sample).map(|_| self.reduce_light(scene, it.clone())).sum::<Vec3f>() / (self.sample as f32)
    }

    pub fn iter<'a>(&'a self, coord: Vec2f, scene: &'a Scene, frame: &Frame) -> RaytraceIterator {
        let w = frame.res.0 as f32 * frame.ssaa;
        let h = frame.res.1 as f32 * frame.ssaa;
        let aspect = w / h;

        let uv = Vec2f {
            x: aspect * (coord.x - 0.5 * w) / w,
            y: (coord.y - 0.5 * h) / h
        };

        let ray = RayTracer::cast(self, uv, frame);

        RaytraceIterator {
            rt: self,
            scene: scene,
            next_ray: ray
        }
    }

    pub fn reduce_light<'a, I>(&self, scene: &'a Scene, it: I) -> Vec3f where I: Iterator<Item = (RayHit<'a>, Option<Vec<&'a Light>>)> + Clone {
        if it.clone().count() == 0 {
            return scene.sky.color;
        }

        let tmp = it.collect::<Vec<_>>();
        let path = tmp.iter().rev();

        path.fold(scene.sky.color * scene.sky.pwr, |col, (hit, lights)| {
            // emit
            let emit = hit.get_emit();

            if rand::thread_rng().gen_bool(emit.into()) {
                return hit.get_color();
            }

            // direct light
            let l_col = lights.as_ref().map_or(Vec3f::zero(), |lights| {
                lights.iter().map(|light| {
                    let l = match light.kind {
                        LightKind::Point{pos} => pos - Vec3f::from(&hit.ray),
                        LightKind::Dir{dir} => -dir.norm()
                    };
    
                    let diff = (l.norm() * hit.norm).max(0.0);
                    let spec = (hit.ray.dir * l.norm().reflect(hit.norm)).max(0.0).powi(32) * (1.0 - hit.get_rough());
    
                    let o_col = hit.get_color() * (1.0 - hit.get_metal());
    
                    ((o_col * diff).hadam(light.color) + spec) * light.pwr
                }).sum()
            });

            // indirect light
            let d_col = 0.5 * col + hit.get_color().hadam(col);

            (d_col + l_col) * hit.ray.pwr
        })
    }

    pub fn rand(&self, n: Vec3f, r: f32) -> Vec3f {
        let th = (1.0 - 2.0 * rand::thread_rng().sample(self.sampler)).acos();
        let phi = rand::thread_rng().sample(self.sampler) * 2.0 * PI;

        let v = Vec3f {
            x: th.sin() * phi.cos(),
            y: th.sin() * phi.sin(),
            z: th.cos()
        };

        (n + r * v).norm()
    }

    pub fn default_sampler() -> Uniform<f32> {
        Uniform::new(0.0, 1.0)
    }
}

impl<'a> Iterator for RaytraceIterator<'a> {
    type Item = (RayHit<'a>, Option<Vec<&'a Light>>);
    fn next(&mut self) -> Option<Self::Item> {
        // check bounce
        if self.next_ray.bounce > self.rt.bounce {
            return None
        }

        // intersect
        if let Some(hit) = RayTracer::closest_hit(self.scene, &self.next_ray) {
            let mut out_light: Option<Vec<&Light>> = None;

            // get light
            if let Some(lights) = self.scene.light.as_ref() {
                for light in lights {
                    let l = match light.kind {
                        LightKind::Point{pos} => pos - Vec3f::from(&hit.0.ray),
                        LightKind::Dir{dir} => -dir.norm()
                    };

                    let ray_l = Ray::cast_default((&hit.0.ray).into(), l.norm());
        
                    if let Some(_) = RayTracer::closest_hit(self.scene, &ray_l) {
                        continue;
                    }

                    if let Some(ref mut out_light) = out_light {
                        out_light.push(light);
                    } else {
                        out_light = Some(vec![light]);
                    }
                }
            }

            // reflect
            self.next_ray = hit.0.ray.reflect(self.rt, &hit.0);
            let mut n_hit = hit.0.clone();
            let opacity = hit.0.get_opacity();

            // 15% chance to reflect for transparent material
            if rand::thread_rng().gen_bool((1.0 - opacity).min(0.85).into()) {
                if let Some(r) = hit.1.ray.refract(self.rt, &hit.1) {
                    self.next_ray = r;
                    n_hit = hit.1.clone();
                }
            }

            return Some((n_hit, out_light))
        }

        None
    }
}
