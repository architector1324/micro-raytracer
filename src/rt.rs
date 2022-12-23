use rand::Rng;
use std::path::PathBuf;
use std::f32::consts::PI;
use image::EncodableLayout;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use rand::distributions::Uniform;
use std::io::prelude::{Read, Write};
use serde::{Serialize, Deserialize};

use crate::lin::{Vec3f, Vec2f, Mat3f, Mat4f, ParseFromStrIter, Vec4f};

const E: f32 = 0.0001;

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RayTracer {
    pub bounce: usize,
    pub sample: usize,
    pub loss: f32,

    #[serde(skip, default = "RayTracer::default_sampler")]
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
    pub ray: (Ray, Ray),
    pub norm: (Vec3f, Vec3f)
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Camera {
    pub pos: Vec3f,
    pub dir: Vec4f,
    pub fov: f32,
    pub gamma: f32,
    pub exp: f32
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Frame {
    pub res: (u16, u16),
    pub ssaa: f32,
    pub cam: Camera
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Color {
    Vec3(Vec3f),
    Hex(String)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct BufferF32 {
    pub w: usize,
    pub h: usize,
    pub dat: Option<Vec<Vec3f>>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Texture {
    Buffer(BufferF32),
    InlineBase64(String),
    File(PathBuf),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct Material {
    pub albedo: Color,
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

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RendererKind {
    Sphere {r: f32},
    Plane {n: Vec3f},
    Box{sizes: Vec3f}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub struct Renderer {
    #[serde(flatten)]
    pub kind: RendererKind,

    #[serde(default)]
    pub mat: Material,

    #[serde(default)]
    pub pos: Vec3f,

    #[serde(default = "Vec4f::forward")]
    pub dir: Vec4f,

    #[serde(default)]
    pub name: Option<String>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LightKind {
    Point {
        pos: Vec3f
    },
    Dir {
        dir: Vec3f
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct Light {
    #[serde(flatten)]
    pub kind: LightKind,
    pub pwr: f32,
    pub color: Color
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Sky {
    pub color: Color,
    pub pwr: f32
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
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

impl Default for RayTracer {
    fn default() -> Self {
        RayTracer {
            bounce: 8,
            sample: 16,
            loss: 0.15,
            sampler: RayTracer::default_sampler()
        }
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

impl Default for Camera {
    fn default() -> Self {
        Camera {
            pos: -Vec3f::forward(),
            dir: Vec4f::forward(),
            fov: 70.0,
            gamma: 0.8,
            exp: 0.2
        }
    }
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            res: (1280, 720),
            ssaa: 1.0,
            cam: Camera::default()
        }
    }
}

impl Default for Sky {
    fn default() -> Self {
        Sky {
            color: Color::Vec3(Vec3f::zero()),
            pwr: 0.5
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Scene {
            renderer: None,
            light: None,
            sky: Sky::default()
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Color::Vec3(Vec3f::from([1.0, 1.0, 1.0]))
    }
}

impl Default for Material {
    fn default() -> Self {
        Material {
            albedo: Color::default(),
            rough: 0.0,
            metal: 0.0,
            glass: 0.0,
            opacity: 1.0,
            emit: 0.0,
            tex: None,
            rmap: None,
            mmap: None,
            gmap: None,
            omap: None,
            emap: None
        }
    }
}

impl Default for BufferF32 {
    fn default() -> Self {
        BufferF32 {
            w: 0,
            h: 0,
            dat: None
        }
    }
}

impl Default for Light {
    fn default() -> Self {
        Light {
            kind: LightKind::Point{
                pos: Vec3f::default()
            },
            pwr: 0.5,
            color: Color::default()
        }
    }
}

impl From<Vec<String>> for Camera {
    fn from(args: Vec<String>) -> Self {
        let mut it = args.iter();
        let mut cam = Camera::default();

        while let Some(param) = it.next() {
            match param.as_str() {
                "pos:" => cam.pos = Vec3f::parse(&mut it),
                "dir:" => cam.dir = Vec4f::parse(&mut it),
                "fov:" => cam.fov = <f32>::parse(&mut it),
                "gamma:" => cam.gamma = <f32>::parse(&mut it),
                "exp:" => cam.exp = <f32>::parse(&mut it),
                _ => panic!("`{}` param for `cam` is unxpected!", param)
            }
        }
        cam 
    }
}

impl From<Vec<String>> for Light {
    fn from(args: Vec<String>) -> Self {
        let t = &args[0];
        let mut it = args.iter();

        // parse object
        let mut light = Light {
            kind: match t.as_str() {
                "pt:" | "point:" => LightKind::Point {pos: Vec3f::default()},
                "dir:" => LightKind::Dir {dir: Vec3f{x: 0.0, y: 1.0, z: 0.0}},
                _ => panic!("`{}` type is unxpected!", t)
            },
            ..Default::default()
        };

        // modify params
        while let Some(param) = it.next() {
            // type params
            let is_type_param = match light.kind {
                LightKind::Point {ref mut pos} => {
                    if param.as_str() == "pt:" || param.as_str() == "point:" {
                        *pos = Vec3f::parse(&mut it);
                        true
                    } else {
                        false
                    }
                },
                LightKind::Dir {ref mut dir} => {
                    if param.as_str() == "dir:" {
                        *dir = Vec3f::parse(&mut it).norm();
                        true
                    } else {
                        false
                    }
                }
            };

            // common params
            match param.as_str() {
                "col:" => light.color = Color::parse(&mut it),
                "pwr:" => light.pwr = <f32>::parse(&mut it),
                _ => {
                    if !is_type_param {
                        panic!("`{}` param for `light` is unxpected!", param);
                    }
                }
            }
        }

        light
    }
}

impl From<Vec<String>> for Renderer {
    fn from(args: Vec<String>) -> Self {
        let t = &args[0];
        let mut it = args.iter().skip(1);

        // parse object
        let mut obj = Renderer {
            kind: match t.as_str() {
                "sph" | "sphere" => RendererKind::Sphere {r: 0.5},
                "pln" | "plane" => RendererKind::Plane {n: Vec3f{x: 0.0, y: 0.0, z: 1.0}},
                "box" => RendererKind::Box {sizes: Vec3f{x: 0.5, y: 0.5, z: 0.5}},
                _ => panic!("`{}` type is unxpected!", t)
            },
            pos: Vec3f::default(),
            dir: Vec4f::forward(),
            mat: Material::default(),
            name: None
        };

        // modify params
        while let Some(param) = it.next() {
            // type params
            let is_type_param = match obj.kind {
                RendererKind::Sphere {ref mut r} => {
                    if param.as_str() == "r:" {
                        *r = <f32>::parse(&mut it);
                        true
                    } else {
                        false
                    }
                },
                RendererKind::Plane{ref mut n} => {
                    if param.as_str() == "n:" {
                        *n = Vec3f::parse(&mut it);
                        true
                    } else {
                        false
                    }
                },
                RendererKind::Box {ref mut sizes} => {
                    if param.as_str() == "size:" {
                        *sizes = Vec3f::parse(&mut it);
                        true
                    } else {
                        false
                    }
                }
            };

            // common params
            match param.as_str() {
                "name:" => obj.name = it.next().cloned(),
                "pos:" => obj.pos = Vec3f::parse(&mut it),
                "dir:" => obj.dir = Vec4f::parse(&mut it),
                "albedo:" => obj.mat.albedo = Color::parse(&mut it),
                "rough:" => obj.mat.rough = <f32>::parse(&mut it),
                "metal:" => obj.mat.metal = <f32>::parse(&mut it),
                "glass:" => obj.mat.glass = <f32>::parse(&mut it),
                "opacity:" => obj.mat.opacity = <f32>::parse(&mut it),
                "emit:" => obj.mat.emit = <f32>::parse(&mut it),
                "tex:" => {
                    let s = String::from(it.next().unwrap());

                    obj.mat.tex = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "rmap:" => {
                    let s = String::from(it.next().unwrap());

                    obj.mat.rmap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "mmap:" => {
                    let s = String::from(it.next().unwrap());

                    obj.mat.mmap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "gmap:" => {
                    let s = String::from(it.next().unwrap());

                    obj.mat.gmap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "omap:" => {
                    let s = String::from(it.next().unwrap());

                    obj.mat.omap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "emap:" => {
                    let s = String::from(it.next().unwrap());

                    obj.mat.emap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                _ => {
                    if !is_type_param {
                        panic!("`{}` param for `{}` is unxpected!", param, t);
                    } 
                }
            };
        }
        obj
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

        let norm = rt.rand(hit.norm.0, rough);
        let dir = self.dir.reflect(norm).norm();

        Ray::cast(self.into(), dir, self.pwr * (1.0 - rt.loss.min(1.0)), self.bounce + 1)
    }

    pub fn refract(&self, rt:&RayTracer, hit: &RayHit) -> Option<Ray> {
        let mut rough = hit.get_rough();
        let opacity = hit.get_opacity();

        // 80% chance to diffuse for dielectric
        if hit.obj.mat.metal == 0.0 && opacity != 0.0 && rand::thread_rng().gen_bool(0.80) {
            rough = 1.0;
        }

        let norm = rt.rand(hit.norm.1, rough);

        let eta = 1.0 + 0.5 * hit.get_glass();
        let dir = self.dir.refract(eta, norm)?.norm();

        Some(Ray::cast(self.into(), dir, self.pwr * (1.0 - rt.loss.min(1.0)), self.bounce + 1))
    }
}

impl <'a> RayHit<'a> {
    pub fn get_color(&self) -> Vec3f {
        self.obj.get_color(Some((&self.ray.0).into()))
    }

    pub fn get_rough(&self) -> f32 {
        self.obj.get_rough(Some((&self.ray.0).into()))
    }

    pub fn get_metal(&self) -> f32 {
        self.obj.get_metal(Some((&self.ray.0).into()))
    }

    pub fn get_glass(&self) -> f32 {
        self.obj.get_glass(Some((&self.ray.0).into()))
    }

    pub fn get_opacity(&self) -> f32 {
        self.obj.get_opacity(Some((&self.ray.0).into()))
    }

    pub fn get_emit(&self) -> f32 {
        self.obj.get_emit(Some((&self.ray.0).into()))
    }
}

impl Color {
    pub fn vec3(&self) -> Vec3f {
        if let Color::Vec3(v) = self {
            return v.clone();
        }
        panic!("color is not ready!")
    }

    pub fn to_vec3(&mut self) {
        match &self {
            Color::Hex(s) => {
                if s.starts_with("#") {
                    let v = <u32>::from_str_radix(&s[1..7], 16).unwrap()
                    .to_le_bytes()[..3]
                    .iter()
                    .rev()
                    .map(|v| *v as f32 / 255.0)
                    .collect::<Vec<_>>();

                    *self = Color::Vec3(Vec3f::from(&v[..]));
                } else {
                    panic!("{} is not a hex color!", s);
                }
            },
            _ => ()
        }
    }
}

impl Texture {
    pub fn load(name: &str) -> Texture {
        let mut tmp = image::open(name).unwrap();
        let img = tmp.as_mut_rgb8().unwrap();
        let size = img.dimensions();

        let out = img.pixels().map(|px| Vec3f::from(px.0.map(|v| v as f32 / 255.0))).collect();

        Texture::Buffer(BufferF32 {
            w: size.0 as usize,
            h: size.1 as usize,
            dat: Some(out)
        })
    }

    pub fn from_inline(s: &str) -> Texture {
        let decoded = base64::decode(s).unwrap();

        let mut dec = GzDecoder::new(decoded.as_bytes());
        let mut self_json = String::new();

        dec.read_to_string(&mut self_json).unwrap();
        serde_json::from_str::<Texture>(&self_json).unwrap()
    }

    pub fn to_buffer(&mut self) {
        match self {
            Texture::File(name) => *self = Texture::load(name.as_os_str().to_str().unwrap()),
            Texture::InlineBase64(s) => {
                if s.contains(".") {
                    *self = Texture::load(s);
                } else {
                    *self = Texture::from_inline(s);
                }
            },
            _ => ()
        }
    }

    pub fn to_inline(&mut self) {
        self.to_buffer();

        let s = serde_json::to_string(self).unwrap();

        let mut enc = GzEncoder::new(vec![], flate2::Compression::best());
        enc.write_all(s.as_bytes()).unwrap();

        let compress = enc.finish().unwrap();
        let encoded = base64::encode(compress);

        *self = Texture::InlineBase64(encoded);
    }

    pub fn get_color(&self, uv: Vec2f) -> Vec3f {
        if let Texture::Buffer(buf) = self {
            let x = (uv.x * buf.w as f32) as usize;
            let y = (uv.y * buf.h as f32) as usize;

            if let Some(dat) = &buf.dat {
                return dat[(x + y * buf.w) as usize];
            }
            return Vec3f::zero()
        }
        panic!("buffer is not ready!")
    }
}

impl Renderer {
    pub fn intersect(&self, ray: &Ray) -> Option<(f32, f32)> {
        match self.kind {
            RendererKind::Sphere{r} => {
                let rot_y = Mat3f::rotate_y(self.dir);
                let look = Mat4f::lookat(self.dir, Vec3f::up());

                let n_orig = self.pos + rot_y * (look * (ray.orig - self.pos));
                let n_dir = rot_y * (look * ray.dir);

                let o = n_orig - self.pos;

                let a = n_dir * n_dir;
                let b = 2.0 * (o * n_dir);
                let c = o * o - r.powi(2);

                let disc = b.powi(2) - 4.0 * a * c;

                if disc < 0.0 {
                    return None
                }

                let t0 = (-b - disc.sqrt()) / (2.0 * a);
                let t1 = (-b + disc.sqrt()) / (2.0 * a);

                if t0 >= 0.0 {
                    return Some((t0, t1));
                }

                None
            },
            RendererKind::Plane{n} => {
                let d = -n.norm() * self.pos;
                let t = -(ray.orig * n.norm() + d) / (ray.dir * n.norm());

                if t > 0.0 {
                    return Some((t, t));
                }
                None
            },
            RendererKind::Box{sizes} => {
                let rot_y = Mat3f::rotate_y(self.dir);
                let look = Mat4f::lookat(self.dir, Vec3f::up());

                let n_orig = self.pos + rot_y * (look * (ray.orig - self.pos));
                let n_dir = rot_y * (look * ray.dir);

                let mut m = n_dir.recip();

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

                let n = (n_orig - self.pos).hadam(m);
                let k = (sizes / 2.0).hadam(m.abs());

                let a = -n - k;
                let b = -n + k;

                let t0 = a.x.max(a.y).max(a.z);
                let t1 = b.x.min(b.y).min(b.z);

                if t0 > t1 || t1 < 0.0 {
                    return None
                }

                Some((t0, t1))
            }
        }
    }

    pub fn normal(&self, hit: Vec3f) -> Vec3f {
        match self.kind {
            RendererKind::Sphere{..} => (hit - self.pos).norm(),
            RendererKind::Plane{n} => n.norm(),
            RendererKind::Box{sizes} => {
                let rot_y = Mat3f::rotate_y(self.dir);
                let look = Mat4f::lookat(self.dir, Vec3f::up());

                let n_hit = self.pos + rot_y * (look * (hit - self.pos));

                let p = (n_hit - self.pos).hadam(sizes.recip() * 2.0);

                let pos_r = 1.0-E..1.0+E;
                let neg_r = -1.0-E..-1.0+E;

                if pos_r.contains(&p.x) {
                    // right
                    return Vec3f::right()
                } else if neg_r.contains(&p.x) {
                    // left
                    return -Vec3f::right()
                } else if pos_r.contains(&p.y) {
                    // forward
                    return Vec3f::forward()
                } else if neg_r.contains(&p.y) {
                    // backward
                    return -Vec3f::forward()
                } if pos_r.contains(&p.z) {
                    // top
                    return Vec3f::up()
                } else if neg_r.contains(&p.z) {
                    // bottom
                    return -Vec3f::up()
                } else {
                    // error
                    Vec3f::zero()
                }
            }
        }
    }

    pub fn to_uv(&self, hit: Vec3f) -> Vec2f {
        match self.kind {
            RendererKind::Sphere{..} => {
                let rot_y = Mat3f::rotate_y(self.dir);
                let look = Mat4f::lookat(self.dir, Vec3f::up());
                let n_hit = self.pos + rot_y * (look * (hit - self.pos));

                let v = (n_hit - self.pos).norm();
                Vec2f {
                    x: 0.5 + 0.5 * v.x.atan2(-v.y) / PI,
                    y: 0.5 - 0.5 * v.z
                }
            },
            RendererKind::Plane{..} => {
                let rot_y = Mat3f::rotate_y(self.dir);
                let look = Mat4f::lookat(self.dir, Vec3f::up());
                let v = rot_y * (look * hit);
                Vec2f {
                    x: (v.x + 0.5).fract().abs(),
                    y: (v.y + 0.5).fract().abs()
                }
            },
            RendererKind::Box {sizes} => {
                let rot_y = Mat3f::rotate_y(self.dir);
                let look = Mat4f::lookat(self.dir, Vec3f::up());

                let n_hit = self.pos + rot_y * (look * (hit - self.pos));

                let p = (n_hit - self.pos).hadam(sizes.recip() * 2.0);

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
            }
        }
    }

    pub fn get_color(&self, v: Option<Vec3f>) -> Vec3f {
        if let Some(tex) = &self.mat.tex {
            if let Some(v) = v {
                return self.mat.albedo.vec3().hadam(tex.get_color(self.to_uv(v)));
            }
        }
        self.mat.albedo.vec3()
    }

    pub fn get_rough(&self, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.rmap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(v)).x
            }
        }
        self.mat.rough
    }

    pub fn get_metal(&self, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.mmap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(v)).x;
            }
        }
        self.mat.metal
    }

    pub fn get_glass(&self, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.gmap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(v)).x;
            }
        }
        self.mat.glass
    }

    pub fn get_opacity(&self, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.omap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(v)).x;
            }
        }
        self.mat.opacity
    }

    pub fn get_emit(&self, v: Option<Vec3f>) -> f32 {
        if let Some(tex) = &self.mat.emap {
            if let Some(v) = v {
                return tex.get_color(self.to_uv(v)).x;
            }
        }
        self.mat.emit
    }
}

impl RayTracer {
    fn closest_hit<'a>(scene: &'a Scene, ray: &Ray) -> Option<RayHit<'a>> {
        let hits = scene.renderer.as_deref()?.iter()
            .map(|obj| (obj, obj.intersect(&ray)))
            .filter_map(|(obj, p)| Some((obj, p?.0, p?.1)));

        hits.min_by(|(_, max, _), (_, p, _)| max.total_cmp(&p)).and_then(|v| {
            let r0 = Ray {t: v.1, ..ray.clone()};
            let r1 = Ray {t: v.2, ..ray.clone()};

            Some(
                RayHit {
                    obj: v.0,
                    norm: (v.0.normal((&r0).into()), v.0.normal((&r1).into())),
                    ray: (r0, r1),
                }
            )
        })
    }

    fn cast(uv: Vec2f, frame: &Frame) -> Ray {
        // get direction
        let tan_fov = (frame.cam.fov / 2.0).to_radians().tan();

        let dir = Vec3f{
            x: uv.x,
            y: 1.0 / (2.0 * tan_fov),
            z: -uv.y
        }.norm();

        let cam_dir = frame.cam.dir;
        let look = Mat4f::lookat(cam_dir, Vec3f::up());
        let rot_y = Mat3f::rotate_y(cam_dir);

        // cast
        Ray::cast_default(frame.cam.pos, rot_y * (look * dir))
    }

    pub fn raytrace<'a, I>(&self, scene: &'a Scene, it: I) -> Vec3f where I: Iterator<Item = (RayHit<'a>, Option<Vec<&'a Light>>)> + Clone {
        (0..self.sample).map(|_| self.reduce_light(scene, it.clone())).sum::<Vec3f>() / (self.sample as f32)
    }

    pub fn iter<'a>(&'a self, coord: Vec2f, scene: &'a Scene, frame: &Frame) -> RaytraceIterator {
        let w = frame.res.0 as f32 * frame.ssaa;
        let h = frame.res.1 as f32 * frame.ssaa;
        let aspect = w / h;

        let uv = Vec2f {
            x: aspect * (coord.x - w / 2.0) / w,
            y: (coord.y - h / 2.0) / h
        };

        let ray = RayTracer::cast(uv, frame);

        RaytraceIterator {
            rt: self,
            scene: scene,
            next_ray: ray
        }
    }

    pub fn reduce_light<'a, I>(&self, scene: &'a Scene, it: I) -> Vec3f where I: Iterator<Item = (RayHit<'a>, Option<Vec<&'a Light>>)> + Clone {
        if it.clone().count() == 0 {
            return scene.sky.color.vec3();
        }

        let tmp = it.collect::<Vec<_>>();
        let path = tmp.iter().rev();

        path.fold(scene.sky.color.vec3() * scene.sky.pwr, |col, (hit, lights)| {
            // emit
            let emit = hit.get_emit();

            if rand::thread_rng().gen_bool(emit.into()) {
                return hit.get_color();
            }

            // direct light
            let l_col = lights.as_ref().map_or(Vec3f::zero(), |lights| {
                lights.iter().map(|light| {
                    let l = match light.kind {
                        LightKind::Point{pos} => pos - Vec3f::from(&hit.ray.0),
                        LightKind::Dir{dir} => -dir.norm()
                    };
    
                    let diff = (l.norm() * hit.norm.0).max(0.0);
                    let spec = (hit.ray.0.dir * l.norm().reflect(hit.norm.0)).max(0.0).powi(32) * (1.0 - hit.get_rough());
    
                    let o_col = hit.get_color() * (1.0 - hit.get_metal());
    
                    ((o_col * diff).hadam(light.color.vec3()) + spec) * light.pwr
                }).sum()
            });

            // indirect light
            let d_col = 0.5 * col + hit.get_color().hadam(col);

            (d_col + l_col) * hit.ray.0.pwr
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
                        LightKind::Point{pos} => pos - Vec3f::from(&hit.ray.0),
                        LightKind::Dir{dir} => -dir.norm()
                    };

                    let ray_l = Ray::cast_default((&hit.ray.0).into(), l.norm());
        
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
            self.next_ray = hit.ray.0.reflect(self.rt, &hit);
            let opacity = hit.get_opacity();

            // 15% chance to reflect for transparent material
            if rand::thread_rng().gen_bool((1.0 - opacity).min(0.85).into()) {
                if let Some(r) = hit.ray.1.refract(self.rt, &hit) {
                    self.next_ray = r;
                }
            }

            return Some((hit, out_light))
        }

        None
    }
}

impl<'a> ParseFromStrIter<'a> for Color {
    fn parse<I: Iterator<Item = &'a String> + Clone>(it: &mut I) -> Self {
        let tmp = it.clone().next().unwrap();

        if tmp.starts_with("#") {
            it.next();
            return Color::Hex(tmp.clone());
        }

        Color::Vec3(Vec3f::parse(it))
    }
}
