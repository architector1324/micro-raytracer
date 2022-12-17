use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::lin::{Vec3f, Vec2f, Mat3f};

const E: f32 = 0.0001;

#[derive(Serialize, Deserialize, Debug)]
pub struct RayTracer {
    pub bounce: usize,
    pub sample: usize,
    pub loss: f32,
}


#[derive(Debug, Clone)]
pub struct Ray {
    pub orig: Vec3f,
    pub dir: Vec3f,
    pub t: f32,
    pub pwr: f32,
    pub bounce: usize
}

#[derive(Debug)]
pub struct RayHit<'a> {
    pub obj: &'a Renderer,
    pub ray: (Ray, Ray),
    pub norm: (Vec3f, Vec3f)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Camera {
    pub pos: Vec3f,
    pub dir: Vec3f,
    pub fov: f32,
    pub gamma: f32,
    pub exp: f32
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Frame {
    pub res: (u16, u16),
    pub ssaa: f32,
    pub cam: Camera
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct BufferF32 {
    pub w: usize,
    pub h: usize,
    pub dat: Option<Vec<Vec3f>>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Texture {
    #[serde(rename = "inl")]
    InlineBase64(String),

    #[serde(rename = "file")]
    File(String),

    #[serde(rename = "buf")]
    Buffer(BufferF32)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct Material {
    pub albedo: Vec3f,
    pub rough: f32,
    pub metal: f32,
    pub glass: f32,
    pub opacity: f32,
    pub emit: f32,
    pub tex: Option<Texture>,
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
pub struct Light {
    #[serde(flatten)]
    pub kind: LightKind,
    pub pwr: f32,
    pub color: Vec3f
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Scene {
    pub renderer: Option<Vec<Renderer>>,
    pub light: Option<Vec<Light>>,

    #[serde(default)]
    pub sky: Vec3f
}

impl Ray {
    pub fn cast(orig: Vec3f, dir: Vec3f, pwr: f32, bounce: usize) -> Ray {
        Ray{orig: orig + dir * E, dir: dir, pwr: pwr, bounce: bounce, t: 0.0}
    }

    pub fn cast_default(orig: Vec3f, dir: Vec3f) -> Ray {
        Ray{orig: orig + dir * E, dir: dir, ..Default::default()}
    }

    pub fn reflect(&self, rt: &RayTracer, hit: &RayHit) -> Ray {
        let norm = (hit.norm.0 + Vec3f::rand(hit.obj.mat.rough)).norm();
        let dir = self.dir.reflect(norm).norm();

        Ray::cast(self.into(), dir, self.pwr * (1.0 - rt.loss.min(1.0)), self.bounce + 1)
    }

    pub fn refract(&self, rt:&RayTracer, hit: &RayHit) -> Option<Ray> {
        let norm = (hit.norm.1 + Vec3f::rand(hit.obj.mat.rough)).norm();

        let eta = 1.0 + hit.obj.mat.glass / 2.0;
        let dir = self.dir.refract(eta, norm)?.norm();

        Some(Ray::cast(self.into(), dir, self.pwr * (1.0 - rt.loss.min(1.0)), self.bounce + 1))
    }
}

impl Texture {
    pub fn load(name: &str) -> Texture {
        let mut tmp = image::open(name).unwrap();
        let img = tmp.as_mut_rgb8().unwrap();
        let size = img.dimensions();

        let mut out = vec![];

        for px in img.pixels() {
            let col = Vec3f(
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0
            );
            out.push(col);
        }

        Texture::Buffer(BufferF32 {
            w: size.0 as usize,
            h: size.1 as usize,
            dat: Some(out)
        })
    }

    pub fn from_inline(s: &str) -> Texture {
        todo!()
    }

    pub fn to_buffer(&mut self) {
        match self {
            Texture::File(name) => *self = Texture::load(name),
            Texture::InlineBase64(s) => *self = Texture::from_inline(s),
            _ => ()
        }
    }

    pub fn get_color(&self, uv: Vec2f) -> Vec3f {
        if let Texture::Buffer(buf) = self {
            let x = (uv.0 * buf.w as f32) as usize;
            let y = (uv.1 * buf.h as f32) as usize;

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
                let o = ray.orig - self.pos;

                let a = ray.dir * ray.dir;
                let b = 2.0 * (o * ray.dir);
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
                let m = ray.dir.recip();
                let n = (ray.orig - self.pos).hadam(m);
                let k = (sizes / 2.0).hadam(m.abs());

                let a = -n - k;
                let b = -n + k;

                let t0 = a.0.max(a.1).max(a.2);
                let t1 = b.0.min(b.1).min(b.2);

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
                let p = (hit - self.pos).hadam(sizes.recip() * 2.0);

                let pos_r = 1.0-E..1.0+E;
                let neg_r = -1.0-E..-1.0+E;

                if pos_r.contains(&p.0) {
                    // right
                    return Vec3f(1.0, 0.0, 0.0);
                } else if neg_r.contains(&p.0) {
                    // left
                    return Vec3f(-1.0, 0.0, 0.0);
                } else if pos_r.contains(&p.1) {
                    // forward
                    return Vec3f(0.0, 1.0, 0.0);
                } else if neg_r.contains(&p.1) {
                    // backward
                    return Vec3f(0.0, -1.0, 0.0);
                } if pos_r.contains(&p.2) {
                    // top
                    return Vec3f(0.0, 0.0, 1.0);
                } else if neg_r.contains(&p.2) {
                    // bottom
                    return Vec3f(0.0, 0.0, -1.0);
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
                let v = (hit - self.pos).norm();
                Vec2f(
                    0.5 + 0.5 * v.0.atan2(-v.1) / std::f32::consts::PI,
                    0.5 - 0.5 * v.2
                )
            }
            _ => todo!()
        }
    }

    pub fn get_color(&self, v: Vec3f) -> Vec3f {
        if let Some(tex) = &self.mat.tex {
            return self.mat.albedo.hadam(tex.get_color(self.to_uv(v)));
        }
        self.mat.albedo
    }
}

impl RayTracer {
    pub fn closest_hit<'a>(scene: &'a Scene, ray: &'a Ray) -> Option<RayHit<'a>> {
        let hits = scene.renderer.as_deref()?.iter().map(|obj| (obj, obj.intersect(&ray))).filter(|p| p.1.is_some()).map(|p| (p.0, p.1.unwrap().0, p.1.unwrap().1));

        hits.min_by(|max, p| max.1.total_cmp(&p.1)).and_then(|v| {
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

    pub fn cast(uv: Vec2f, frame: &Frame) -> Ray {
        // get direction
        let tan_fov = (frame.cam.fov / 2.0).to_radians().tan();

        let dir = Vec3f(uv.0, 1.0 / (2.0 * tan_fov), -uv.1).norm();
        let cam_dir = frame.cam.dir.norm();

        // rotate direction
        let rot_x: Mat3f = [
            1.0, 0.0, 0.0,
            0.0, cam_dir.1, -cam_dir.2,
            0.0, cam_dir.2, cam_dir.1
        ];

        // let rot_y: Mat3f = [
        //     cam_dir.0, 0.0, cam_dir.2,
        //     0.0, 1.0, 0.0,
        //     -cam_dir.2, 0.0, cam_dir.0
        // ];

        let rot_z: Mat3f = [
            cam_dir.1, cam_dir.0, 0.0,
            -cam_dir.0, cam_dir.1, 0.0,
            0.0, 0.0, 1.0
        ];

        // cast
        Ray::cast_default(frame.cam.pos, rot_x * (rot_z * dir))
    }

    pub fn pathtrace_direct_light(scene: &Scene, hit: &RayHit) -> Option<Vec3f> {
        let mut col = Vec3f::zero();

        let lights = scene.light.as_ref()?;

        for light in lights {
            let l = match light.kind {
                LightKind::Point{pos} => pos - Vec3f::from(&hit.ray.0),
                LightKind::Dir{dir} => -dir.norm()
            };

            let ray_l = Ray::cast_default((&hit.ray.0).into(), l.norm());

            if let Some(_) = RayTracer::closest_hit(scene, &ray_l) {
                continue;
            }

            let diff = (l.norm() * hit.norm.0).max(0.0);
            let spec = (hit.ray.0.dir * l.norm().reflect(hit.norm.0)).max(0.0).powi(32) * (1.0 - hit.obj.mat.rough);

            let o_col = hit.obj.get_color((&hit.ray.0).into()) * (1.0 - hit.obj.mat.metal);

            // col += ((o_col * diff).hadam(light.color) + spec) * light.pwr / (l.mag().powi(2));
            col += ((o_col * diff).hadam(light.color) + spec) * light.pwr;
        }

        Some(col * hit.ray.0.pwr)
    }

    pub fn pathtrace_indirect_light(&self, scene: &Scene, hit: &RayHit) -> Vec3f {
        let mut next_ray = hit.ray.0.reflect(self, hit);

        // 15% chance to reflect for transparent material
        if rand::thread_rng().gen_bool((1.0 - hit.obj.mat.opacity).min(0.85).into()) {
            if let Some(r) = hit.ray.1.refract(self, hit) {
                next_ray = r;
            }
        }

        let next_col = self.pathtrace(scene, &mut next_ray);
        let col = next_col / 2.0 + hit.obj.get_color((&hit.ray.0).into()).hadam(next_col);

        col * next_ray.pwr
    }

    pub fn pathtrace<'a>(&self, scene: &'a Scene, ray: &Ray) -> Vec3f {
        // check bounce
        if ray.bounce > self.bounce {
            return scene.sky
        }

        // intersect
        if let Some(hit) = RayTracer::closest_hit(scene, ray) {
            // emit
            if rand::thread_rng().gen_bool(hit.obj.mat.emit.into()) {
                return hit.obj.get_color((&hit.ray.0).into());
            }

            // total light
            let l_col = RayTracer::pathtrace_direct_light(scene, &hit);
            let d_col = self.pathtrace_indirect_light(scene, &hit);

            return d_col + l_col.unwrap_or(Vec3f::default());
        }

        scene.sky
    }

    pub fn raytrace(&self, coord: Vec2f, scene: &Scene, frame: &Frame) -> Vec3f {
        let w = frame.res.0 as f32 * frame.ssaa;
        let h = frame.res.1 as f32 * frame.ssaa;
        let aspect = w / h;

        let uv = Vec2f(
            aspect * (coord.0 - w / 2.0) / w,
            (coord.1 - h / 2.0) / h
        );

        let mut ray = RayTracer::cast(uv, frame);
        self.pathtrace(scene, &mut ray)
    }
}

impl <'a> From<&'a Ray> for Vec3f {
    fn from(ray: &'a Ray) -> Self {
        ray.orig + ray.dir * ray.t
    }
}

impl Default for Ray {
    fn default() -> Self {
        Ray{
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
            pos: Vec3f(0.0, -1.0, 0.0),
            dir: Vec3f(0.0, 1.0, 0.0),
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

impl Default for Scene {
    fn default() -> Self {
        Scene {
            renderer: None,
            light: None,
            sky: Vec3f::default()
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Material {
            albedo: Vec3f(1.0, 1.0, 1.0),
            rough: 0.0,
            metal: 0.0,
            glass: 0.0,
            opacity: 1.0,
            emit: 0.0,
            tex: None
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
            color: Vec3f(1.0, 1.0, 1.0)
        }
    }
}
