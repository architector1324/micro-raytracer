use serde::{Serialize, Deserialize};
use clap::{Parser};
use rand::Rng;
use serde_json::json;


// extra;
trait ParseFromStrIter<'a> {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self;
}

// cli
#[derive(Parser)]
#[command(author, version, about = "Tiny raytracing microservice.", long_about = None)]
struct CLI {
    #[arg(short, long, action, next_line_help = true, help = "Print full info in json")]
    verbose: bool,

    #[arg(long, action, next_line_help = true, help = "Print full info in json with prettifier")]
    pretty: bool,

    #[arg(short, long, action, next_line_help = true, help = "Dry run (useful with verbose)")]
    dry: bool,

    #[arg(short, long, next_line_help = true, help = "Final image output filename", value_name = "FILE.EXT")]
    output: Option<std::path::PathBuf>,

    #[arg(long, next_line_help = true, help="Max ray bounce")]
    bounce: Option<usize>,

    #[arg(long, next_line_help = true, help="Max path-tracing samples")]
    sample: Option<usize>,

    #[arg(long, next_line_help = true, help="Ray bounce energy loss")]
    loss: Option<f32>,

    #[arg(short, long, action, next_line_help = true, help="Save output on each sample")]
    update: bool,

    #[arg(short, long, next_line_help = true, help="Parallel workers count")]
    worker: Option<usize>,

    #[arg(short, long, next_line_help = true, help = "Scene description json input filename", value_name = "FILE.json")]
    scene: Option<std::path::PathBuf>,

    #[arg(short, long, next_line_help = true, help = "Frame description json input filename", value_name = "FILE.json")]
    frame: Option<std::path::PathBuf>,

    #[arg(long, value_names = ["w", "h"], next_line_help = true, help = "Frame output image resolution")]
    res: Option<Vec<u16>>,

    #[arg(long, next_line_help = true, help = "Output image SSAAx antialiasing")]
    ssaa: Option<f32>,

    // scene builder
    #[arg(long, value_names = ["pos: <f32 f32 f32>", "dir: <f32 f32 f32>", "fov: <f32>", "gamma: <f32>", "exp: <f32>"], num_args = 1..,  allow_negative_numbers = true, next_line_help = true, help = "Add camera to the scene")]
    cam: Option<Vec<String>>,

    #[arg(long, value_names = ["<type: sphere(sph)|plane(pln)>", "param: <sphere: r: <f32>>|<plane: n: <f32 f32 f32>>", "pos: <f32 f32 f32>" , "albedo: <f32 f32 f32>", "rough: <f32>", "metal: <f32>", "glass: <f32>", "opacity: <f32>", "emit"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add renderer to the scene")]
    obj: Option<Vec<String>>,

    #[arg(long, value_names = ["param: <point(pt): <f32 f32 f32>>|<dir: <f32 f32 f32>>", "pwr: <f32>", "col: <f32 f32 f32>"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add light source to the scene")]
    light: Option<Vec<String>>,

    #[arg(long, value_names = ["r", "g", "b"], next_line_help = true, action = clap::ArgAction::Append, help="Scene sky color")]
    sky: Option<Vec<String>>
}


// raytracer
#[derive(Serialize, Deserialize, Debug)]
struct RayTracer {
    bounce: usize,
    sample: usize,
    loss: f32,
}

#[derive(Debug, Clone, Copy)]
struct Vec2f (f32, f32);
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
struct Vec3f (f32, f32, f32);
type Mat3f = [f32; 9];
#[derive(Debug, Clone)]
struct Ray {
    orig: Vec3f,
    dir: Vec3f,
    t: f32,
    pwr: f32,
    bounce: usize
}

#[derive(Serialize, Deserialize, Debug)]
struct Camera {
    pos: Vec3f,
    dir: Vec3f,
    fov: f32,
    gamma: f32,
    exp: f32
}

#[derive(Serialize, Deserialize, Debug)]
struct Frame {
    res: (u16, u16),
    ssaa: f32,
    cam: Camera
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
struct Material {
    albedo: Vec3f,
    rough: f32,
    metal: f32,
    glass: f32,
    opacity: f32,
    emit: bool
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
enum RendererKind {
    Sphere {r: f32},
    Plane {n: Vec3f}
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
struct Renderer {
    #[serde(flatten)]
    kind: RendererKind,

    #[serde(default)]
    mat: Material,

    #[serde(default)]
    pos: Vec3f,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
enum LightKind {
    Point {
        pos: Vec3f
    },
    Dir {
        dir: Vec3f
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Light {
    #[serde(flatten)]
    kind: LightKind,
    pwr: f32,
    color: Vec3f
}

#[derive(Serialize, Deserialize, Debug)]
struct Scene {
    renderer: Option<Vec<Renderer>>,
    light: Option<Vec<Light>>,

    #[serde(default)]
    sky: Vec3f
}

impl std::ops::Mul<Vec3f> for Mat3f {
    type Output = Vec3f;

    fn mul(self, rhs: Vec3f) -> Self::Output {
        Vec3f(
            self[0] * rhs.0 + self[1] * rhs.1 + self[2] * rhs.2,
            self[3] * rhs.0 + self[4] * rhs.1 + self[5] * rhs.2,
            self[6] * rhs.0 + self[7] * rhs.1 + self[8] * rhs.2,
        )
    }
}

impl Vec3f {
    fn mag(self) -> f32 {
        (self.0.powi(2) + self.1.powi(2) + self.2.powi(2)).sqrt()
    }

    fn norm(self) -> Vec3f {
        self * self.mag().recip()
    }

    fn reflect(self, n: Vec3f) -> Vec3f {
        self - n * (2.0 * (self * n))
    }

    fn hadam(self, rhs: Vec3f) -> Vec3f {
        Vec3f(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }

    fn rand(max_v: f32) -> Vec3f {
        Vec3f(
            rand::thread_rng().gen_range(-0.5..0.5) * max_v,
            rand::thread_rng().gen_range(-0.5..0.5) * max_v, 
            rand::thread_rng().gen_range(-0.5..0.5) * max_v
        )
    }

    fn zero() -> Vec3f {
        Vec3f(0.0, 0.0, 0.0)
    }
}

impl From<Vec3f> for [f32; 3] {
    fn from(v: Vec3f) -> Self {
        [v.0, v.1, v.2]
    }
}

impl std::ops::Add for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3f(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl std::ops::Add<f32> for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: f32) -> Self::Output {
        Vec3f(self.0 + rhs, self.1 + rhs, self.2 + rhs)
    }
}

impl std::ops::Sub<f32> for Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs: f32) -> Self::Output {
        Vec3f(self.0 - rhs, self.1 - rhs, self.2 - rhs)
    }
}

impl std::ops::Sub for Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3f(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl std::ops::Mul for Vec3f {
    type Output = f32;
    fn mul(self, rhs: Self) -> Self::Output {
        self.0 * rhs.0 + self.1 * rhs.1 + self.2 * rhs.2
    }
}

impl std::ops::Mul<f32> for Vec3f {
    type Output = Vec3f;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec3f(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl std::ops::Div<f32> for Vec3f {
    type Output = Vec3f;

    fn div(self, rhs: f32) -> Self::Output {
        self * rhs.recip()
    }
}

impl std::ops::Neg for Vec3f {
    type Output = Vec3f;

    fn neg(self) -> Self::Output {
        Vec3f(-self.0, -self.1, -self.2)
    }
}

impl std::ops::Div for Vec3f {
    type Output = Vec3f;

    fn div(self, rhs: Self) -> Self::Output {
        Vec3f(self.0 / rhs.0, self.1 / rhs.1, self.2 / rhs.2)
    }
}

impl std::ops::AddAssign for Vec3f {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl std::ops::SubAssign for Vec3f {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
        self.2 -= rhs.2;
    }
}

impl <'a> ParseFromStrIter<'a> for Vec3f {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self {
        Vec3f(
            <f32>::parse(it),
            <f32>::parse(it),
            <f32>::parse(it)
        )
    }
}

impl <'a> ParseFromStrIter<'a> for f32 {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self {
        it.next().unwrap().parse::<f32>().expect("should be <f32>!")
    }
}

impl Ray {
    fn reflect(&self, rt: &RayTracer, obj: &Renderer) -> Ray {
        let hit = self.orig + self.dir * self.t;
        let norm = (obj.normal(hit) + Vec3f::rand(obj.mat.rough)).norm();

        let dir = self.dir.reflect(norm);

        Ray {
            dir: dir,
            orig: hit + dir * 0.001,
            pwr: self.pwr * (1.0 - rt.loss.min(1.0)),
            t: 0.0,
            bounce: self.bounce + 1
        }
    }

    fn refract(&self, rt:&RayTracer, obj: &Renderer) -> Option<Ray> {
        let hit = self.orig + self.dir * self.t;
        let norm = (obj.normal(hit) + Vec3f::rand(obj.mat.rough)).norm();

        let eta = 1.0 + obj.mat.glass / 2.0;
        let cos = -norm * self.dir;

        let k = 1.0 - eta.powi(2) * (1.0 - cos.powi(2));
        if k < 0.0 {
            return None
        }

        let dir = self.dir * eta + norm * (cos * eta + k.sqrt());

        Some(Ray {
            dir: dir,
            orig: hit + dir.norm() * 0.001,
            pwr: self.pwr * (1.0 - rt.loss.min(1.0)),
            t: 0.0,
            bounce: self.bounce + 1
        })
    }
}

impl Default for Vec3f {
    fn default() -> Self {
        Vec3f::zero()
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            pos: Vec3f(0.0, -1.0, 0.0),
            dir: Vec3f(0.0, 1.0, 0.0),
            fov: 90.0,
            gamma: 0.8,
            exp: 0.2
        }
    }
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            res: (800, 600),
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
            emit: false
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

impl Renderer {
    fn intersect(&self, ray: &Ray) -> Option<(f32, f32)> {
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
            }
        }
    }

    fn normal(&self, hit: Vec3f) -> Vec3f {
        match self.kind {
            RendererKind::Sphere{r: _} => (hit - self.pos).norm(),
            RendererKind::Plane{n} => n.norm()
        }
    }
}

impl From<&Vec<String>> for Camera {
    fn from(args: &Vec<String>) -> Self {
        let mut it = args.iter();
        let mut cam = Camera::default();

        while let Some(param) = it.next() {
            match param.as_str() {
                "pos:" => cam.pos = Vec3f::parse(&mut it),
                "dir:" => cam.dir = Vec3f::parse(&mut it),
                "fov:" => cam.fov = <f32>::parse(&mut it),
                "gamma:" => cam.gamma = <f32>::parse(&mut it),
                "exp:" => cam.exp = <f32>::parse(&mut it),
                _ => panic!("`{}` param for `cam` is unxpected!", param)
            }
        }
        cam   
    }
}

impl From<&Vec<String>> for Light {
    fn from(args: &Vec<String>) -> Self {
        let t = &args[0];
        let mut it = args.iter();

        // parse object
        let mut light = Light {
            kind: match t.as_str() {
                "pt:" | "point:" => LightKind::Point {pos: Vec3f::default()},
                "dir:" => LightKind::Dir {dir: Vec3f(0.0, 1.0, 0.0)},
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
                        *dir = Vec3f::parse(&mut it);
                        true
                    } else {
                        false
                    }
                }
            };

            // common params
            match param.as_str() {
                "col:" => light.color = Vec3f::parse(&mut it),
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

impl From<&Vec<String>> for Renderer {
    fn from(args: &Vec<String>) -> Self {
        let t = &args[0];
        let mut it = args.iter().skip(1);

        // parse object
        let mut obj = Renderer {
            kind: match t.as_str() {
                "sph" | "sphere" => RendererKind::Sphere {r: 0.5},
                "pln" | "plane" => RendererKind::Plane {n: Vec3f(0.0, 0.0, 1.0)},
                _ => panic!("`{}` type is unxpected!", t)
            },
            pos: Vec3f::default(),
            mat: Material::default()
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
                }
            };

            // common params
            match param.as_str() {
                "pos:" => obj.pos = Vec3f::parse(&mut it),
                "albedo:" => obj.mat.albedo = Vec3f::parse(&mut it),
                "rough:" => obj.mat.rough = <f32>::parse(&mut it),
                "metal:" => obj.mat.metal = <f32>::parse(&mut it),
                "glass:" => obj.mat.glass = <f32>::parse(&mut it),
                "opacity:" => obj.mat.opacity = <f32>::parse(&mut it),
                "emit" => obj.mat.emit = true,
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

impl RayTracer {
    fn find_closest_intersection<'a>(scene: &'a Scene, ray: &Ray) -> Option<(&'a Renderer, f32, f32)> {
        let hits = scene.renderer.as_deref()?.iter().map(|obj| (obj, obj.intersect(&ray))).filter(|p| p.1.is_some()).map(|p| (p.0, p.1.unwrap().0, p.1.unwrap().1));
        hits.min_by(|max, p| max.1.total_cmp(&p.1))
    }

    fn cast(coord: Vec2f, frame: &Frame) -> Ray {
        let w = frame.res.0 as f32 * frame.ssaa;
        let h = frame.res.1 as f32 * frame.ssaa;

        let aspect = w / h;
        let tan_fov = (frame.cam.fov / 2.0).to_radians().tan();

        // get direction
        let dir = Vec3f(
            aspect * (2.0 * (coord.0 + 0.5) / w - 1.0),
            tan_fov.recip(),
            -(2.0 * (coord.1 + 0.5) / h - 1.0)
        ).norm();

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
        Ray {
            orig: frame.cam.pos,
            dir: rot_x * (rot_z * dir), // rot_x * (rot_y * (rot_z * dir)),
            t: 0.0,
            pwr: 1.0,
            bounce: 0
        }
    }

    fn pathtrace<'a>(&self, scene: &'a Scene, ray: &mut Ray) -> (Vec3f, Option<(&'a Renderer, f32, f32)>) {
        // check bounce
        if ray.bounce > self.bounce {
            return (scene.sky, None)
        }

        // intersect
        let hit = RayTracer::find_closest_intersection(scene, ray);
        if let None = hit {
            return (scene.sky, None)
        }

        ray.t = hit.unwrap().1;

        let hit_obj = hit.unwrap().0;
        let o_col = hit_obj.mat.albedo * hit_obj.mat.opacity * (1.0 - hit_obj.mat.metal);

        // emit
        if hit_obj.mat.emit {
            return (hit_obj.mat.albedo, hit)
        }

        let hit_p = ray.orig + ray.dir * ray.t;
        let n = hit_obj.normal(hit_p);

        // direct light
        let mut l_col = Vec3f::zero();

        if let Some(lights) = scene.light.as_ref() {
            for light in lights {
                let l = match light.kind {
                    LightKind::Point { pos } => pos - hit_p,
                    LightKind::Dir {dir} => -dir.norm()
                };

                if let None = RayTracer::find_closest_intersection(scene, &Ray{orig: hit_p + l.norm() * 0.001, dir: l.norm(), pwr:0.0, t:0.0, bounce:0}) {
                    let diff = (l.norm() * n).max(0.0);
                    let spec = (ray.dir * l.norm().reflect(n)).max(0.0).powi(32) * (1.0 - hit_obj.mat.rough);
    
                    // l_col += ((o_col * diff).hadam(light.color) + spec) * light.pwr / (l.mag().powi(2));
                    l_col += ((o_col * diff).hadam(light.color) + spec) * light.pwr;
                }
            }
        }

        // indirect light
        let mut r_ray = ray.reflect(self, hit_obj);

        // 15% chance to reflect
        if hit_obj.mat.opacity != 1.0 && rand::thread_rng().gen_bool(0.85) {
            let mut r_tmp = ray.clone();
            r_tmp.t = hit.unwrap().2;

            if let Some(r) = r_tmp.refract(self, hit_obj) {
                r_ray = r;
            }
        }

        let path = self.pathtrace(scene, &mut r_ray);
        let d_col = (path.0 + hit_obj.mat.albedo.hadam(path.0)) / 2.0;

        // total light
        (l_col * ray.pwr + d_col * r_ray.pwr, hit)
    }

    fn raytrace(&self, coord: Vec2f, scene: &Scene, frame: &Frame) -> Vec3f {
        let mut ray = RayTracer::cast(coord, frame);
        let path = self.pathtrace(scene, &mut ray);

        path.0
    }
}


fn main() {
    // parse cli
    let cli = CLI::parse();

    // get frame
    let mut frame = Frame::default();

    if let Some(frame_json_filename) = cli.frame {
        let frame_json = std::fs::read_to_string(frame_json_filename).unwrap();
        frame = serde_json::from_str(frame_json.as_str()).unwrap();
    }

    if let Some(pair) = cli.res {
        frame.res = (
            pair.get(0).unwrap().clone(),
            pair.get(1).unwrap().clone()
        );
    }

    if let Some(ssaa) = cli.ssaa {
        frame.ssaa = ssaa;
    }

    if let Some(cam_args) = cli.cam {
        frame.cam = Camera::from(&cam_args);
    }

    // get scene
    let mut scene = Scene::default();

    if let Some(scene_json_filename) = cli.scene {
        let scene_json = std::fs::read_to_string(scene_json_filename).unwrap();
        scene = serde_json::from_str(scene_json.as_str()).unwrap();
    }

    if let Some(objs_flat) = cli.obj {
        let args_rev: Vec<_> = objs_flat.iter().rev().map(|v| String::from(v)).collect();
        let objs = args_rev.split_inclusive(|t| ["sphere", "sph", "plane", "pln"].contains(&t.as_str())).map(|v| v.iter().rev());

        if scene.renderer.is_none() {
            scene.renderer = Some(vec![]);
        }

        for obj in objs {
            scene.renderer.as_mut().unwrap().push(Renderer::from(&obj.map(|v| String::from(v)).collect()));
        }
    }

    if let Some(lights_flat) = cli.light {
        let args_rev: Vec<_> = lights_flat.iter().rev().map(|v| String::from(v)).collect();
        let objs = args_rev.split_inclusive(|t| ["pt:", "point:", "dir:"].contains(&t.as_str())).map(|v| v.iter().rev());

        if scene.light.is_none() {
            scene.light = Some(vec![]);
        }

        for obj in objs {
            scene.light.as_mut().unwrap().push(Light::from(&obj.map(|v| String::from(v)).collect()));
        }
    }

    if let Some(sky) = cli.sky {
        scene.sky = Vec3f::parse(&mut sky.iter());
    }

    // setup raytacer
    let rt = RayTracer{
        bounce: cli.bounce.unwrap_or(8),
        sample: cli.sample.unwrap_or(16),
        loss: cli.loss.unwrap_or(0.15),
    };

    // verbose
    let info_json = json!({
        "scene": scene,
        "frame": frame,
        "rt": rt,
    });

    if cli.verbose {
        if cli.pretty {
            println!("{}", serde_json::to_string_pretty(&info_json).unwrap());
        } else {
            println!("{}", info_json.to_string());
        }
    }

    if cli.dry {
        return;
    }

    // raytrace
    let nw = (frame.res.0 as f32 * frame.ssaa) as u32;
    let nh = (frame.res.1 as f32 * frame.ssaa) as u32;

    let filename = cli.output.unwrap_or(std::path::PathBuf::from("out.png"));
    let img: image::RgbImage = image::ImageBuffer::new(nw, nh);

    let pool = threadpool::ThreadPool::new(cli.worker.unwrap_or(24));

    let img_mtx = std::sync::Arc::new(std::sync::Mutex::new(img));
    let scene_sync = std::sync::Arc::new(scene);
    let frame_sync = std::sync::Arc::new(frame);
    let rt_sync = std::sync::Arc::new(rt);

    for x in 0..nw {
        for y in 0..nh {
            let img_mtx_c = std::sync::Arc::clone(&img_mtx);
            let rt_syc_c = std::sync::Arc::clone(&rt_sync);
            let scene_syc_c = std::sync::Arc::clone(&scene_sync);
            let frame_sync_c = std::sync::Arc::clone(&frame_sync);

            pool.execute(move || {
                // raycast
                let samples = (0..rt_syc_c.sample).map(|_| rt_syc_c.raytrace(Vec2f(x as f32, y as f32), &scene_syc_c, &frame_sync_c));
                let col = samples.fold(Vec3f::zero(), |acc, v| acc + v) / (rt_syc_c.sample as f32);

                // gamma correction
                let mut final_col = <[f32; 3]>::from(col).map(|v| (v).powf(frame_sync_c.cam.gamma));

                // tone mapping (Reinhard's)
                final_col = final_col.map(|v| v * (1.0 + v / ((1.0 - frame_sync_c.cam.exp) as f32).powi(2)) / (1.0 + v));

                // set pixel
                let mut guard = img_mtx_c.lock().unwrap();
                let px = guard.get_pixel_mut(x.clone().into(), y.clone().into());
                *px = image::Rgb(final_col.map(|v| (255.0 * v) as u8));
            });
        }
    }
    pool.join();

    // save output
    let out_img = image::imageops::resize(&img_mtx.lock().unwrap().to_owned(), frame_sync.res.0 as u32, frame_sync.res.1 as u32, image::imageops::FilterType::Lanczos3);
    out_img.save(filename).unwrap();
}
