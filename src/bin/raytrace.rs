use serde::{Serialize, Deserialize};
use clap::{Parser};
use rand::Rng;
use serde_json::json;


// cli
#[derive(Parser)]
#[command(author, version, about = "Tiny raytracing microservice.", long_about = None)]
struct CLI {
    #[arg(short, long, action, help = "Print frame and scene json info")]
    verbose: bool,

    #[arg(long, action, help = "Print frame and scene json info with pretty")]
    pretty: bool,

    #[arg(short, long, help = "Final image output filename", value_name = "FILE.EXT")]
    output: Option<std::path::PathBuf>,

    #[arg(long, help="Max ray bounce")]
    bounce: Option<usize>,

    #[arg(long, help="Max path-tracing samples")]
    sample: Option<usize>,

    #[arg(long, help="Ray bounce energy loss")]
    loss: Option<f32>,

    #[arg(long, action, help="Save output on each sample")]
    update: bool,

    #[arg(short, long, help = "Scene description json input filename", value_name = "FILE.json")]
    scene: Option<std::path::PathBuf>,

    #[arg(short, long, help = "Frame description json input filename", value_name = "FILE.json")]
    frame: Option<std::path::PathBuf>,

    #[arg(long, value_names = ["w", "h"], help = "Frame output image resolution")]
    res: Option<Vec<u16>>,

    #[arg(long, value_names = ["pos", "dir", "fov"], num_args = 1..=10,  help = "Frame camera")]
    cam: Option<Vec<String>>,

    #[arg(long, value_names = ["pos", "r", "albedo"], num_args = 0.., action = clap::ArgAction::Append, help = "Render sphere")]
    sphere: Option<Vec<String>>,

    #[arg(long, value_names = ["pos", "pwr", "col"], num_args = 0.., action = clap::ArgAction::Append, help = "Light source")]
    light: Option<Vec<String>>
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
    fov: f32
}

#[derive(Serialize, Deserialize, Debug)]
struct Frame {
    res: (u16, u16),
    cam: Camera
}

#[derive(Serialize, Deserialize, Debug)]
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
    mat: Material,
    pos: Vec3f,
}

#[derive(Serialize, Deserialize, Debug)]
struct Light {
    pos: Vec3f,
    pwr: f32,
    color: Vec3f
}

#[derive(Serialize, Deserialize, Debug)]
struct Scene {
    renderer: Option<Vec<Renderer>>,
    light: Option<Vec<Light>>,
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

    fn to_array(self) -> [f32; 3] {
        [self.0, self.1, self.2]
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

    fn refract(&self, rt:&RayTracer, obj: &Renderer) -> Ray {
        let hit = self.orig + self.dir * self.t;
        let norm = (obj.normal(hit) + Vec3f::rand(obj.mat.rough)).norm();

        let eta = 1.0 - obj.mat.glass;
        let k = 1.0 - eta.powi(2) * (1.0 - (norm * (-self.dir)).powi(2));

        let dir = (norm * (eta * (norm * (-self.dir)) + k.sqrt()) - (-self.dir) * eta).norm();

        Ray {
            dir: dir,
            orig: hit + dir * 0.001,
            pwr: self.pwr * (1.0 - rt.loss.min(1.0)),
            t: 0.0,
            bounce: self.bounce + 1
        }
    }
}

impl Default for Vec3f {
    fn default() -> Self {
        Vec3f(0.0, 0.0, 0.0)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            pos: Vec3f(0.0, -1.0, 0.0),
            dir: Vec3f(0.0, 1.0, 0.0),
            fov: 90.0
        }
    }
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            res: (800, 600),
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

impl RayTracer {
    fn find_closest_intersection<'a>(scene: &'a Scene, ray: &Ray) -> Option<(&'a Renderer, f32, f32)> {
        let hits = scene.renderer.as_deref()?.iter().map(|obj| (obj, obj.intersect(&ray))).filter(|p| p.1.is_some()).map(|p| (p.0, p.1.unwrap().0, p.1.unwrap().1));
        hits.min_by(|max, p| max.1.total_cmp(&p.1))
    }

    fn cast(coord: Vec2f, frame: &Frame) -> Ray {
        let w = frame.res.0 as f32;
        let h = frame.res.1 as f32;

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
        let mut l_col = Vec3f::default();

        if let Some(lights) = scene.light.as_ref() {
            for light in lights {
                let l = light.pos - hit_p;

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

        // 20% chance to reflect
        if hit_obj.mat.opacity != 1.0 && rand::thread_rng().gen_bool(0.8) {
            let mut r_tmp = ray.clone();
            r_tmp.t = hit.unwrap().2;
            r_ray = r_tmp.refract(self, hit_obj);
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

    if let Some(cam_args) = cli.cam {
        let mut it = cam_args.iter();

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "pos:" => {
                    frame.cam.pos = Vec3f(
                        it.next().unwrap().parse::<f32>().unwrap(),
                        it.next().unwrap().parse::<f32>().unwrap(),
                        it.next().unwrap().parse::<f32>().unwrap()
                    )
                },
                "dir:" => {
                    frame.cam.dir = Vec3f(
                        it.next().unwrap().parse::<f32>().unwrap(),
                        it.next().unwrap().parse::<f32>().unwrap(),
                        it.next().unwrap().parse::<f32>().unwrap()
                    )
                },
                "fov:" => frame.cam.fov = it.next().unwrap().parse::<f32>().unwrap(),
                _ => ()
            }
        }
    }

    // get scene
    let mut scene = Scene::default();

    if let Some(scene_json_filename) = cli.scene {
        let scene_json = std::fs::read_to_string(scene_json_filename).unwrap();
        scene = serde_json::from_str(scene_json.as_str()).unwrap();
    }

    if let Some(spheres) = cli.sphere {
        let sphere = Renderer {
            kind: RendererKind::Sphere{r: 0.5},
            pos: Vec3f(0.0, 0.0, 0.0),
            mat: Material {
                albedo: Vec3f(1.0, 1.0, 1.0),
                rough: 0.0,
                metal: 0.0,
                glass: 0.0,
                opacity: 1.0,
                emit: false
            }
        };
    
        if spheres.is_empty() {
            if let Some(scene) = &mut scene.renderer {
                scene.push(sphere);
            } else {
                scene.renderer = Some(vec![sphere]);
            }
        }
    }

    if let Some(lights) = cli.light {
        let light = Light {
            pos: Vec3f(-0.5, -1.0, 0.5),
            pwr: 0.5,
            color: Vec3f(1.0, 1.0, 1.0)
        };
    
        if lights.is_empty() {
            if let Some(scene) = &mut scene.light {
                scene.push(light);
            } else {
                scene.light = Some(vec![light]);
            }
        }
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

    // raytrace
    let filename = cli.output.unwrap_or(std::path::PathBuf::from("out.png"));
    let mut img: image::RgbImage = image::ImageBuffer::new(frame.res.0.into(), frame.res.1.into());

    for (x, y, px) in img.enumerate_pixels_mut() {
        // raycast
        let samples = (0..rt.sample).map(|_| rt.raytrace(Vec2f(x as f32, y as f32), &scene, &frame));
        let col = samples.fold(Vec3f::default(), |acc, v| acc + v) / (rt.sample as f32);

        // set pixel
        *px = image::Rgb(col.to_array().map(|v| (255.0 * v) as u8));
    }
 
    // save output
    img.save(filename).unwrap();
}
