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

    #[serde(skip_serializing, skip_deserializing)]
    rand: rand::prelude::ThreadRng
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
struct Vec3f (f32, f32, f32);
type Mat3f = [f32; 9];

#[derive(Debug)]
struct Ray {
    orig: Vec3f,
    dir: Vec3f,
    t: f32,
    pwr: f32
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
    gloss: f32,
    rough: f32,
    metal: f32
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
}

impl std::ops::Add for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3f(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
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
    fn cast(x: f32, y: f32, frame: &Frame) -> Ray {
        let w = frame.res.0 as f32;
        let h = frame.res.1 as f32;

        let aspect = w / h;
        let tan_fov = (frame.cam.fov / 2.0).to_radians().tan();

        // get direction
        let dir = Vec3f(
            aspect * (2.0 * (x + 0.5) / w - 1.0),
            tan_fov.recip(),
            -(2.0 * (y + 0.5) / h - 1.0)
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
            orig: frame.cam.pos.clone(),
            dir: rot_x * (rot_z * dir), // rot_x * (rot_y * (rot_z * dir)),
            t: 20000.0,
            pwr: 1.0
        }
    }

    fn reflect(&mut self, rt: &mut RayTracer, obj: &Renderer) {
        let hit = self.orig + self.dir * self.t;
        let mut norm = obj.normal(hit);

        self.orig = obj.pos;

        let rough = Vec3f(
            rt.rand.gen_range(-0.5..0.5) * obj.mat.rough,
            rt.rand.gen_range(-0.5..0.5) * obj.mat.rough, 
            rt.rand.gen_range(-0.5..0.5) * obj.mat.rough
        );
        norm = (norm + rough).norm();

        self.dir = self.dir.reflect(norm);
        self.pwr *= obj.mat.gloss;
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
    fn intersect(&self, ray: &Ray) -> Option<f32> {
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

                let t = (-b - disc.sqrt()) / (2.0 * a);

                if t >= 0.0 {
                    return Some(t);
                }

                None
            },
            RendererKind::Plane{n} => {
                let t = (ray.orig * n.norm() + self.pos.2) / (ray.dir * n.norm());

                if t >= 0.0 {
                    return Some(t);
                }
                None
            }
        }
    }

    fn normal(&self, hit: Vec3f) -> Vec3f {
        (self.pos - hit).norm()
    }

    fn get_color(&self, rt: &mut RayTracer, ray: &Ray, scene: &Scene) -> Vec3f {
        let mut color = Vec3f::default();

        if let Some(lights) = &scene.light {
            for light in lights {
                let hit = ray.orig + ray.dir * ray.t;
                let mut norm = self.normal(hit);
                let l = hit - light.pos;

                let rough = Vec3f(
                    rt.rand.gen_range(-0.5..0.5) * self.mat.rough,
                    rt.rand.gen_range(-0.5..0.5) * self.mat.rough, 
                    rt.rand.gen_range(-0.5..0.5) * self.mat.rough
                );
                norm = (norm + rough).norm();

                let diffuse = (norm * l.norm()).max(0.0);
                let specular = self.mat.gloss * (-ray.dir.reflect(norm) * l.norm()).max(0.0).powi(32);

                let power = light.pwr * (diffuse + specular) / (2.0 * l.mag().powi(2));
                color += (self.mat.albedo + light.color / 2.0) * power;
            }
        }
        color
    }
}

impl RayTracer {
    fn find_closest_intersection<'a>(&self, scene: &'a Scene, ray: &mut Ray) -> Option<&'a Renderer> {
        let hits = scene.renderer.as_deref()?.iter().map(|obj| (obj, obj.intersect(&ray))).filter(|p| p.1.is_some()).map(|p| (p.0, p.1.unwrap()));
        let hit = hits.min_by(|max, p| max.1.total_cmp(&p.1));

        if let Some((obj, t)) = hit {
            ray.t = t;
            return Some(obj)
        }

        None
    }

    fn raytrace_sample(&mut self, x: u32, y: u32, scene: &Scene, frame: &Frame) -> Vec3f {
        let mut ray = Ray::cast(x as f32, y as f32, frame);
        let mut col = Vec3f::default();

        for _ in 0..self.bounce {
            let hit = self.find_closest_intersection(scene, &mut ray);

            if let Some(obj) = hit {
                col += obj.get_color(self, &ray, scene) * ray.pwr;
                ray.reflect(self, obj);
            } else {
                col += scene.sky * ray.pwr;
                break;
            }
        }

        col
    }

    fn raytrace(&mut self, scene: &Scene, frame: &Frame, out: &mut image::RgbImage) {
        for (x, y, pixel) in out.enumerate_pixels_mut() {
            *pixel = image::Rgb([0, 0, 0]);

            // raycast
            let samples = (0..self.sample).map(|_| self.raytrace_sample(x, y, scene, frame));
            let col = samples.fold(Vec3f::default(), |acc, v| acc + v) / (self.sample as f32);

            // set pixel
            *pixel = image::Rgb([(255.0 * col.0) as u8, (255.0 * col.1) as u8, (255.0 * col.2) as u8]);
        }
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
                gloss: 0.0,
                rough: 0.0,
                metal: 0.0
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
    let mut rt = RayTracer{
        bounce: cli.bounce.unwrap_or(8),
        sample: cli.sample.unwrap_or(16),
        rand: rand::thread_rng()
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
    let mut img = image::ImageBuffer::new(frame.res.0.into(), frame.res.1.into());
    rt.raytrace(&scene, &frame, &mut img);

    // save output
    match cli.output {
        Some(filename) => img.save(filename).unwrap(),
        None => img.save("out.png").unwrap()
    }
}
