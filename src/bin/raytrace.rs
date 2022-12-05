use serde::{Serialize, Deserialize};
use clap::{Parser};


// cli
#[derive(Parser)]
#[command(author, version, about = "Tiny raytracing microservice.", long_about = None)]
struct CLI {
    #[arg(short, long, help = "Final image output filename", value_name = "FILE.EXT")]
    output: Option<std::path::PathBuf>,

    #[arg(short, long, help = "Scene description json input filename", value_name = "FILE.json")]
    scene: Option<std::path::PathBuf>,

    #[arg(short, long, help = "Frame description json input filename", value_name = "FILE.json")]
    frame: Option<std::path::PathBuf>,

    #[arg(long, value_names = ["w", "h"], help = "Frame output image resolution")]
    res: Option<Vec<u16>>,

    #[arg(long, value_names = ["pos", "dir", "fov"], num_args = 1..=10,  help = "Frame camera")]
    cam: Option<Vec<String>>,

    #[arg(long, value_names = ["pos", "r", "col"], num_args = 1.., action = clap::ArgAction::Append, help = "Render sphere")]
    sphere: Option<Vec<String>>
}


// raytracer
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Vec3f (f32, f32, f32);

#[derive(Debug)]
struct Ray {
    orig: Vec3f,
    dir: Vec3f,
    t: f32
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
    albedo: Vec3f
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", content = "body")]
#[serde(rename_all = "lowercase")]
enum Renderer {
    Sphere {
        pos: Vec3f,
        r: f32,
        mat: Material
    }
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
    light: Option<Vec<Light>>
}

struct RayTracer;

impl Vec3f {
    fn add(&self, a: &Vec3f) -> Vec3f {
        Vec3f(self.0 + a.0, self.1 + a.1, self.2 + a.2)
    }

    fn sub(&self, a: &Vec3f) -> Vec3f {
        Vec3f(self.0 - a.0, self.1 - a.1, self.2 - a.2)
    }

    fn mul_s(&self, a: f32) -> Vec3f {
        Vec3f(self.0 * a, self.1 * a, self.2 * a)
    }

    fn dot(&self, a: &Vec3f) -> f32 {
        self.0 * a.0 + self.1 * a.1 + self.2 * a.2
    }

    fn mag(&self) -> f32 {
        f32::sqrt(self.0.powf(2.0) + self.1.powf(2.0) + self.2.powf(2.0))
    }

    fn norm(&self) -> Vec3f {
        let mag = 1.0 / self.mag();
        self.mul_s(mag)
    }
}

impl Ray {
    fn cast(x: f32, y: f32, frame: &Frame) -> Ray {
        let w = frame.res.0 as f32;
        let h = frame.res.1 as f32;

        let aspect = w / h;
        let tan_fov = (frame.cam.fov / 2.0).to_radians().tan();

        Ray {
            orig: frame.cam.pos.clone(),
            dir: Vec3f(
                aspect * (2.0 * (x + 0.5) / w - 1.0),
                1.0 / tan_fov,
                -(2.0 * (y + 0.5) / h - 1.0)
            ).norm(),
            t: 20000.0
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
            pos: Vec3f(0.0, 0.0, 0.0),
            dir: Vec3f(0.0, 1.0, 0.0),
            fov: 90.0
        }
    }
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            res: (800, 600),
            cam: Default::default()
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Scene {
            renderer: None,
            light: None
        }
    }
}

impl Renderer {
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        match self {
            Renderer::Sphere {pos, r, mat: _} => {
                let o = ray.orig.sub(pos);

                let a = ray.dir.dot(&ray.dir);
                let b = 2.0 * o.dot(&ray.dir);
                let c = o.dot(&o) - r.powf(2.0);

                let disc = b.powf(2.0) - 4.0 * a * c;

                if disc < 0.0 {
                    return None
                }

                let t = (-b - disc.sqrt()) / (2.0 * a);

                if t >= 0.0 {
                    return Some(t);
                }

                None
            }
        }
    }

    fn normal(&self, hit: &Vec3f) -> Vec3f {
        match self {
            Renderer::Sphere {pos, r: _, mat: _} => {
                pos.sub(hit).norm()
            }
        }
    }

    fn get_color(&self, ray: &Ray, scene: &Scene) -> Vec3f {
        match self {
            Renderer::Sphere {pos: _, r: _, mat} => {
                if let Some(lights) = &scene.light {
                    let mut color = mat.albedo.clone();

                    for light in lights {
                        let hit = ray.orig.add(&ray.dir.mul_s(ray.t));
                        let norm = self.normal(&hit);
                        let l = hit.sub(&light.pos);

                        let dt = norm.dot(&l.norm()).max(0.0);

                        color = color.add(&light.color.mul_s(light.pwr)).mul_s(dt / 2.0);
                    }

                    return color;
                }
                Default::default()
            }
        }
    }
}

impl RayTracer {
    fn find_closest_intersection<'a>(scene: &'a Scene, ray: &Ray) -> Option<(&'a Renderer, f32)> {
        let hits = scene.renderer.as_deref()?.iter().map(|obj| (obj, obj.intersect(&ray))).filter(|p| p.1.is_some()).map(|p| (p.0, p.1.unwrap()));
        hits.min_by(|max, p| max.1.total_cmp(&p.1))
    }

    fn raytrace(scene: &Scene, frame: &Frame, out: &mut image::RgbImage) {
        for (x, y, pixel) in out.enumerate_pixels_mut() {
            *pixel = image::Rgb([0, 0, 0]);
    
            let mut ray = Ray::cast(x as f32, y as f32, frame);
            let hit = RayTracer::find_closest_intersection(scene, &ray);

            if let Some((obj, t)) = hit {
                ray.t = t;
                let col = obj.get_color(&ray, scene);
                *pixel = image::Rgb([(255.0 * col.0) as u8, (255.0 * col.1) as u8, (255.0 * col.2) as u8]);
            }
        }
    }
}


fn main() {
    // parse cli
    let cli = CLI::parse();

    // get frame
    let mut frame: Frame = Default::default();

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

    println!("{:?}", frame);

    // get scene
    let mut scene: Scene = Default::default();

    if let Some(scene_json_filename) = cli.scene {
        let scene_json = std::fs::read_to_string(scene_json_filename).unwrap();
        scene = serde_json::from_str(scene_json.as_str()).unwrap();
    }

    println!("{:?}", scene);

    // raytrace
    let mut img = image::ImageBuffer::new(frame.res.0.into(), frame.res.1.into());
    RayTracer::raytrace(&scene, &frame, &mut img);

    // save output
    match cli.output {
        Some(filename) => img.save(filename).unwrap(),
        None => img.save("out.png").unwrap()
    }
}
