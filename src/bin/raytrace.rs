use clap::{Parser};
use serde_json::json;

use micro_raytracer::lin::{Vec3f, Vec2f};
use micro_raytracer::rt::{RayTracer, Scene, Frame, Camera, LightKind, RendererKind, Light, Renderer, Material, Texture};


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

    #[arg(long, value_names = ["<type: sphere(sph)|plane(pln)|box>", "param: <sphere: r: <f32>>|<plane: n: <f32 f32 f32>>|<box: size: <f32 f32 f32>>", "pos: <f32 f32 f32>" , "albedo: <f32 f32 f32>", "rough: <f32>", "metal: <f32>", "glass: <f32>", "opacity: <f32>", "emit: <f32>", "tex: <FILE.ext|<base64 str>>"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add renderer to the scene")]
    obj: Option<Vec<String>>,

    #[arg(long, value_names = ["param: <point(pt): <f32 f32 f32>>|<dir: <f32 f32 f32>>", "pwr: <f32>", "col: <f32 f32 f32>"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add light source to the scene")]
    light: Option<Vec<String>>,

    #[arg(long, value_names = ["r", "g", "b"], next_line_help = true, action = clap::ArgAction::Append, help="Scene sky color")]
    sky: Option<Vec<String>>
}

trait ParseFromStrIter<'a> {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self;
}

trait ParseFromArgs<T: From<Vec<String>>> {
    fn parse_args(args: &Vec<String>, pat: &[&str], out: &mut Option<Vec<T>>) {
        let args_rev: Vec<_> = args.iter().rev().map(|v| String::from(v)).collect();
        let objs = args_rev.split_inclusive(|t| pat.contains(&t.as_str())).map(|v| v.iter().rev());

        if out.is_none() {
            *out = Some(vec![]);
        }

        for obj in objs {
            out.as_mut().unwrap().push(T::from(obj.map(|v| String::from(v)).collect::<Vec<_>>()));
        }
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

struct CameraWrap(Camera);
struct RendererWrap(Renderer);
struct LightWrap(Light);

impl ParseFromArgs<RendererWrap> for Scene {}
impl ParseFromArgs<LightWrap> for Scene {}

impl From<Vec<String>> for CameraWrap {
    fn from(args: Vec<String>) -> Self {
        let mut it = args.iter();
        let mut cam = Camera::default();

        while let Some(param) = it.next() {
            match param.as_str() {
                "pos:" => cam.pos = Vec3f::parse(&mut it),
                "dir:" => cam.dir = Vec3f::parse(&mut it).norm(),
                "fov:" => cam.fov = <f32>::parse(&mut it),
                "gamma:" => cam.gamma = <f32>::parse(&mut it),
                "exp:" => cam.exp = <f32>::parse(&mut it),
                _ => panic!("`{}` param for `cam` is unxpected!", param)
            }
        }
        CameraWrap(cam)   
    }
}

impl From<Vec<String>> for LightWrap {
    fn from(args: Vec<String>) -> Self {
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
                        *dir = Vec3f::parse(&mut it).norm();
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

        LightWrap(light)
    }
}

impl From<Vec<String>> for RendererWrap {
    fn from(args: Vec<String>) -> Self {
        let t = &args[0];
        let mut it = args.iter().skip(1);

        // parse object
        let mut obj = Renderer {
            kind: match t.as_str() {
                "sph" | "sphere" => RendererKind::Sphere {r: 0.5},
                "pln" | "plane" => RendererKind::Plane {n: Vec3f(0.0, 0.0, 1.0)},
                "box" => RendererKind::Box {sizes: Vec3f(0.5, 0.5, 0.5)},
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
                "pos:" => obj.pos = Vec3f::parse(&mut it),
                "albedo:" => obj.mat.albedo = Vec3f::parse(&mut it),
                "rough:" => obj.mat.rough = <f32>::parse(&mut it),
                "metal:" => obj.mat.metal = <f32>::parse(&mut it),
                "glass:" => obj.mat.glass = <f32>::parse(&mut it),
                "opacity:" => obj.mat.opacity = <f32>::parse(&mut it),
                "emit:" => obj.mat.emit = <f32>::parse(&mut it),
                "tex:" => obj.mat.tex = Some(Texture::File(String::from(it.next().unwrap()))), 
                _ => {
                    if !is_type_param {
                        panic!("`{}` param for `{}` is unxpected!", param, t);
                    } 
                }
            };
        }
        RendererWrap(obj)
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
        frame.cam = CameraWrap::from(cam_args).0;
    }

    // get scene
    let mut scene = Scene::default();

    if let Some(scene_json_filename) = cli.scene {
        let scene_json = std::fs::read_to_string(scene_json_filename).unwrap();
        scene = serde_json::from_str(scene_json.as_str()).unwrap();
    }

    if let Some(objs_args) = cli.obj {
        let mut tmp = match scene.renderer {
            Some(ref objs) => Some(objs.iter().map(|v| RendererWrap(v.clone())).collect::<Vec<_>>()),
            _ => None
        };
        Scene::parse_args(&objs_args, &["sphere", "sph", "plane", "pln", "box"], &mut tmp);
    }

    if let Some(lights_args) = cli.light {
        let mut tmp = match scene.light {
            Some(ref objs) => Some(objs.iter().map(|v| LightWrap(v.clone())).collect::<Vec<_>>()),
            _ => None
        };
        Scene::parse_args(&lights_args, &["pt:", "point:", "dir:"], &mut tmp);
    }

    if let Some(sky) = cli.sky {
        scene.sky = Vec3f::parse(&mut sky.iter());
    }

    if let Some(ref mut objs) = scene.renderer {
        for obj in objs {
            if let Some(tex) = &mut obj.mat.tex {
                tex.to_buffer();
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
