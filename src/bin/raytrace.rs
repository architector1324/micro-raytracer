use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use clap::Parser;
use serde_json::json;

use micro_raytracer::lin::{Vec3f, Vec2f, ParseFromStrIter};
use micro_raytracer::rt::{RayTracer, Scene, Frame, Camera, Light, Renderer, Color};


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

    #[arg(long, next_line_help = true, help="Parallel jobs count on each dimension")]
    dim: Option<usize>,

    #[arg(short, long, next_line_help = true, help = "Scene description json input filename", value_name = "FILE.json")]
    scene: Option<std::path::PathBuf>,

    #[arg(short, long, next_line_help = true, help = "Frame description json input filename", value_name = "FILE.json")]
    frame: Option<std::path::PathBuf>,

    #[arg(long, value_names = ["w", "h"], next_line_help = true, help = "Frame output image resolution")]
    res: Option<Vec<u16>>,

    #[arg(long, next_line_help = true, help = "Output image SSAAx antialiasing")]
    ssaa: Option<f32>,

    // scene builder
    #[arg(long, value_names = ["pos: <f32 f32 f32>", "dir: <f32 f32 f32 f32>", "fov: <f32>", "gamma: <f32>", "exp: <f32>"], num_args = 1..,  allow_negative_numbers = true, next_line_help = true, help = "Add camera to the scene")]
    cam: Option<Vec<String>>,

    #[arg(long, value_names = ["type: sphere(sph)|plane(pln)|box", "name: <str>", "param: <sphere: r: <f32>>|<plane: n: <f32 f32 f32>>|<box: size: <f32 f32 f32>>", "pos: <f32 f32 f32>", "dir: <f32 f32 f32 f32>", "albedo: <f32 f32 f32>|hex", "rough: <f32>", "metal: <f32>", "glass: <f32>", "opacity: <f32>", "emit: <f32>", "tex: <FILE.ext|<base64 str>>", "rmap: <FILE.ext|<base64 str>>", "mmap: <FILE.ext|<base64 str>>", "gmap: <FILE.ext|<base64 str>>", "omap: <FILE.ext|<base64 str>>", "emap: <FILE.ext|<base64 str>>"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add renderer to the scene")]
    obj: Option<Vec<String>>,

    #[arg(long, value_names = ["param: <point(pt): <f32 f32 f32>>|<dir: <f32 f32 f32>>", "pwr: <f32>", "col: <f32 f32 f32>|hex"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add light source to the scene")]
    light: Option<Vec<String>>,

    #[arg(long, value_names = ["<f32 f32 f32>|hex", "pwr"], num_args = 1.., next_line_help = true, action = clap::ArgAction::Append, help="Scene sky color")]
    sky: Option<Vec<String>>
}

trait ParseFromArgs<T: From<Vec<String>>> {
    fn parse_args(args: &Vec<String>, pat: &[&str]) -> Vec<T>{
        let args_rev: Vec<_> = args.iter()
            .rev()
            .map(|v| String::from(v)).collect();

        args_rev.split_inclusive(|t| pat.contains(&t.as_str()))
            .map(|v| v.iter().rev())
            .map(|obj| T::from(obj.map(|v| String::from(v)).collect::<Vec<_>>()))
            .collect()
    }
}

impl ParseFromArgs<Renderer> for Scene {}
impl ParseFromArgs<Light> for Scene {}


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
        frame.cam = Camera::from(cam_args);
    }

    // get scene
    let mut scene = Scene::default();

    if let Some(scene_json_filename) = cli.scene {
        let scene_json = std::fs::read_to_string(scene_json_filename).unwrap();
        scene = serde_json::from_str(scene_json.as_str()).unwrap();
    }

    if let Some(objs_args) = cli.obj {
        let new_objs = Scene::parse_args(&objs_args, &["sphere", "sph", "plane", "pln", "box"]);

        if let Some(ref mut objs) = scene.renderer {
            objs.extend(new_objs);
        } else {
            scene.renderer = Some(new_objs);
        }
    }

    if let Some(lights_args) = cli.light {
        let new_lights = Scene::parse_args(&lights_args, &["pt:", "point:", "dir:"]);

        if let Some(ref mut lights) = scene.light {
            lights.extend(new_lights);
        } else {
            scene.light = Some(new_lights);
        }
    }

    if let Some(sky) = cli.sky {
        let mut it = sky.iter();
        scene.sky.color = Color::parse(&mut it);
        scene.sky.pwr = <f32>::parse(&mut it);
    }

    // setup raytacer
    let rt = RayTracer{
        bounce: cli.bounce.unwrap_or(8),
        sample: cli.sample.unwrap_or(16),
        loss: cli.loss.unwrap_or(0.15),
        ..RayTracer::default()
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

    // unwrap textures
    scene.sky.color.to_vec3();

    if let Some(ref mut lights) = scene.light {
        for light in lights {
            light.color.to_vec3()
        }
    }

    if let Some(ref mut objs) = scene.renderer {
        for obj in objs {
            obj.mat.albedo.to_vec3();

            if let Some(tex) = &mut obj.mat.tex {
                tex.to_buffer();
            }
            if let Some(rmap) = &mut obj.mat.rmap {
                rmap.to_buffer();
            }
            if let Some(mmap) = &mut obj.mat.mmap {
                mmap.to_buffer();
            }
            if let Some(gmap) = &mut obj.mat.gmap {
                gmap.to_buffer();
            }
            if let Some(omap) = &mut obj.mat.omap {
                omap.to_buffer();
            }
            if let Some(emap) = &mut obj.mat.emap {
                emap.to_buffer();
            }
        }
    }

    // raytrace
    let nw = (frame.res.0 as f32 * frame.ssaa) as usize;
    let nh = (frame.res.1 as f32 * frame.ssaa) as usize;

    let workers = cli.worker.unwrap_or(24);
    let pool = threadpool::ThreadPool::new(workers);

    let scene_sync = Arc::new(scene);
    let frame_sync = Arc::new(frame);
    let rt_sync = Arc::new(rt);

    let colors = Arc::new(Mutex::new(HashMap::new()));

    let n_dim = cli.dim.unwrap_or(64);

    let g_w = (nw as f32 / n_dim as f32).ceil() as usize;
    let g_h = (nh as f32 / n_dim as f32).ceil() as usize;

    for g_x in 0usize..n_dim {
        for g_y in 0usize..n_dim {
            let rt_syc_c = Arc::clone(&rt_sync);
            let scene_syc_c = Arc::clone(&scene_sync);
            let frame_sync_c = Arc::clone(&frame_sync);
            let colors_c = Arc::clone(&colors);

            pool.execute(move || {
                let l_colors = (0..g_w).flat_map(
                    |x| std::iter::repeat(x)
                        .zip(0..g_h)
                        .map(|(x, y)| (x + g_w * g_x, y + g_h * g_y))
                        .map(|(x, y)| (
                            (x, y),
                            rt_syc_c.raytrace(&scene_syc_c, rt_syc_c.iter(Vec2f{x: x as f32, y: y as f32}, &scene_syc_c, &frame_sync_c))
                        ))
                ).collect::<Vec<_>>();

                colors_c.lock().unwrap().extend(l_colors);
            });
        }
    }
    pool.join();

    // save output
    let filename = cli.output.unwrap_or(std::path::PathBuf::from("out.png"));

    let img = image::ImageBuffer::from_fn(nw as u32, nh as u32, |x, y| {
        let col = colors.lock().unwrap().get(&(x as usize, y as usize)).unwrap_or(&Vec3f::zero()).clone();
    
        // gamma correction
        let gamma_col = col.into_iter().map(|v| (v).powf(frame_sync.cam.gamma));
    
        // tone mapping (Reinhard's)
        let final_col = gamma_col.map(|v| v * (1.0 + v / ((1.0 - frame_sync.cam.exp) as f32).powi(2)) / (1.0 + v));
    
        // set pixel
        image::Rgb(final_col.map(|v| (255.0 * v) as u8)
            .collect::<Vec<_>>().as_slice().try_into().unwrap())
    });

    let out_img = image::imageops::resize(&img, frame_sync.res.0 as u32, frame_sync.res.1 as u32, image::imageops::FilterType::Lanczos3);
    out_img.save(filename).unwrap();

    // // test raytracer sampler
    // let mut img = image::ImageBuffer::new(512, 512);

    // for _ in 0..10000 {
    //     let n = Vec3f::from([0.5, 0.0, 1.0]).norm();
    //     let r = 1.0;
    //     let v = rt.rand(n, r);

    //     img.put_pixel((128.0 + 96.0 * v.x) as u32, (128.0 - 96.0 * v.y) as u32, image::Rgb::<u8>([255, 255, 255]));
    //     img.put_pixel((384.0 + 96.0 * v.x) as u32, (128.0 - 96.0 * v.z) as u32, image::Rgb::<u8>([255, 255, 255]));
    //     img.put_pixel((256.0 + 96.0 * v.y) as u32, (384.0 - 96.0 * v.z) as u32, image::Rgb::<u8>([255, 255, 255]));
    // }

    // img.save(std::path::PathBuf::from("test.png")).unwrap();
}
