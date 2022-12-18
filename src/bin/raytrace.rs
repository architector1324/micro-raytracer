use clap::Parser;
use serde_json::json;

use micro_raytracer::lin::{Vec3f, Vec2f, ParseFromStrIter};
use micro_raytracer::rt::{RayTracer, Scene, Frame, Camera, Light, Renderer};


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
    #[arg(long, value_names = ["pos: <f32 f32 f32>", "dir: <f32 f32 f32 f32>", "fov: <f32>", "gamma: <f32>", "exp: <f32>"], num_args = 1..,  allow_negative_numbers = true, next_line_help = true, help = "Add camera to the scene")]
    cam: Option<Vec<String>>,

    #[arg(long, value_names = ["<type: sphere(sph)|plane(pln)|box>", "param: <sphere: r: <f32>>|<plane: n: <f32 f32 f32>>|<box: size: <f32 f32 f32>>", "pos: <f32 f32 f32>", "dir: <f32 f32 f32 f32>", "albedo: <f32 f32 f32>", "rough: <f32>", "metal: <f32>", "glass: <f32>", "opacity: <f32>", "emit: <f32>", "tex: <FILE.ext|<base64 str>>"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add renderer to the scene")]
    obj: Option<Vec<String>>,

    #[arg(long, value_names = ["param: <point(pt): <f32 f32 f32>>|<dir: <f32 f32 f32>>", "pwr: <f32>", "col: <f32 f32 f32>"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add light source to the scene")]
    light: Option<Vec<String>>,

    #[arg(long, value_names = ["r", "g", "b"], next_line_help = true, action = clap::ArgAction::Append, help="Scene sky color")]
    sky: Option<Vec<String>>
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
        Scene::parse_args(&objs_args, &["sphere", "sph", "plane", "pln", "box"], &mut scene.renderer);
    }

    if let Some(lights_args) = cli.light {
        Scene::parse_args(&lights_args, &["pt:", "point:", "dir:"], &mut scene.light);
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

    // unwrap textures
    if let Some(ref mut objs) = scene.renderer {
        for obj in objs {
            if let Some(tex) = &mut obj.mat.tex {
                tex.to_buffer();
            }
        }
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
                let samples = (0..rt_syc_c.sample).map(|_| rt_syc_c.raytrace(Vec2f{x: x as f32, y: y as f32}, &scene_syc_c, &frame_sync_c));
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
