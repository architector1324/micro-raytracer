use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use scoped_threadpool::Pool;

use clap::Parser;
use serde::{Serialize, Deserialize};

use micro_raytracer::lin::{Vec3f, Vec2f, ParseFromStrIter};
use micro_raytracer::rt::{RayTracer, Scene, Frame, Camera, Light, Renderer, Color, FromArgs};


#[derive(Parser)]
#[command(author, version, about = "Tiny raytracing microservice.", long_about = None)]
struct CLI {
    #[arg(short, long, action, next_line_help = true, help = "Print full render info in json")]
    verbose: bool,

    #[arg(long, action, next_line_help = true, help = "Print full render info in json with prettifier")]
    pretty: bool,

    #[arg(short, long, action, next_line_help = true, help = "Dry run (useful with verbose)")]
    dry: bool,

    #[arg(short, long, next_line_help = true, help = "Final image output filename", value_name = "FILE.EXT")]
    output: Option<PathBuf>,

    #[arg(long, next_line_help = true, help="Max ray bounce")]
    bounce: Option<usize>,

    #[arg(long, next_line_help = true, help="Max path-tracing samples")]
    sample: Option<usize>,

    #[arg(long, next_line_help = true, help="Ray bounce energy loss")]
    loss: Option<f32>,

    #[arg(short, long, action, next_line_help = true, help="Save output on each sample")]
    update: bool,

    #[arg(short, long, next_line_help = true, help="Parallel workers count")]
    worker: Option<u32>,

    #[arg(long, next_line_help = true, help="Parallel jobs count on each dimension")]
    dim: Option<usize>,

    #[arg(short, long, next_line_help = true, help = "Scene description json input filename", value_name = "FILE.json")]
    scene: Option<PathBuf>,

    #[arg(short, long, next_line_help = true, help = "Frame description json input filename", value_name = "FILE.json")]
    frame: Option<PathBuf>,

    #[arg(long, next_line_help = true, help = "Full render description json input filename", value_name = "FILE.json")]
    full: Option<PathBuf>,

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

trait ParseFromArgs<T: FromArgs> {
    fn parse_args(args: &Vec<String>, pat: &[&str]) -> Result<Vec<T>, String> {
        let args_rev: Vec<_> = args.iter()
            .rev()
            .map(|v| String::from(v)).collect();

        args_rev.split_inclusive(|t| pat.contains(&t.as_str()))
            .map(|v| v.iter().rev())
            .map(|obj| T::from_args(&obj.map(|v| String::from(v)).collect::<Vec<_>>()))
            .collect()
    }
}

impl ParseFromArgs<Renderer> for Scene {}
impl ParseFromArgs<Light> for Scene {}

struct Sampler {
    n_dim: usize,
    pool: Pool,
    colors: HashMap<(usize, usize), Vec3f>,
    last_count: usize
}

impl Sampler {
    fn new(workers: u32, n_dim: usize) -> Sampler {
        Sampler {
            n_dim,
            pool: Pool::new(workers),
            colors: HashMap::new(),
            last_count: 0
        }
    }

    fn execute<'a>(&mut self, scene: &'a Scene, frame: &Frame, rt: &'a RayTracer) -> Duration {
        let nw = (frame.res.0 as f32 * frame.ssaa) as usize;
        let nh = (frame.res.1 as f32 * frame.ssaa) as usize;
        
        let g_w = (nw as f32 / self.n_dim as f32).ceil() as usize;
        let g_h = (nh as f32 / self.n_dim as f32).ceil() as usize;
        
        let total_time = std::time::Instant::now();

        let colors = Arc::new(Mutex::new(&mut self.colors));
        
        self.pool.scoped(|s| {
            for g_x in 0usize..self.n_dim {
                for g_y in 0usize..self.n_dim {
                    let colors_c = Arc::clone(&colors);
    
                    s.execute(move || {
                        let l_colors = (0..g_w).flat_map(
                            |x| std::iter::repeat(x)
                                .zip(0..g_h)
                                .map(|(x, y)| ((x + g_w * g_x, y + g_h * g_y), std::time::Instant::now()))
                                .map(|((x, y), time)| (
                                    (x, y),
                                    (
                                        rt.reduce_light(scene, rt.iter(Vec2f{x: x as f32, y: y as f32}, &scene, &frame)),
                                        time.elapsed()
                                    )
                                ))
                                // .inspect(|((x, y), (_, time))| println!("{} {}: {:?}", x, y, time))
                                .map(|((x, y), (col, _))| ((x, y), col))
                        ).collect::<HashMap<_, _>>();

                        let mut guard = colors_c.lock().unwrap();

                        l_colors.into_iter().for_each(|((x, y), col)| {
                            let entry = guard.get_mut(&(x, y));

                            if let Some(old_v) = entry {
                                *old_v += col;
                            } else {
                                guard.insert((x, y), col);
                            }
                        });
                    });
                }
            }
        });

        self.last_count += 1;
        total_time.elapsed()
    }

    fn save(&self, filename: &PathBuf, frame: &Frame) -> Result<(), String> {
        let nw = (frame.res.0 as f32 * frame.ssaa) as usize;
        let nh = (frame.res.1 as f32 * frame.ssaa) as usize;
    
        let img = image::ImageBuffer::from_fn(nw as u32, nh as u32, |x, y| {
            let col = self.colors.get(&(x as usize, y as usize)).unwrap().clone() / self.last_count as f32;

            // gamma correction
            let gamma_col = col.into_iter().map(|v| (v).powf(frame.cam.gamma));

            // tone mapping (Reinhard's)
            let final_col = gamma_col.map(|v| v * (1.0 + v / ((1.0 - frame.cam.exp) as f32).powi(2)) / (1.0 + v));

            // set pixel
            image::Rgb(final_col.map(|v| (255.0 * v) as u8)
                .collect::<Vec<_>>().as_slice().try_into().unwrap())
        });

        let out_img = image::imageops::resize(&img, frame.res.0 as u32, frame.res.1 as u32, image::imageops::FilterType::Lanczos3);
        out_img.save(&filename).map_err(|e| e.to_string())
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
struct Render {
    rt: RayTracer,
    frame: Frame,
    scene: Scene
}

impl Default for Render {
    fn default() -> Self {
        Render {
            rt: RayTracer {
                bounce: 8,
                sample: 16,
                loss: 0.15,
                ..RayTracer::default()
            },
            frame: Frame::default(),
            scene: Scene::default()
        }
    }
}

impl CLI {
    fn parse_render(&self) -> Result<Render, String> {
        let mut render = Render::default();

        // prase full
        if let Some(full_json_filename) = &self.full {
            let full_json = std::fs::read_to_string(full_json_filename).map_err(|e| e.to_string())?;
            render = serde_json::from_str(full_json.as_str()).map_err(|e| e.to_string())?;
        }

        if let Some(bounce) = self.bounce {
            render.rt.bounce = bounce
        }

        if let Some(sample) = self.sample {
            render.rt.sample = sample;
        }

        if let Some(loss) = self.loss {
            render.rt.loss = loss;
        }

        // parse frame
        if let Some(frame_json_filename) = &self.frame {
            let frame_json = std::fs::read_to_string(frame_json_filename).map_err(|e| e.to_string())?;
            render.frame = serde_json::from_str(frame_json.as_str()).map_err(|e| e.to_string())?;
        }

        if let Some(pair) = &self.res {
            render.frame.res = (
                pair.get(0).ok_or("unexpected ends!")?.clone(),
                pair.get(1).ok_or("unexpected ends!")?.clone()
            );
        }

        if let Some(ssaa) = self.ssaa {
            render.frame.ssaa = ssaa;
        }

        if let Some(cam_args) = &self.cam {
            render.frame.cam = Camera::from_args(cam_args)?;
        }

        // get scene
        if let Some(scene_json_filename) = &self.scene {
            let scene_json = std::fs::read_to_string(scene_json_filename).map_err(|e| e.to_string())?;
            render.scene = serde_json::from_str(scene_json.as_str()).map_err(|e| e.to_string())?;
        }

        if let Some(objs_args) = &self.obj {
            let new_objs = Scene::parse_args(&objs_args, &["sphere", "sph", "plane", "pln", "box"])?;

            if let Some(ref mut objs) = render.scene.renderer {
                objs.extend(new_objs);
            } else {
                render.scene.renderer = Some(new_objs);
            }
        }

        if let Some(lights_args) = &self.light {
            let new_lights = Scene::parse_args(&lights_args, &["pt:", "point:", "dir:"])?;

            if let Some(ref mut lights) = render.scene.light {
                lights.extend(new_lights);
            } else {
                render.scene.light = Some(new_lights);
            }
        }

        if let Some(sky) = &self.sky {
            let mut it = sky.iter();
            render.scene.sky.color = Color::parse(&mut it)?;
            render.scene.sky.pwr = <f32>::parse(&mut it)?;
        }

        Ok(render)
    }
}

fn main_wrapped() -> Result<(), String> {
    // parse render
    let cli = CLI::parse();

    let mut render = cli.parse_render()?;

    // verbose
    if cli.verbose {
        if cli.pretty {
            println!("{}", serde_json::to_string_pretty(&render).map_err(|e| e.to_string())?);
        } else {
            println!("{}", serde_json::to_string(&render).map_err(|e| e.to_string())?);
        }
    }

    if cli.dry {
        return Ok(());
    }

    // unwrap textures
    render.scene.sky.color.to_vec3()?;

    if let Some(ref mut lights) = render.scene.light {
        for light in lights {
            light.color.to_vec3()?
        }
    }

    if let Some(ref mut objs) = render.scene.renderer {
        for obj in objs {
            obj.mat.albedo.to_vec3()?;

            if let Some(tex) = &mut obj.mat.tex {
                tex.to_buffer()?;
            }
            if let Some(rmap) = &mut obj.mat.rmap {
                rmap.to_buffer()?;
            }
            if let Some(mmap) = &mut obj.mat.mmap {
                mmap.to_buffer()?;
            }
            if let Some(gmap) = &mut obj.mat.gmap {
                gmap.to_buffer()?;
            }
            if let Some(omap) = &mut obj.mat.omap {
                omap.to_buffer()?;
            }
            if let Some(emap) = &mut obj.mat.emap {
                emap.to_buffer()?;
            }
        }
    }

    // raytrace
    let mut sampler = Sampler::new(cli.worker.unwrap_or(24), cli.dim.unwrap_or(64));
    let filename = cli.output.unwrap_or(PathBuf::from("out.png"));

    for _ in 0..render.rt.sample {
        let time = sampler.execute(&render.scene, &render.frame, &render.rt);

        // println!("total {:?}", time);

        if cli.update {
            sampler.save(&filename, &render.frame)?;
        }
    }

    // save output
    sampler.save(&filename, &render.frame)?;
    Ok(())
}

fn main() {
    if let Err(e) = main_wrapped() {
        println!("{{err: \"{}\"}}", e.to_string());
    }
}
