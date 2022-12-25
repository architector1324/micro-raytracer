use std::path::PathBuf;
use std::time::{Duration, Instant};
use clap::Parser;
use log::info;

use crate::cli_parser::{ParseFromStrIter, ParseFromArgs, FromArgs};
use crate::rt::{Render, Scene, Camera, Color};
use crate::sampler::Sampler;


#[derive(Parser)]
#[command(author, version, about = "Tiny raytracing microservice.", long_about = None)]
pub struct CLI {
    #[arg(short, long, action, next_line_help = true, help = "Print full render info in json")]
    pub verbose: bool,

    #[arg(long, action, next_line_help = true, help = "Print full render info in json with prettifier")]
    pub pretty: bool,

    #[arg(short, long, action, next_line_help = true, help = "Dry run (useful with verbose)")]
    pub dry: bool,

    #[arg(short, long, next_line_help = true, help = "Final image output filename", value_name = "FILE.EXT")]
    pub output: Option<PathBuf>,

    #[arg(long, next_line_help = true, help = "Launch http server", value_name = "address")]
    pub http: Option<String>,

    #[arg(long, next_line_help = true, help="Max ray bounce")]
    pub bounce: Option<usize>,

    #[arg(long, next_line_help = true, help="Max path-tracing samples")]
    pub sample: Option<usize>,

    #[arg(long, next_line_help = true, help="Ray bounce energy loss")]
    pub loss: Option<f32>,

    #[arg(short, long, action, next_line_help = true, help="Save output on each sample")]
    pub update: bool,

    #[arg(short, long, next_line_help = true, help="Parallel workers count")]
    pub worker: Option<u32>,

    #[arg(long, next_line_help = true, help="Parallel jobs count on each dimension")]
    pub dim: Option<usize>,

    #[arg(short, long, next_line_help = true, help = "Scene description json input filename", value_name = "FILE.json")]
    pub scene: Option<PathBuf>,

    #[arg(short, long, next_line_help = true, help = "Frame description json input filename", value_name = "FILE.json")]
    pub frame: Option<PathBuf>,

    #[arg(long, next_line_help = true, help = "Full render description json input filename", value_name = "FILE.json")]
    pub full: Option<PathBuf>,

    #[arg(long, value_names = ["w", "h"], next_line_help = true, help = "Frame output image resolution")]
    pub res: Option<Vec<u16>>,

    #[arg(long, next_line_help = true, help = "Output image SSAAx antialiasing")]
    pub ssaa: Option<f32>,

    // scene builder
    #[arg(long, value_names = ["pos: <f32 f32 f32>", "dir: <f32 f32 f32 f32>", "fov: <f32>", "gamma: <f32>", "exp: <f32>"], num_args = 1..,  allow_negative_numbers = true, next_line_help = true, help = "Add camera to the scene")]
    pub cam: Option<Vec<String>>,

    #[arg(long, value_names = ["type: sphere(sph)|plane(pln)|box", "name: <str>", "param: <sphere: r: <f32>>|<plane: n: <f32 f32 f32>>|<box: size: <f32 f32 f32>>", "pos: <f32 f32 f32>", "dir: <f32 f32 f32 f32>", "albedo: <f32 f32 f32>|hex", "rough: <f32>", "metal: <f32>", "glass: <f32>", "opacity: <f32>", "emit: <f32>", "tex: <FILE.ext|<base64 str>>", "rmap: <FILE.ext|<base64 str>>", "mmap: <FILE.ext|<base64 str>>", "gmap: <FILE.ext|<base64 str>>", "omap: <FILE.ext|<base64 str>>", "emap: <FILE.ext|<base64 str>>"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add renderer to the scene")]
    pub obj: Option<Vec<String>>,

    #[arg(long, value_names = ["param: <point(pt): <f32 f32 f32>>|<dir: <f32 f32 f32>>", "pwr: <f32>", "col: <f32 f32 f32>|hex"], num_args = 0.., action = clap::ArgAction::Append, allow_negative_numbers = true, next_line_help = true, help = "Add light source to the scene")]
    pub light: Option<Vec<String>>,

    #[arg(long, value_names = ["<f32 f32 f32>|hex", "pwr"], num_args = 1.., next_line_help = true, action = clap::ArgAction::Append, help="Scene sky color")]
    pub sky: Option<Vec<String>>
}


impl CLI {
    pub fn parse_render(&self) -> Result<Render, String> {
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

        // parse scene
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

    pub fn raytrace(&self, render: &mut Render) -> Result<Duration, String> {
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
        let mut sampler = Sampler::new(self.worker.unwrap_or(24), self.dim.unwrap_or(64));
        let filename = self.output.clone().unwrap_or(PathBuf::from("out.png"));

        let time = Instant::now();

        for sample in 0..render.rt.sample {
            let time = sampler.execute(&render.scene, &render.frame, &render.rt);
            info!("cli:sample:{}: {:?}", sample, time);

            if self.update {
                let img = sampler.img(&render.frame)?;
                img.save(&filename).map_err(|e| e.to_string())?;
            }
        }

        // save output
        let img = sampler.img(&render.frame)?;
        img.save(&filename).map_err(|e| e.to_string())?;

        Ok(time.elapsed())
    }
}
