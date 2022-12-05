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
#[derive(Serialize, Deserialize, Debug)]
struct Camera {
    pos: (f32, f32, f32),
    dir: (f32, f32, f32),
    fov: f32
}

#[derive(Serialize, Deserialize, Debug)]
struct Frame {
    res: (u16, u16),
    cam: Camera
}


fn main() {
    // parse cli
    let cli = CLI::parse();

    // get frame
    let mut frame = Frame {
        res: (800, 600),
        cam: Camera {
            pos: (0.5, 0.0, 0.5),
            dir: (0.0, 1.0, 0.0),
            fov: 60.0
        }
    };

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
                    frame.cam.pos = (
                        it.next().unwrap().parse::<f32>().unwrap(),
                        it.next().unwrap().parse::<f32>().unwrap(),
                        it.next().unwrap().parse::<f32>().unwrap()
                    )
                },
                "dir:" => {
                    frame.cam.dir = (
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

    // raytrace
    let img = image::ImageBuffer::from_fn(frame.res.0.into(), frame.res.1.into(), |_, _| {
        image::Rgb([255u8, 255u8, 255u8])
    });

    // save output
    match cli.output {
        Some(filename) => img.save(filename).unwrap(),
        None => img.save("out.png").unwrap()
    }
}
