use serde::{Serialize, Deserialize};

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
    let cli = clap::Command::new("raytrace")
        .version("0.1.0")
        .author("Architector1324 <olegsajaxov@yandex.ru>")
        .about("Tiny raytracing microservice.")
        .args([
            clap::Arg::new("output")
                .short('o')
                .long("output")
                .help("Final image output filename"),

            clap::Arg::new("scene")
                .short('s')
                .long("scene")
                .help("Scene description json input filename"),

            clap::Arg::new("frame")
                .short('f')
                .long("frame")
                .help("Frame description json input filename"),

            clap::Arg::new("res")
                .long("res")
                .value_names(["w", "h"])
                .help("Frame output image resolution"),

            clap::Arg::new("cam")
                .long("cam")
                .value_names(["pos", "dir", "fov"])
                .num_args(1..)
                .help("Frame camera"),

            clap::Arg::new("sphere")
                .long("sphere")
                .value_names(["pos", "r", "col"])
                .num_args(1..)
                .action(clap::ArgAction::Append)
                .help("Render sphere")
        ])
        .get_matches();

    // get frame
    let mut frame = Frame {
        res: (800, 600),
        cam: Camera {
            pos: (0.5, 0.0, 0.5),
            dir: (0.0, 1.0, 0.0),
            fov: 60.0
        }
    };

    if let Some(frame_json_filename) = cli.get_one::<String>("frame") {
        let frame_json = std::fs::read_to_string(frame_json_filename).unwrap();
        frame = serde_json::from_str(frame_json.as_str()).unwrap();
    }

    if let Some(mut pair) = cli.get_many::<String>("res") {
        frame.res = (
            pair.next().unwrap().parse::<u16>().unwrap(),
            pair.next().unwrap().parse::<u16>().unwrap()
        );
    }

    if let Some(mut cam_args) = cli.get_many::<String>("cam") {
        while let Some(arg) = cam_args.next() {
            match arg.as_str() {
                "pos:" => {
                    frame.cam.pos = (
                        cam_args.next().unwrap().parse::<f32>().unwrap(),
                        cam_args.next().unwrap().parse::<f32>().unwrap(),
                        cam_args.next().unwrap().parse::<f32>().unwrap()
                    )
                },
                "dir:" => {
                    frame.cam.dir = (
                        cam_args.next().unwrap().parse::<f32>().unwrap(),
                        cam_args.next().unwrap().parse::<f32>().unwrap(),
                        cam_args.next().unwrap().parse::<f32>().unwrap()
                    )
                },
                "fov:" => frame.cam.fov = cam_args.next().unwrap().parse::<f32>().unwrap(),
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
    match cli.get_one::<String>("output") {
        Some(filename) => img.save(filename).unwrap(),
        None => img.save("out.png").unwrap()
    }
}
