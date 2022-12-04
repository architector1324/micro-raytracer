use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Camera {
    pos: (f32, f32, f32),
    dir: (f32, f32, f32),
    fov: f32
}

#[derive(Serialize, Deserialize)]
struct Frame {
    res: (u16, u16),
    cam: Camera
}


fn main() {
    // match arguments
    let args = clap::Command::new("raytrace")
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
                .num_args(2)
                .help("Output image resolution")
        ])
        .get_matches();

    // get frame
    let mut frame = Frame {
        res: (0, 0),
        cam: Camera {
            pos: (0.5, 0.0, 0.5),
            dir: (0.0, 1.0, 0.0),
            fov: 60.0
        }
    };

    if let Some(mut pair) = args.get_many::<String>("res") {
        let w =  pair.next().unwrap().parse::<u16>().unwrap();
        let h =  pair.next().unwrap().parse::<u16>().unwrap();
        
        frame.res = (w, h);
    } else if let Some(frame_json_filename) = args.get_one::<String>("frame") {
        let frame_json = std::fs::read_to_string(frame_json_filename).unwrap();
        frame = serde_json::from_str(frame_json.as_str()).unwrap();
    }

    println!("{:?}", frame.cam.pos);

    // raytrace
    let img = image::ImageBuffer::from_fn(frame.res.0.into(), frame.res.1.into(), |_, _| {
        image::Rgb([255u8, 255u8, 255u8])
    });

    // save output
    match args.get_one::<String>("output") {
        Some(filename) => img.save(filename).unwrap(),
        None => img.save("out.png").unwrap()
    }
}
