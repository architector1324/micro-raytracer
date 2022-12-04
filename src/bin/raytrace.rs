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
    println!("Hello, raytracer!");

    // match arguments
    let args = clap::Command::new("raytrace")
        .version("0.1.0")
        .author("Architector1324 <olegsajaxov@yandex.ru>")
        .about("Tiny raytracing microservice.")
        .args([
            clap::Arg::new("output")
                .short('o')
                .help("Final image output filename"),

            clap::Arg::new("scene")
                .short('s')
                .help("Scene description json input filename"),

            clap::Arg::new("frame")
                .short('s')
                .help("Frame description json input filename")
        ])
        .get_matches();


    // raytrace
    let frame_json = r#"
        {
            "res": [800, 600],
            "cam": {
                "pos": [0.5, 0, 0.5],
                "dir": [0, 1, 0],
                "fov": 60
            }
        }
    "#;

    let frame: Frame = serde_json::from_str(frame_json).unwrap();

    println!("{:?}", frame.res);

    let img = image::ImageBuffer::from_fn(800, 600, |_, _| {
        image::Rgb([255u8, 255u8, 255u8])
    });

    // save output
    match args.get_one::<String>("output") {
        Some(filename) => img.save(filename).unwrap(),
        None => img.save("out.png").unwrap()
    }
}
