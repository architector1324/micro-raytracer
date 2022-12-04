use image;


fn main() {
    println!("Hello, raytracer!");

    let img = image::ImageBuffer::from_fn(800, 600, |_, _| {
        image::Rgb([255u8, 255u8, 255u8])
    });

    img.save("out.png").unwrap();
}
