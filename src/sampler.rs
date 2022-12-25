use image::RgbImage;
use std::time::Duration;
use std::sync::{Arc, Mutex};
use scoped_threadpool::Pool;
use std::collections::HashMap;

use crate::lin::{Vec3f, Vec2f};
use crate::rt::{RayTracer, Scene, Frame};


pub struct Sampler {
    n_dim: usize,
    pool: Pool,
    colors: HashMap<(usize, usize), Vec3f>,
    last_count: usize
}

impl Sampler {
    pub fn new(workers: u32, n_dim: usize) -> Sampler {
        Sampler {
            n_dim,
            pool: Pool::new(workers),
            colors: HashMap::new(),
            last_count: 0
        }
    }

    pub fn execute<'a>(&mut self, scene: &'a Scene, frame: &Frame, rt: &'a RayTracer) -> Duration {
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

    pub fn img(&self, frame: &Frame) -> Result<RgbImage, String> {
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

        Ok(image::imageops::resize(&img, frame.res.0 as u32, frame.res.1 as u32, image::imageops::FilterType::Lanczos3))
    }
}
