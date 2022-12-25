use std::path::PathBuf;

use crate::lin::{Vec3f, Vec4f};
use crate::rt::{Scene, Renderer, RendererKind, Light, LightKind, Camera, Material, Texture, Color};


pub trait ParseFromStrIter<'a>: Sized {
    fn parse<I: Iterator<Item = &'a String> + Clone>(it: &mut I) -> Result<Self, String>;
}

impl <'a> ParseFromStrIter<'a> for f32 {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Result<Self, String> {
        it.next().ok_or("unexpected ends!")?.parse::<f32>().map_err(|_| "should be <f32>!".to_string())
    }
}

impl <'a> ParseFromStrIter<'a> for Vec3f {
    fn parse<I: Iterator<Item = &'a String> + Clone>(it: &mut I) -> Result<Self, String> {
        Ok(Vec3f {
            x: <f32>::parse(it)?,
            y: <f32>::parse(it)?,
            z: <f32>::parse(it)?
        })
    }
}

impl <'a> ParseFromStrIter<'a> for Vec4f {
    fn parse<I: Iterator<Item = &'a String> + Clone>(it: &mut I) -> Result<Self, String> {
        Ok(Vec4f {
            w: <f32>::parse(it)?,
            x: <f32>::parse(it)?,
            y: <f32>::parse(it)?,
            z: <f32>::parse(it)?
        })
    }
}

impl IntoIterator for Vec3f {
    type Item = f32;
    type IntoIter = std::array::IntoIter<f32, 3>;

    fn into_iter(self) -> Self::IntoIter {
        <[f32; 3]>::from(self).into_iter()
    }
}

impl<'a> ParseFromStrIter<'a> for Color {
    fn parse<I: Iterator<Item = &'a String> + Clone>(it: &mut I) -> Result<Color, String> {
        let tmp = it.clone().next().ok_or("unexpected ends!")?;

        if tmp.starts_with("#") {
            it.next();
            return Ok(Color::Hex(tmp.clone()));
        }

        Ok(Color::Vec3(Vec3f::parse(it)?))
    }
}

pub trait FromArgs: Sized {
    fn from_args(args: &Vec<String>) -> Result<Self, String>;
}

impl FromArgs for Camera {
    fn from_args(args: &Vec<String>) -> Result<Self, String> {
        let mut it = args.iter();
        let mut cam = Camera::default();

        while let Some(param) = it.next() {
            match param.as_str() {
                "pos:" => cam.pos = Vec3f::parse(&mut it)?,
                "dir:" => cam.dir = Vec4f::parse(&mut it)?,
                "fov:" => cam.fov = <f32>::parse(&mut it)?,
                "gamma:" => cam.gamma = <f32>::parse(&mut it)?,
                "exp:" => cam.exp = <f32>::parse(&mut it)?,
                "aprt:" => cam.aprt = <f32>::parse(&mut it)?,
                "foc:" => cam.foc = <f32>::parse(&mut it)?,
                _ => return Err(format!("`{}` param for `cam` is unxpected!", param))
            }
        }
        Ok(cam) 
    }
}

impl FromArgs for Light {
    fn from_args(args: &Vec<String>) -> Result<Self, String> {
        let t = &args[0];
        let mut it = args.iter();

        // parse object
        let mut light = Light {
            kind: match t.as_str() {
                "pt:" | "point:" => LightKind::Point {pos: Vec3f::default()},
                "dir:" => LightKind::Dir {dir: Vec3f{x: 0.0, y: 1.0, z: 0.0}},
                _ => return Err(format!("`{}` type is unxpected!", t))
            },
            ..Default::default()
        };

        // modify params
        while let Some(param) = it.next() {
            // type params
            let is_type_param = match light.kind {
                LightKind::Point {ref mut pos} => {
                    if param.as_str() == "pt:" || param.as_str() == "point:" {
                        *pos = Vec3f::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                },
                LightKind::Dir {ref mut dir} => {
                    if param.as_str() == "dir:" {
                        *dir = Vec3f::parse(&mut it)?.norm();
                        true
                    } else {
                        false
                    }
                }
            };

            // common params
            match param.as_str() {
                "col:" => light.color = Color::parse(&mut it)?,
                "pwr:" => light.pwr = <f32>::parse(&mut it)?,
                _ => {
                    if !is_type_param {
                        return Err(format!("`{}` param for `light` is unxpected!", param));
                    }
                }
            }
        }

        Ok(light)
    }
}

impl FromArgs for Renderer {
    fn from_args(args: &Vec<String>) -> Result<Self, String> {
        let t = &args[0];
        let mut it = args.iter().skip(1);

        // parse object
        let mut obj = Renderer {
            kind: match t.as_str() {
                "sph" | "sphere" => RendererKind::Sphere {r: 0.5},
                "pln" | "plane" => RendererKind::Plane {n: Vec3f{x: 0.0, y: 0.0, z: 1.0}},
                "box" => RendererKind::Box {sizes: Vec3f{x: 0.5, y: 0.5, z: 0.5}},
                _ => return Err(format!("`{}` type is unxpected!", t))
            },
            pos: Vec3f::default(),
            dir: Vec4f::forward(),
            mat: Material::default(),
            name: None
        };

        // modify params
        while let Some(param) = it.next() {
            // type params
            let is_type_param = match obj.kind {
                RendererKind::Sphere {ref mut r} => {
                    if param.as_str() == "r:" {
                        *r = <f32>::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                },
                RendererKind::Plane{ref mut n} => {
                    if param.as_str() == "n:" {
                        *n = Vec3f::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                },
                RendererKind::Box {ref mut sizes} => {
                    if param.as_str() == "size:" {
                        *sizes = Vec3f::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                }
            };

            // common params
            match param.as_str() {
                "name:" => obj.name = it.next().cloned(),
                "pos:" => obj.pos = Vec3f::parse(&mut it)?,
                "dir:" => obj.dir = Vec4f::parse(&mut it)?,
                "albedo:" => obj.mat.albedo = Color::parse(&mut it)?,
                "rough:" => obj.mat.rough = <f32>::parse(&mut it)?,
                "metal:" => obj.mat.metal = <f32>::parse(&mut it)?,
                "glass:" => obj.mat.glass = <f32>::parse(&mut it)?,
                "opacity:" => obj.mat.opacity = <f32>::parse(&mut it)?,
                "emit:" => obj.mat.emit = <f32>::parse(&mut it)?,
                "tex:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.tex = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "rmap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.rmap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "mmap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.mmap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "gmap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.gmap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "omap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.omap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                "emap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.emap = if s.contains(".") {
                        Some(Texture::File(PathBuf::from(s)))
                    } else {
                        Some(Texture::InlineBase64(s))
                    }
                },
                _ => {
                    if !is_type_param {
                        return Err(format!("`{}` param for `{}` is unxpected!", param, t));
                    } 
                }
            };
        }
        Ok(obj)
    }
}

pub trait ParseFromArgs<T: FromArgs> {
    fn parse_args(args: &Vec<String>, pat: &[&str]) -> Result<Vec<T>, String> {
        let args_rev: Vec<_> = args.iter()
            .rev()
            .map(|v| v.to_string()).collect();

        args_rev.split_inclusive(|t| pat.contains(&t.as_str()))
            .map(|v| v.iter().rev())
            .map(|obj| T::from_args(&obj.map(|v| v.to_string()).collect::<Vec<_>>()))
            .collect()
    }
}

impl ParseFromArgs<Renderer> for Scene {}
impl ParseFromArgs<Light> for Scene {}
