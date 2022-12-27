use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use image::EncodableLayout;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use std::io::prelude::{Read, Write};
use serde::{Serialize, Deserialize};

use crate::lin::{Vec3f, Vec4f};
use crate::rt::{Render, RayTracer, Frame, Scene, Renderer, RendererKind, Light, LightKind, Camera, Material, Texture, Sky};


pub trait Wrapper<T> {
    fn unwrap(self) -> Result<T, String>;
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct CameraWrapper {
    pub pos: Vec3f,
    pub dir: Vec4f,
    pub fov: f32,
    pub gamma: f32,
    pub exp: f32,
    pub aprt: f32,
    pub foc: f32
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RayTracerWrapper {
    pub bounce: usize,
    pub sample: usize,
    pub loss: f32
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct FrameWrapper {
    pub res: (u16, u16),
    pub ssaa: f32,
    pub cam: CameraWrapper
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ColorWrapper {
    Vec3(Vec3f),
    Hex(String)
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct SkyWrapper {
    pub color: ColorWrapper,
    pub pwr: f32
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LightKindWrapper {
    Point {
        pos: Vec3f
    },
    Dir {
        dir: Vec3f
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct LightWrapper {
    #[serde(flatten)]
    pub kind: LightKindWrapper,
    pub pwr: f32,
    pub color: ColorWrapper
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct TextureBufferWrapper {
    pub w: usize,
    pub h: usize,
    pub dat: Option<Vec<Vec3f>>
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum TextureWrapper {
    Buffer(TextureBufferWrapper),
    InlineBase64(String),
    File(PathBuf),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct MaterialWrapper {
    pub albedo: ColorWrapper,
    pub rough: f32,
    pub metal: f32,
    pub glass: f32,
    pub opacity: f32,
    pub emit: f32,

    pub tex: Option<TextureWrapper>,
    pub rmap: Option<TextureWrapper>, // rough map
    pub mmap: Option<TextureWrapper>, // metal map
    pub gmap: Option<TextureWrapper>, // glass map
    pub omap: Option<TextureWrapper>, // opacity map
    pub emap: Option<TextureWrapper>, // emit map
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum MeshWrapper {
    Mesh(Vec<(Vec3f, Vec3f, Vec3f)>),
    InlineBase64(String),
    File(PathBuf)
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RendererKindWrapper {
    Sphere{r: f32},
    Plane{n: Vec3f},
    Box{sizes: Vec3f},
    Triangle{vtx: (Vec3f, Vec3f, Vec3f)},
    Mesh{mesh: MeshWrapper}
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub struct RendererWrapper {
    #[serde(flatten)]
    pub kind: RendererKindWrapper,

    #[serde(default)]
    pub mat: MaterialWrapper,

    #[serde(default)]
    pub pos: Vec3f,

    #[serde(default = "Vec4f::backward")]
    pub dir: Vec4f,

    #[serde(default)]
    pub name: Option<String>
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct SceneWrapper {
    pub renderer: Option<Vec<RendererWrapper>>,
    pub light: Option<Vec<LightWrapper>>,
    pub sky: SkyWrapper
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RenderWrapper {
    pub rt: RayTracerWrapper,
    pub frame: FrameWrapper,
    pub scene: SceneWrapper
}

impl Default for TextureBufferWrapper {
    fn default() -> Self {
        TextureBufferWrapper {
            w: 0,
            h: 0,
            dat: None
        }
    }
}

impl Default for RenderWrapper {
    fn default() -> Self {
        RenderWrapper {
            rt: RayTracerWrapper::default(),
            frame: FrameWrapper::default(),
            scene: SceneWrapper::default()
        }
    }
}

impl Default for RayTracerWrapper {
    fn default() -> Self {
        RayTracerWrapper {
            bounce: 8,
            sample: 16,
            loss: 0.15
        }
    }
}

impl Default for CameraWrapper {
    fn default() -> Self {
        CameraWrapper {
            pos: -Vec3f::forward(),
            dir: Vec4f::forward(),
            fov: 70.0,
            gamma: 0.8,
            exp: 0.2,
            aprt: 0.001,
            foc: 100.0
        }
    }
}

impl Default for FrameWrapper {
    fn default() -> Self {
        FrameWrapper {
            res: (1280, 720),
            ssaa: 1.0,
            cam: CameraWrapper::default()
        }
    }
}

impl Default for SkyWrapper {
    fn default() -> Self {
        SkyWrapper {
            color: ColorWrapper::Vec3(Vec3f::zero()),
            pwr: 0.5
        }
    }
}

impl Default for SceneWrapper {
    fn default() -> Self {
        SceneWrapper {
            renderer: None,
            light: None,
            sky: SkyWrapper::default()
        }
    }
}


impl Default for MaterialWrapper {
    fn default() -> Self {
        MaterialWrapper {
            albedo: ColorWrapper::Vec3(Vec3f::from([1.0, 1.0, 1.0])),
            rough: 0.0,
            metal: 0.0,
            glass: 0.0,
            opacity: 1.0,
            emit: 0.0,
            tex: None,
            rmap: None,
            mmap: None,
            gmap: None,
            omap: None,
            emap: None
        }
    }
}

impl Default for LightWrapper {
    fn default() -> Self {
        LightWrapper {
            kind: LightKindWrapper::Point {
                pos: Vec3f::default()
            },
            pwr: 0.5,
            color: ColorWrapper::Vec3(Vec3f::from([1.0, 1.0, 1.0]))
        }
    }
}


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

impl<'a> ParseFromStrIter<'a> for ColorWrapper {
    fn parse<I: Iterator<Item = &'a String> + Clone>(it: &mut I) -> Result<Self, String> {
        let tmp = it.clone().next().ok_or("unexpected ends!")?;

        if tmp.starts_with("#") {
            it.next();
            return Ok(ColorWrapper::Hex(tmp.clone()));
        }

        Ok(ColorWrapper::Vec3(Vec3f::parse(it)?))
    }
}

pub trait FromArgs: Sized {
    fn from_args(args: &Vec<String>) -> Result<Self, String>;
}

impl FromArgs for CameraWrapper {
    fn from_args(args: &Vec<String>) -> Result<Self, String> {
        let mut it = args.iter();
        let mut cam = CameraWrapper::default();

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

impl FromArgs for LightWrapper {
    fn from_args(args: &Vec<String>) -> Result<Self, String> {
        let t = &args[0];
        let mut it = args.iter();

        // parse object
        let mut light = LightWrapper {
            kind: match t.as_str() {
                "pt:" | "point:" => LightKindWrapper::Point {pos: Vec3f::default()},
                "dir:" => LightKindWrapper::Dir {dir: Vec3f{x: 0.0, y: 1.0, z: 0.0}},
                _ => return Err(format!("`{}` type is unxpected!", t))
            },
            ..Default::default()
        };

        // modify params
        while let Some(param) = it.next() {
            // type params
            let is_type_param = match light.kind {
                LightKindWrapper::Point {ref mut pos} => {
                    if param.as_str() == "pt:" || param.as_str() == "point:" {
                        *pos = Vec3f::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                },
                LightKindWrapper::Dir {ref mut dir} => {
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
                "col:" => light.color = ColorWrapper::parse(&mut it)?,
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

impl FromArgs for RendererWrapper {
    fn from_args(args: &Vec<String>) -> Result<Self, String> {
        let t = &args[0];
        let mut it = args.iter().skip(1);

        // parse object
        let mut obj = RendererWrapper {
            kind: match t.as_str() {
                "sph" | "sphere" => RendererKindWrapper::Sphere {r: 0.5},
                "pln" | "plane" => RendererKindWrapper::Plane {n: Vec3f{x: 0.0, y: 0.0, z: 1.0}},
                "box" => RendererKindWrapper::Box {sizes: Vec3f{x: 0.5, y: 0.5, z: 0.5}},
                "tri" | "triangle" => RendererKindWrapper::Triangle {vtx: (
                    Vec3f{x: 0.5, y: 0.0, z: -0.25},
                    Vec3f{x: 0.0, y: 0.0, z: 0.5},
                    Vec3f{x: -0.5, y: 0.0, z: -0.25},
                )},
                "mesh" => RendererKindWrapper::Mesh {
                    mesh: MeshWrapper::Mesh(
                        vec![(
                            Vec3f{x: 0.5, y: 0.0, z: -0.25},
                            Vec3f{x: 0.0, y: 0.0, z: 0.5},
                            Vec3f{x: -0.5, y: 0.0, z: -0.25},
                        )]
                    )
                },
                _ => return Err(format!("`{}` type is unxpected!", t))
            },
            pos: Vec3f::default(),
            dir: Vec4f::backward(),
            mat: MaterialWrapper::default(),
            name: None
        };

        // modify params
        while let Some(param) = it.next() {
            // type params
            let is_type_param = match obj.kind {
                RendererKindWrapper::Sphere {ref mut r} => {
                    if param.as_str() == "r:" {
                        *r = <f32>::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                },
                RendererKindWrapper::Plane{ref mut n} => {
                    if param.as_str() == "n:" {
                        *n = Vec3f::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                },
                RendererKindWrapper::Box{ref mut sizes} => {
                    if param.as_str() == "size:" {
                        *sizes = Vec3f::parse(&mut it)?;
                        true
                    } else {
                        false
                    }
                },
                RendererKindWrapper::Triangle{ref mut vtx} => {
                    if param.as_str() == "vtx:" {
                        *vtx = (
                            Vec3f::parse(&mut it)?,
                            Vec3f::parse(&mut it)?,
                            Vec3f::parse(&mut it)?
                        );
                        true
                    } else {
                        false
                    }
                },
                RendererKindWrapper::Mesh{ref mut mesh} => {
                    if param.as_str() == "mesh:" {
                        let mut tmp = vec![(
                            Vec3f::parse(&mut it)?,
                            Vec3f::parse(&mut it)?,
                            Vec3f::parse(&mut it)?
                        )];

                        loop {
                            let v0 = Vec3f::parse(&mut it);
                            let v1 = Vec3f::parse(&mut it);
                            let v2 = Vec3f::parse(&mut it);

                            if v0.is_err() || v1.is_err() || v2.is_err() {
                                break;
                            }

                            tmp.push((v0?, v1?, v2?));
                        }

                        *mesh = MeshWrapper::Mesh(tmp);

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
                "albedo:" => obj.mat.albedo = ColorWrapper::parse(&mut it)?,
                "rough:" => obj.mat.rough = <f32>::parse(&mut it)?,
                "metal:" => obj.mat.metal = <f32>::parse(&mut it)?,
                "glass:" => obj.mat.glass = <f32>::parse(&mut it)?,
                "opacity:" => obj.mat.opacity = <f32>::parse(&mut it)?,
                "emit:" => obj.mat.emit = <f32>::parse(&mut it)?,
                "tex:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.tex = if s.contains(".") {
                        Some(TextureWrapper::File(PathBuf::from(s)))
                    } else {
                        Some(TextureWrapper::InlineBase64(s))
                    }
                },
                "rmap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.rmap = if s.contains(".") {
                        Some(TextureWrapper::File(PathBuf::from(s)))
                    } else {
                        Some(TextureWrapper::InlineBase64(s))
                    }
                },
                "mmap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.mmap = if s.contains(".") {
                        Some(TextureWrapper::File(PathBuf::from(s)))
                    } else {
                        Some(TextureWrapper::InlineBase64(s))
                    }
                },
                "gmap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.gmap = if s.contains(".") {
                        Some(TextureWrapper::File(PathBuf::from(s)))
                    } else {
                        Some(TextureWrapper::InlineBase64(s))
                    }
                },
                "omap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.omap = if s.contains(".") {
                        Some(TextureWrapper::File(PathBuf::from(s)))
                    } else {
                        Some(TextureWrapper::InlineBase64(s))
                    }
                },
                "emap:" => {
                    let s = it.next().ok_or("unexpected ended!".to_string())?.to_string();

                    obj.mat.emap = if s.contains(".") {
                        Some(TextureWrapper::File(PathBuf::from(s)))
                    } else {
                        Some(TextureWrapper::InlineBase64(s))
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

impl ParseFromArgs<RendererWrapper> for SceneWrapper {}
impl ParseFromArgs<LightWrapper> for SceneWrapper {}


impl MeshWrapper {
    pub fn load(name: &str) -> Result<Self, String> {
        let tmp = BufReader::new(File::open(name).map_err(|e| e.to_string())?);
        let obj: obj::Obj = obj::load_obj(tmp).map_err(|e| e.to_string())?;

        let mut mesh = Vec::new();

        for idx in obj.indices.chunks(3) {
            mesh.push((
                Vec3f::from(obj.vertices[idx[0] as usize].position),
                Vec3f::from(obj.vertices[idx[1] as usize].position),
                Vec3f::from(obj.vertices[idx[2] as usize].position),
            ));
        }

        Ok(MeshWrapper::Mesh(mesh))
    }

    pub fn from_inline(s: &str) -> Result<Self, String> {
        let decoded = base64::decode(s).map_err(|e| e.to_string())?;

        let mut dec = GzDecoder::new(decoded.as_bytes());
        let mut self_json = String::new();

        dec.read_to_string(&mut self_json).map_err(|e| e.to_string())?;
        Ok(serde_json::from_str::<MeshWrapper>(&self_json).map_err(|e| e.to_string())?)
    }

    pub fn to_buffer(self) -> Result<Self, String> {
        match self {
            MeshWrapper::File(name) => MeshWrapper::load(name.as_os_str().to_str().ok_or("cannot convert to string!".to_string())?),
            MeshWrapper::InlineBase64(s) => {
                if s.contains(".") {
                    MeshWrapper::load(s.as_str())
                } else {
                    MeshWrapper::from_inline(s.as_str())
                }
            },
            _ => Ok(self)
        }
    }

    pub fn to_inline(self) -> Result<Self, String> {
        let buf = self.to_buffer()?;

        let s = serde_json::to_string(&buf).map_err(|e| e.to_string())?;

        let mut enc = GzEncoder::new(vec![], flate2::Compression::best());
        enc.write_all(s.as_bytes()).map_err(|e| e.to_string())?;

        let compress = enc.finish().map_err(|e| e.to_string())?;
        let encoded = base64::encode(compress);

        Ok(MeshWrapper::InlineBase64(encoded))
    }
}

impl TextureWrapper {
    pub fn load(name: &str) -> Result<Self, String> {
        let mut tmp = image::open(name).map_err(|e| e.to_string())?;
        let img = tmp.as_mut_rgb8().ok_or("is not rgb888 image!".to_string())?;
        let size = img.dimensions();

        let out = img.pixels().map(|px| Vec3f::from(px.0.map(|v| v as f32 / 255.0))).collect();

        Ok(TextureWrapper::Buffer(TextureBufferWrapper{
            w: size.0 as usize,
            h: size.1 as usize,
            dat: Some(out)
        }))
    }

    pub fn from_inline(s: &str) -> Result<Self, String> {
        let decoded = base64::decode(s).map_err(|e| e.to_string())?;

        let mut dec = GzDecoder::new(decoded.as_bytes());
        let mut self_json = String::new();

        dec.read_to_string(&mut self_json).map_err(|e| e.to_string())?;
        Ok(serde_json::from_str::<TextureWrapper>(&self_json).map_err(|e| e.to_string())?)
    }

    pub fn to_buffer(self) -> Result<Self, String> {
        match self {
            TextureWrapper::File(name) => TextureWrapper::load(name.as_os_str().to_str().ok_or("cannot convert to string!".to_string())?),
            TextureWrapper::InlineBase64(s) => {
                if s.contains(".") {
                    TextureWrapper::load(s.as_str())
                } else {
                    TextureWrapper::from_inline(s.as_str())
                }
            },
            _ => Ok(self)
        }
    }

    pub fn to_inline(self) -> Result<Self, String> {
        let buf = self.to_buffer()?;

        let s = serde_json::to_string(&buf).map_err(|e| e.to_string())?;

        let mut enc = GzEncoder::new(vec![], flate2::Compression::best());
        enc.write_all(s.as_bytes()).map_err(|e| e.to_string())?;

        let compress = enc.finish().map_err(|e| e.to_string())?;
        let encoded = base64::encode(compress);

        Ok(TextureWrapper::InlineBase64(encoded))
    }
}

impl Wrapper<Vec3f> for ColorWrapper {
    fn unwrap(self) -> Result<Vec3f, String> {
        match &self {
            ColorWrapper::Hex(s) => {
                if s.starts_with("#") {
                    let v = <u32>::from_str_radix(&s[1..7], 16).map_err(|e| e.to_string())?
                    .to_le_bytes()[..3]
                    .iter()
                    .rev()
                    .map(|v| *v as f32 / 255.0)
                    .collect::<Vec<_>>();

                    Ok(Vec3f::from(&v[..]))
                } else {
                    Err(format!("{} is not a hex color!", s))
                }
            },
            ColorWrapper::Vec3(v) => Ok(v.clone())
        }
    }
}

impl Wrapper<RayTracer> for RayTracerWrapper {
    fn unwrap(self) -> Result<RayTracer, String> {
        Ok(RayTracer {
            bounce: self.bounce,
            sample: self.sample,
            loss: self.loss,
            sampler: RayTracer::default_sampler()
        })
    }
}

impl Wrapper<Camera> for CameraWrapper {
    fn unwrap(self) -> Result<Camera, String> {
        Ok(Camera {
            pos: self.pos,
            dir: self.dir,
            fov: self.fov,
            gamma: self.gamma,
            exp: self.exp,
            aprt: self.aprt,
            foc: self.foc
        })
    }
}

impl Wrapper<Frame> for FrameWrapper {
    fn unwrap(self) -> Result<Frame, String> {
        Ok(Frame {
            res: self.res,
            ssaa: self.ssaa,
            cam: self.cam.unwrap()?
        })
    }
}

impl Wrapper<Texture> for TextureWrapper {
    fn unwrap(self) -> Result<Texture, String> {
        let buf = self.to_buffer()?;

        if let TextureWrapper::Buffer(buf) = buf {
            return Ok(Texture {
                w: buf.w,
                h: buf.h,
                dat: buf.dat
            })
        }

        unreachable!()
    }
}

impl Wrapper<Material> for MaterialWrapper {
    fn unwrap(self) -> Result<Material, String> {
        Ok(Material{
            albedo: self.albedo.unwrap()?,
            rough: self.rough,
            metal: self.metal,
            glass: self.glass,
            opacity: self.opacity,
            emit: self.emit,
            tex: if let Some(tex) = self.tex {Some(tex.unwrap()?)} else {None},
            rmap: if let Some(rmap) = self.rmap {Some(rmap.unwrap()?)} else {None},
            mmap: if let Some(mmap) = self.mmap {Some(mmap.unwrap()?)} else {None},
            gmap: if let Some(gmap) = self.gmap {Some(gmap.unwrap()?)} else {None},
            omap: if let Some(omap) = self.omap {Some(omap.unwrap()?)} else {None},
            emap: if let Some(emap) = self.emap {Some(emap.unwrap()?)} else {None}
        })
    }
}

impl Wrapper<RendererKind> for MeshWrapper {
    fn unwrap(self) -> Result<RendererKind, String> {
        let buf = self.to_buffer()?;

        if let MeshWrapper::Mesh(mesh) = buf {
            return Ok(RendererKind::Mesh(mesh))
        }

        unreachable!()
    }
}

impl Wrapper<RendererKind> for RendererKindWrapper {
    fn unwrap(self) -> Result<RendererKind, String> {
        match self {
            RendererKindWrapper::Sphere{r} => Ok(RendererKind::Sphere{r}),
            RendererKindWrapper::Plane{n} => Ok(RendererKind::Plane{n}),
            RendererKindWrapper::Box{sizes} => Ok(RendererKind::Box{sizes}),
            RendererKindWrapper::Triangle{vtx} => Ok(RendererKind::Triangle{vtx}),
            RendererKindWrapper::Mesh{mesh} => mesh.unwrap()
        }
    }
}

impl Wrapper<Renderer> for RendererWrapper {
    fn unwrap(self) -> Result<Renderer, String> {
        Ok(Renderer{
            kind: self.kind.unwrap()?,
            mat: self.mat.unwrap()?,
            pos: self.pos,
            dir: self.dir
        })
    }
}

impl Wrapper<Sky> for SkyWrapper {
    fn unwrap(self) -> Result<Sky, String> {
        Ok(Sky {
            color: self.color.unwrap()?,
            pwr: self.pwr
        })
    }
}

impl Wrapper<LightKind> for LightKindWrapper {
    fn unwrap(self) -> Result<LightKind, String> {
        match self {
            LightKindWrapper::Point {pos} => Ok(LightKind::Point{pos}),
            LightKindWrapper::Dir {dir} => Ok(LightKind::Dir{dir})
        }
    }
}

impl Wrapper<Light> for LightWrapper {
    fn unwrap(self) -> Result<Light, String> {
        Ok(Light {
            kind: self.kind.unwrap()?,
            pwr: self.pwr,
            color: self.color.unwrap()?
        })
    }
}

impl Wrapper<Vec<Renderer>> for Vec<RendererWrapper> {
    fn unwrap(self) -> Result<Vec<Renderer>, String> {
        let mut objs = Vec::new();

        for obj in self {
            objs.push(obj.unwrap()?);
        }

        Ok(objs)
    }
}

impl Wrapper<Vec<Light>> for Vec<LightWrapper> {
    fn unwrap(self) -> Result<Vec<Light>, String> {
        let mut lights = Vec::new();

        for light in self {
            lights.push(light.unwrap()?);
        }

        Ok(lights)
    }
}

impl Wrapper<Scene> for SceneWrapper {
    fn unwrap(self) -> Result<Scene, String> {
        Ok(Scene {
            renderer: if let Some(objs) = self.renderer {Some(objs.unwrap()?)} else {None},
            light: if let Some(lights) = self.light {Some(lights.unwrap()?)} else {None},
            sky: self.sky.unwrap()?
        })
    }
}

impl Wrapper<Render> for RenderWrapper {
    fn unwrap(self) -> Result<Render, String> {
        Ok(Render {
            rt: self.rt.unwrap()?,
            frame: self.frame.unwrap()?,
            scene: self.scene.unwrap()?
        })
    }
}
