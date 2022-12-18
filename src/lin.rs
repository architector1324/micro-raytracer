use serde::{Serialize, Deserialize};
use rand::Rng;


#[derive(Debug, Clone)]
pub struct Vec2f {
    pub x: f32,
    pub y: f32
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(into = "[f32; 3]", from = "[f32; 3]")]
pub struct Vec3f {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(into = "[f32; 4]", from = "[f32; 4]")]
pub struct Vec4f {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[derive(Debug, Clone, Copy)]
pub struct Mat3f([f32; 9]);

impl Vec2f {
    pub fn zero() -> Self {
        Vec2f {x: 0.0, y: 0.0}
    }
}

impl Vec4f {
    pub fn forward() -> Self {
        Vec4f {
            w: 0.0,
            x: 0.0,
            y: 1.0,
            z: 0.0
        }
    }

    pub fn proj(self) -> Vec3f {
        Vec3f {
            x: self.x,
            y: self.y,
            z: self.z
        }
    }
}

impl Vec3f {
    pub fn forward() -> Self {
        Vec3f {x: 0.0, y: 1.0, z: 0.0}
    }

    pub fn right() -> Self {
        Vec3f {x: 1.0, y: 0.0, z: 0.0}
    }

    pub fn up() -> Self {
        Vec3f {x: 0.0, y: 0.0, z: 1.0}
    }
    
    pub fn mag(self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    pub fn norm(self) -> Self {
        self * self.mag().recip()
    }

    pub fn reflect(self, n: Vec3f) -> Self {
        self - n * (2.0 * (self * n))
    }

    pub fn recip(self) -> Self {
        Vec3f {
            x: self.x.recip(),
            y: self.y.recip(),
            z: self.z.recip()
        }
    }

    pub fn abs(self) -> Self {
        Vec3f {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs()
        }
    }

    pub fn clamp(self, min: f32, max: f32) -> Self {
        Vec3f {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max)
        }
    }

    pub fn refract(self, eta: f32, n: Vec3f) -> Option<Vec3f> {
        let cos = -n * self;

        let k = 1.0 - eta.powi(2) * (1.0 - cos.powi(2));
        if k < 0.0 {
            return None
        }

        Some(self * eta + n * (cos * eta + k.sqrt()))
    }

    pub fn hadam(self, rhs: Vec3f) -> Self {
        Vec3f {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z
        }
    }

    pub fn rand(max_v: f32) -> Self {
        Vec3f {
            x: rand::thread_rng().gen_range(-0.5..0.5) * max_v,
            y: rand::thread_rng().gen_range(-0.5..0.5) * max_v, 
            z: rand::thread_rng().gen_range(-0.5..0.5) * max_v
        }
    }

    pub fn zero() -> Self {
        Vec3f {
            x: 0.0,
            y: 0.0,
            z: 0.0
        }
    }
}

impl Mat3f {
    pub fn rotate_dir(dir: Vec4f) -> (Mat3f, Mat3f, Mat3f) {
        let n_dir = dir.proj().norm();

        let rot_x = Mat3f([
            1.0, 0.0, 0.0,
            0.0, n_dir.y, -n_dir.z,
            0.0, n_dir.z, n_dir.y
        ]);

        let cw = (1.0 - dir.w.powi(2)).sqrt();

        let rot_y = Mat3f([
            cw, 0.0, dir.w,
            0.0, 1.0, 0.0,
            -dir.w, 0.0, cw
        ]);

        let rot_z = Mat3f([
            n_dir.y, n_dir.x, 0.0,
            -n_dir.x, n_dir.y, 0.0,
            0.0, 0.0, 1.0
        ]);

        (rot_x, rot_y, rot_z)
    }
}

impl std::ops::Add for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3f {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z
        }
    }
}

impl std::ops::Add<f32> for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: f32) -> Self::Output {
        Vec3f {
            x: self.x + rhs,
            y: self.y + rhs,
            z: self.z + rhs
        }
    }
}

impl std::ops::Sub<f32> for Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs: f32) -> Self::Output {
        Vec3f {
            x: self.x - rhs,
            y: self.y - rhs,
            z: self.z - rhs
        }
    }
}

impl std::ops::Sub for Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3f {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z
        }
    }
}

impl std::ops::Mul for Vec3f {
    type Output = f32;
    fn mul(self, rhs: Self) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl std::ops::Mul<f32> for Vec3f {
    type Output = Vec3f;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec3f {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs
        }
    }
}

impl std::ops::Div<f32> for Vec3f {
    type Output = Vec3f;

    fn div(self, rhs: f32) -> Self::Output {
        self * rhs.recip()
    }
}

impl std::ops::Neg for Vec3f {
    type Output = Vec3f;

    fn neg(self) -> Self::Output {
        Vec3f {
            x: -self.x,
            y: -self.y,
            z: -self.z
        }
    }
}

impl std::ops::Div for Vec3f {
    type Output = Vec3f;

    fn div(self, rhs: Self) -> Self::Output {
        Vec3f {
            x: self.x / rhs.x,
            y: self.y / rhs.y, 
            z: self.z / rhs.z
        }
    }
}

impl std::ops::AddAssign for Vec3f {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::SubAssign for Vec3f {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl std::ops::Mul<Vec3f> for Mat3f {
    type Output = Vec3f;

    fn mul(self, rhs: Vec3f) -> Self::Output {
        Vec3f {
            x: self.0[0] * rhs.x + self.0[1] * rhs.y + self.0[2] * rhs.z,
            y: self.0[3] * rhs.x + self.0[4] * rhs.y + self.0[5] * rhs.z,
            z: self.0[6] * rhs.x + self.0[7] * rhs.y + self.0[8] * rhs.z,
        }
    }
}

impl std::fmt::Display for Vec3f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({} {} {})", self.x, self.y, self.z))
    }
}

impl std::fmt::Display for Vec2f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({} {})", self.x, self.y))
    }
}

impl From<Vec3f> for [f32; 3] {
    fn from(v: Vec3f) -> Self {
        [v.x, v.y, v.z]
    }
}

impl From<[f32; 3]> for Vec3f {
    fn from(v: [f32; 3]) -> Self {
        Vec3f {
            x: v[0],
            y: v[1],
            z: v[2]
        }
    }
}

impl Default for Vec3f {
    fn default() -> Self {
        Vec3f::zero()
    }
}

impl From<Vec4f> for [f32; 4] {
    fn from(v: Vec4f) -> Self {
        [v.w, v.x, v.y, v.z]
    }
}

impl From<[f32; 4]> for Vec4f {
    fn from(v: [f32; 4]) -> Self {
        Vec4f {
            w: v[0],
            x: v[1],
            y: v[2],
            z: v[3]
        }
    }
}

pub trait ParseFromStrIter<'a> {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self;
}

impl <'a> ParseFromStrIter<'a> for Vec3f {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self {
        Vec3f {
            x: <f32>::parse(it),
            y: <f32>::parse(it),
            z: <f32>::parse(it)
        }
    }
}

impl <'a> ParseFromStrIter<'a> for Vec4f {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self {
        Vec4f {
            w: <f32>::parse(it),
            x: <f32>::parse(it),
            y: <f32>::parse(it),
            z: <f32>::parse(it)
        }
    }
}

impl <'a> ParseFromStrIter<'a> for f32 {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self {
        it.next().unwrap().parse::<f32>().expect("should be <f32>!")
    }
}
