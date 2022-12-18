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

pub type Mat3f = [f32; 9];

impl Vec2f {
    pub fn zero() -> Vec2f {
        Vec2f {x: 0.0, y: 0.0}
    }
}

impl Vec3f {
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
            x: self[0] * rhs.x + self[1] * rhs.y + self[2] * rhs.z,
            y: self[3] * rhs.x + self[4] * rhs.y + self[5] * rhs.z,
            z: self[6] * rhs.x + self[7] * rhs.y + self[8] * rhs.z,
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

impl <'a> ParseFromStrIter<'a> for f32 {
    fn parse<I: Iterator<Item = &'a String>>(it: &mut I) -> Self {
        it.next().unwrap().parse::<f32>().expect("should be <f32>!")
    }
}
