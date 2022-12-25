use std::collections::HashMap;
use std::io::{Write, Read};
use std::net::{TcpListener, TcpStream};

use image::{RgbImage, EncodableLayout};

use crate::rt::Render;
use crate::sampler::Sampler;


pub struct HttpServer {
    pub hlr: TcpListener
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    uri: String,
    version: String,
    headers: HashMap<String, String>,
    body: String
}

impl HttpRequest {
    fn parse(s: &String) -> Result<HttpRequest, String> {
        let header_body = s.splitn(2, "\r\n\r\n").collect::<Vec<_>>();

        // parse body
        let body = header_body[1].trim_end_matches('\0').to_string();

        // parse headers
        let headers = header_body[0].splitn(2, "\r\n").collect::<Vec<_>>();

        let status = headers[0].split(" ").collect::<Vec<_>>();
        let method = status[0].to_string();
        let uri = status[1].to_string();
        let version = status[2].to_string();

        let mut map = HashMap::new();

        let headers = headers[1].split("\r\n").collect::<Vec<_>>();

        for header in headers {
            let header = header.splitn(2, ": ").collect::<Vec<_>>();
            map.insert(header[0].to_string(), header[1].to_string());
        }

        Ok(HttpRequest {
            method,
            uri,
            version,
            headers: map,
            body
        })
    }
}

impl HttpServer {
    pub fn handle(mut s: TcpStream) -> Result<(), String> {
        // request
        let mut buf = [0; 1024 * 1024]; // 1mb
        s.read(&mut buf[..]).map_err(|e| e.to_string())?;

        let req_s = String::from_utf8(Vec::from(buf)).map_err(|e| e.to_string())?;
        let req = HttpRequest::parse(&req_s)?;

        // validate
        if req.version != "HTTP/1.1" {
            let res =  "HTTP/1.1 505 HTTP Version Not Supported\r\n";
            s.write_all(res.as_bytes()).map_err(|e| e.to_string())?;

            return Ok(());
        }

        if req.method != "POST" {
            let res =  "HTTP/1.1 405 Method Not Allowed\r\n";
            s.write_all(res.as_bytes()).map_err(|e| e.to_string())?;

            return Ok(());
        }

        if !req.headers.contains_key("Content-Type") {
            let res =  "HTTP/1.1 400 Bad Request\r\n";
            s.write_all(res.as_bytes()).map_err(|e| e.to_string())?;

            return Ok(());
        }

        if !req.headers.get("Content-Type").unwrap().starts_with("application/json") {
            println!("{}", req.headers.get("Content-Type").unwrap());

            let res =  "HTTP/1.1 415 Unsupported Media Type\r\n";
            s.write_all(res.as_bytes()).map_err(|e| e.to_string())?;

            return Ok(());
        }

        if !req.headers.contains_key("Content-Length") {
            let res =  "HTTP/1.1 411 Length Required\r\n";
            s.write_all(res.as_bytes()).map_err(|e| e.to_string())?;

            return Ok(());
        }

        if req.headers.get("Content-Length").unwrap().parse::<usize>().map_err(|e| e.to_string())? != req.body.as_bytes().len() {
            let res =  "HTTP/1.1 400 Bad Request\r\n";
            s.write_all(res.as_bytes()).map_err(|e| e.to_string())?;

            return Ok(());
        }

        let mut render = serde_json::from_str(&req.body).map_err(|e| e.to_string())?;
        let img = HttpServer::raytrace(&mut render)?;

        let mut img_jpg: Vec<u8> = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut img_jpg), image::ImageOutputFormat::Jpeg(90)).map_err(|e| e.to_string())?;

        // response
        let res = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n{}\r\n",
            img_jpg.as_bytes().len(),
            unsafe {String::from_utf8_unchecked(img_jpg)}
        );

        s.write_all(res.as_bytes()).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn raytrace(render: &mut Render) -> Result<RgbImage, String> {
        // unwrap textures
        render.scene.sky.color.to_vec3()?;

        if let Some(ref mut lights) = render.scene.light {
            for light in lights {
                light.color.to_vec3()?
            }
        }

        if let Some(ref mut objs) = render.scene.renderer {
            for obj in objs {
                obj.mat.albedo.to_vec3()?;

                if let Some(tex) = &mut obj.mat.tex {
                    tex.to_buffer()?;
                }
                if let Some(rmap) = &mut obj.mat.rmap {
                    rmap.to_buffer()?;
                }
                if let Some(mmap) = &mut obj.mat.mmap {
                    mmap.to_buffer()?;
                }
                if let Some(gmap) = &mut obj.mat.gmap {
                    gmap.to_buffer()?;
                }
                if let Some(omap) = &mut obj.mat.omap {
                    omap.to_buffer()?;
                }
                if let Some(emap) = &mut obj.mat.emap {
                    emap.to_buffer()?;
                }
            }
        }

        // raytrace
        let mut sampler = Sampler::new(24, 64);

        for _ in 0..render.rt.sample {
            sampler.execute(&render.scene, &render.frame, &render.rt);

            // let time = sampler.execute(&render.scene, &render.frame, &render.rt);
            // println!("total {:?}", time);
        }

        // convert to image
        sampler.img(&render.frame)
    }

    pub fn start(&self) -> Result<(), String> {
        for s in self.hlr.incoming() {
            let stream = s.map_err(|e| e.to_string())?;

            std::thread::spawn(|| HttpServer::handle(stream).unwrap());
        }

        Ok(())
    }
}