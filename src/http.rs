use std::collections::HashMap;
use std::io::{Write, Read};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::time::{Duration, Instant};

use image::{RgbImage, EncodableLayout};
use log::{info, error};

use crate::parser::{RenderWrapper, Wrapper};
use crate::rt::Render;
use crate::sampler::Sampler;


pub struct HttpServer {
    pub hlr: TcpListener
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    _uri: String,
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
            _uri: uri,
            version,
            headers: map,
            body
        })
    }
}

impl HttpServer {
    pub fn handle(mut s: TcpStream) -> Result<(), String> {
        let addr = s.peer_addr().map_err(|e| e.to_string())?;

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

        let render: RenderWrapper = serde_json::from_str(&req.body).map_err(|e| e.to_string())?;
        info!("http:render[{}]: {}", addr, serde_json::to_string(&render).map_err(|e| e.to_string())?);

        let (img, time) = HttpServer::raytrace(addr, &mut render.unwrap()?)?;
        info!("http:done[{}]: {:?}", addr, time);

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

    pub fn raytrace(addr: SocketAddr, render: &mut Render) -> Result<(RgbImage, Duration), String> {
        // raytrace
        let mut sampler = Sampler::new(24, 64);
        let time = Instant::now();

        for sample in 0..render.rt.sample {
            let time = sampler.execute(&render.scene, &render.frame, &render.rt);
            info!("http:sample[{}]:{}: {:?}", addr, sample, time);
        }

        // convert to image
        Ok((sampler.img(&render.frame)?, time.elapsed()))
    }

    pub fn start(&self) -> Result<(), String> {
        for s in self.hlr.incoming() {
            let stream = s.map_err(|e| e.to_string())?;
            info!("http:connected: {}", stream.peer_addr().map_err(|e| e.to_string())?);

            std::thread::spawn(|| {
                if let Err(e) = HttpServer::handle(stream) {
                    error!("http: {}", e.to_string());
                }
            });
        }

        Ok(())
    }
}
