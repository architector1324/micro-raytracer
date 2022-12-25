use clap::Parser;
use std::net::TcpListener;

use micro_raytracer::http::HttpServer;
use micro_raytracer::cli::CLI;


fn main_wrapped() -> Result<(), String> {
    let cli = CLI::parse();

    // launch http server
    if let Some(addr) = &cli.http {
        let server = HttpServer {
            hlr: TcpListener::bind(addr).map_err(|e| e.to_string())?
        };

        server.start()?;
    }

    // parse render
    let mut render = cli.parse_render()?;

    // verbose
    if cli.verbose {
        if cli.pretty {
            println!("{}", serde_json::to_string_pretty(&render).map_err(|e| e.to_string())?);
        } else {
            println!("{}", serde_json::to_string(&render).map_err(|e| e.to_string())?);
        }
    }

    if cli.dry {
        return Ok(());
    }

    cli.raytrace(&mut render)
}

fn main() {
    if let Err(e) = main_wrapped() {
        println!("{{err: \"{}\"}}", e.to_string());
    }
}
