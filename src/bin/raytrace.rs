use std::net::TcpListener;

use clap::Parser;
use log::{LevelFilter, info, error};
use simplelog::{TermLogger, TerminalMode, Config, ColorChoice};

use micro_raytracer::http::HttpServer;
use micro_raytracer::cli::CLI;


fn main_wrapped() -> Result<(), String> {
    TermLogger::init(LevelFilter::Info, Config::default(), TerminalMode::Stdout, ColorChoice::Auto).map_err(|e| e.to_string())?;

    let cli = CLI::parse();

    if !cli.verbose {
        log::set_max_level(LevelFilter::Off);
    }

    // launch http server
    if let Some(addr) = &cli.http {
        log::set_max_level(LevelFilter::Info);

        let server = HttpServer {
            hlr: TcpListener::bind(addr).map_err(|e| e.to_string())?
        };

        server.start()?;
    }

    // parse render
    let mut render = cli.parse_render()?;

    // verbose
    if cli.pretty {
        info!("cli:render: {}", serde_json::to_string_pretty(&render).map_err(|e| e.to_string())?);
    } else {
        info!("cli:render: {}", serde_json::to_string(&render).map_err(|e| e.to_string())?);
    }

    if cli.dry {
        return Ok(());
    }

    info!("cli:done: {:?}", cli.raytrace(&mut render)?);

    Ok(())
}

fn main() {
    if let Err(e) = main_wrapped() {
        error!("cli: {}", e.to_string());
    }
}
