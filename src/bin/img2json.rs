use std::path::PathBuf;

use clap::Parser;
use serde_json::json;

use micro_raytracer::parser::TextureWrapper;


#[derive(Parser)]
#[command(author, version, about = "Convert images to json for micro-rt.", long_about = None)]
struct CLI {
    #[arg(next_line_help = true, help = "Input image filename")]
    img: String,

    #[arg(long, action, next_line_help = true, help = "Print json with prettifier")]
    pretty: bool,

    #[arg(short, long, value_name="fmt: <buf|inl>", next_line_help = true, help = "Texture format")]
    fmt: Option<String>
}

fn main() {
    let cli = CLI::parse();

    let mut tex = TextureWrapper::File(PathBuf::from(cli.img));

    if let Some(fmt) = cli.fmt {
        match fmt.as_str() {
            "buf" => tex = tex.to_buffer().unwrap(),
            "inl" => tex = tex.to_inline().unwrap(),
            _ => panic!("unknown texture format {}!", fmt.as_str())
        }
    } else {
        tex = tex.to_buffer().unwrap();
    }

    let tex_json = json!({
        "tex": tex
    });

    let final_json = if cli.pretty {
        serde_json::to_string_pretty(&tex_json).unwrap()
    } else {
        serde_json::to_string(&tex_json).unwrap()
    };

    println!("{}", final_json);
}
