use std::path::PathBuf;

use clap::Parser;
use serde_json::json;

use micro_raytracer::parser::{TextureWrapper, MeshWrapper};


#[derive(Parser)]
#[command(author, version, about = "Convert images to json for micro-rt.", long_about = None)]
struct CLI {
    #[arg(long, next_line_help = true, help = "Input image filename")]
    img: Option<String>,

    #[arg(long, next_line_help = true, help = "Input wavefont object filename")]
    obj: Option<String>,

    #[arg(long, action, next_line_help = true, help = "Print json with prettifier")]
    pretty: bool,

    #[arg(short, long, value_name="fmt: <buf|inl>", next_line_help = true, help = "Texture format")]
    fmt: Option<String>
}

fn main() {
    let cli = CLI::parse();

    let mut inner_json = json!({});

    if let Some(img) = cli.img {
        let mut tex = TextureWrapper::File(PathBuf::from(img));
    
        if let Some(fmt) = cli.fmt {
            match fmt.as_str() {
                "buf" => tex = tex.to_buffer().unwrap(),
                "inl" => tex = tex.to_inline().unwrap(),
                _ => panic!("unknown texture format {}!", fmt.as_str())
            }
        } else {
            tex = tex.to_buffer().unwrap();
        }
    
        inner_json = json!({
            "tex": tex
        });
    
    } else if let Some(obj) = cli.obj {
        let mut obj = MeshWrapper::File(PathBuf::from(obj));
    
        if let Some(fmt) = cli.fmt {
            match fmt.as_str() {
                "buf" => obj = obj.to_buffer().unwrap(),
                "inl" => obj = obj.to_inline().unwrap(),
                _ => panic!("unknown object format {}!", fmt.as_str())
            }
        } else {
            obj = obj.to_buffer().unwrap();
        }

        inner_json = json!({
            "mesh": obj
        });
    }

    let final_json = if cli.pretty {
        serde_json::to_string_pretty(&inner_json).unwrap()
    } else {
        serde_json::to_string(&inner_json).unwrap()
    };

    println!("{}", final_json);
}
