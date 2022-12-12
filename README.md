# Raytracing Microservice

Lightweight raytracing microservice written in [Rust](https://www.rust-lang.org/).

The main idea is to use flags or [json](https://www.json.org/json-en.html) for generating images in terminal, shell scripts, http servers and etc.

It's something like [Zenity](https://github.com/GNOME/zenity), that provides you to create simple UI in terminal.

## Build
Build statically for [linux](https://en.wikipedia.org/wiki/Linux) using [musl](https://musl.libc.org/). This executable may run on any linux system without any additional libs.

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## Usage
```bash
$ ./raytrace -h
Tiny raytracing microservice.

Usage: raytrace [OPTIONS]

Options:
  -v, --verbose                         Print full info in json
      --pretty                          Print full info in json with prettifier
  -d, --dry                             Dry run (useful with verbose)
  -o, --output <FILE.EXT>               Final image output filename
      --bounce <BOUNCE>                 Max ray bounce
      --sample <SAMPLE>                 Max path-tracing samples
      --gamma <GAMMA>                   Final image gamma correction
      --exp <EXP>                       Final image camera exposure
      --loss <LOSS>                     Ray bounce energy loss
  -u, --update                          Save output on each sample
  -w, --worker <WORKER>                 Parallel workers count
  -s, --scene <FILE.json>               Scene description json input filename
  -f, --frame <FILE.json>               Frame description json input filename
      --res <w> <h>                     Frame output image resolution
      --cam <pos> <dir> <fov>...        Frame camera
      --sphere [<pos> <r> <albedo>...]  Render sphere
      --light [<pos> <pwr> <col>...]    Light source
  -h, --help                            Print help information
  -V, --version                         Print version information
```

### In-place in terminal
Let's render simple scene with sphere in terminal:

/!\ CLI make scene is not ready yet!
```bash
raytrace --sphere --light pos: -0.5 -1 0.5
```

It will produce an PNG image 800x600:
![image](doc/out0.png)

Now let's change a resolution and output file:
```bash
raytrace --sphere --light pos: -0.5 -1 0.5 --res 1280 720 -o final.ppm
```

![image](doc/out1.png)

Let's make something interesting (it will take some time):

```bash
raytrace --sphere r: 0.2 pos: 0 0 0.7 albedo: 1 0.92 0.6 emit \
         --sphere r: 0.2 pos: 0 0.5 0 albedo: 1 0 0 \
         --sphere r: 0.2 pos: 0.5 0 0 metal: 1 \
         --sphere r: 0.2 pos: -0.25 -0.5 0 glass: 0.2 opacity: 0 \
         --sphere r: 0.2 pos: -0.5 0 0 rough: 1 \
         --sphere r: 100 pos: 0 0 -100.2 rough: 1 \
         --sphere r: 100 pos: 0 0 101.2 rough: 1 \
         --sphere r: 100 pos: 0 101 0 rough: 1 \
         --sphere r: 100 pos: -101 0 0 albedo: 1 0 0 rough: 1 \
         --sphere r: 100 pos: 101 0 0 albedo: 0 1 0 rough: 1 \
         --cam pos: 0 -2 0.5 fov: 60 gamma 0.4 exp: 0.6 \
         --sample 1024
```

![image](doc/out2.png)

### JSON frame and scene description
1. First create `scene.json` file contains scene information:
```json
{
    "renderer": [
        {
            "type": "sphere",
            "r": 0.2,
            "pos": [0, 0, 0.7],
            "mat": {
                "albedo": [1, 0.917647058824, 0.596078431372],
                "emit": true
            }
        },
        {
            "type": "sphere",
            "r": 0.2,
            "pos": [0, 0.5, 0],
            "mat": {
                "albedo": [1, 0, 0]
            }
        },
        {
            "type": "sphere",
            "pos": [0.5, 0, 0],
            "r": 0.2,
            "mat": {
                "metal": 1
            }
        },
        {
            "type": "sphere",
            "pos": [-0.25, -0.5, 0],
            "r": 0.2,
            "mat": {
                "glass": 0.2,
                "opacity": 0
            }
        },
        {
            "type": "sphere",
            "pos": [-0.5, 0, 0],
            "r": 0.2,
            "mat": {
                "rough": 1
            }
        },
        {
            "type": "sphere",
            "pos": [0, 0, -100.2],
            "r": 100,
            "mat": {
                "rough": 1
            }
        },
        {
            "type": "sphere",
            "pos": [0, 0, 101.2],
            "r": 100,
            "mat": {
                "rough": 1
            }
        },
        {
            "type": "sphere",
            "pos": [0, 101, 0],
            "r": 100,
            "mat": {
                "rough": 1
            }
        },
        {
            "type": "sphere",
            "pos": [-101, 0, 0],
            "r": 100,
            "mat": {
                "albedo": [1, 0, 0],
                "rough": 1
            }
        },
        {
            "type": "sphere",
            "pos": [101, 0, 0],
            "r": 100,
            "mat": {
                "albedo": [0, 1, 0],
                "rough": 1
            }
        }
    ],
    "light": []
}
```

2. Next create `frame.json` file contains output frame information:
```json
{
    "res": [800, 600],
    "cam": {
        "pos": [0, -2, 0.5],
        "dir": [0, 1, 0],
        "fov": 60,
        "gamma": 0.4,
        "exp": 0.6
    }
}
```

3. Finally, run following command (it will take some time):

```bash
raytrace --scene scene.json --frame frame.json --sample 1024
```

![image](doc/out2.png)

### Terminal to json

1. Also you can use `--verbose,-v` flag with `--dry,-d` to get full info in json from cli command:
```bash
raytrace -v -d --sphere --light pos: -0.5 -1 0.5
```

```json
{"frame":{"cam":{"dir":[0.0,1.0,0.0],"fov":90.0,"gamma":0.8,"exp":0.2,"pos":[0.0,-1.0,0.0]},"res":[800,600]},"rt":{"bounce":8,"loss":0.15,"sample":16},"scene":{"light":[{"color":[1.0,1.0,1.0],"pos":[-0.5,-1.0,0.5],"pwr":0.5}],"renderer":[{"mat":{"albedo":[1.0,1.0,1.0],"emit":false,"glass":0.0,"metal":0.0,"opacity":1.0,"rough":0.0},"pos":[0.0,0.0,0.0],"r":0.5,"type":"sphere"}],"sky":[0.0,0.0,0.0]}}
```

2. With prettifier:
```bash
raytrace -v -d --pretty --sphere --light pos: -0.5 -1 0.5
```

```json
{
    "frame": {
        "cam": {
            "dir": [0, 1, 0],
            "fov": 90,
            "gamma": 0.8,
            "exp": 0.2,
            "pos": [0, -1, 0]
        },
        "res": [800, 600]
    },
    "rt": {
        "bounce": 8,
        "loss": 0.15,
        "sample": 16
    },
    "scene": {
        "light": [
            {
                "color": [1, 1, 1],
                "pos": [-0.5, -1, 0.5],
                "pwr": 0.5
            }
        ],
        "renderer": [
            {
                "mat": {
                    "albedo": [1, 1, 1],
                    "emit": false,
                    "glass": 0,
                    "metal": 0,
                    "opacity": 1,
                    "rough": 0
                },
                "pos": [0, 0, 0],
                "r": 0.5,
                "type": "sphere"
            }
        ],
        "sky": [0, 0, 0]
    }
}
```

## API
TBD...
