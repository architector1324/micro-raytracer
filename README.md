# Raytracing Microservice

Tiny raytracing microservice written in [Rust](https://www.rust-lang.org/).

The main idea is to use flags or [json](https://www.json.org/json-en.html) for generating images in terminal, shell scripts, http servers and etc.

It's something like [Zenity](https://github.com/GNOME/zenity), that provides you to create simple UI in terminal.

## Usage
### In-place in terminal
Let's render simple scene with sphere in terminal:
```bash
raytrace --sphere --pos 0.5 0.5 10 --r 20 --color 1 0 0 --light --pos 0 0.5 1 --pwr 1 --color 1 0 0 --frame 800 600 -o out.png
```

### JSON frame and scene description
1. First create `scene.json` file contains scene information:
```json
{
    "renderer": [
        {
            "type": "sphere",
            "pos": [0.5, 0.5, 10],
            "r": 20,
            "mat": {
                "color": [1, 0, 0]
            }
        }
    ],
    "lights": [
        {
            "pos": [0, 0.5, 1],
            "pwr": 1,
            "color": [1, 0, 0]
        }
    ]
}
```

2. Next create `frame.json` file contains output frame information:
```json
{
    "resolution": [800, 600],
    "cam": {
        "pos": [0.5, 0, 0.5],
        "dir": [0, 1, 0],
        "fov": 60
    }
}
```

3. Finally, run following command:

```bash
raytrace scene.json frame.json -o out.png
```

## API
TBD...