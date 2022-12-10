# Raytracing Microservice

Lightweight raytracing microservice written in [Rust](https://www.rust-lang.org/).

The main idea is to use flags or [json](https://www.json.org/json-en.html) for generating images in terminal, shell scripts, http servers and etc.

It's something like [Zenity](https://github.com/GNOME/zenity), that provides you to create simple UI in terminal.

## Usage
### In-place in terminal
Let's render simple scene with sphere in terminal:
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

Let's change some color of sphere:
```bash
raytrace --sphere albedo: 1 0 0 --light pos: -0.5 -1 0.5 --res 1280 720 -o final.ppm
```

TBD...

### JSON frame and scene description
1. First create `scene.json` file contains scene information:
```json
{
    "renderer": [
        {
            "type": "sphere",
            "pos": [0, 0, 0],
            "r": 0.5,
            "mat": {
                "albedo": [1, 1, 1],
                "rough": 0.5,
                "metal": 0,
                "glass": 0,
                "opacity": 1
            }
        }
    ],
    "light": [
        {
            "pos": [0, -1, -0.5],
            "pwr": 0.5,
            "color": [0, 1, 0]
        },
        {
            "pos": [-0.5, -1, 0.5],
            "pwr": 0.8,
            "color": [1, 0, 0]
        },
        {
            "pos": [0.5, -1, 0.5],
            "pwr": 0.8,
            "color": [0, 0, 1]
        }
    ],
    "sky": [0, 0, 0]
}
```

2. Next create `frame.json` file contains output frame information:
```json
{
    "res": [800, 600],
    "cam": {
        "pos": [0, -1, 0],
        "dir": [0, 1, 0],
        "fov": 90
    }
}
```

3. Finally, run following command:

```bash
raytrace --scene scene.json --frame frame.json
```

![image](doc/out2.png)

## API
TBD...
