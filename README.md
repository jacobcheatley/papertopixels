# 🖼️ PaperToPixels

**From doodles to digital domains**
A Flask web app that transforms hand-drawn maps into playable game levels using OpenCV magic.

---

## 🧠 What is this?

**PaperToPixels** is a creative tool that bridges the gap between analog sketches and digital gameplay.
By uploading an image of a hand-drawn map, the app processes it using computer vision techniques to generate a playable game map.
It's a fusion of art and technology, turning your paper designs into interactive digital experiences.

---

## 🎯 Why does it exist?

Because designing game levels should be as intuitive as sketching on paper.
This project empowers artists, game designers, and hobbyists to bring their ideas to life without the need for complex digital tools.
It's about making game development more accessible and fun.

---

## 🛠️ How it works

1. **Upload Your Sketch**
   Draw your game map on paper and upload a clear image of it.

2. **Image Processing**
   The app uses OpenCV to detect lines, shapes, and patterns, interpreting them as game elements.

3. **Map Generation**
   The processed data is converted into a digital game map format.

4. **Play**
   Use the generated map in your game engine or platform of choice.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.6+
* pip

### Installation

```bash
git clone https://github.com/jacobcheatley/papertopixels.git
cd papertopixels
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python papertopixels.py
```

Open your browser and go to:
`http://localhost:5000`

---

## 📁 Project Structure

* `papertopixels.py` – Main Flask app
* `api.py` – API routes and image processing logic
* `image_config.py` – Image processing parameters
* `app_config.py` – Application configuration
* `templates/` – HTML templates
* `static/` – CSS, JavaScript, and static assets
* `image/` – Image processing scripts/utilities

---

## 🧪 Example

Upload a sketch like this:

```
(static/images/sketch_example.png)
```

And get a digital map like this:

```
(static/images/map_example.png)
```

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📄 License

MIT License. See the [LICENSE](LICENSE) file for details.
