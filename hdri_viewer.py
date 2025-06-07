import os

import imageio
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
import moderngl
from pyrr import Matrix44, Vector3


class GLWindow(QtWidgets.QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.ctx = None
        self.prog = None
        self.mvp = None
        self.vao = None
        self.texture = None
        self.rotation = 0.0
        self.exposure = 1.0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # about 60 FPS

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                uniform mat4 mvp;
                out vec3 normal;
                out vec3 pos;
                void main() {
                    pos = in_position;
                    normal = in_normal;
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D env_map;
                uniform float rotation;
                uniform float exposure;
                in vec3 normal;
                in vec3 pos;
                out vec4 fragColor;
                vec2 env_map_coord(vec3 dir){
                    float phi = atan(dir.z, dir.x) + rotation;
                    float theta = acos(clamp(dir.y, -1.0, 1.0));
                    return vec2(phi/(2.0*3.14159265) + 0.5, theta/3.14159265);
                }
                void main() {
                    vec3 n = normalize(normal);
                    vec3 r = reflect(normalize(pos), n);
                    vec2 uv = env_map_coord(r);
                    vec3 color = texture(env_map, uv).rgb * exposure;
                    fragColor = vec4(color, 1.0);
                }
            '''
        )
        self.mvp = self.prog['mvp']
        self.rotation_uniform = self.prog['rotation']
        self.exposure_uniform = self.prog['exposure']
        self.texture_uniform = self.prog['env_map']

        sphere = create_sphere()
        vbo = self.ctx.buffer(np.array(sphere['v'], dtype='f4'))
        nbo = self.ctx.buffer(np.array(sphere['n'], dtype='f4'))
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, '3f', 'in_position'),
             (nbo, '3f', 'in_normal')]
        )

    def load_texture(self, path):
        hdr = imageio.imread(path)
        hdr = np.flipud(hdr)  # image origin
        self.texture = self.ctx.texture(hdr.shape[1::-1], 3, hdr.astype('f4').tobytes(), dtype='f4')
        self.texture.build_mipmaps()

    def paintGL(self):
        if self.texture is None:
            self.ctx.clear(0.2, 0.2, 0.2)
            return
        self.ctx.clear(0.2, 0.2, 0.2)
        self.texture.use(location=0)
        proj = Matrix44.perspective_projection(45.0, self.width()/self.height(), 0.1, 100.0)
        look = Matrix44.look_at(Vector3([0.0, 0.0, 3.0]), Vector3([0.0, 0.0, 0.0]), Vector3([0.0,1.0,0.0]))
        self.mvp.write((proj*look).astype('f4').tobytes())
        self.rotation_uniform.value = self.rotation
        self.exposure_uniform.value = self.exposure
        self.vao.render(moderngl.TRIANGLES)


def create_sphere(segments=64):
    vertices = []
    normals = []
    for y in range(segments):
        for x in range(segments):
            x0 = x / segments
            y0 = y / segments
            x1 = (x + 1) / segments
            y1 = (y + 1) / segments

            theta0 = x0 * 2.0 * np.pi
            theta1 = x1 * 2.0 * np.pi
            phi0 = y0 * np.pi
            phi1 = y1 * np.pi

            p0 = sph(theta0, phi0)
            p1 = sph(theta1, phi0)
            p2 = sph(theta1, phi1)
            p3 = sph(theta0, phi1)

            vertices.extend([p0, p1, p2, p0, p2, p3])
            normals.extend([p0, p1, p2, p0, p2, p3])
    return {'v': vertices, 'n': normals}

def sph(theta, phi):
    sin_phi = np.sin(phi)
    return [np.cos(theta)*sin_phi, np.cos(phi), np.sin(theta)*sin_phi]


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDRI Browser")
        self.resize(800, 600)
        self.gl = GLWindow()
        self.rotate_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rotate_slider.setRange(0, 360)
        self.rotate_slider.valueChanged.connect(self.update_rotation)
        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exposure_slider.setRange(-10, 10)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self.update_exposure)
        open_btn = QtWidgets.QPushButton("Open HDRI")
        open_btn.clicked.connect(self.open_file)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.gl)
        layout.addWidget(QtWidgets.QLabel("Rotate"))
        layout.addWidget(self.rotate_slider)
        layout.addWidget(QtWidgets.QLabel("Exposure"))
        layout.addWidget(self.exposure_slider)
        layout.addWidget(open_btn)
        self.setLayout(layout)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open HDRI', os.getcwd(), 'HDR files (*.hdr *.exr)')
        if path:
            self.gl.load_texture(path)

    def update_rotation(self, value):
        self.gl.rotation = np.radians(value)

    def update_exposure(self, value):
        self.gl.exposure = np.power(2.0, value)


def main():
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()


if __name__ == '__main__':
    main()
