import numpy as np
import vispy.scene
from vispy.scene import visuals
from threading import Thread


class Visualizer():

    def __init__(self, planes):
        #Thread.__init__(self)
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.view = canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        for p in planes:
            mesh = self.__define_plane(p)
            self.view.add(mesh)
        axis = visuals.XYZAxis(parent=self.view.scene)
        #scatter = visuals.Markers()


    def __define_plane(self, w):
        x, y, z = 0.5, 0.375, 1.0
        v = np.array([
            [-x*w,  y*w, z*w],
            [x*w,  -y*w, z*w],
            [-x*w, -y*w, z*w],
            [-x*w,  y*w, z*w],
            [x*w,   y*w, z*w],
            [x*w,  -y*w, z*w]
        ])
        colors = [(0.8, 0.5, 0.7) for i in range(len(v))]
        return visuals.Mesh(v, color=(0.5,0.5,0.8,0.8))

    
    def __define_eyeballs(self):
        left = np.array((-0.12, -0.08, -0.05), float)
        right = np.array((-0.02, -0.08, -0.05), float)
        #draw sphere


    def run(self):
        vispy.app.run()


if __name__=='__main__':
    vis = Visualizer([0.75, 1.35, 2.0])
    vis.run()