import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from threading import Thread
from multiprocessing import Process


class Visualizer():

    def __init__(self, planes, leyeball, reyeball, pipe):
        '''
        NOTE: the R200 camera uses a right-handed coordinate system!
        This means that relative to the image projection, x-values
        increase to the right and y-values increase as they go to the bottom
        '''
        self.leyeball = leyeball
        self.reyeball = reyeball
        self.le_vtx = None
        self.re_vtx = None
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        for p in planes:
            mesh = self.__define_plane(p)
            self.view.add(mesh)
        self.__define_eyeballs()
        axis = visuals.XYZAxis(parent=self.view.scene)
        self.__build_gaze_rays()
        self.pipe = pipe
        self.quit = False
        update_t = Thread(target=self.update, args=())
        update_t.start()
        self.run()
        update_t.join()
        #scatter = visuals.Markers()
        #scatter.set_data(pos, size=0.020)


    def __define_plane(self, w):
        x, y, z = 0.5, 0.375, 1.0
        v = np.array([
            [-x*w, -y*w, z*w],
            [-x*w,  y*w, z*w],
            [x*w,   y*w, z*w],
            [-x*w, -y*w, z*w],
            [x*w,   y*w, z*w],
            [x*w,  -y*w, z*w]
        ])
        colors = [(0.8, 0.5, 0.7) for i in range(len(v))]
        return visuals.Mesh(v, color=(0.5,0.5,0.8,0.8))

    
    def __define_eyeballs(self):
        lsphere = visuals.Sphere(radius = 0.024, method='ico', color='red')
        rsphere = visuals.Sphere(radius = 0.024, method='ico', color='green')
        lsphere.transform = STTransform(translate=self.leyeball)
        rsphere.transform = STTransform(translate=self.reyeball)
        self.view.add(lsphere)
        self.view.add(rsphere)


    def __build_gaze_rays(self):
        self.le_vtx = np.vstack((self.leyeball, self.leyeball))
        self.re_vtx = np.vstack((self.reyeball, self.reyeball))
        le_ray = visuals.Line(self.le_vtx, color='red', width=3)
        re_ray = visuals.Line(self.re_vtx, color='green', width=3)
        self.view.add(le_ray)
        self.view.add(re_ray)

    def __draw_gaze_rays(self, leye, reye):
        leye = leye * 50.0
        reye = reye * 50.0
        self.le_vtx[1] = leye
        self.re_vtx[1] = reye
        self.canvas.update()


    def update(self):
        while not self.quit:
            leye, reye = self.pipe.recv()
            #print("GOT:", leye, reye)
            self.__draw_gaze_rays(leye, reye)


    def run(self):
        self.canvas.app.run()
        self.quit = True


if __name__=='__main__':
    pass
    # left = np.array((-0.12, 0.08, -0.05), float)
    # right = np.array((-0.02, 0.08, -0.05), float)
    # vis = Visualizer([0.75, 1.35, 2.0], left, right)
    # vis.run()