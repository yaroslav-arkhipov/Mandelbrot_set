import pygame as pg
import numpy as np
import numba

#settings
res = width, height = 800, 450
offset = np.array([1.3 * width, height]) // 2

#texture
texture = pg.image.load('img/img.png')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture)

class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((width, height, 3), [0, 0, 0], dtype=np.uint8)
        self.vel = 0.01
        self.increment = [0, 0]
        self.max_iter = 30
        self.max_iter_limit = 5000
        self.zoom = 2.2 / height
        self.scale = 0.993
        self.app_speed = 1 / 4000
        self.prev_time = pg.time.get_ticks()

    def delta_time(self):
        time_now = pg.time.get_ticks() - self.prev_time
        self.prev_time = time_now
        return time_now * self.app_speed

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(screen_array, max_iter, zoom):
        for x in numba.prange(width):
            for y in numba.prange(height):
                c = (x - offset[0]) * zoom + 1j * (y - offset[1]) * zoom
                z = 0
                num_iter = 0
                for i in range(max_iter):
                    z = z ** 2 + c
                    if z.real ** 2 + z.imag ** 2 > 2:
                        break
                    num_iter += 1
                col = int(texture_size * num_iter / max_iter)
                screen_array[x, y] = texture_array[col, col]
        return screen_array

    def control(self):
        pressed_key = pg.key.get_pressed()
        dt = self.delta_time()
        if pressed_key[pg.K_a]:
            self.increment[0] += self.vel * dt
        if pressed_key[pg.K_d]:
            self.increment[0] -= self.vel * dt
        if pressed_key[pg.K_w]:
            self.increment[1] += self.vel * dt
        if pressed_key[pg.K_w]:
            self.increment[1] -= self.vel * dt

        if pressed_key[pg.K_UP] or pressed_key[pg.K_DOWN]:
            inv_scale = 2 - self.scale
            if pressed_key[pg.K_UP]:
                self.zoom *= self.scale
                self.vel *= self.scale
            if pressed_key[pg.K_DOWN]:
                self.zoom *= inv_scale
                self.vel *= inv_scale

        if pressed_key[pg.K_LEFT]:
            self.max_iter -= 1
        if pressed_key[pg.K_RIGHT]:
            self.max_iter += 1
        self.max_iter = min(max(self.max_iter, 2), self.max_iter_limit)

    def update(self):
        self.screen_array = self.render(self.screen_array, self.max_iter, self.zoom)

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self):
        self.control()
        self.update()
        self.draw()

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps()}')

if __name__ == '__main__':
    app = App()
    app.run()