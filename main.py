import taichi as ti
ti.init(arch=ti.cpu)

# ===================== 你固定的参数 完全不动 =====================
GRID_W = 20
GRID_H = 20
PARTICLE_NUM = GRID_W * GRID_H
MASS = 6
DT = 0.0022
KS = 10000
KD = 5
GRAVITY = ti.Vector([0.0, -9.8, 0.0])
MAX_VEL = 100.0
REST_LEN = 0.5

FIX_LEFT = 0
FIX_RIGHT = GRID_W - 1

# 物理场
pos = ti.Vector.field(3, ti.f32, PARTICLE_NUM)
pos_predict = ti.Vector.field(3, ti.f32, PARTICLE_NUM)
vel = ti.Vector.field(3, ti.f32, PARTICLE_NUM)
force = ti.Vector.field(3, ti.f32, PARTICLE_NUM)

# ===================== 正确：结构弹簧 = 4 个方向（上下左右） =====================
dirs = ti.Vector.field(2, int, 4)
dirs[0] = (1, 0)
dirs[1] = (0, 1)
dirs[2] = (-1, 0)
dirs[3] = (0, -1)

LINE_MAX = PARTICLE_NUM * 8
line_idx = ti.field(int, LINE_MAX)

# ===================== 初始化 =====================
@ti.kernel
def init_cloth():
    for idx in range(PARTICLE_NUM):
        gx = idx % GRID_W
        gy = idx // GRID_W
        scale = 0.5
        pos[idx] = ti.Vector([gx * REST_LEN, 0.0, gy * REST_LEN])
        pos_predict[idx] = pos[idx]
        vel[idx] = ti.Vector([0.0, 0.0, 0.0])

# ===================== 正确构建结构弹簧网格 =====================
@ti.kernel
def build_spring_line():
    cnt = 0
    for i in range(PARTICLE_NUM):
        gx = i % GRID_W
        gy = i // GRID_W
        for d in ti.static(range(4)):
            dx, dy = dirs[d]
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                j = ny * GRID_W + nx
                if i < j:
                    line_idx[cnt] = i
                    line_idx[cnt + 1] = j
                    cnt += 2

# ===================== 受力计算（正确 4 邻域） =====================
@ti.func
def compute_force(i: int, use_predict: int):
    p_i = pos[i] if use_predict == 0 else pos_predict[i]
    force[i] = MASS * GRAVITY - KD * vel[i]
    gx = i % GRID_W
    gy = i // GRID_W

    for d in ti.static(range(4)):
        dx, dy = dirs[d]
        nx = gx + dx
        ny = gy + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            j = ny * GRID_W + nx
            p_j = pos[j] if use_predict == 0 else pos_predict[j]
            disp = p_i - p_j
            dist = disp.norm()
            if dist > 1e-6:
                f = -KS * (dist - REST_LEN) * (disp / dist)
                ti.atomic_add(force[i], f)

@ti.func
def speed_limit(i):
    v_len = vel[i].norm()
    if v_len > MAX_VEL:
        vel[i] = vel[i].normalized() * MAX_VEL

# ===================== 三种积分器 =====================
@ti.kernel
def explicit_update():
    # 1. 先算旧力
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        compute_force(i, 0)

    # 2. 先用 旧速度 更新 位置！！（显式核心）
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        pos[i] += vel[i] * DT

    # 3. 再更新 速度
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        vel[i] += force[i] / MASS * DT
        speed_limit(i)

@ti.kernel
def semi_update():
    # 1. 算力
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        compute_force(i, 0)

    # 2. 先更新 速度
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        vel[i] += force[i] / MASS * DT

    # 3. 用新速度 更新 位置！（半隐核心）
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        pos[i] += vel[i] * DT
        speed_limit(i)

@ti.kernel
def implicit_update():
    # 1. 预测下一位置
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        pos_predict[i] = pos[i] + vel[i] * DT

    # 2. 用未来位置算力（隐式核心）
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        compute_force(i, 1)

    # 3. 更新速度
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        vel[i] += force[i] / MASS * DT

    # 4. 更新位置
    for i in range(PARTICLE_NUM):
        if i == FIX_LEFT or i == FIX_RIGHT: continue
        pos[i] += vel[i] * DT
        speed_limit(i)

# ===================== 主函数 =====================
def main():
    init_cloth()
    build_spring_line()
    mode = 1
    pause = False

    win = ti.ui.Window("Cloth Simulation", (1100, 720))
    scene = win.get_scene()
    cam = ti.ui.Camera()
    cam.position(5, -5, 30)
    cam.lookat(5, -5, 0)

    while win.running:
        with win.get_gui().sub_window("Control Panel", 0.02, 0.02, 0.26, 0.36):
            if win.get_gui().button("Explicit Euler"):
                mode = 0
                init_cloth()
            if win.get_gui().button("Semi-Implicit"):
                mode = 1
                init_cloth()
            if win.get_gui().button("Implicit Euler"):
                mode = 2
                init_cloth()
            if win.get_gui().button("Pause / Resume"):
                pause = not pause
            if win.get_gui().button("Reset"):
                init_cloth()

        if not pause:
            if mode == 0:
                explicit_update()
            elif mode == 1:
                semi_update()
            else:
                implicit_update()

        scene.set_camera(cam)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light((8, 2, 8), (1, 1, 1))
        scene.lines(pos, 0.5, line_idx, color=(0.7, 0.5, 0.3))
        scene.particles(pos, radius=0.07, color=(1.0, 0.3, 0.3))
        win.get_canvas().scene(scene)
        win.show()

if __name__ == "__main__":
    main()