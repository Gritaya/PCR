"""graph_slam controller."""

from controller import Robot, PositionSensor, Display
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random


# ─── Robot Setup: Our own robot ─────────────────────────────────────────────────────────────

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Wheels / Odometry
lps = robot.getDevice('ps_1')
rps = robot.getDevice('ps_2')
lps.enable(timestep)
rps.enable(timestep)

leftMotor = robot.getDevice('motor1')
rightMotor = robot.getDevice('motor2')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0)
rightMotor.setVelocity(0)

WHEEL_RADIUS = 0.1  # Standard e-puck radius .. changed to our robot
WHEEL_BASE = 0.3      # Standard e-puck base .. changed o our robot
MAX_SPEED = 6.28
FORWARD = 1 * MAX_SPEED
TURN = 0.3 * MAX_SPEED
ASSOCIATE_DISTANCE = 0.05

# Lidar
lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()
num_beams = lidar.getHorizontalResolution()
fov = lidar.getFov()
angle_step = fov / num_beams

# Display
display = robot.getDevice("display")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def world_to_display(x, y, origin_x, origin_y, scale):
    px = int(origin_x - x * scale)
    py = int(origin_y - y * scale)  # flip Y
    return px, py

# ─── Odometry Update ─────────────────────────────────────────────────────────

def odometry_update(current_x, current_y, current_theta, prev_ps):
    left_val = lps.getValue()
    right_val = rps.getValue()

    # If sensors aren't ready, don't update
    if math.isnan(left_val) or math.isnan(right_val):
        return (current_x, current_y, current_theta), prev_ps, [0, 0], 0.0

    dl = (left_val - prev_ps[0]) * WHEEL_RADIUS
    dr = (right_val - prev_ps[1]) * WHEEL_RADIUS
    d_center = (dl + dr) / 2
    dtheta = (dr - dl) / WHEEL_BASE

    dx = d_center * np.cos(current_theta + dtheta / 2)
    dy = d_center * np.sin(current_theta + dtheta / 2)

    current_x += dx
    current_y += dy
    current_theta = current_theta + dtheta # it was wrapped angle

    prev_ps = (left_val, right_val)
    return (current_x, current_y, current_theta), prev_ps, [dx, dy], dtheta


## -------------------- Not yet worked here ----------------------------------------
def frontier_exploration():
# At each step: robot scans map --> robot add possible path --> robot uses dijkstra to select most optimal path


# relevane metrics: distance to start position (how far we've been), current2unexplored(how sooner we get new info?)
# graph in nodes: find shortest path using Dkjkastra/A*
# proximity to other unexplored positions (Manhattan to the rims)
# Level of encroacment: BFS/DFS
# frontier: how to decide (amount we'll gain, how long it will take us): sum of info + nearest in highest scoring region

    pass
# 2. Find frontiers in the occupancy grid
# frontiers = get_frontier_centroids(grid)

# 3. Find the 'closest' frontier to the robot (or most interesting)
# best_frontier = frontiers[0] 

# 4. Pathfind using Dijkstra from current_pose to the node nearest best_frontier
# path = dijkstra(graph, current_pose, best_frontier)



# ─── GraphSLAM Class ──────────────────────────────────────────────────────────

class GraphSLAM:
    def __init__(self):
        self.poses = []
        self.landmarks = []
        self.motion_edges = []
        self.obs_edges = []
        self.descriptors = []
        self.loop_edges = []

    def add_pose(self, x, y, theta):
        self.poses.append(np.array([x, y, theta]))
        return len(self.poses) - 1

    def add_motion_edge(self, i, j, delta_x, delta_y, delta_theta, noise_std=(0.05,0.05,0.05)):
        sigma = np.diag([noise_std[0]**2, noise_std[1]**2, noise_std[2]**2])
        omega = np.linalg.inv(sigma)
        delta = np.array([delta_x, delta_y, delta_theta])
        self.motion_edges.append((i, j, delta, omega))

    def add_observation_edge(self, pose_i, lm_j, dist, angle, noise_std=(0.05,0.02)):
        sigma = np.diag([noise_std[0]**2, noise_std[1]**2])
        omega = np.linalg.inv(sigma)
        z = np.array([dist, angle])
        self.obs_edges.append((pose_i, lm_j, z, omega))

    def add_loop_edge(self, i, j, delta_x, delta_y, delta_theta, noise_std=(0.05,0.05,0.02)):
        sigma = np.diag([noise_std[0]**2, noise_std[1]**2, noise_std[2]**2])
        omega = np.linalg.inv(sigma)
        delta = np.array([delta_x, delta_y, delta_theta])
        self.loop_edges.append((i, j, delta, omega))

    # to calculate backend
    # xi is x,y,theta
    def motion_error(self, xi, xj, delta):
        dx = xj[0] - xi[0]
        dy = xj[1] - xi[1]
        dtheta = wrap_angle(xj[2] - xi[2])
        predicted = np.array([
            np.cos(xi[2]) * dx + np.sin(xi[2]) * dy,
            -np.sin(xi[2]) * dx + np.cos(xi[2]) * dy,
            dtheta
        ])
        return delta - predicted
    
    # got position of pose, lm  and measurement from lidar
    def observation_error(self, xi, zj, measurement):
        dx = zj[0] - xi[0]
        dy = zj[1] - xi[1]
        predicted_dist = np.sqrt(dx**2 + dy**2)
        predicted_angle = wrap_angle(np.arctan2(dy, dx) - xi[2])
        predicted = np.array([predicted_dist, predicted_angle])
        return measurement - predicted

    def total_cost(self, state_vec):
        # print("In total cost...")
        
        n_poses = len(self.poses)
        n_lm = len(self.landmarks)
       
        #print(f"poses: {self.poses}\nlen:{n_poses}")
        #print(f"landmarks: {self.landmarks}\nlen:{n_lm}")
        
        poses_flat = state_vec[:n_poses*3].reshape(n_poses,3)
        lm_flat = state_vec[n_poses*3:].reshape(n_lm,2)
        cost = 0.0
        for (i,j,delta,omega) in self.motion_edges:
            e = self.motion_error(poses_flat[i], poses_flat[j], delta)
            cost += e @ omega @ e
        for (pi, li, meas, omega) in self.obs_edges: # position i, landmark i, (dist, theta), omega
            e = self.observation_error(poses_flat[pi], lm_flat[li], meas)
            cost += e @ omega @ e
        for (i,j,delta,omega) in self.loop_edges:
            e = self.motion_error(poses_flat[i], poses_flat[j], delta)
            cost += e @ omega @ e
        return cost

    def prune(self, correspondence_threshold=0.5):
        # remove all poses that contain no landmark information
        # prune redundant landmark
        # should be able to merge by kalman filter but nah, not here pls
        if not self.poses: return 
        
        # keeping only poses that contain landmark(s)
        observed_poses = set(pi for (pi,_,_,_) in self.obs_edges)
        observed_poses.add(0)
        observed_poses.add(len(self.poses)-1)
        keep_pose = sorted(observed_poses)
        
        # just for referencing index
        remap_pose = {old:new for new,old in enumerate(keep_pose)}
        
        self.poses = [self.poses[i] for i in keep_pose]
        self.motion_edges = [(remap_pose[i], remap_pose[j], d, o)
                             for (i,j,d,o) in self.motion_edges if i in remap_pose and j in remap_pose]
        self.obs_edges = [(remap_pose[pi], li, m, o)
                          for (pi,li,m,o) in self.obs_edges if pi in remap_pose]

        if len(self.landmarks)<2: return

        lm_array = np.array(self.landmarks)
        dists = cdist(lm_array, lm_array)
        np.fill_diagonal(dists, np.inf)
        merge_map = list(range(len(self.landmarks)))
        
        # remove redundant landmarks
        for i in range(len(self.landmarks)):
            for j in range(i+1,len(self.landmarks)):
                if dists[i,j] < correspondence_threshold:
                    merge_map[j] = merge_map[i]

        self.obs_edges = [(pi, merge_map[li], m, o) for (pi,li,m,o) in self.obs_edges]
        canonical = sorted(set(merge_map))
        remap_lm = {old:new for new,old in enumerate(canonical)}
        self.landmarks = [self.landmarks[i] for i in canonical]
        self.obs_edges = [(pi, remap_lm[li], m, o) for (pi,li,m,o) in self.obs_edges]

    def optimize(self):
        #print(f"poses: {self.poses}")
        #print(f"landmarks: {self.landmarks}")
        n_poses = len(self.poses)
        n_lm = len(self.landmarks)
        if n_poses == 0: return {'converged': False}
        x0 = np.concatenate([np.array(self.poses).flatten(),
                             np.array(self.landmarks).flatten() if n_lm>0 else []])
        fixed = self.poses[0].copy()
        def cost_with_anchor(state_vec):
            state_vec[:3] = fixed
            return self.total_cost(state_vec)
        result = minimize(cost_with_anchor, x0, method='L-BFGS-B',
                          options={'maxiter':500,'ftol':1e-9})
        optimized = result.x
        opt_poses = optimized[:n_poses*3].reshape(n_poses,3)
        opt_lm = optimized[n_poses*3:].reshape(n_lm,2) if n_lm>0 else []
        self.poses = [opt_poses[i] for i in range(n_poses)]
        self.landmarks = [opt_lm[i] for i in range(n_lm)]
        return {'trajectory':opt_poses, 'landmarks':opt_lm,
                'final_cost':result.fun, 'converged':result.success}
                

# ─── Display Class ───────────────────────────────────────────────────────────

class DisplaySLAM:
    def __init__(self, display):
        self.display = display
        self.width = display.getWidth()
        self.height = display.getHeight()
        self.origin_x = self.width // 2
        self.origin_y = self.height // 2
        self.scale = 10 # Increased scale to enlarge

    def clear(self):
        self.display.setColor(0x1e1e2e)
        self.display.fillRectangle(0, 0, self.width, self.height)

    def draw_robot(self, x, y, theta): 
        px, py = world_to_display(x, y, self.origin_x, self.origin_y, self.scale)
        self.display.setColor(0x5DCAA5) # green
        self.display.drawPixel(px, py)
        hx = x + 0.05 * np.cos(theta)
        hy = y + 0.05 * np.sin(theta)
        hpx, hpy = world_to_display(hx, hy, self.origin_x, self.origin_y, self.scale)
        self.display.drawLine(px, py, hpx, hpy)
        
    def draw_trajectory(self, poses):
        self.display.setColor(0x7F77DD) # purple
        for i in range(1, len(poses)):
            x1, y1 = poses[i-1][:2]
            x2, y2 = poses[i][:2]
            p1 = world_to_display(x1, y1, self.origin_x, self.origin_y, self.scale)
            p2 = world_to_display(x2, y2, self.origin_x, self.origin_y, self.scale)
            self.display.drawLine(p1[0], p1[1], p2[0], p2[1])
            
    def draw_observation_edges(self, slam):
        self.display.setColor(0x44475A)  # dark grey lines
        for (pose_i, lm_j, z, omega) in slam.obs_edges:
            # Guard: check indices are valid
            if pose_i >= len(slam.poses) or lm_j >= len(slam.landmarks):
                continue
            
            rx, ry = slam.poses[pose_i][:2]
            lx, ly = slam.landmarks[lm_j][:2]
            
            p1 = world_to_display(rx, ry, self.origin_x, self.origin_y, self.scale)
            p2 = world_to_display(lx, ly, self.origin_x, self.origin_y, self.scale)
            self.display.drawLine(p1[0], p1[1], p2[0], p2[1])

    def draw_landmarks(self, landmarks):
        self.display.setColor(0xEF9F27) # yellow
        for (x, y) in landmarks:
            px, py = world_to_display(x, y, self.origin_x, self.origin_y, self.scale)
            # self.display.fillOval(px-2, py-2, 4, 4)
            self.display.drawPixel(px, py)
            
    # def draw_lidar(self, rx, ry, hits):
        # self.display.setColor(0xD85A30)
        # rpx, rpy = world_to_display(rx, ry, self.origin_x, self.origin_y, self.scale)
        # for (lx, ly) in hits:
            # px, py = world_to_display(lx, ly, self.origin_x, self.origin_y, self.scale)
            # self.display.drawLine(rpx, rpy, px, py)

    def update(self, slam, x, y, theta, lidar_hits):
        self.clear()
    
        # Draw ONLY the confirmed landmarks stored in the SLAM graph
        if len(slam.landmarks) > 0:
            self.draw_observation_edges(slam)
            self.draw_landmarks(slam.landmarks)
        
        # Draw the history of where the robot has been (pose graph)
        if len(slam.poses) > 1:
            self.draw_trajectory(slam.poses)    
        # if lidar_hits:
        #     self.draw_lidar(x, y, lidar_hits)
        
        # Draw the robot on top
        self.draw_robot(x, y, theta)


# Returning index of landmarks, working with prune function
def associate_landmark(slam, lm_x, lm_y):
    slam.landmarks.append([lm_x, lm_y])
    return len(slam.landmarks) -1

def compute_similarity(desc1, desc2):
    return np.dot(desc1, desc2)  # cosine similarity

def compute_scan_descriptor(range_image, num_bins=20):
    ranges = np.array(range_image)
    ranges = ranges[np.isfinite(ranges)]  # remove inf

    if len(ranges) == 0:
        return np.zeros(num_bins)

    hist, _ = np.histogram(ranges, bins=num_bins, range=(0, np.max(ranges)))
    hist = hist / (np.linalg.norm(hist) + 1e-6)  # normalize

    return hist

def detect_loop_closure(slam, current_pose_id,
                        distance_thresh=0.3,
                        similarity_thresh=0.8,
                        min_separation=30):

    current_pose = slam.poses[current_pose_id]
    print(f"current pose id: {current_pose_id}")
    current_desc = slam.descriptors[current_pose_id]

    for i in range(0, current_pose_id - min_separation):
        past_pose = slam.poses[i]

        # 1. Distance check
        dx = current_pose[0] - past_pose[0]
        dy = current_pose[1] - past_pose[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist > distance_thresh:
            continue

        # 2. Similarity check
        past_desc = slam.descriptors[i]
        sim = compute_similarity(current_desc, past_desc)

        if sim < similarity_thresh:
            continue

        # 3. Valid loop closure
        dtheta = wrap_angle(current_pose[2] - past_pose[2])
        return i, dx, dy, dtheta

    return None
    

# ─── Main Loop ──────────────────────────────────────────────────────────────

def run_robot(robot):
    slam = GraphSLAM()
    viz = DisplaySLAM(display)
    
    # FIX: Step once so sensors initialize
    robot.step(timestep)
    
    current_x, current_y, current_theta = 0.0, 0.0, 0.0
    prev_ps = (lps.getValue(), rps.getValue())
    viz = DisplaySLAM(display)

    VIZ_EVERY = 5 # visualize frequency
    step_count = 0

    # pose_id = slam.add_pose(current_x, current_y, current_theta)
    #print(f"pose_id: {pose_id}")
    # print(f"slam.descriptors: {slam.descriptors}")

    action = "fffffff" # for hard_walk: f = forward, r = right, l = left
    timer = 0
    pose_id = 0
    TIMESTEP = 5
    
    # action = "forward" # for random_walk
    timer = 0
    step_count = 5 # MAX_STEPS
    
    def forward():
        leftMotor.setVelocity(FORWARD)
        rightMotor.setVelocity(FORWARD)
        return
    
    def left():
        leftMotor.setVelocity(-TURN)
        rightMotor.setVelocity(TURN)
        return
    
    def right():
        leftMotor.setVelocity(TURN)
        rightMotor.setVelocity(-TURN) 
        return
    
    def random_walk(action, timer):
        import random
        new_action = False
    
        if timer <= 0:
            action = random.choice(["forward", "forward", "left", "right"])
            timer = 200
            new_action = True
    
        if action == "forward":
            forward()
        elif action == "left":
            left()
        elif action == "right":
            right()
    
        timer -= 1
        return action, timer, new_action
            
    
    def hard_walk(motion, timer):
        
        if timer <= 0:
            motion = motion[1::]
            timer = 20
        
        if len(motion) == 0:
            return motion, timer
       
        action = motion[0]
        
        if action == 'f':
            forward()
        elif action == 'l':
            left()
        elif action == 'r':
            right()
        
        timer -= 1
        
        return motion, timer
        
    
    while robot.step(TIMESTEP) != -1:
        ## for random walk
        # print("random walk..")
        # action, timer, new_action = random_walk(action, timer)
        
        print("hardcode walking..")
        print(f"action: {action}, {timer}")
        action, timer = hard_walk(action, timer)
        
        if len(action) == 0:
            print("Simulation Stopped")
            break

        # Collecting odometry 
        pose, prev_ps, motion_delta, dtheta = odometry_update(current_x, current_y, current_theta, prev_ps)
        print(pose, prev_ps, motion_delta, dtheta)
        dx, dy = motion_delta
        current_x, current_y, current_theta = pose

        # Add pose & motion edge
        print(f"add pose and motion edge: {current_x, current_y, current_theta}")
        new_pose_id = slam.add_pose(current_x, current_y, current_theta)
        print(f"new_pose_id: {new_pose_id}")
        
        slam.add_motion_edge(pose_id, new_pose_id, dx, dy, dtheta)
        pose_id = new_pose_id
        print(f"pose_id:{pose_id}")
        

        # Lidar scanning
        range_image = lidar.getRangeImage()
        lidar_hits = []
        descriptor = compute_scan_descriptor(range_image)
        slam.descriptors.append(descriptor)
        print("done lidarr")
        # print(f"slam.descriptor after lidar: {slam.descriptors}")
           
        # collect slam information from lidar
        for i, dist in enumerate(range_image):
            if dist == float('inf') or dist > lidar.getMaxRange() or dist < 0.05:
                continue
            angle = -fov/2 + (i / num_beams) * fov
            lm_x = current_x + dist * np.cos(current_theta + angle)
            lm_y = current_y + dist * np.sin(current_theta + angle)
            lm_id = associate_landmark(slam, lm_x, lm_y) # just add and give index
            slam.add_observation_edge(pose_id, lm_id, dist, angle)
            lidar_hits.append((lm_x, lm_y))
        # print(f"pose_id before loop: {pose_id}")
        # print(f"slam.descriptor before loop: {slam.descriptors}")
        
        # when it moves to same position --> do loop closure to help add information to optimization
        print("loop detection..")
        loop = detect_loop_closure(slam, pose_id)
        if loop is not None:
            i, dx, dy, dtheta = loop
            slam.add_loop_edge(i, pose_id, dx, dy, dtheta)
            print(f"Loop closure: {i} ↔ {pose_id}", flush=True)

        step_count += 1
        
        display.setColor(0xFF0000)
        display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
        print(f"Display size: {display.getWidth()} x {display.getHeight()}")
        
        print(f"VIZ | poses={len(slam.poses)} landmarks={len(slam.landmarks)} obs={len(slam.obs_edges)}")

        slam.prune(correspondence_threshold=0.3)
        viz.update(slam, current_x, current_y, current_theta, lidar_hits)
        print(f"Origin: ({viz.origin_x}, {viz.origin_y})")
        print(f"Display size: {viz.width} x {viz.height}")
        print(f"Pose range X: {min(p[0] for p in slam.poses):.2f} to {max(p[0] for p in slam.poses):.2f}")
        print(f"Pose range Y: {min(p[1] for p in slam.poses):.2f} to {max(p[1] for p in slam.poses):.2f}")
        print("\n ///////////////////////////////////////////////////////")
        # result = slam.optimize()
        # if 'converged' in result:
           # print("Converged:", result['converged'], flush=True)
    print("returning values..")
    return slam, viz       
           
        
        
            

if __name__ == "__main__":
    print("Starting GraphSLAM...", flush=True)
    slam, viz = run_robot(robot)
    print(f"poses: {slam.poses},\nlandmarks: {slam.landmarks}")
    print("optimizing...")
    
    # This will takes time (500 iterations)
    # Hyp: It is shown after MANUALLY pausing simulation (not in simulation loop)
    result = slam.optimize()
    if 'converged' in result:
        print("Converged:", result['converged'], flush=True)
    print("Done!")