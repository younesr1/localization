import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray,Pose,Point,Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
from matplotlib import pyplot as plt

class MapSub(Node):
    def __init__(self):
        super().__init__('map_sub')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.occupancy_grid = None
        self.resolution = 0.03
        self.x_center = -1.94
        self.y_center = -8.63
        self.width = 356
        self.height = 457
        

    def to_cartesian(self,msg):
        data = msg.data
        assert all([d in [0,100] for d in data])
        mat = np.reshape(data,(-1,self.width))
        assert mat.shape == (self.height, self.width)
        map_obstacles = np.argwhere(mat>0)
        map_obstacles = map_obstacles*self.resolution
        map_obstacles = np.fliplr(map_obstacles)
        map_obstacles[:,0] += self.x_center
        map_obstacles[:,1] += self.y_center
        return map_obstacles

    def listener_callback(self, msg):
        # the map is initially just a 1d array of 0s and 100s
        # we convert this 1d array to a list of cartesian coordinates in the odom frame
        # each coordinate represents an occupied cell
        self.occupancy_grid = self.to_cartesian(msg)
        self.get_logger().info('Got the map')

class PoseGenerator(Node):
    def __init__(self):
        super().__init__('pose_generator')
        self.publisher_ = self.create_publisher(PoseArray, '/lab/poses', 10)

    def generate_poses(self):
        population = 500
        x_min = -2
        x_max = 0
        y_min = -3
        y_max = 1
        yaw_min = 0
        yaw_max = 2*np.pi
        ret = []
        for _ in range(population):
            x_rand = np.random.uniform(x_min,x_max)
            y_rand = np.random.uniform(y_min,y_max)
            yaw_rand = np.random.uniform(yaw_min,yaw_max)
            p = Point(x=x_rand, y=y_rand, z=0.0)
            q = self.yaw_to_quaternion(yaw_rand)
            ret.append(Pose(position=p,orientation=q))
        return ret

    def yaw_to_quaternion(self,yaw):
        assert 0 <= yaw <= 2*np.pi
        q = R.from_euler('z', yaw).as_quat()
        return Quaternion(x=q[0],y=q[1],z=q[2],w=q[3])

    def pub(self):
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = self.generate_poses()
        self.publisher_.publish(msg)
        self.get_logger().info("publishing pose array")

    def run(self):
        for _ in range(1):
            self.pub()

def main(args=None):
    rclpy.init(args=args)

    map_sub = MapSub()

    map_sub.get_logger().info('waiting for map')
    rclpy.spin_once(map_sub)
    assert map_sub.occupancy_grid is not None
    #plt.scatter(map_sub.occupancy_grid[:,0],map_sub.occupancy_grid[:,1],s=0.5)
    #plt.show()

    pose_pub = PoseGenerator()
    pose_pub.run()

    map_sub.get_logger().info(f'Terminating')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    map_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
