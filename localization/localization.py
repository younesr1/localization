import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray,Pose,Point,Quaternion
from scipy.spatial.transform import Rotation as R

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

    def listener_callback(self, msg):
        self.occupancy_grid = msg
        self.get_logger().info('Got the map')

class PoseGenerator(Node):

    def __init__(self):
        super().__init__('pose_generator')
        self.publisher_ = self.create_publisher(PoseArray, '/lab/poses', 10)

    def generate_poses(self):
        population = 1000
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3
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
    map_sub.get_logger().info(f'Map Info: {map_sub.occupancy_grid.info}')

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
