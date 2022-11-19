import rclpy                                                                                         
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

class Echoer(Node):
    def __init__(self):
        super().__init__('map_sub')
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.save_map,
            10)
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.save_scan,
            10)
        self.map_subscription  # prevent unused variable warning
        self.scan_subscription  # prevent unused variable warning
        self.occupancy_grid = None
        self.scan = None

        self.map_publisher = self.create_publisher(OccupancyGrid, '/lab/map', 10)
        self.scan_publisher = self.create_publisher(LaserScan, '/lab/scan', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def save_map(self,msg):
        if self.occupancy_grid is None:
            self.occupancy_grid = msg
            self.get_logger().info('Got the map')

    def save_scan(self,msg):
        if self.scan is None:
            self.scan = msg
            self.get_logger().info('Got the scan')

    def timer_callback(self):
        if self.occupancy_grid is None:
            self.get_logger().info('Waiting for the map')
            return
        if self.scan is None:
            self.get_logger().info('Waiting for the scan')
            return
        self.map_publisher.publish(self.occupancy_grid)
        self.scan_publisher.publish(self.scan)

def main(args=None):
    rclpy.init(args=args)

    echoer = Echoer()

    rclpy.spin(echoer)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

