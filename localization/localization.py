import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray,Pose,Point,Quaternion
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.neighbors import KDTree

class DataSub(Node):
    def __init__(self):
        super().__init__('data_sub')
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/lab/map',
            self.map_cb,
            10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/lab/scan',
            self.scan_cb,
            10)
        self.map_sub # prevent unused variable warning
        self.scan_sub # prevent unused variable warning
        self.occupancy_grid = None
        self.scan= None
        self.resolution = 0.03
        self.x_center = -1.94
        self.y_center = -8.63
        self.width = 356
        self.height = 457
        self.scan = None

    def map_to_cartesian(self,msg):
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

    # returns the laser scan in the sensor frame
    def scan_to_cartesian(self,scan):
        #assert all(scan.range_min <= d <= scan.range_max or math.isinf(d) for d in scan.ranges)
        assert all(0 <= d <= scan.range_max or math.isinf(d) for d in scan.ranges)
        ret = np.array([[r*np.cos(scan.angle_increment*i),r*np.sin(scan.angle_increment*i)] for i,r in enumerate(scan.ranges)])
        assert ret.shape == (len(scan.ranges),2)
        ret = ret[~np.isinf(ret).any(axis=1)]
        assert all(~np.isinf(ret).any(axis=1))
        return ret

    def map_cb(self, msg):
        # the map is initially just a 1d array of 0s and 100s
        # we convert this 1d array to a list of cartesian coordinates in the odom frame
        # each coordinate represents an occupied cell
        if self.occupancy_grid is None:
            self.occupancy_grid = self.map_to_cartesian(msg)
            self.get_logger().info('Got the map')

    def scan_cb(self, msg):
        if self.scan is None:
            self.scan = self.scan_to_cartesian(msg)
            self.get_logger().info('Got the scan')

class PoseGenerator(Node):
    def __init__(self,occupancy_grid,scan):
        super().__init__('pose_generator')
        self.occupancy_grid = occupancy_grid
        self.scan = scan
        self.publisher_ = self.create_publisher(PoseArray, '/lab/poses', 10)

        self.population = 2 #5000
        self.lidar_standard_deviation = 0.2
        self.iterations = 1
        x_min = -2
        x_max = 0
        y_min = -3
        y_max = 1
        yaw_min = 0
        yaw_max = 2*np.pi
        self.poses = []
        '''for _ in range(self.population):
            x_rand = np.random.uniform(x_min,x_max)
            y_rand = np.random.uniform(y_min,y_max)
            yaw_rand = np.random.uniform(yaw_min,yaw_max)
            p = Point(x=x_rand, y=y_rand, z=0.0)
            q = self.yaw_to_quaternion(yaw_rand)
            self.poses.append(Pose(position=p,orientation=q))'''
        p = Point(x=-1.9986503222697123, y=0.22951129057694164, z=0.0)
        q = Quaternion(x=0.0, y=0.0, z=0.999807757390197, w=-0.019607352253300746)
        self.poses.append(Pose(position=p,orientation=q))

        p = Point(x=-0.2,y= -1.9, z=0.000)
        q = self.yaw_to_quaternion(np.radians(130))
        self.poses.append(Pose(position=p,orientation=q))

        self.kdt=KDTree(occupancy_grid)

    def yaw_to_quaternion(self,yaw):
        # younes todo uncomment assert 0 <= yaw <= 2*np.pi
        q = R.from_euler('z', yaw).as_quat()
        return Quaternion(x=q[0],y=q[1],z=q[2],w=q[3])

    def quaternion_to_yaw(self,quat):
        x,y,z = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_euler('xyz',degrees=False)
        # younes todo uncomment assert x == y == 0 
        return z

    def publish(self):
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = self.poses
        self.publisher_.publish(msg)
        self.get_logger().info("publishing pose array")

    def scan_to_map_frame(self,pose,scan):
        yaw = self.quaternion_to_yaw(pose.orientation)
        x_hat = pose.position.x
        y_hat = pose.position.y
        assert pose.position.z == 0
        ret = np.array([[a*np.cos(yaw)-b*np.sin(yaw)+x_hat, a*np.sin(yaw)+b*np.cos(yaw)+y_hat] for a,b in scan])
        assert len(ret) == len(scan)
        return ret


    def update(self):
        assert len(self.poses) == self.population
        weights = []
        for pose in self.poses:
            map_frame_scan = self.scan_to_map_frame(pose,self.scan)
            distances=self.kdt.query(map_frame_scan, k=1)[0][:]
            weight= np.sum(np.exp(-(distances**2)/(2*self.lidar_standard_deviation**2)))
            weights.append(weight)
        assert len(weights) == self.population
        weights = np.array(weights)/sum(weights)
        assert all(0<=w<=1 for w in weights)
        #plt.hist(weights)
        #plt.show()
        # younes todo delete this. just debugging
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(weights)
        print(self.poses)
        for pose in self.poses:
            temp = self.scan_to_map_frame(pose,self.scan)
            ax1.scatter(temp[:,0],temp[:,1],s=0.5,c='b',marker='o',label='scan')
        ax1.scatter(self.occupancy_grid[:,0],self.occupancy_grid[:,1],s=0.5,c='r',marker='o',label='map')
        plt.legend(loc='upper left')
        plt.show()
        assert len(self.poses) == len(weights) == self.population
        self.poses = np.random.choice(self.poses,size=self.population,replace=True,p=weights).tolist()
        assert type(self.poses) is list

    def run(self):
        self.get_logger().info("waiting for laser scan")
        self.publish()
        for _ in range(self.iterations):
            self.update()
            self.publish()

def main(args=None):
    rclpy.init(args=args)

    data_sub = DataSub()

    data_sub.get_logger().info('waiting for map')
    rclpy.spin_once(data_sub)
    rclpy.spin_once(data_sub)
    assert data_sub.occupancy_grid is not None
    assert data_sub.scan is not None
    #plt.scatter(data_sub.occupancy_grid[:,0],data_sub.occupancy_grid[:,1],s=0.5)
    #plt.show()

    pose_pub = PoseGenerator(data_sub.occupancy_grid, data_sub.scan)
    pose_pub.run()

    data_sub.get_logger().info(f'Terminating')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    data_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
