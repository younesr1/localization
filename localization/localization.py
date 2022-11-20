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

# i created another node called the echoer that constantly echoes the map
# and the same laser scan over and over again
# i also define the the center and res of the map
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
        # all these constants come from the yaml file
        self.resolution = 0.03
        self.x_center = -1.94
        self.y_center = -8.63
        self.width = 356
        self.height = 457
        self.scan = None

    # this converts the map sent over the /lab/map topic
    # to a numpy array representing occupancy in cartesian coordinates
    # so its a list of coordinates that are occupied in the map frame
    def map_to_cartesian(self,msg):
        data = msg.data
        # assert the map is a binary map
        assert all([d in [0,100] for d in data])
        mat = np.reshape(data,(-1,self.width))
        assert mat.shape == (self.height, self.width)
        # keep the obstacles
        map_obstacles = np.argwhere(mat>0)
        # scale map from pixels to meters
        map_obstacles = map_obstacles*self.resolution
        map_obstacles = np.fliplr(map_obstacles)
        # shift the map s.t the cetner is the map frame
        map_obstacles[:,0] += self.x_center
        map_obstacles[:,1] += self.y_center
        return map_obstacles

    # returns the laser scan in the sensor frame
    def scan_to_cartesian(self,scan):
        #assert all(scan.range_min <= d <= scan.range_max or math.isinf(d) for d in scan.ranges)
        assert all(0 <= d <= scan.range_max or math.isinf(d) for d in scan.ranges)
        # polar to rect form
        ret = np.array([[r*np.cos(scan.angle_increment*i),r*np.sin(scan.angle_increment*i)] for i,r in enumerate(scan.ranges)])
        assert ret.shape == (len(scan.ranges),2)
        # remove all the 'inf' values since theyre essentailly values where the scan didnt find anything
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
        # this ensures we echo the same scan over and over and that it never changes
        if self.scan is None:
            self.scan = self.scan_to_cartesian(msg)
            self.get_logger().info('Got the scan')

# this class generates poses
class PoseGenerator(Node):
    def __init__(self,occupancy_grid,scan):
        super().__init__('pose_generator')
        # store the occupancy grid and scan
        self.occupancy_grid = occupancy_grid
        self.scan = scan
        self.publisher_ = self.create_publisher(PoseArray, '/lab/poses', 10)

        # these params were recommended by the TA
        self.population = 5000
        self.lidar_standard_deviation = 0.2
        self.iterations = 100
        # these were empirically foudn. essentially bounds around the real maze
        # we start by generating a random population just on the real maze, not the entire map
        x_min = -2
        x_max = 0
        y_min = -3
        y_max = 1
        yaw_min = 0
        yaw_max = 2*np.pi
        # this is where we store the list of possible poses
        self.poses = []
        # generate many random samples within the maze only
        for _ in range(self.population):
            x_rand = np.random.uniform(x_min,x_max)
            y_rand = np.random.uniform(y_min,y_max)
            yaw_rand = np.random.uniform(yaw_min,yaw_max)
            p = Point(x=x_rand, y=y_rand, z=0.0)
            q = self.yaw_to_quaternion(yaw_rand)
            self.poses.append(Pose(position=p,orientation=q))
        '''p = Point(x=-1.9986503222697123, y=0.22951129057694164, z=0.0)
        q = Quaternion(x=0.0, y=0.0, z=0.999807757390197, w=-0.019607352253300746)
        self.poses.append(Pose(position=p,orientation=q))

        p = Point(x=-0.2,y= -1.9, z=0.000)
        q = self.yaw_to_quaternion(np.radians(130))
        self.poses.append(Pose(position=p,orientation=q))'''

        # this will let us compare our scans to our map
        self.kdt=KDTree(occupancy_grid)

    def yaw_to_quaternion(self,yaw):
        # this converts a yaw in rad to a quat
        assert 0 <= yaw <= 2*np.pi
        q = R.from_euler('z', yaw).as_quat()
        return Quaternion(x=q[0],y=q[1],z=q[2],w=q[3])

    def quaternion_to_yaw(self,quat):
        # this converts a quat to a yaw in rad
        x,y,z = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_euler('xyz',degrees=False)
        assert x == y == 0 
        return z

    def publish(self):
        # this publishes the list of possible poses
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = self.poses
        self.publisher_.publish(msg)
        self.get_logger().info("publishing pose array")

    # this converts a cartesian scan from the sensor frame to the map frame
    def scan_to_map_frame(self,pose,scan):
        yaw = self.quaternion_to_yaw(pose.orientation)
        x_hat = pose.position.x
        y_hat = pose.position.y
        assert pose.position.z == 0
        # i derived this expression by writing out the homogenous transformation for a general 2D transform
        # by hand and then just filled in the constants
        # Really, this is the transfrom from the sensor frame to the map frame
        # we know the distance from the sensor frame to the map frame, its just the position of the particle of interest
        # we also know the orientation of the sensor frame to the map frame, its just the orientation of the particle
        # we apply this transformation for each point in the scan (note we deleted all 'inf'
        ret = np.array([[a*np.cos(yaw)-b*np.sin(yaw)+x_hat, a*np.sin(yaw)+b*np.cos(yaw)+y_hat] for a,b in scan])
        assert len(ret) == len(scan)
        return ret


    def update(self):
        assert len(self.poses) == self.population
        weights = []
        for pose in self.poses:
            # for each of the poses we are considering, we take the scan and transform it from the pose to the map frame
            map_frame_scan = self.scan_to_map_frame(pose,self.scan)
            # next use NN algo to determine how similar the scan in the map frame is to the map we got
            # if the pose is very close to our actual pose, then the 2 will be very similar which means the distance is small
            # so a small distance is good
            distances=self.kdt.query(map_frame_scan, k=1)[0][:]
            # next we convert the distances to weights. by raising each distance to the negative exponential,
            # we make distance have a high weight. we also normalize by the std deviation of the sensor
            # we also square to make it more disparate
            # we sum over all the distance to get a sense of the goodness of the scan
            weight= np.sum(np.exp(-(distances**2)/(2*self.lidar_standard_deviation**2)))
            weights.append(weight)
        assert len(weights) == self.population
        # now that we have all weights we normalize.
        # each weight is now between 0 and 1
        weights = np.array(weights)/sum(weights)
        assert all(0<=w<=1 for w in weights)
        #plt.hist(weights)
        #plt.show()
        # younes todo delete this. just debugging
        '''fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(weights)
        print(self.poses)
        for pose in self.poses:
            temp = self.scan_to_map_frame(pose,self.scan)
            ax1.scatter(temp[:,0],temp[:,1],s=0.5,c='b',marker='o',label='scan')
        ax1.scatter(self.occupancy_grid[:,0],self.occupancy_grid[:,1],s=0.5,c='r',marker='o',label='map')
        plt.legend(loc='upper left')
        plt.show()'''
        assert len(self.poses) == len(weights) == self.population
        # finally we randomly resample based on the weights.
        # the weights are probabilities. so higher weights are more likely to be selected
        self.poses = np.random.choice(self.poses,size=self.population,replace=True,p=weights).tolist()
        assert type(self.poses) is list

    def run(self):
        self.get_logger().info("waiting for laser scan")
        self.publish()
        np.set_printoptions(threshold=np.inf)
        # just loop for the iterations. publish the pose array every time u update
        # it usually take > 15 mins to converge to something clear
        for it in range(self.iterations):
            self.get_logger().info(f"iteration {it}")
            print(self.poses)
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
