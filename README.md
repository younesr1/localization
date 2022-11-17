build and run the node colcon with build && . install/setup.bash && ros2 run localization node

run the map server in one window with ros2 run nav2_map_server map_server --ros-args --params-file launch/map_server_params.yaml

step through the lifecucle state machine in other terminal with ros2 lifecycle set /map_server configure && ros2 lifecycle set /map_server activate


open up rviz2 and look at the map and and pose arrays. map topic is /map. pose array topic is /lab/poses
