#!/usr/bin/env python

import carla

def main():
    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        client.load_world_if_different('Town06')
        
        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()
        #print(client.get_available_maps())
        
        spawn_points = world.get_map().get_spawn_points()

        # Draw the spawn point locations as numbers in the map
        print("Drawing spawn points")
        for i, spawn_point in enumerate(spawn_points):
            world.debug.draw_string(spawn_point.location, str(i), life_time=10000)
        print("Starting infinite loop")
        # In synchronous mode, we need to run the simulation to fly the spectator
        while True:
            world.tick()
    except KeyboardInterrupt:
        print('Done')

if __name__ == '__main__':

    main()

    
