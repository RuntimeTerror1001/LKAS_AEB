#!/usr/bin/env python
import carla
import random
import time
import logging

# Configure logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Ego vehicle spawn point
EGO_SPAWN_POINT = carla.Location(x=109.5, y=-50.0, z=0.2)
SAFE_RADIUS = 50.0  # Minimum distance from ego spawn point

def is_near_ego_spawn(location):
    """Check if location is too close to ego spawn point"""
    return location.distance(EGO_SPAWN_POINT) < SAFE_RADIUS


def spawn_traffic():
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    
    # Get existing ego vehicle
    ego_vehicle = None
    for actor in world.get_actors():
        if 'hero' in actor.attributes.get('role_name', ''):
            ego_vehicle = actor
            break
    
    if not ego_vehicle:
        logging.warning("Ego vehicle not found! Using spawn point protection only")
    
    # Traffic Manager setup (asynchronous mode)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    # No synchronous mode
    
    # ========================
    # VEHICLE SPAWNING (AVOID EGO SPAWN AREA)
    # ========================
    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    
    # Filter spawn points away from ego spawn location
    spawn_points = [pt for pt in spawn_points 
                    if not is_near_ego_spawn(pt.location)]
    
    # Create batch commands
    batch = []
    num_vehicles = min(40, len(spawn_points))
    for _ in range(num_vehicles):
        bp = random.choice(vehicle_bps)
        transform = random.choice(spawn_points)
        batch.append(carla.command.SpawnActor(bp, transform))
        spawn_points.remove(transform)

    # Apply batch command
    vehicle_ids = []
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logging.error(f"Vehicle spawn error: {response.error}")
        else:
            vehicle_ids.append(response.actor_id)
    
    # Set autopilot
    tm_port = traffic_manager.get_port()
    batch = [
        carla.command.SetAutopilot(vid, True, tm_port)
        for vid in vehicle_ids
    ]
    client.apply_batch_sync(batch, True)

    # ========================
    # PEDESTRIAN SPAWNING (AVOID EGO SPAWN AREA)
    # ========================
    walker_bps    = world.get_blueprint_library().filter('walker.pedestrian.*')
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    
    percentagePedestriansRunning = 0.0
    percentagePedestriansCrossing = 0.3
    num_pedestrians = 80
    
    # Generate safe spawn points
    spawn_points = []
    while len(spawn_points) < num_pedestrians:
        loc = world.get_random_location_from_navigation()
        if loc and not is_near_ego_spawn(loc):
            spawn_points.append(carla.Transform(loc))

    # Spawn walkers
    batch = []
    walker_speeds = []
    for pt in spawn_points:
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # speed attribute or default
        if walker_bp.has_attribute('speed'):
            speeds = walker_bp.get_attribute('speed').recommended_values
            walker_speeds.append(speeds[1] if random.random()>percentagePedestriansRunning else speeds[2])
        else:
            walker_speeds.append(1.5)
        batch.append(carla.command.SpawnActor(walker_bp, pt))
    results = client.apply_batch_sync(batch, True)

    walkers = []
    for i, res in enumerate(results):
        if res.error:
            logging.error(f"Walker spawn error: {res.error}")
        else:
            walkers.append({'id': res.actor_id, 'speed': walker_speeds[i]})

    # Spawn controllers
    batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), w['id']) for w in walkers]
    results = client.apply_batch_sync(batch, True)
    for i, res in enumerate(results):
        if res.error:
            logging.error(f"Controller spawn error: {res.error}")
        else:
            walkers[i]['con'] = res.actor_id

    # Initialize controllers
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    all_ids = [c for w in walkers for c in (w['con'], w['id'])]
    all_actors = world.get_actors(all_ids)
    for idx in range(0, len(all_ids), 2):
        con = all_actors[idx]
        walker = all_actors[idx+1]
        con.start()
        con.go_to_location(world.get_random_location_from_navigation())
        con.set_max_speed(float(walkers[idx//2]['speed']))

    logging.info(f"Spawned {len(vehicle_ids)} vehicles and {len(walkers)} pedestrians")
    logging.info(f"Protected ego spawn area at {EGO_SPAWN_POINT} with {SAFE_RADIUS}m radius")

    # ========================
    # MAIN LOOP (async)
    # ========================
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("Destroying traffic actors...")
        # Destroy vehicles
        if vehicle_ids:
            client.apply_batch([carla.command.DestroyActor(v) for v in vehicle_ids])
        # Stop controllers
        for w in walkers:
            if 'con' in w:
                actor = world.get_actor(w['con'])
                if actor:
                    actor.stop()
        # Destroy walkers and controllers
        batch = []
        for w in walkers:
            batch.append(carla.command.DestroyActor(w['con']))
            batch.append(carla.command.DestroyActor(w['id']))
        client.apply_batch_sync(batch, True)

if __name__ == '__main__':
    spawn_traffic()
