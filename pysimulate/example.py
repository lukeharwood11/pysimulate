from agent import Agent, GameControlDriver
from simulation import Simulation, CarTestSimulation
from vehicle import Vehicle, SensorBuilder, Car
from qlearn import QLearningAgent
from os.path import join


def main():
    car = Car(
        driver=None,
        debug=False,
        acceleration_multiplier=.5,
        normalize=True
    )

    simulation = CarTestSimulation(
        debug=False,
        fps=35,  # None means simulation fps is not tracked (Suggested for training)
        num_episodes=None,
        caption="Default Simulation",
        car=car,
        track_offset=(0, 0),
        screen_size=(1400, 800),
        track_size=(1400, 800)
    )
    simulation.set_track_paths(
        join("assets", "track-border.png"),
        join("assets", "track.png"),
        join("assets", "track-rewards.png")
    )
    simulation.set_start_pos(start_pos=(875, 100))
    # Create Sensors
    sb = SensorBuilder(
        sim=simulation,
        depth=500,
        default_value=None,
        color=(255, 0, 0),
        width=2,
        pointer=True
    )
    sensors = sb.generate_sensors(sensor_range=(-90, 90, 5))

    driver_map = {
        'user': GameControlDriver(
            num_inputs=len(sensors),
            num_outputs=car.num_outputs
        ),
        'qlearn': QLearningAgent(
            simulation=simulation,
            alpha=0.01,
            alpha_decay=0.01,
            y=0.90,
            epsilon=.98,
            num_sensors=len(sensors),
            num_actions=car.num_outputs,
            batch_size=64,
            replay_mem_max=400,
            save_after=100,
            load_latest_model=True,
            training_model=False,
            model_path=None,
            train_each_step=False,
            debug=False,
            other_inputs=car.get_external_inputs(),
            timeout=10
        )
    }

    # Change this line to select different drivers
    driver = driver_map['qlearn']
    driver.set_model_dir(join("assets", "models"))

    # sensors = sb.generate_sensors([0])
    # Attach sensors to car
    car.init_sensors(sensors=sensors)
    # Throw driver in the vehicle
    car.driver = driver
    simulation.initialize()
    simulation.simulate()


if __name__ == "__main__":
    main()
